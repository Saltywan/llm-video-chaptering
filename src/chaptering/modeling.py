"""Pinned local model paths, native formatting and budgeted joint generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .inference import strip_qwen_terminal_tokens
from .supervision import prepare_causal_example
from .training import prepare_seq2seq_example
from .windowing import build_window_prompt, parse_window_outline

GPU_UUID = "GPU-05f1cfd1-0219-e850-913d-159b6ed72d60"
AUTHORIZED_GPU_UUIDS = {GPU_UUID, "GPU-d7e5651f-45ed-fd48-2bd4-7f38cc45e13b"}  # GPU2: authorized 2026-11-17


def _token_position(sequence: list[int], marker: list[int]) -> int | None:
    return next((i for i in range(len(sequence) - len(marker) + 1)
                 if sequence[i:i + len(marker)] == marker), None)


def model_entry(root: Path, alias: str) -> dict[str, Any]:
    inventory = json.loads((root / "artifacts/runtime/model_inventory.json").read_text())
    item = dict(inventory["models"][alias])
    item["directory"] = item.get("directory", str(root / "models" / alias))
    selected = json.loads((root / "configs/models.json").read_text())["models"][alias]
    if item["revision"] != selected["revision"]:
        raise ValueError("Inventory and selected checkpoint revisions disagree")
    item["architecture"] = selected["architecture"]
    return item


class ChapterModel:
    """Caller must hold the exclusive GPU0 lock for this object's lifetime."""

    def __init__(self, root: Path, alias: str, *, adapter: str | None = None,
                 attention: str | None = None, gemma_final_context: str = "direct_v1"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in AUTHORIZED_GPU_UUIDS:
            raise RuntimeError("Exact authorized GPU UUID mask required (GPU0, or authorized GPU2)")
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device")
        self.root, self.alias, self.torch = root, alias, torch
        self.gemma_final_context = gemma_final_context
        self.entry = model_entry(root, alias)
        self.seq2seq = self.entry["architecture"] == "seq2seq"
        directory = self.entry["directory"]
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        cls = AutoModelForSeq2SeqLM if self.seq2seq else getattr(transformers, config.architectures[0])
        self.attention = attention or ("eager" if self.seq2seq else "sdpa")
        self.model = cls.from_pretrained(directory, local_files_only=True, dtype=torch.bfloat16,
                                         device_map={"": 0}, attn_implementation=self.attention)
        self.encoder_limit = config.encoder.max_position_embeddings if self.seq2seq else config.text_config.max_position_embeddings
        self.decoder_limit = config.decoder.max_position_embeddings if self.seq2seq else self.encoder_limit
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()

    def render(self, prompt: str) -> str:
        kwargs = {} if self.seq2seq else {"enable_thinking": True}
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True, **kwargs)

    def prepare_training(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.torch
        window, target = item["window"], item["target"]
        prompt = build_window_prompt(window)
        kwargs = {"window": window, "masked_target_spans": target["masked_target_spans"]}
        if self.seq2seq:
            prepared = prepare_seq2seq_example(self.tokenizer, prompt, target["target_json"],
                                               encoder_limit=self.encoder_limit, decoder_limit=self.decoder_limit, **kwargs)
        else:
            prepared = prepare_causal_example(self.tokenizer, prompt, target["target_json"], self.alias,
                                               gemma_final_context=self.gemma_final_context,
                                               thought_prefix=item.get("thought_prefix"), **kwargs)
            if len(prepared["input_ids"]) > self.encoder_limit:
                raise ValueError("Causal full training sequence exceeds pinned context")
        if target["masked_target_spans"] and prepared["audit"]["masked_partial_title_tokens"] < 1:
            raise ValueError("Partial titles were not masked by actual tokenizer")
        if not prepared["audit"]["supervision_gate_passed"]:
            raise ValueError("Training supervision gate failed")
        inputs = {name: torch.tensor([prepared[name]], dtype=torch.long, device="cuda:0")
                  for name in ["input_ids", "attention_mask", "labels"]}
        if self.seq2seq:
            full = torch.tensor([prepared["full_target_ids"]], dtype=torch.long, device="cuda:0")
            inputs["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=full)
            inputs["decoder_attention_mask"] = torch.ones_like(inputs["decoder_input_ids"])
            prepared["audit"]["explicit_unmasked_decoder_shift_applied"] = True
        return inputs, prepared["audit"]

    def seq2seq_prompt_length(self, window: dict[str, Any]) -> int:
        """Exact encoder token count of one window's rendered prompt."""
        prompt = self.render(build_window_prompt(window))
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def generate_seq2seq_batch(self, windows: list[dict[str, Any]], *, cap: int,
                               final_cap: int) -> list[dict[str, Any] | None]:
        """Greedy batched generation for seq2seq backbones.

        Generates at most ``cap`` new tokens. A row that has not emitted an end token is
        returned as ``None`` when ``cap < final_cap`` (the caller regenerates it with a
        larger allowance) and as ``truncated`` when ``cap == final_cap``. Parsing and
        statuses are the same strict ones used by ``generate_window``.
        """
        if not self.seq2seq:
            raise RuntimeError("Batched greedy generation is only for seq2seq backbones")
        if cap > final_cap or final_cap + 1 > self.decoder_limit:
            raise ValueError("Decoder generation allowance exceeds supported context")
        from .batching import split_generated
        torch = self.torch
        prompts = [self.render(build_window_prompt(window)) for window in windows]
        self.tokenizer.padding_side = "right"
        encoded = self.tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > self.encoder_limit:
            raise ValueError("Inference input exceeds supported encoder context")
        inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
        eos = self.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (tuple, list)) else [eos]
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        results: list[dict[str, Any] | None] = []
        for window, n, row in zip(windows, lengths, output[:, 1:].tolist()):
            tail, finished = split_generated(row, eos)
            if not finished and cap < final_cap:
                results.append(None)
                continue
            raw = self.tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = {"video_id": window["video_id"], "window_index": window["window_index"],
                      "status": "error", "chapters": [], "error": None,
                      "usage": {"prompt_tokens": n, "completion_tokens": len(tail),
                                "total_tokens": n + len(tail), "reasoning_tokens": 0},
                      "model_revision": self.entry["revision"], "attention": self.attention,
                      "enable_thinking": None, "reasoning_budget": None, "raw_text": raw,
                      "backend": "hf_batched_greedy_v1", "wall_seconds": None,
                      "batch": {"size": len(windows), "cap": cap, "seconds": elapsed}}
            if not finished:
                result.update(status="truncated", error="Total generation budget exhausted")
            else:
                result["final_text"] = raw.strip()
                try:
                    result["chapters"] = parse_window_outline(result["final_text"], window)
                    result["status"] = "ok"
                except (ValueError, TypeError, KeyError) as exc:
                    result.update(status="invalid", error=type(exc).__name__ + ": " + str(exc))
            results.append(result)
        return results

    def generate_window(self, window: dict[str, Any], *, max_new_tokens: int = 4096,
                        reasoning_budget: int = 1024, seed: int = 393) -> dict[str, Any]:
        torch = self.torch
        result = {"video_id": window["video_id"], "window_index": window["window_index"],
                  "status": "error", "chapters": [], "error": None, "usage": {},
                  "model_revision": self.entry["revision"], "attention": self.attention,
                  "enable_thinking": None if self.seq2seq else True,
                  "reasoning_budget": None if self.seq2seq else reasoning_budget}
        started = time.perf_counter()
        try:
            prompt = self.render(build_window_prompt(window))
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            n = inputs["input_ids"].shape[1]
            if n > self.encoder_limit or (not self.seq2seq and n + max_new_tokens > self.decoder_limit):
                raise ValueError("Inference input plus output allowance exceeds supported context")
            if self.seq2seq and max_new_tokens + 1 > self.decoder_limit:
                raise ValueError("Decoder generation allowance exceeds supported context")
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            request_seed = (seed + int(hashlib.sha256(f'{window["video_id"]}:{window["window_index"]}'.encode()).hexdigest()[:8], 16)) % (2**32)
            torch.manual_seed(request_seed)
            self.model.eval()
            kwargs = {"max_new_tokens": max_new_tokens, "do_sample": not self.seq2seq}
            processor = None
            if not self.seq2seq:
                from .reasoning import make_thinking_budget_processor
                processor = make_thinking_budget_processor(self.tokenizer, self.alias, n, reasoning_budget)
                kwargs.update(temperature=0.6, top_p=0.95, top_k=20, logits_processor=[processor])
                tokens = [self.tokenizer.eos_token_id]
                terminal = "<|im_end|>" if self.alias == "M1" else "<turn|>"
                tokens.append(self.tokenizer.convert_tokens_to_ids(terminal))
                endoftext = self.tokenizer.convert_tokens_to_ids("<|endoftext|>") if self.alias == "M1" else None
                if type(endoftext) is int and endoftext != self.tokenizer.unk_token_id:
                    tokens.append(endoftext)
                kwargs["eos_token_id"] = list(set(tokens))
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
            torch.cuda.synchronize()
            tail = output[0, 1:] if self.seq2seq else output[0, n:]
            raw = self.tokenizer.decode(tail, skip_special_tokens=self.seq2seq,
                                        clean_up_tokenization_spaces=False)
            result["raw_text"] = raw
            result["raw_text"] = raw
            result["usage"] = {"prompt_tokens": n, "completion_tokens": tail.numel(),
                               "total_tokens": n + tail.numel(), "reasoning_tokens": 0 if self.seq2seq else None}
            if not self.seq2seq:
                close = "</think>" if self.alias == "M1" else "<channel|>"
                position = _token_position(tail.tolist(), self.tokenizer.encode(close, add_special_tokens=False))
                if position is not None:
                    result["usage"]["reasoning_tokens"] = position
                elif self.alias == "M2":
                    opening = self.tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
                    if tail.tolist()[:len(opening)] != opening:
                        result["usage"]["reasoning_tokens"] = 0
                result["reasoning_token_accounting"] = "generated tokens before native close; includes generated channel opener"
            if processor is not None:
                audit = getattr(processor, "audit", None)
                result["thinking_budget_audit"] = audit() if callable(audit) else audit
            eos = kwargs.get("eos_token_id", self.model.generation_config.eos_token_id)
            eos = eos if isinstance(eos, (tuple, list)) else [eos]
            truncated = tail.numel() >= max_new_tokens and int(tail[-1]) not in eos
            if truncated:
                result.update(status="truncated", error="Total generation budget exhausted")
                return result
            if self.seq2seq:
                final = raw.strip()
            elif self.alias == "M1":
                if "</think>" not in raw:
                    raise ValueError("No native final-answer delimiter")
                reasoning, final = raw.split("</think>", 1)
                final = strip_qwen_terminal_tokens(final)
                result["reasoning_characters"] = len(reasoning)
            else:
                parsed = self.tokenizer.parse_response(raw, prefix=prompt)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                    raise ValueError("Official parser returned no final text")
                final = parsed["content"].strip()
                reasoning = parsed.get("thinking")
                result["reasoning_characters"] = len(reasoning) if isinstance(reasoning, str) else 0
            result["final_text"] = final
            result["chapters"] = parse_window_outline(final, window)
            result["status"] = "ok"
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            result.update(status="invalid" if isinstance(exc, ValueError) else "error",
                          error=type(exc).__name__ + ": " + str(exc))
        finally:
            result["wall_seconds"] = time.perf_counter() - started
        return result
