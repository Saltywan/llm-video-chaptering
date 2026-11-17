#!/usr/bin/env python
"""One synthetic final-outline update with native thinking-enabled inference."""

import argparse
import json
from pathlib import Path
import sys
import time
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.inference import build_prompt, strip_qwen_terminal_tokens
from pilot_seq2seq import GPU_UUID, gpu0_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", choices=["M1", "M2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    item = json.loads((root / "artifacts/runtime/model_inventory.json").read_text())["models"][args.alias]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"alias": args.alias, "revision": item["revision"], "pid": os.getpid(),
              "gpu_uuid": GPU_UUID, "purpose": "synthetic compatibility only; not effectiveness",
              "status": "running", "enable_thinking": True,
              "sentence_count": 4, "native_parser_version": "native-final-v2"}
    started = time.perf_counter()
    try:
        with gpu0_lock():
            import torch
            import transformers
            from transformers import AutoConfig, AutoTokenizer
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
            from chaptering.supervision import prepare_causal_example
            from chaptering.evaluation import parse_outline

            torch.manual_seed(393)
            directory = item["directory"]
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            config = AutoConfig.from_pretrained(directory, local_files_only=True)
            row = {"video_id": "synthetic-c-pilot", "sentences": [
                "To bake bread, mix flour with water and yeast.",
                "Knead the dough, let it rise, then bake it in a hot oven.",
                "Now let us discuss basic bicycle maintenance.",
                "Check tire pressure and lubricate the chain before a long ride."]}
            target = json.dumps({"chapters": [{"start": 0, "title": "Baking bread"},
                                             {"start": 2, "title": "Bicycle maintenance"}]}, separators=(",", ":"))
            prompt = build_prompt(row)
            prepared = prepare_causal_example(tokenizer, prompt, target, args.alias)
            gate = json.loads((root / "artifacts/runtime/causal_supervision.json").read_text())["models"][args.alias]
            if (not prepared["audit"].get("supervision_gate_passed")
                    or not gate.get("supervision_gate_passed") or not gate.get("parser_gate_passed")
                    or gate.get("revision") != item["revision"]):
                raise RuntimeError("Actual pinned template/parser CPU gates must pass before CUDA model loading")
            # This audit must be written and checked before a CUDA model is loaded.
            report["supervision_audit"] = prepared["audit"]
            (args.output / "supervision_audit.json").write_text(json.dumps(prepared["audit"], indent=2))
            generation_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                            tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=True)
            inference_inputs = tokenizer(generation_text, add_special_tokens=False, return_tensors="pt")
            architecture = getattr(transformers, config.architectures[0])
            attention = "sdpa" if getattr(architecture, "_supports_sdpa", False) else "eager"
            phase = time.perf_counter()
            base = architecture.from_pretrained(directory, local_files_only=True, dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation=attention)
            torch.cuda.synchronize()
            report.update(model_class=config.architectures[0], attention_implementation=attention,
                          load_seconds=time.perf_counter() - phase, precision="bfloat16",
                          input_tokens=inference_inputs["input_ids"].shape[1])
            inputs = {key: value.to("cuda:0") for key, value in inference_inputs.items()}
            training = {key: torch.tensor([prepared[key]], dtype=torch.long, device="cuda:0")
                        for key in ["input_ids", "attention_mask", "labels"]}

            def generate(model, phase_name, limit):
                model.eval()
                torch.manual_seed(393)
                phase_start = time.perf_counter()
                with torch.no_grad():
                    result = model.generate(**inputs, do_sample=True, temperature=0.6, top_p=0.95,
                                            top_k=20, max_new_tokens=limit)
                torch.cuda.synchronize()
                tail = result[0, inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                entry = {"generated_tokens": tail.numel(), "seconds": time.perf_counter() - phase_start,
                         "max_new_tokens": limit, "status": "invalid", "reasoning_characters": None}
                if not raw.strip():
                    raise RuntimeError("Generation decoded to empty text")
                if tail.numel() >= limit:
                    entry["status"] = "truncated"
                else:
                    try:
                        if args.alias == "M1":
                            if "</think>" not in raw:
                                raise ValueError("Thinking-enabled output has no final-answer delimiter")
                            reasoning, final = raw.split("</think>", 1)
                            final = strip_qwen_terminal_tokens(final)
                            entry["reasoning_characters"] = len(reasoning)
                            entry["parser"] = "explicit native closing-think delimiter"
                        else:
                            parsed = tokenizer.parse_response(raw, prefix=generation_text)
                            if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                                raise ValueError("Official parser did not return final text")
                            final = parsed["content"].strip()
                            reasoning = parsed.get("thinking")
                            entry["reasoning_characters"] = len(reasoning) if isinstance(reasoning, str) else None
                            entry["parser"] = "pinned official tokenizer.parse_response"
                        entry["final_text"] = final
                        entry["chapters"] = parse_outline(final, len(row["sentences"]))
                        entry["status"] = "ok"
                    except (ValueError, TypeError, KeyError) as exc:
                        entry["error_type"] = type(exc).__name__
                        entry["error"] = str(exc)
                report[phase_name] = entry
                print(json.dumps({"event": phase_name, **{k: v for k, v in entry.items()
                                                         if k not in {"final_text", "chapters"}}}), flush=True)
                return tail.detach().cpu()

            generate(base, "unadapted_generation", args.max_new_tokens)
            model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                                  lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
            report["adapter_module_policy"] = "q_proj/v_proj only for compatibility; not final selection"
            report["adapter_modules"] = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train()
            trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
            phase = time.perf_counter()
            loss = model(**training).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in trainable):
                raise RuntimeError("Nonfinite gradient")
            nonzero = [(name, p) for name, p in trainable if p.grad is not None and torch.count_nonzero(p.grad).item()]
            if not nonzero:
                raise RuntimeError("No nonzero adapter gradients")
            parameter_name, parameter = nonzero[0]
            before = parameter.detach().clone()
            optimizer.step()
            if any(not torch.isfinite(p).all() for _, p in trainable):
                raise RuntimeError("Nonfinite adapter weights")
            changed = torch.count_nonzero(before != parameter).item()
            if not changed:
                raise RuntimeError("No adapter parameter update")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            report.update(loss=loss.item(), one_update_seconds=time.perf_counter() - phase,
                          nonzero_gradient_tensors=len(nonzero), changed_parameter_elements=changed,
                          updated_parameter=parameter_name, trainable_parameters=sum(p.numel() for _, p in trainable))
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            generate(model, "adapted_generation", args.max_new_tokens)
            directory = args.output / "adapter"
            model.save_pretrained(directory)
            saved = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            base = model.unload()
            reloaded = PeftModel.from_pretrained(base, directory, is_trainable=False)
            loaded = get_peft_model_state_dict(reloaded)
            equal = saved.keys() == loaded.keys() and all(torch.equal(v, loaded[k].detach().cpu()) for k, v in saved.items())
            if not equal:
                raise RuntimeError("Saved/reloaded adapter tensors differ")
            report["save_reload_adapter_tensors_equal"] = equal
            generate(reloaded, "reloaded_generation_smoke", 64)
            report.update(status="training_compatibility_pass", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          final_output_gate_passed=all(report[name]["status"] == "ok"
                                                      for name in ["unadapted_generation", "adapted_generation"]))
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
    report["total_seconds"] = time.perf_counter() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "adapter_modules"}), flush=True)
    return 0 if report["status"] == "training_compatibility_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
