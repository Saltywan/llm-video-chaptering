#!/usr/bin/env python
"""Batch native-thinking causal ZS/LoRA inference on the same GPU0 engine.

Use a separate vLLM interpreter read-only. Training and metric evaluation
continue in the project's isolated environment. No services are reconfigured.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.inference import strip_qwen_terminal_tokens
from chaptering.modeling import model_entry, _token_position
from chaptering.windowing import build_window_prompt, parse_window_outline, stitch_window_predictions
from pilot_seq2seq import gpu0_lock


def decode_result(tokenizer, alias, window, rendered, request, revision):
    result = {"video_id": window["video_id"], "window_index": window["window_index"],
              "status": "error", "chapters": [], "error": None, "usage": {},
              "model_revision": revision, "backend": "vllm", "enable_thinking": True,
              "thinking_protocol": "native_unforced_v1", "reasoning_budget": None,
              "wall_seconds": None}
    completion = request.outputs[0]
    tokens = list(completion.token_ids)
    raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result.update(raw_text=raw, finish_reason=completion.finish_reason,
                  stop_reason=completion.stop_reason)
    marker = "</think>" if alias == "M1" else "<channel|>"
    position = _token_position(tokens, tokenizer.encode(marker, add_special_tokens=False))
    reasoning_tokens = position
    if alias == "M2" and position is None and "<|channel>thought" not in raw:
        reasoning_tokens = 0
    result["usage"] = {"prompt_tokens": len(request.prompt_token_ids),
                       "completion_tokens": len(tokens),
                       "total_tokens": len(request.prompt_token_ids) + len(tokens),
                       "reasoning_tokens": reasoning_tokens}
    result["reasoning_token_accounting"] = "generated tokens before first native close, including generated opener"
    metrics = getattr(request, "metrics", None)
    if metrics and metrics.finished_time is not None and metrics.arrival_time is not None:
        result["latency_seconds"] = metrics.finished_time - metrics.arrival_time
    if completion.finish_reason != "stop":
        result.update(status="truncated" if completion.finish_reason == "length" else "error",
                      error="Generation did not finish normally")
        return result
    try:
        if alias == "M1":
            if "</think>" not in raw:
                raise ValueError("No native final-answer delimiter")
            reasoning, final = raw.split("</think>", 1)
            final = strip_qwen_terminal_tokens(final)
            result["reasoning_characters"] = len(reasoning)
        else:
            parsed = tokenizer.parse_response(raw, prefix=rendered)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                raise ValueError("Official parser returned no final text")
            final = parsed["content"].strip()
            thought = parsed.get("thinking")
            result["reasoning_characters"] = len(thought) if isinstance(thought, str) else 0
        result["final_text"] = final
        result["chapters"] = parse_window_outline(final, window)
        result["status"] = "ok"
    except (ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid", error=type(exc).__name__ + ": " + str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", required=True, choices=["M1", "M2"])
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--arms", nargs="+", choices=["ZS", "FT"], default=["ZS", "FT"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=393)
    parser.add_argument("--purpose", default="validation-only zero-shot/adapted generation; not held-out test")
    args = parser.parse_args()
    if args.max_num_seqs < 1 or not 1 <= args.max_new_tokens <= 32768:
        parser.error("Positive concurrency and output budget up to 32768 required")
    if args.batch_size < 1:
        parser.error("Positive batch size required")
    if "FT" in args.arms and (args.adapter is None or not (args.adapter / "adapter_model.safetensors").is_file()):
        parser.error("An existing saved pilot adapter is required")
    items = [json.loads(line) for line in args.windows.read_text().splitlines() if line.strip()]
    if not items or any(set(item) != {"window"} for item in items):
        parser.error("Require a nonempty gold-free window manifest")
    windows = [item["window"] for item in items]
    if len({(w["video_id"], w["window_index"]) for w in windows}) != len(windows):
        parser.error("Duplicate video/window keys")
    root = Path(__file__).resolve().parents[1]
    entry = model_entry(root, args.alias)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20 if args.alias == "M1" else 64,
                "presence_penalty": 1.5 if args.alias == "M1" else 0.0,
                "repetition_penalty": 1.0, "min_p": 0.0}
    max_model_len = 8192 + args.max_new_tokens
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_revision=entry["revision"], dtype="bfloat16", lora_dtype="bfloat16",
                  backend="vllm", thinking_protocol="native_unforced_v1", enable_thinking=True,
                  **sampling, retry_budget=0,
                  sampling_source=f'https://huggingface.co/{entry["model_id"]}#best-practices',
                  max_model_len=max_model_len, max_num_batched_tokens=8192,
                  gpu_memory_utilization=0.70, enable_prefix_caching=False,
                  purpose=args.purpose,
                  windows_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest(),
                  adapter_sha256=hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest() if args.adapter else None,
                  packages={name: importlib.metadata.version(name) for name in ["vllm", "torch", "transformers"]})
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2))
    started = time.perf_counter()
    with gpu0_lock():
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        tokenizer = AutoTokenizer.from_pretrained(entry["directory"], local_files_only=True)
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": build_window_prompt(w)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True) for w in windows]
        prompts = [{"prompt_token_ids": tokenizer.encode(text, add_special_tokens=False)} for text in rendered]
        if any(len(p["prompt_token_ids"]) > 8192 for p in prompts):
            raise ValueError("Prepared input exceeds common 8192-token limit")
        terminal = "<|im_end|>" if args.alias == "M1" else "<turn|>"
        stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(terminal)}
        stops = [t for t in stops if type(t) is int and t != tokenizer.unk_token_id]
        params = []
        for window in windows:
            seed = (args.seed + int(hashlib.sha256(f'{window["video_id"]}:{window["window_index"]}'.encode()).hexdigest()[:8], 16)) % (2**32)
            params.append(SamplingParams(max_tokens=args.max_new_tokens, **sampling,
                          seed=seed, stop_token_ids=stops,
                          skip_special_tokens=False))
        engine = LLM(model=entry["directory"], tokenizer=entry["directory"], dtype="bfloat16",
                     tensor_parallel_size=1, pipeline_parallel_size=1,
                     distributed_executor_backend="uni", max_model_len=max_model_len,
                     max_num_seqs=args.max_num_seqs, max_num_batched_tokens=8192,
                     gpu_memory_utilization=0.70, language_model_only=True,
                     enable_prefix_caching=False, enable_chunked_prefill=True,
                     enable_lora=True, max_lora_rank=16, max_loras=1, lora_dtype="bfloat16",
                     generation_config="vllm", seed=args.seed)
        loaded_seconds = time.perf_counter() - started
        expected = Counter(w["video_id"] for w in windows)
        for arm in args.arms:
            adapter = None if arm == "ZS" else LoRARequest("pilot-adapter", 1, str(args.adapter.resolve()))
            output = args.output / arm
            output.mkdir()
            groups, counts = defaultdict(list), Counter()
            generation_seconds, token_total = 0.0, 0
            with (output / "windows.jsonl").open("x") as raw_log, (output / "predictions.jsonl").open("x") as whole_log:
                for offset in range(0, len(windows), args.batch_size):
                    end = min(offset + args.batch_size, len(windows))
                    phase = time.perf_counter()
                    requests = engine.generate(prompts[offset:end], params[offset:end], lora_request=adapter, use_tqdm=False)
                    generation_seconds += time.perf_counter() - phase
                    if len(requests) != end - offset:
                        raise RuntimeError("Engine did not return every requested window")
                    for window, text, prompt, request in zip(windows[offset:end], rendered[offset:end], prompts[offset:end], requests):
                        if list(request.prompt_token_ids) != prompt["prompt_token_ids"]:
                            raise RuntimeError("Engine request order/prompt mismatch")
                        result = decode_result(tokenizer, args.alias, window, text, request, entry["revision"])
                        raw_log.write(json.dumps(result, ensure_ascii=False) + "\n")
                        token_total += len(request.outputs[0].token_ids)
                        key = window["video_id"]
                        groups[key].append((window, result))
                        if len(groups[key]) == expected[key]:
                            pairs = groups.pop(key)
                            prediction = stitch_window_predictions([p[0] for p in pairs], [p[1] for p in pairs], run_id=args.output.name + "-" + arm)
                            whole_log.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                            counts[prediction["status"]] += 1
                    raw_log.flush()
                    whole_log.flush()
                    checkpoint = {"arm": arm, "completed_windows": end, "requested_windows": len(windows),
                                  "completed_videos": sum(counts.values()), "status_counts": dict(counts),
                                  "generation_seconds": generation_seconds, "completion_tokens": token_total}
                    temp = output / "checkpoint.tmp"
                    temp.write_text(json.dumps(checkpoint, indent=2))
                    temp.replace(output / "checkpoint.json")
                    print(json.dumps({"event": "batch_complete", **checkpoint}), flush=True)
                    del requests
            if groups:
                raise RuntimeError("Incomplete whole-video groups after generation")
            summary = {"alias": args.alias, "arm": arm, "requested_videos": len(expected),
                       "requested_windows": len(windows), "status_counts": dict(counts),
                       "generation_seconds": generation_seconds, "completion_tokens": token_total,
                       "output_tokens_per_second": token_total / generation_seconds,
                       "timing_policy": "sum of batch elapsed times; concurrent latencies are not summed",
                       "configured_max_num_seqs": args.max_num_seqs, "batch_size": args.batch_size,
                       "submitted_windows": len(windows)}
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
        (args.output / "summary.json").write_text(json.dumps({"status": "requested_generation_complete", "arms": args.arms,
            "loaded_seconds": loaded_seconds, "total_seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
