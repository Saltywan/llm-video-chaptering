#!/usr/bin/env python
"""Zero-shot generation through an already-running vLLM endpoint (M1/M2).

The prompt is rendered and tokenized here with the pinned tokenizer and sent as token
IDs to the OpenAI-compatible /completions route, with the same sampling settings,
per-window seeds, stop tokens, strict parser and stitching as infer_vllm_pair.py.
The service itself is never managed. The API key is read from the first line of
stdin, kept in memory only, and never written to any file or log.

Endpoint outputs are a separate inference path: label them as such and do not pair
them with adapted outputs from the owned GPU0 engine without a recorded agreement check.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.inference import strip_qwen_terminal_tokens
from chaptering.modeling import model_entry
from chaptering.windowing import build_window_prompt, parse_window_outline, stitch_window_predictions


def interpret(tokenizer, alias, window, rendered, raw, finish_reason, usage, revision):
    """Turn one raw completion into a strict per-window result."""
    result = {"video_id": window["video_id"], "window_index": window["window_index"],
              "status": "error", "chapters": [], "error": None, "usage": dict(usage),
              "model_revision": revision, "backend": "vllm_endpoint", "enable_thinking": True,
              "thinking_protocol": "native_unforced_v1", "reasoning_budget": None,
              "wall_seconds": None, "raw_text": raw, "finish_reason": finish_reason}
    marker = "</think>" if alias == "M1" else "<channel|>"
    position = raw.find(marker)
    if position >= 0:
        result["usage"]["reasoning_tokens"] = len(tokenizer.encode(raw[:position], add_special_tokens=False))
    elif alias == "M2" and "<|channel>thought" not in raw:
        result["usage"]["reasoning_tokens"] = 0
    else:
        result["usage"]["reasoning_tokens"] = None
    result["reasoning_token_accounting"] = "tokens of generated text before first native close, re-tokenized"
    if finish_reason != "stop":
        result.update(status="truncated" if finish_reason == "length" else "error",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:48761/v1")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=393)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--purpose", default="validation-only zero-shot through an existing service; not held-out test")
    args = parser.parse_args()
    parsed_url = urlsplit(args.base_url)
    loopback = parsed_url.hostname in {"127.0.0.1", "localhost"}
    if (parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment
            or not (parsed_url.scheme == "https" or (parsed_url.scheme == "http" and loopback))):
        parser.error("Endpoint must be credential-free HTTPS, or HTTP on loopback only")
    api_key = sys.stdin.readline().strip()
    if not api_key:
        parser.error("API key must be supplied on the first line of stdin")
    items = [json.loads(line) for line in args.windows.read_text().splitlines() if line.strip()]
    if not items or any(set(item) != {"window"} for item in items):
        parser.error("Require a nonempty gold-free window manifest")
    windows = [item["window"] for item in items]
    root = Path(__file__).resolve().parents[1]
    entry = model_entry(root, args.alias)
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20 if args.alias == "M1" else 64,
                "presence_penalty": 1.5 if args.alias == "M1" else 0.0,
                "repetition_penalty": 1.0, "min_p": 0.0}
    done = {}
    if args.resume:
        for line in (args.output / "windows.jsonl").read_text().splitlines():
            row = json.loads(line)
            done[(row["video_id"], row["window_index"])] = row
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(model_id=entry["model_id"], model_revision=entry["revision"], backend="vllm_endpoint",
                      thinking_protocol="native_unforced_v1", enable_thinking=True, retry_budget=0, **sampling,
                      purpose=args.purpose,
                      windows_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest(),
                      credential="first line of stdin; not stored")
        (args.output / "run_config.json").write_text(json.dumps(config, indent=2))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(entry["directory"], local_files_only=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    listing = urllib.request.Request(args.base_url.rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + api_key, "User-Agent": "llm-video-chaptering/1.0"})
    served = json.load(opener.open(listing, timeout=30))["data"]
    if not any(model.get("id") == entry["model_id"] and str(model.get("root", "")).rstrip("/").endswith(
            Path(entry["directory"]).name) for model in served):
        raise RuntimeError("Endpoint does not serve the pinned model directory")
    terminal = "<|im_end|>" if args.alias == "M1" else "<turn|>"
    stops = sorted({token for token in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(terminal))
                    if type(token) is int and token != tokenizer.unk_token_id})

    def request(window):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": build_window_prompt(window)}],
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(prompt_ids) > args.max_input_tokens:
            raise ValueError("Prepared input exceeds the configured input limit")
        seed = (args.seed + int(hashlib.sha256(
            f'{window["video_id"]}:{window["window_index"]}'.encode()).hexdigest()[:8], 16)) % (2**32)
        payload = {"model": entry["model_id"], "prompt": prompt_ids, "max_tokens": args.max_new_tokens,
                   "seed": seed, "stop_token_ids": stops, "skip_special_tokens": False,
                   "add_special_tokens": False, "stream": False, **sampling}
        call = urllib.request.Request(args.base_url.rstrip("/") + "/completions",
                                      data=json.dumps(payload).encode(),
                                      headers={"Authorization": "Bearer " + api_key,
                                               "Content-Type": "application/json", "User-Agent": "llm-video-chaptering/1.0"})
        started = time.perf_counter()
        try:
            data = json.load(opener.open(call, timeout=args.timeout))
            choice = data["choices"][0]
            result = interpret(tokenizer, args.alias, window, rendered, choice["text"],
                               choice.get("finish_reason"), data.get("usage") or {}, entry["revision"])
        except urllib.error.HTTPError as exc:
            result = {"video_id": window["video_id"], "window_index": window["window_index"], "status": "error",
                      "chapters": [], "error": f"HTTP {exc.code}", "usage": {}, "backend": "vllm_endpoint"}
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            result = {"video_id": window["video_id"], "window_index": window["window_index"], "status": "error",
                      "chapters": [], "error": type(exc).__name__, "usage": {}, "backend": "vllm_endpoint"}
        result["latency_seconds"] = time.perf_counter() - started
        return result

    todo = [window for window in windows if (window["video_id"], window["window_index"]) not in done]
    started = time.perf_counter()
    with (args.output / "windows.jsonl").open("a") as log, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(request, window) for window in todo]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            done[(result["video_id"], result["window_index"])] = result
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if count % 50 == 0 or count == len(todo):
                print(json.dumps({"event": "progress", "completed": count, "of": len(todo),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    elapsed = time.perf_counter() - started
    groups = defaultdict(list)
    for window in windows:
        groups[window["video_id"]].append(window)
    counts, window_counts = Counter(), Counter(row["status"] for row in done.values())
    with (args.output / "predictions.jsonl").open("w") as whole:
        for video_id, group in groups.items():
            prediction = stitch_window_predictions(
                group, [done[(video_id, window["window_index"])] for window in group], run_id=args.output.name)
            whole.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            counts[prediction["status"]] += 1
    tokens = sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in done.values())
    summary = {"alias": args.alias, "arm": "ZS", "backend": "vllm_endpoint", "requested_videos": len(groups),
               "requested_windows": len(windows), "status_counts": dict(counts),
               "window_status_counts": dict(window_counts), "completion_tokens": tokens,
               "this_invocation_windows": len(todo), "this_invocation_seconds": elapsed,
               "concurrency": args.concurrency}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
