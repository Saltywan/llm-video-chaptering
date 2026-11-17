#!/usr/bin/env python
"""Batched greedy inference for the seq2seq backbones (E1/E2).

Same prompts, strict parser, statuses and stitching as infer_local.py; only the
execution is batched. Prompts are bucketed by length. Generation runs in rising
allowance tiers: a row with no end token at a lower tier is regenerated from scratch
at the next tier, and only the final tier (--max-new-tokens) can mark it truncated.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.batching import plan_batches
from chaptering.modeling import ChapterModel
from chaptering.windowing import stitch_window_predictions
from pilot_seq2seq import gpu0_lock


def key(window):
    return (window["video_id"], window["window_index"])


def run_batch(model, torch, windows, cap, final_cap, log):
    """Generate one batch; on CUDA out-of-memory, halve it and try again."""
    try:
        return model.generate_seq2seq_batch(windows, cap=cap, final_cap=final_cap)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(windows) == 1:
            raise
        log({"event": "oom_split", "size": len(windows), "cap": cap})
        half = len(windows) // 2
        return (run_batch(model, torch, windows[:half], cap, final_cap, log)
                + run_batch(model, torch, windows[half:], cap, final_cap, log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", choices=["E1", "E2"], required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--attention")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--tiers", default="384,1536",
                        help="comma-separated lower allowances tried before --max-new-tokens")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--square-budget", type=int, default=16 * 8192 * 8192,
                        help="upper bound on batch_size * longest_prompt_tokens**2")
    parser.add_argument("--max-videos", type=int, help="only the first N videos in manifest order")
    parser.add_argument("--purpose", default="validation only, not final test results")
    args = parser.parse_args()
    tiers = sorted({int(t) for t in args.tiers.split(",") if t.strip()} | {args.max_new_tokens})
    if tiers[0] < 1 or tiers[-1] != args.max_new_tokens:
        raise ValueError("Tiers must be positive and no larger than --max-new-tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]

    def log(event):
        print(json.dumps(event), flush=True)

    groups = defaultdict(list)
    for line in args.windows.read_text().splitlines():
        item = json.loads(line)
        if set(item) != {"window"}:
            raise ValueError("Inference requires gold-free window records only")
        groups[item["window"]["video_id"]].append(item["window"])
    if not groups:
        raise ValueError("Empty inference manifest")
    if args.max_videos is not None:
        groups = dict(list(groups.items())[:args.max_videos])
    run_config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    run_config.update(purpose=args.purpose, tiers=tiers, decoding="greedy",
                      window_manifest_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest(),
                      requested_videos=len(groups),
                      requested_windows=sum(len(windows) for windows in groups.values()),
                      packages={name: importlib.metadata.version(name)
                                for name in ["torch", "transformers", "peft"]})
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2))
    started = time.perf_counter()
    with gpu0_lock():
        model = ChapterModel(root, args.alias, adapter=args.adapter, attention=args.attention)
        torch = model.torch
        loaded = time.perf_counter() - started
        pending = [window for windows in groups.values() for window in windows]
        results, tier_stats, generation_seconds = {}, [], 0.0
        with (args.output / "windows.jsonl").open("x") as raw:
            for cap in tiers:
                if not pending:
                    break
                lengths = [model.seq2seq_prompt_length(window) for window in pending]
                batches = plan_batches(lengths, max_batch=args.max_batch, square_budget=args.square_budget)
                carried, tier_started = [], time.perf_counter()
                for number, indices in enumerate(batches):
                    batch = [pending[i] for i in indices]
                    batch_started = time.perf_counter()
                    outputs = run_batch(model, torch, batch, cap, tiers[-1], log)
                    for window, result in zip(batch, outputs):
                        if result is None:
                            carried.append(window)
                            continue
                        results[key(window)] = result
                        raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                    raw.flush()
                    log({"event": "batch", "alias": args.alias, "cap": cap, "batch": number + 1,
                         "batches": len(batches), "size": len(batch), "longest_prompt": lengths[indices[0]],
                         "seconds": time.perf_counter() - batch_started, "carried": len(carried),
                         "completed_windows": len(results)})
                seconds = time.perf_counter() - tier_started
                generation_seconds += seconds
                tier_stats.append({"cap": cap, "windows": len(pending), "batches": len(batches),
                                   "carried_to_next": len(carried), "seconds": seconds})
                pending = carried
        if pending:
            raise RuntimeError("Windows left without a result after the final tier")
        counts, window_counts = defaultdict(int), defaultdict(int)
        with (args.output / "predictions.jsonl").open("x") as whole:
            for video_id, windows in groups.items():
                outputs = [results[key(window)] for window in windows]
                for item in outputs:
                    window_counts[item["status"]] += 1
                prediction = stitch_window_predictions(windows, outputs, run_id=args.output.name)
                whole.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                counts[prediction["status"]] += 1
        completion = sum(item["usage"]["completion_tokens"] for item in results.values())
        summary = {"alias": args.alias, "backend": "hf_batched_greedy_v1", "requested_videos": len(groups),
                   "requested_windows": len(results), "status_counts": dict(counts),
                   "window_status_counts": dict(window_counts), "tiers": tier_stats,
                   "completion_tokens": completion, "load_seconds": loaded,
                   "generation_seconds": generation_seconds,
                   "kept_output_tokens_per_second": completion / generation_seconds,
                   "total_seconds": time.perf_counter() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        log(summary)


if __name__ == "__main__":
    main()
