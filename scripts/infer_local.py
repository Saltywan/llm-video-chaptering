#!/usr/bin/env python
"""Sequential local window inference, with complete failure accounting."""

import argparse
from collections import defaultdict
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.modeling import ChapterModel
from chaptering.windowing import stitch_window_predictions
from pilot_seq2seq import gpu0_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", choices=["E1", "E2", "M1", "M2"], required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--attention")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=393)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config["purpose"] = "real-data pilot; validation only, not final test results"
    run_config["packages"] = {name: importlib.metadata.version(name)
                              for name in ["torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"]}
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2))
    run_config["window_manifest_sha256"] = hashlib.sha256(args.windows.read_bytes()).hexdigest()
    run_config["packages"] = {name: importlib.metadata.version(name)
                               for name in ["torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"]}
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2))
    groups = defaultdict(list)
    for line in args.windows.read_text().splitlines():
        item = json.loads(line)
        if set(item) != {"window"}:
            raise ValueError("Inference requires gold-free window records only")
        groups[item["window"]["video_id"]].append(item["window"])
    if not groups:
        raise ValueError("Empty inference manifest")
    started = time.perf_counter()
    with gpu0_lock():
        model = ChapterModel(root, args.alias, adapter=args.adapter, attention=args.attention)
        counts = defaultdict(int)
        with (args.output / "windows.jsonl").open("x") as raw, (args.output / "predictions.jsonl").open("x") as whole:
            for video_id, windows in groups.items():
                results = []
                for window in windows:
                    result = model.generate_window(window, max_new_tokens=args.max_new_tokens,
                                                   reasoning_budget=args.reasoning_budget, seed=args.seed)
                    raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                    raw.flush()
                    results.append(result)
                    print(json.dumps({"event": "window", "alias": args.alias, "video_id": video_id,
                                      "window_index": window["window_index"], "status": result["status"],
                                      "wall_seconds": result["wall_seconds"], "error": result["error"],
                                      "usage": result["usage"]}), flush=True)
                prediction = stitch_window_predictions(windows, results, run_id=args.output.name)
                prediction["wall_seconds"] = sum(item["wall_seconds"] for item in results)
                whole.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                whole.flush()
                counts[prediction["status"]] += 1
        summary = {"alias": args.alias, "requested_videos": len(groups), "status_counts": dict(counts),
                   "total_seconds": time.perf_counter() - started,
                   "peak_allocated_bytes": model.torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": model.torch.cuda.max_memory_reserved()}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
