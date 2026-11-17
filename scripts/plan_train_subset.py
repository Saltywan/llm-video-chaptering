#!/usr/bin/env python
"""Pick one shared training-video subset for all four backbones under a wall-clock budget.

Each backbone's speed comes from its own measured benchmark run (report.json + steps.jsonl of a
short train_lora.py run with the same settings as the full run). Speed is measured in the
same cheap length estimate the batch planner uses, so it transfers to unseen windows. Videos are
taken whole, in a seeded shuffle, until the summed predicted time of all four runs reaches the
budget. Every training window of a chosen video is kept; nothing is truncated.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import estimated_tokens


def measured_rate(run, items_by_key, alias):
    """Estimated tokens per training second, from a finished benchmark run."""
    report = json.loads((run / "report.json").read_text())
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]
    seconds = sum(row["seconds"] for row in rows)
    tokens = sum(estimated_tokens(items_by_key[tuple(key)], alias) for row in rows for key in row["windows"])
    if not rows or seconds <= 0:
        raise ValueError(f"Benchmark {run} has no timed steps")
    return tokens / seconds, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True, help="full train_windows.jsonl of the pack")
    parser.add_argument("--benchmark", nargs=2, action="append", metavar=("ALIAS", "RUN_DIR"), required=True)
    parser.add_argument("--budget-hours", type=float, required=True, help="summed training time of all runs")
    parser.add_argument("--overhead-seconds", type=float, default=300, help="per run: load, checks, save")
    parser.add_argument("--seed", type=int, default=393)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent != args.windows.parent:
        parser.error("Output must be new and next to the pack's train_windows.jsonl (manifest checks)")
    items = [json.loads(line) for line in args.windows.read_text().splitlines() if line.strip()]
    by_key = {(item["window"]["video_id"], item["window"]["window_index"]): item for item in items}
    rates, reports = {}, {}
    for alias, run in args.benchmark:
        rates[alias], report = measured_rate(Path(run), by_key, alias)
        reports[alias] = {"run": run, "status": report.get("status"), "batch_tokens": report.get("batch_tokens")}
    videos = {}
    for item in items:
        videos.setdefault(item["window"]["video_id"], []).append(item)
    order = sorted(videos)
    random.Random(args.seed).shuffle(order)
    budget = args.budget_hours * 3600 - args.overhead_seconds * len(rates)
    chosen, spent = [], 0.0
    for video in order:
        cost = sum(estimated_tokens(item, alias) / rate for item in videos[video] for alias, rate in rates.items())
        if spent + cost > budget:
            break
        chosen.append(video)
        spent += cost
    if not chosen:
        raise ValueError("Budget admits no training video")
    keep = set(chosen)
    lines = [line for line in args.windows.read_text().splitlines()
             if line.strip() and json.loads(line)["window"]["video_id"] in keep]
    args.output.write_text("\n".join(lines) + "\n")
    per_alias = {alias: sum(estimated_tokens(item, alias) for video in chosen for item in videos[video]) / rate / 3600
                 for alias, rate in rates.items()}
    summary = {"purpose": "shared training subset sized to the wall-clock budget; whole videos, seeded order",
               "source": str(args.windows), "source_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "seed": args.seed, "budget_hours": args.budget_hours, "overhead_seconds_per_run": args.overhead_seconds,
               "available_videos": len(videos), "available_windows": len(items),
               "chosen_videos": len(chosen), "chosen_windows": len(lines),
               "estimated_tokens_per_second": rates, "predicted_training_hours": per_alias,
               "benchmarks": reports}
    args.output.with_suffix(".plan.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
