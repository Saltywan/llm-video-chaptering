#!/usr/bin/env python
"""Rejection-sampling fine-tuning (RFT) data: two modes, both CPU only and train split only.

``manifest``: seeded choice of whole training videos from a pack's train_windows.jsonl, written
as a gold-free {"window"} manifest for endpoint sampling (gold never leaves this script).

``select``: for each training window, score every sampled completion by exact F1 of its internal
owned chapter starts (processing anchor excluded) against the gold window target, and keep the best one when it
finished normally, parsed under the strict parser and reaches --min-f1. Kept completions are the
model's own unedited text. Writes {"window","target","rft_raw","rft_meta"} rows next to the pack.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def internal_starts(chapters, window):
    """Global starts the window owns (A <= start < B), without the processing anchor L."""
    return {chapter["start"] for chapter in chapters
            if window["A"] <= chapter["start"] < window["B"] and chapter["start"] != window["L"]}


def window_f1(predicted, gold):
    if not predicted and not gold:
        return 1.0
    hits = len(predicted & gold)
    return 2 * hits / (len(predicted) + len(gold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("manifest")
    make.add_argument("--train-windows", type=Path, required=True)
    make.add_argument("--max-windows", type=int, required=True)
    make.add_argument("--seed", type=int, default=393)
    make.add_argument("--exclude", type=Path, nargs="*", default=[], help="earlier manifests whose windows are skipped")
    make.add_argument("--output", type=Path, required=True)
    pick = sub.add_parser("select")
    pick.add_argument("--train-windows", type=Path, required=True)
    pick.add_argument("--manifest", type=Path, required=True)
    pick.add_argument("--samples", type=Path, nargs="+", required=True, help="windows.jsonl of each sampling run")
    pick.add_argument("--min-f1", type=float, default=0.5)
    pick.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.train_windows.read_text().splitlines() if line.strip()]
    if args.mode == "manifest":
        skip = {(json.loads(line)["window"]["video_id"], json.loads(line)["window"]["window_index"])
                for path in args.exclude for line in path.read_text().splitlines() if line.strip()}
        videos = {}
        for item in items:
            if (item["window"]["video_id"], item["window"]["window_index"]) not in skip:
                videos.setdefault(item["window"]["video_id"], []).append(item["window"])
        order = sorted(videos)
        random.Random(args.seed).shuffle(order)
        chosen = []
        for video in order:
            if len(chosen) + len(videos[video]) > args.max_windows:
                continue
            chosen.extend(videos[video])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            for window in chosen:
                handle.write(json.dumps({"window": window}, ensure_ascii=False) + "\n")
        print(json.dumps({"windows": len(chosen), "videos": len({w["video_id"] for w in chosen})}))
        return
    if args.output.exists() or args.output.parent != args.train_windows.parent:
        parser.error("Output must be new and next to train_windows.jsonl (training manifest checks)")
    wanted = {(json.loads(line)["window"]["video_id"], json.loads(line)["window"]["window_index"])
              for line in args.manifest.read_text().splitlines() if line.strip()}
    gold = {(item["window"]["video_id"], item["window"]["window_index"]): item for item in items}
    if not wanted <= gold.keys():
        raise SystemExit("Sampling manifest is not a subset of the training windows")
    samples = {key: [] for key in wanted}
    for path in args.samples:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["video_id"], row["window_index"])
            if key in samples:
                # infer_vllm_pair.py writes <run>/ZS/windows.jsonl; the endpoint driver writes <run>/windows.jsonl.
                run = path.parent.parent.name if path.parent.name in ("ZS", "FT") else path.parent.name
                samples[key].append((run, row))
    reasons, kept, scores = Counter(), [], []
    for key in sorted(wanted):
        window = gold[key]["window"]
        target = internal_starts(json.loads(gold[key]["target"]["target_json"])["chapters"], window)
        best = None
        for run, row in samples[key]:
            if row.get("status") != "ok" or row.get("finish_reason") != "stop":
                reasons["sample_not_ok"] += 1
                continue
            score = window_f1(internal_starts(row["chapters"], window), target)
            scores.append(score)
            if best is None or score > best[0]:
                best = (score, run, row)
        if best is None:
            reasons["window_without_valid_sample"] += 1
        elif best[0] < args.min_f1:
            reasons["window_best_below_min_f1"] += 1
        else:
            kept.append({"window": gold[key]["window"], "target": gold[key]["target"], "rft_raw": best[2]["raw_text"],
                         "rft_meta": {"run": best[1], "window_f1": best[0], "candidates": len(samples[key]),
                                      "reasoning_tokens": (best[2].get("usage") or {}).get("reasoning_tokens")}})
    with args.output.open("x") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"recipe": "rft_self_samples_v1", "min_f1": args.min_f1, "manifest_windows": len(wanted),
               "samples_read": sum(len(v) for v in samples.values()), "kept_windows": len(kept),
               "drop_counts": dict(reasons), "mean_valid_sample_f1": sum(scores) / len(scores) if scores else None,
               "mean_kept_f1": sum(r["rft_meta"]["window_f1"] for r in kept) / len(kept) if kept else None,
               "sample_files": [str(p) for p in args.samples],
               "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
