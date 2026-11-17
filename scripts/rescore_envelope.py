#!/usr/bin/env python
"""Re-read saved window outputs with the declared lenient ``envelope-v1`` policy.

CPU only; nothing is regenerated. A window is re-read only when its strict status
is ``invalid`` and it has a saved ``final_text``. Truncated and errored windows are
unchanged. Writes a new predictions file for evaluate_predictions.py and a summary;
the strict files are never modified.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.envelope import ENVELOPE_VERSION, strip_envelope_v1
from chaptering.windowing import parse_window_outline, stitch_window_predictions


def reread(result, window):
    """Return a lenient copy of one strict window result and what was removed."""
    if result["status"] != "invalid" or not isinstance(result.get("final_text"), str):
        return result, []
    inner, removed = strip_envelope_v1(result["final_text"])
    if not removed:
        return result, []
    try:
        chapters = parse_window_outline(inner, window)
    except (ValueError, TypeError, KeyError):
        return result, []
    lenient = dict(result, status="ok", chapters=chapters, error=None,
                   envelope={"version": ENVELOPE_VERSION, "removed": removed,
                             "strict_status": result["status"], "strict_error": result["error"]})
    return lenient, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True, help="gold-free window pack")
    parser.add_argument("--results", type=Path, required=True, help="strict per-window results (windows.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="new lenient predictions.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; refusing to overwrite")
    groups = defaultdict(list)
    for line in args.windows.read_text().splitlines():
        window = json.loads(line)["window"]
        groups[window["video_id"]].append(window)
    results = {}
    for line in args.results.read_text().splitlines():
        item = json.loads(line)
        results[(item["video_id"], item["window_index"])] = item
    wanted = {video_id for video_id, _ in results}
    strict_windows, lenient_windows, removed_counts = Counter(), Counter(), Counter()
    strict_videos, lenient_videos = Counter(), Counter()
    run_id = f"{args.results.parent.name}-{ENVELOPE_VERSION}"
    with args.output.open("x") as handle:
        for video_id, windows in groups.items():
            if video_id not in wanted:
                continue
            strict = [results[(video_id, window["window_index"])] for window in windows]
            lenient = []
            for window, item in zip(windows, strict):
                new, removed = reread(item, window)
                lenient.append(new)
                strict_windows[item["status"]] += 1
                lenient_windows[new["status"]] += 1
                removed_counts.update(["+".join(removed)] if removed else [])
            strict_videos[stitch_window_predictions(windows, strict, run_id=run_id)["status"]] += 1
            prediction = stitch_window_predictions(windows, lenient, run_id=run_id)
            prediction["format_policy"] = ENVELOPE_VERSION
            lenient_videos[prediction["status"]] += 1
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    summary = {"format_policy": ENVELOPE_VERSION, "results": str(args.results), "output": str(args.output),
               "videos": sum(lenient_videos.values()), "strict_window_status": dict(strict_windows),
               "lenient_window_status": dict(lenient_windows), "accepted_after_removing": dict(removed_counts),
               "strict_video_status": dict(strict_videos), "lenient_video_status": dict(lenient_videos)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
