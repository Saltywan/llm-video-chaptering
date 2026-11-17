#!/usr/bin/env python3
"""Evaluate canonical JSONL predictions with explicit optional metric settings.

Metric packages and the exact semantic model revision must already be present
in the selected task environment/cache. This script does not install or fetch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chaptering.reporting import (  # noqa: E402
    make_bertscore_scorer, make_rouge_l_scorer, paired_video_bootstrap,
    report_predictions,
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in input record")
        result[key] = value
    return result


def _records(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON record") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-predictions", type=Path,
                        help="Optional right-hand prediction arm; bootstrap reports primary minus this arm")
    parser.add_argument("--skip-boundary-similarity", action="store_true")
    parser.add_argument("--rouge-stemmer", action="store_true")
    parser.add_argument("--semantic-model")
    parser.add_argument("--semantic-revision", help="Complete 40-character commit SHA; must already be cached")
    parser.add_argument("--semantic-layers", type=int)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--semantic-cache-dir")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--span-iou", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261116)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    pins = (args.semantic_model, args.semantic_revision, args.semantic_layers)
    if any(value is not None for value in pins) and not all(value is not None for value in pins):
        parser.error("semantic scoring requires --semantic-model, --semantic-revision and --semantic-layers together")
    inputs = [args.references, args.predictions]
    if args.compare_predictions is not None:
        inputs.append(args.compare_predictions)
    if args.output.resolve() in [path.resolve() for path in inputs]:
        parser.error("output must not overwrite an input record file")
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; choose a new path or explicitly use --overwrite")
    references = _records(args.references)
    predictions = _records(args.predictions)
    rouge = make_rouge_l_scorer(use_stemmer=args.rouge_stemmer)
    semantic = None
    if args.semantic_model is not None:
        semantic = make_bertscore_scorer(
            model_id=args.semantic_model, revision=args.semantic_revision,
            num_layers=args.semantic_layers, device=args.semantic_device,
            batch_size=args.semantic_batch_size, cache_dir=args.semantic_cache_dir)

    def evaluate(rows):
        return report_predictions(references, rows, rouge_scorer=rouge,
                                  semantic_scorer=semantic,
                                  include_boundary_similarity=not args.skip_boundary_similarity,
                                  tolerance=args.tolerance, iou_threshold=args.span_iou)

    result = evaluate(predictions)
    if args.compare_predictions is not None:
        other = evaluate(_records(args.compare_predictions))
        result["comparison"] = {"right_arm": other, "paired_exact_f1_bootstrap": paired_video_bootstrap(
            result["canonical"]["per_video"], other["canonical"]["per_video"],
            resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)}
    result["input_files"] = {"references": str(args.references), "left_predictions": str(args.predictions),
                             "right_predictions": str(args.compare_predictions) if args.compare_predictions else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    print(json.dumps({"output": str(args.output),
                      "requested_videos": result["canonical"]["aggregate"]["requested_videos"],
                      "failed_videos": result["canonical"]["aggregate"]["failed_videos"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
