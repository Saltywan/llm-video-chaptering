#!/usr/bin/env python
"""CPU-only, pinned YTSeg text audit. Prints aggregates, never transcripts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chaptering.data import (  # noqa: E402
    CONFIG, DATASET, LABEL_SEMANTICS, DataValidationError,
    canonicalize_row, indexed_transcript, outline_target,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    quantiles = {}
    for q in [0, .1, .5, .9, .95, .99, 1]:
        quantiles[str(q)] = ordered[int((len(ordered) - 1) * q + .5)]
    return {"count": len(values), "total": sum(values), "mean": sum(values) / len(values), "quantiles": quantiles}


def select_smoke(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick fixed 10th/50th/90th sentence-count quantiles, before inference."""
    ordered = sorted(records, key=lambda r: (len(r["sentences"]), r["video_id"]))
    if len(ordered) <= 3:
        return ordered
    indices = {int((len(ordered) - 1) * q + .5) for q in [.1, .5, .9]}
    return [ordered[index] for index in sorted(indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Resolved to and recorded as a full commit SHA")
    parser.add_argument("--limit", type=int, help="Optional per-split train/validation limit; marks audit partial")
    parser.add_argument("--tokenizer", action="append", default=[], metavar="ALIAS=PATH_OR_REPO",
                        help="Optional CPU tokenizer census; no model weights are loaded")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    # Import heavy/optional dependencies only inside the CLI entry point.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("Dataset API did not return a full pinned commit SHA")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Refusing to overwrite a nonempty audit directory; select a new --output")
    output.mkdir(parents=True, exist_ok=True)
    tokenizers = {}
    tokenizer_metadata = {}
    for spec in args.tokenizer:
        if "=" not in spec:
            raise SystemExit("--tokenizer must use ALIAS=PATH_OR_REPO")
        alias, source = spec.split("=", 1)
        if not alias or not source or alias in tokenizers:
            raise SystemExit("Tokenizer aliases and sources must be nonempty and aliases unique")
        from transformers import AutoTokenizer
        if Path(source).is_dir():
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
            tokenizer_metadata[alias] = {"source": source, "revision": tokenizer_revision,
                                         "local_files_only": True,
                                         "revision_verified": bool(tokenizer_revision)}
        else:
            tokenizer_revision = HfApi().model_info(source).sha
            if not isinstance(tokenizer_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
                raise SystemExit("Tokenizer model API did not return a full pinned commit SHA")
            tokenizer = AutoTokenizer.from_pretrained(source, revision=tokenizer_revision)
            tokenizer_metadata[alias] = {"source": source, "revision": tokenizer_revision,
                                         "local_files_only": False, "revision_verified": True}
        tokenizers[alias] = tokenizer

    report: dict[str, Any] = {
        "schema_version": 1, "dataset": DATASET, "config": CONFIG, "revision": revision,
        "label_semantics": LABEL_SEMANTICS,
        "alignment_policy": "all rows: binary starts equal marked-transcript starts, titles and sentences align",
        "partial_train_validation": args.limit is not None,
        "test_policy": "video_id column only; no canonical test examples or test predictions",
        "smoke_policy": "fixed sentence-count quantiles 0.1, 0.5, 0.9; ties ordered by video_id; not effectiveness sample",
        "splits": {}, "invalid_rows": 0,
        "tokenizer_sources": tokenizer_metadata,
        "tokenizer_measure": "indexed transcript without instruction/chat template; outline JSON without template",
    }
    split_ids: dict[str, set[str]] = {}
    errors: Counter[str] = Counter()
    invalid_path = output / "invalid_rows.jsonl"
    with invalid_path.open("w", encoding="utf-8") as invalid_file:
        for split in ["train", "validation"]:
            dataset = load_dataset(DATASET, CONFIG, revision=revision, split=split, streaming=True)
            records = []
            ids: set[str] = set()
            duplicate_ids = []
            sentences, chapters, title_lengths = [], [], []
            token_counts = {alias: {"input": [], "target": []} for alias in tokenizers}
            seen = invalid = 0
            with (output / f"{split}.jsonl").open("w", encoding="utf-8") as destination:
                for index, row in enumerate(dataset):
                    if args.limit is not None and index >= args.limit:
                        break
                    seen += 1
                    video_id = row.get("video_id")
                    if isinstance(video_id, str):
                        if video_id in ids:
                            duplicate_ids.append(video_id)
                        ids.add(video_id)
                    try:
                        record = canonicalize_row(row, split, revision)
                    except DataValidationError as exc:
                        invalid += 1
                        errors[str(exc)] += 1
                        invalid_file.write(json.dumps({"split": split, "row_index": index, "video_id": video_id,
                                                       "error": str(exc)}, ensure_ascii=False) + "\n")
                        continue
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    sentences.append(len(record["sentences"]))
                    chapters.append(len(record["reference"]["chapter_starts"]))
                    title_lengths.extend(len(title.split()) for title in record["reference"]["chapter_titles"])
                    for alias, tokenizer in tokenizers.items():
                        token_counts[alias]["input"].append(len(tokenizer.encode(indexed_transcript(record), add_special_tokens=False)))
                        target = json.dumps(outline_target(record), ensure_ascii=False, separators=(",", ":"))
                        token_counts[alias]["target"].append(len(tokenizer.encode(target, add_special_tokens=False)))
            smoke = select_smoke(records)
            with (output / f"smoke_{split}.jsonl").open("w", encoding="utf-8") as handle:
                for record in smoke:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            split_ids[split] = ids
            positive_count = sum(chapters) - len(records)
            candidate_gaps = sum(sentences) - len(records)
            report["splits"][split] = {
                "rows_seen": seen, "valid_rows": len(records), "invalid_rows": invalid,
                "unique_ids": len(ids), "duplicate_ids": duplicate_ids,
                "sentence_counts": distribution(sentences), "chapter_counts": distribution(chapters),
                "title_word_counts": distribution(title_lengths),
                "internal_boundary_positive_count": positive_count,
                "candidate_gap_count": candidate_gaps,
                "internal_boundary_positive_fraction": positive_count / candidate_gaps if candidate_gaps else None,
                "smoke": [{"video_id": row["video_id"], "sentences": len(row["sentences"])} for row in smoke],
                "tokenizers": {alias: {kind: distribution(values) for kind, values in counts.items()}
                               for alias, counts in token_counts.items()},
                "indexed_transcript_token_coverage_without_instruction_or_template": {
                    alias: {str(budget): sum(value <= budget for value in counts["input"]) / len(records)
                            if records else None for budget in [2048, 4096, 8192, 16384, 32768, 65536]}
                    for alias, counts in token_counts.items()},
            }
            report["invalid_rows"] += invalid

        # Projection happens at the Parquet reader: test transcript columns are not requested.
        test_ids: set[str] = set()
        test_duplicates = []
        test_count = 0
        test = load_dataset(DATASET, CONFIG, revision=revision, split="test", columns=["video_id"], streaming=True)
        for row in test:
            if set(row) != {"video_id"}:
                raise RuntimeError("Test reader did not honor video_id-only projection")
            video_id = row["video_id"]
            if not isinstance(video_id, str) or not video_id.strip():
                errors["invalid_test_video_id"] += 1
                continue
            test_count += 1
            if video_id in test_ids:
                test_duplicates.append(video_id)
            test_ids.add(video_id)
        split_ids["test"] = test_ids
        report["splits"]["test"] = {"rows_seen": test_count, "unique_ids": len(test_ids),
                                            "duplicate_ids": test_duplicates, "content_audited": False}

    overlaps = {}
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = sorted(split_ids[first] & split_ids[second])
        overlaps[f"{first}__{second}"] = {"count": len(overlap), "video_ids": overlap}
    report["split_overlaps"] = overlaps
    report["error_counts"] = dict(errors)
    valid = not errors and not any(info["duplicate_ids"] for info in report["splits"].values()) and not any(
        info["count"] for info in overlaps.values())
    report["passed"] = valid
    report["full_integrity_gate_passed"] = valid and args.limit is None
    report["files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.glob("*.jsonl")
    }
    write_json(output / "audit.json", report)
    # Small aggregates only: the full report and data files stay on disk.
    print(json.dumps({"output": str(output), "revision": revision, "passed": valid,
                      "partial": args.limit is not None, "invalid_rows": report["invalid_rows"],
                      "split_counts": {split: info["rows_seen"] for split, info in report["splits"].items()},
                      "error_counts": dict(errors)}, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
