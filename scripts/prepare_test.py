#!/usr/bin/env python
"""Build the held-out test pack once, after the freeze. CPU only, prints aggregates only.

1. Canonicalize the official test split at the pinned dataset revision (same validator as
   train/validation; invalid rows are recorded, never repaired) into a new audit directory.
2. Check test IDs: unique, equal to the full source audit's test-ID count, and disjoint from
   the resolved train/validation files.
3. Window every test video with exactly the window settings of the frozen validation pack.
   Windows are gold-free; references go to test_records.jsonl for scoring only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chaptering.data import CONFIG, DATASET, DataValidationError, canonicalize_row  # noqa: E402
from chaptering.windowing import WINDOW_PROMPT_VERSION, build_windows  # noqa: E402
from prepare_windows import ALIASES, distribution, load_tokenizers, read_canonical, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True, help="frozen pack whose settings are copied")
    parser.add_argument("--source-audit", type=Path, required=True, help="full original audit (test ID count)")
    parser.add_argument("--resolved", type=Path, required=True, help="resolved train/validation directory")
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("CPU only: set CUDA_VISIBLE_DEVICES to empty")
    if args.audit_output.exists() or args.output.exists():
        raise SystemExit("Outputs must be new directories")
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen":
        raise SystemExit("configuration must be frozen before the test pack is built")
    pack = json.loads((args.validation_pack / "manifest.json").read_text())
    settings = pack["settings"]
    revision = pack["dataset_revision"]
    audit = json.loads((args.source_audit / "audit.json").read_text())
    # The source audit fails its overall gate only because of train duplicates, which the
    # resolution handled; the test-relevant parts must hold on their own.
    if (audit["revision"] != revision or audit.get("partial_train_validation") is not False
            or audit["splits"]["test"]["duplicate_ids"]
            or audit["split_overlaps"]["train__test"]["count"] or audit["split_overlaps"]["validation__test"]["count"]):
        raise SystemExit("Source audit revision/test-integrity mismatch")

    from datasets import load_dataset
    args.audit_output.mkdir(parents=True)
    invalid, ids, duplicates = [], set(), []
    with (args.audit_output / "test.jsonl").open("x", encoding="utf-8") as handle:
        for index, row in enumerate(load_dataset(DATASET, CONFIG, revision=revision, split="test", streaming=True)):
            video_id = row.get("video_id")
            if video_id in ids:
                duplicates.append(video_id)
            ids.add(video_id)
            try:
                record = canonicalize_row(row, "test", revision)
            except DataValidationError as exc:
                invalid.append({"row_index": index, "video_id": video_id, "error": str(exc)})
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    known = set()
    for split in ["train", "validation"]:
        known |= {row["video_id"] for row in read_canonical(args.resolved / f"{split}.jsonl", split)}
    overlap = sorted(ids & known)
    test_audit = {"dataset": DATASET, "config": CONFIG, "revision": revision, "rows_seen": index + 1,
                  "valid_rows": index + 1 - len(invalid), "invalid_rows": invalid, "duplicate_ids": duplicates,
                  "source_audit_test_unique_ids": audit["splits"]["test"]["unique_ids"],
                  "unique_ids": len(ids), "overlap_with_resolved_train_validation": overlap,
                  "freeze_sha256": sha256(args.freeze),
                  "test_sha256": sha256(args.audit_output / "test.jsonl")}
    test_audit["passed"] = (not duplicates and not overlap
                            and len(ids) == audit["splits"]["test"]["unique_ids"])
    (args.audit_output / "test_audit.json").write_text(json.dumps(test_audit, indent=2) + "\n")
    if not test_audit["passed"]:
        print(json.dumps({k: v for k, v in test_audit.items() if k != "invalid_rows"}, indent=2))
        return 2

    records = read_canonical(args.audit_output / "test.jsonl", "test")
    _, counters, model_metadata, config_hashes = load_tokenizers(ROOT, settings["input_tokens"])
    args.output.mkdir(parents=True)
    failures, window_count, tokens = [], 0, {alias: [] for alias in ALIASES}
    with (args.output / "test_records.jsonl").open("x", encoding="utf-8") as record_file, \
            (args.output / "test_windows.jsonl").open("x", encoding="utf-8") as window_file:
        for row in records:
            try:
                windows = build_windows(row, counters, settings["input_tokens"],
                                        max_core_sentences=settings["core_sentences"],
                                        left_context=settings["context_sentences"],
                                        right_context=settings["context_sentences"],
                                        core_search=settings.get("core_search", "exhaustive"))
            except Exception as exc:  # recorded, never silently dropped
                failures.append({"video_id": row["video_id"], "error_type": type(exc).__name__})
                continue
            record_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            for window in windows:
                window_file.write(json.dumps({"window": window}, ensure_ascii=False) + "\n")
                for alias in ALIASES:
                    tokens[alias].append(window["input_token_counts"][alias])
            window_count += len(windows)
    manifest = {"schema_version": 1, "purpose": "held-out test pack, built once after the freeze",
                "prompt_version": WINDOW_PROMPT_VERSION, "dataset_revision": revision,
                "settings": settings, "settings_copied_from": str(args.validation_pack),
                "validation_pack_manifest_sha256": sha256(args.validation_pack / "manifest.json"),
                "models": model_metadata, "config_files_sha256": config_hashes,
                "videos": len(records), "windowed_videos": len(records) - len(failures), "windows": window_count,
                "failures": failures, "invalid_canonical_rows": len(invalid),
                "input_tokens": {alias: distribution(values) for alias, values in tokens.items()},
                "test_audit_sha256": sha256(args.audit_output / "test_audit.json"),
                "preparation_gate_passed": not failures,
                "files_sha256": {path.name: sha256(path) for path in sorted(args.output.glob("*.jsonl"))}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["videos", "windowed_videos", "windows", "invalid_canonical_rows",
                                                "preparation_gate_passed"]}, indent=2))
    return 0 if manifest["preparation_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
