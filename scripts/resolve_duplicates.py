#!/usr/bin/env python
"""Resolve a pinned audit's duplicate IDs without changing source partitions.

Only stdlib is required. Identical means exact canonical sentences+reference,
disregarding provenance. Conflicting IDs are quarantined in full. Every original
validation ID is excluded from training even if its validation rows conflict.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(record: dict[str, Any]) -> str:
    """Hash exact supervision content, independent of provenance or split."""
    content = {"sentences": record["sentences"], "reference": record["reference"]}
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_record(record: Any, expected_split: str) -> None:
    """Reject malformed canonical records; this resolver does not repair them."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("invalid canonical schema_version")
    if record.get("split") != expected_split:
        raise ValueError("canonical record split does not match its input file")
    if not isinstance(record.get("video_id"), str) or not record["video_id"].strip():
        raise ValueError("invalid canonical video_id")
    sentences = record.get("sentences")
    if not isinstance(sentences, list) or not sentences or not all(
        isinstance(sentence, str) and sentence.strip() for sentence in sentences
    ):
        raise ValueError("invalid canonical sentences")
    reference = record.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("invalid canonical reference")
    starts, titles = reference.get("chapter_starts"), reference.get("chapter_titles")
    if not isinstance(starts, list) or not starts or starts[0] != 0 or not all(
        type(start) is int and 0 <= start < len(sentences) for start in starts
    ) or any(first >= second for first, second in zip(starts, starts[1:])):
        raise ValueError("invalid canonical chapter_starts")
    if not isinstance(titles, list) or len(titles) != len(starts) or not all(
        isinstance(title, str) and title.strip() for title in titles
    ):
        raise ValueError("invalid canonical chapter_titles")


def resolve_records(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return disjoint clean subsets, complete conflict quarantine, and counts."""
    original_validation_ids = {record["video_id"] for record in validation}
    clean, quarantined, summaries = {}, {}, {}
    excluded_train = []
    for split, records in [("train", train), ("validation", validation)]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            validate_record(record, split)
            grouped[record["video_id"]].append(record)
        variants = {video_id: sorted({content_fingerprint(row) for row in group})
                    for video_id, group in grouped.items()}
        conflict_ids = {video_id for video_id, fingerprints in variants.items() if len(fingerprints) > 1}
        identical_duplicate_ids = {video_id for video_id, group in grouped.items()
                                   if len(group) > 1 and video_id not in conflict_ids}
        excluded_ids = set(grouped) & original_validation_ids if split == "train" else set()
        quarantine = [row for row in records if row["video_id"] in conflict_ids]
        excluded = [row for row in records if row["video_id"] in excluded_ids]
        seen = set()
        retained = []
        identical_removed = Counter()
        for record in records:
            video_id = record["video_id"]
            if video_id in excluded_ids or video_id in conflict_ids:
                continue
            if video_id in seen:
                identical_removed[video_id] += 1
            else:
                retained.append(record)
                seen.add(video_id)
        clean[split], quarantined[split] = retained, quarantine
        if split == "train":
            excluded_train = excluded
        # Conflict quarantine and cross-split exclusion may overlap. The
        # mutually-exclusive accounting below counts the overlap only once.
        conflict_only_removed = sum(row["video_id"] not in excluded_ids for row in quarantine)
        summaries[split] = {
            "original_rows": len(records), "original_unique_ids": len(grouped),
            "resolved_rows": len(retained), "resolved_unique_ids": len(seen),
            "original_duplicate_id_occurrences": len(records) - len(grouped),
            "identical_duplicate_ids": sorted(identical_duplicate_ids),
            "identical_duplicates_removed_after_exclusions": dict(sorted(identical_removed.items())),
            "identical_duplicate_rows_removed_after_exclusions": sum(identical_removed.values()),
            "conflict_ids": sorted(conflict_ids),
            "conflict_variants": {video_id: variants[video_id] for video_id in sorted(conflict_ids)},
            "quarantined_conflict_rows": len(quarantine),
            "validation_priority_excluded_ids": sorted(excluded_ids),
            "validation_priority_excluded_rows": len(excluded),
            "conflict_rows_also_excluded_by_validation_priority": sum(
                row["video_id"] in excluded_ids for row in quarantine),
            "exclusive_removed_counts": {
                "validation_priority": len(excluded),
                "conflict_not_already_excluded": conflict_only_removed,
                "identical_duplicates_after_exclusions": sum(identical_removed.values()),
            },
        }
        if len(retained) + len(excluded) + conflict_only_removed + sum(identical_removed.values()) != len(records):
            raise AssertionError("duplicate resolution row accounting failed")
    train_ids = {row["video_id"] for row in clean["train"]}
    validation_ids = {row["video_id"] for row in clean["validation"]}
    postconditions = {
        "train_nonempty": bool(clean["train"]),
        "validation_nonempty": bool(clean["validation"]),
        "train_ids_unique": len(train_ids) == len(clean["train"]),
        "validation_ids_unique": len(validation_ids) == len(clean["validation"]),
        "no_train_validation_overlap": not bool(train_ids & validation_ids),
        "no_original_validation_id_in_train": not bool(train_ids & original_validation_ids),
        "partitions_unchanged": all(row["split"] == split for split, rows in clean.items() for row in rows),
    }
    return {"clean": clean, "quarantine": quarantined, "excluded_train": excluded_train,
            "summary": summaries, "postconditions": postconditions,
            "original_validation_ids": sorted(original_validation_ids)}


def source_gate_evidence(
    audit: dict[str, Any], records: dict[str, list[dict[str, Any]]], hashes: dict[str, str]
) -> dict[str, bool]:
    """Duplicate-only source audit failures are expected, other failures are not."""
    splits = audit.get("splits", {})
    overlaps = audit.get("split_overlaps", {})
    test = splits.get("test", {})
    expected_hashes = audit.get("files_sha256", {})
    full = audit.get("partial_train_validation") is False
    counts_match = all(
        splits.get(split, {}).get("valid_rows") == len(rows)
        and splits.get(split, {}).get("rows_seen") == len(rows)
        for split, rows in records.items()
    )
    test_isolation = all(
        isinstance(overlaps.get(key), dict) and overlaps[key].get("count") == 0
        and overlaps[key].get("video_ids") == []
        for key in ["train__test", "validation__test"]
    )
    return {
        "source_audit_full": full,
        "source_invalid_rows_zero": audit.get("invalid_rows") == 0
            and all(splits.get(split, {}).get("invalid_rows") == 0 for split in records),
        "source_error_counts_empty": audit.get("error_counts") == {},
        "source_counts_match_canonical_files": counts_match,
        "source_canonical_hashes_match_audit": all(
            expected_hashes.get(name) == digest for name, digest in hashes.items()),
        "source_has_no_test_overlap": test_isolation,
        "source_test_ids_unique": test.get("duplicate_ids") == []
            and type(test.get("rows_seen")) is int and test["rows_seen"] > 0
            and test.get("unique_ids") == test.get("rows_seen"),
    }


def select_smoke(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match audit_data.py: fixed 10th/50th/90th sentence-count quantiles."""
    ordered = sorted(records, key=lambda row: (len(row["sentences"]), row["video_id"]))
    if len(ordered) <= 3:
        return ordered
    indices = {int((len(ordered) - 1) * quantile + .5) for quantile in [.1, .5, .9]}
    return [ordered[index] for index in sorted(indices)]


def read_records(path: Path, split: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank canonical JSONL line in {split} at line {line_index}")
            try:
                record = json.loads(line)
                validate_record(record, split)
            except (ValueError, TypeError, KeyError) as exc:
                # Exception content may contain source text; report location only.
                raise ValueError(f"invalid canonical row in {split} at line {line_index}") from None
            records.append(record)
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_directory(source: Path, output: Path) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output must be a new directory; refusing to overwrite")
    audit_path = source / "audit.json"
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    input_hashes = {f"{split}.jsonl": file_sha256(source / f"{split}.jsonl")
                    for split in ["train", "validation"]}
    audit_hash = file_sha256(audit_path)
    records = {split: read_records(source / f"{split}.jsonl", split) for split in ["train", "validation"]}
    evidence = source_gate_evidence(audit, records, input_hashes)
    resolved = resolve_records(records["train"], records["validation"])
    output.mkdir(parents=True, exist_ok=False)
    for split in ["train", "validation"]:
        write_records(output / f"{split}.jsonl", resolved["clean"][split])
        write_records(output / f"smoke_{split}.jsonl", select_smoke(resolved["clean"][split]))
        write_records(output / f"quarantine_{split}.jsonl", resolved["quarantine"][split])
    write_records(output / "excluded_train_validation_ids.jsonl", resolved["excluded_train"])
    unchanged = all(file_sha256(source / name) == digest for name, digest in input_hashes.items())
    unchanged = unchanged and file_sha256(audit_path) == audit_hash
    evidence["source_files_unchanged_during_resolution"] = unchanged
    # Clean records are exact subsets of the verified source files, with no new
    # partition moves. Their test-isolation evidence therefore remains valid.
    resolved["postconditions"]["test_isolation_inherited_from_verified_source_subsets"] = (
        evidence["source_has_no_test_overlap"] and evidence["source_canonical_hashes_match_audit"]
        and unchanged
    )
    report = {
        "schema_version": 1, "source_directory": str(source), "output_directory": str(output),
        "dataset": audit.get("dataset"), "config": audit.get("config"), "revision": audit.get("revision"),
        "policy": {
            "training_exclusion": "Remove every training row whose video_id occurs in the ORIGINAL validation file, even if that validation ID is quarantined.",
            "identical_duplicates": "Keep first source occurrence within each split; identity is exact sentences+reference, excluding provenance.",
            "conflicts": "Quarantine all occurrences of any within-split ID with multiple content fingerprints; never choose one variant.",
            "overlapping_removal_reasons": "Conflict quarantine includes cross-split-excluded rows; exclusive_removed_counts avoids double counting.",
            "partitions": "No record changes partition; output files are unmodified source-record subsets.",
            "smoke": "Regenerated at sentence-count quantiles .1/.5/.9, ties by video_id, before model results.",
            "test": "No test content read. Inherit zero overlap only from hash-verified full source audit and subset-preserving resolution.",
        },
        "source_files_sha256": {**input_hashes, "audit.json": audit_hash},
        "source_gate_evidence": evidence,
        "original_and_resolved_counts": resolved["summary"],
        "postconditions": resolved["postconditions"],
        "resolved_integrity_gate_passed": all(evidence.values()) and all(resolved["postconditions"].values()),
        "smoke_manifests": {
            split: [{"video_id": row["video_id"], "sentences": len(row["sentences"])}
                    for row in select_smoke(resolved["clean"][split])]
            for split in ["train", "validation"]
        },
        "output_files_sha256": {path.name: file_sha256(path) for path in sorted(output.glob("*.jsonl"))},
    }
    (output / "resolution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original pinned audit directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    args = parser.parse_args()
    try:
        report = resolve_directory(args.source, args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        # File-system/schema diagnostics contain no source transcripts.
        raise SystemExit(str(exc)) from None
    print(json.dumps({"output": report["output_directory"],
                      "resolved_integrity_gate_passed": report["resolved_integrity_gate_passed"],
                      "counts": report["original_and_resolved_counts"],
                      "source_gate_evidence": report["source_gate_evidence"]}, indent=2))
    return 0 if report["resolved_integrity_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
