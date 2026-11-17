"""Duplicate resolution preserves held-out priority and exposes conflicts."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_duplicates.py"
spec = importlib.util.spec_from_file_location("resolve_duplicates", SCRIPT)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def record(video_id, split="train", text="Sentence.", title="Topic", provenance=None):
    return {"schema_version": 1, "video_id": video_id, "split": split,
            "sentences": [text], "reference": {"chapter_starts": [0], "chapter_titles": [title]},
            "provenance": provenance or {"revision": "fixture"}}


def write_source(path, train, validation, **changes):
    path.mkdir()
    for split, rows in [("train", train), ("validation", validation)]:
        (path / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = {"dataset": "fixture", "config": "text", "revision": "fixture",
             "partial_train_validation": False, "invalid_rows": 0, "error_counts": {},
             "splits": {"train": {"rows_seen": len(train), "valid_rows": len(train), "invalid_rows": 0},
                        "validation": {"rows_seen": len(validation), "valid_rows": len(validation), "invalid_rows": 0},
                        "test": {"rows_seen": 2, "unique_ids": 2, "duplicate_ids": []}},
             "split_overlaps": {"train__test": {"count": 0, "video_ids": []},
                                "validation__test": {"count": 0, "video_ids": []}},
             "files_sha256": {f"{split}.jsonl": resolver.file_sha256(path / f"{split}.jsonl")
                              for split in ["train", "validation"]}}
    audit.update(changes)
    (path / "audit.json").write_text(json.dumps(audit))
    return audit


class DuplicateResolutionTests(unittest.TestCase):
    def test_identical_duplicates_ignore_provenance_keep_first(self):
        first = record("a", provenance={"source": "first"})
        second = record("a", provenance={"source": "second"})
        result = resolver.resolve_records([first, record("b"), second], [])
        self.assertEqual(result["clean"]["train"], [first, record("b")])
        self.assertEqual(result["summary"]["train"]["identical_duplicates_removed_after_exclusions"], {"a": 1})
        self.assertEqual(result["quarantine"]["train"], [])

    def test_text_or_label_conflicts_quarantine_all_occurrences(self):
        for different in [record("a", text="Changed."), record("a", title="Changed")]:
            train = [record("a"), different, record("a"), record("b")]
            with self.subTest(different=different):
                result = resolver.resolve_records(train, [])
                self.assertEqual([row["video_id"] for row in result["clean"]["train"]], ["b"])
                self.assertEqual(len(result["quarantine"]["train"]), 3)
                self.assertEqual(result["summary"]["train"]["conflict_ids"], ["a"])

    def test_cross_split_exclusion_preserves_validation(self):
        validation = [record("shared", "validation", text="Held out.")]
        result = resolver.resolve_records([record("shared"), record("keep")], validation)
        self.assertEqual([row["video_id"] for row in result["clean"]["train"]], ["keep"])
        self.assertEqual(result["clean"]["validation"], validation)
        self.assertEqual(len(result["excluded_train"]), 1)

    def test_quarantined_original_validation_id_still_excluded_from_train(self):
        validation = [record("shared", "validation"), record("shared", "validation", title="Conflict")]
        result = resolver.resolve_records([record("shared"), record("keep")], validation)
        self.assertEqual(result["clean"]["validation"], [])
        self.assertEqual([row["video_id"] for row in result["clean"]["train"]], ["keep"])
        self.assertTrue(result["postconditions"]["no_original_validation_id_in_train"])

    def test_conflict_and_cross_split_overlap_is_counted_once(self):
        train = [record("shared"), record("shared", title="Conflict"), record("keep")]
        result = resolver.resolve_records(train, [record("shared", "validation")])
        summary = result["summary"]["train"]
        self.assertEqual(summary["quarantined_conflict_rows"], 2)
        self.assertEqual(summary["conflict_rows_also_excluded_by_validation_priority"], 2)
        self.assertEqual(sum(summary["exclusive_removed_counts"].values()), 2)

    def test_resolution_does_not_mutate_records(self):
        train = [record("a"), record("a")]
        snapshot = copy.deepcopy(train)
        resolver.resolve_records(train, [])
        self.assertEqual(train, snapshot)

    def test_smoke_manifest_is_deterministic_and_regenerated(self):
        rows = [record(str(index), text="Sentence.") for index in range(20)]
        for index, row in enumerate(rows):
            row["sentences"] *= index + 1
        chosen = resolver.select_smoke(rows)
        self.assertEqual([len(row["sentences"]) for row in chosen], [3, 11, 18])
        self.assertEqual(resolver.select_smoke(list(reversed(rows))), chosen)

    def test_full_verified_source_can_pass_resolved_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, [record("a"), record("a"), record("shared")],
                         [record("shared", "validation"), record("shared", "validation")])
            original = {path.name: path.read_bytes() for path in source.iterdir()}
            report = resolver.resolve_directory(source, root / "resolved")
            self.assertTrue(report["resolved_integrity_gate_passed"])
            self.assertEqual(report["original_and_resolved_counts"]["train"]["resolved_rows"], 1)
            self.assertEqual(report["original_and_resolved_counts"]["validation"]["resolved_rows"], 1)
            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, original)
            self.assertTrue((root / "resolved" / "smoke_train.jsonl").exists())

    def test_partial_or_invalid_source_never_passes_resolved_gate(self):
        for changes in [{"partial_train_validation": True}, {"invalid_rows": 1},
                        {"error_counts": {"source_error": 1}}, {"files_sha256": {}}]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_source(root / "source", [record("a")], [record("b", "validation")], **changes)
                report = resolver.resolve_directory(root / "source", root / "resolved")
                self.assertFalse(report["resolved_integrity_gate_passed"])

    def test_test_overlap_or_missing_evidence_never_passes_gate(self):
        for overlaps in [{}, {"train__test": {"count": 1, "video_ids": ["a"]},
                              "validation__test": {"count": 0, "video_ids": []}}]:
            with self.subTest(overlaps=overlaps), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_source(root / "source", [record("a")], [record("b", "validation")],
                             split_overlaps=overlaps)
                report = resolver.resolve_directory(root / "source", root / "resolved")
                self.assertFalse(report["resolved_integrity_gate_passed"])

    def test_changed_source_hash_fails_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, [record("a")], [record("b", "validation")])
            (source / "train.jsonl").write_text(json.dumps(record("changed")) + "\n")
            report = resolver.resolve_directory(source, root / "resolved")
            self.assertFalse(report["resolved_integrity_gate_passed"])
            self.assertFalse(report["source_gate_evidence"]["source_canonical_hashes_match_audit"])

    def test_empty_resolved_validation_fails_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_source(root / "source", [record("a")],
                         [record("b", "validation"), record("b", "validation", title="Conflict")])
            report = resolver.resolve_directory(root / "source", root / "resolved")
            self.assertFalse(report["resolved_integrity_gate_passed"])
            self.assertFalse(report["postconditions"]["validation_nonempty"])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_source(root / "source", [record("a")], [record("b", "validation")])
            (root / "resolved").mkdir()
            with self.assertRaisesRegex(ValueError, "new directory"):
                resolver.resolve_directory(root / "source", root / "resolved")

    def test_record_split_never_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "split"):
            resolver.resolve_records([record("a", "validation")], [])


if __name__ == "__main__":
    unittest.main()
