"""Selection/pack fixtures use only stdlib and fake tokenizer counters."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_windows.py"
spec = importlib.util.spec_from_file_location("prepare_windows", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(count=40, split="train"):
    return [{"schema_version": 1, "video_id": f"{split}-{index:03}", "split": split,
             "sentences": [f"Sentence {position}." for position in range(index + 1)],
             "reference": {"chapter_starts": [0], "chapter_titles": [f"Secret title {index}"]}}
            for index in range(count)]


class SelectionTests(unittest.TestCase):
    def test_four_bins_receive_equal_quotas(self):
        selected = module.select_length_stratified(rows(), 8, 393)
        counts = [sum(lower <= len(row["sentences"]) <= upper for row in selected)
                  for lower, upper in [(1, 10), (11, 20), (21, 30), (31, 40)]]
        self.assertEqual(counts, [2, 2, 2, 2])

    def test_selection_order_independent_and_repeatable(self):
        source = rows()
        first = module.select_length_stratified(source, 12, 393)
        self.assertEqual(first, module.select_length_stratified(list(reversed(source)), 12, 393))
        self.assertNotEqual([row["video_id"] for row in first],
                            [row["video_id"] for row in module.select_length_stratified(source, 12, 394)])

    def test_gold_cannot_change_selection(self):
        source = rows()
        changed = copy.deepcopy(source)
        for row in changed:
            row["reference"] = {"chapter_starts": [0, 999], "chapter_titles": ["Different", "Gold"]}
        first = module.select_length_stratified(source, 12)
        second = module.select_length_stratified(changed, 12)
        self.assertEqual([row["video_id"] for row in first], [row["video_id"] for row in second])

    def test_small_bins_zero_and_unequal_quotas(self):
        self.assertEqual(module.select_length_stratified([], 0), [])
        self.assertEqual(len(module.select_length_stratified(rows(3), 3)), 3)
        self.assertEqual(len(module.select_length_stratified(rows(11), 7)), 7)
        with self.assertRaises(ValueError):
            module.select_length_stratified(rows(3), 4)

    def test_duplicate_ids_rejected_and_ties_use_id(self):
        source = rows(10)
        for row in source:
            row["sentences"] = ["Same length."]
        self.assertEqual(module.select_length_stratified(source, 4),
                         module.select_length_stratified(list(reversed(source)), 4))
        source.append(source[0])
        with self.assertRaises(ValueError):
            module.select_length_stratified(source, 4)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def fake_load(root, input_tokens):
    counter = lambda text: len(re.findall(r"^\[\d+\] ", text, flags=re.MULTILINE))
    return ({alias: FakeTokenizer() for alias in module.ALIASES},
            {alias: counter for alias in module.ALIASES},
            {alias: {"revision": "fixture"} for alias in module.ALIASES}, {})


def source_fixture(root):
    source = root / "data"
    source.mkdir()
    for split in ["train", "validation"]:
        (source / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows(8, split)))
    resolution = {"revision": "fixture", "resolved_integrity_gate_passed": True,
                  "output_files_sha256": {f"{split}.jsonl": module.sha256(source / f"{split}.jsonl")
                                          for split in ["train", "validation"]}}
    (source / "resolution.json").write_text(json.dumps(resolution))
    return source


def args(source, output, **changes):
    value = argparse.Namespace(data=source, output=output, train_videos=4, validation_videos=4,
                               input_tokens=8, core_sentences=4, context_sentences=1,
                               seed=393, census_all=False)
    for key, item in changes.items():
        setattr(value, key, item)
    return value


class PreparationTests(unittest.TestCase):
    def test_pack_separates_validation_gold_and_hashes_every_output(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(module, "load_tokenizers", fake_load):
            root = Path(temp)
            source = source_fixture(root)
            manifest = module.prepare(args(source, root / "out"))
            self.assertTrue(manifest["preparation_gate_passed"])
            self.assertFalse(manifest["full_corpus_census_passed"])
            validation = [json.loads(line) for line in (root / "out/validation_windows.jsonl").read_text().splitlines()]
            self.assertTrue(validation)
            for packet in validation:
                self.assertEqual(set(packet), {"window"})
                self.assertNotIn("reference", packet["window"])
                self.assertNotIn("Secret title", json.dumps(packet))
            training = [json.loads(line) for line in (root / "out/train_windows.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(packet) == {"window", "target"} for packet in training))
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(module.sha256(root / "out" / name), digest)

    def test_census_zero_selection_processes_all_but_writes_no_pack_records(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(module, "load_tokenizers", fake_load):
            root = Path(temp)
            source = source_fixture(root)
            manifest = module.prepare(args(source, root / "out", train_videos=0, validation_videos=0, census_all=True))
            self.assertTrue(manifest["full_corpus_census_passed"])
            for split in ["train", "validation"]:
                self.assertEqual(manifest["statistics"][split]["processed"]["attempted_videos"], 8)
                self.assertEqual((root / "out" / f"{split}_windows.jsonl").read_text(), "")
                self.assertEqual((root / "out" / f"{split}_records.jsonl").read_text(), "")

    def test_impossible_video_is_recorded_and_fails_gate(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(module, "load_tokenizers", fake_load):
            root = Path(temp)
            source = source_fixture(root)
            manifest = module.prepare(args(source, root / "out", input_tokens=1))
            self.assertFalse(manifest["preparation_gate_passed"])
            self.assertGreater(manifest["failure_count"], 0)
            failures = [json.loads(line) for line in (root / "out/failures.jsonl").read_text().splitlines()]
            self.assertEqual(len(failures), manifest["failure_count"])
            self.assertTrue(all(failure["selected"] for failure in failures))

    def test_empty_cuda_mask_new_output_and_integrity_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_fixture(root)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaisesRegex(ValueError, "empty CUDA"):
                    module.prepare(args(source, root / "out"))
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                (root / "existing").mkdir()
                with self.assertRaisesRegex(ValueError, "new directory"):
                    module.prepare(args(source, root / "existing"))
                with self.assertRaisesRegex(ValueError, "only with --census-all"):
                    module.prepare(args(source, root / "out", train_videos=0))
                (source / "train.jsonl").write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "audit hashes"):
                    module.prepare(args(source, root / "out"))


if __name__ == "__main__":
    unittest.main()
