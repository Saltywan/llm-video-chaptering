"""Schema/label checks run with stdlib unittest and no network dependencies."""

import copy
import unittest

from chaptering.data import DataValidationError, canonicalize_row, inference_record, indexed_transcript, outline_target


def example():
    return {"video_id": "fixture", "text_ref": ["Alpha.", "Still alpha.", "Beta.", "Still beta."],
            "target_binary_ref": "|=0010", "chapter_titles": ["Alpha", "Beta"],
            "target_text_ref": "[CSTART] Alpha [CEND] Alpha. Still alpha. [CSTART] Beta [CEND] Beta. Still beta.",
            "chapter_timestamps": [0, 10], "channel_id": "must_not_leak"}


class CanonicalDataTests(unittest.TestCase):
    def test_internal_bits_are_sentence_starts(self):
        record = canonicalize_row(example(), "train", "fixture_revision")
        self.assertEqual(record["reference"]["chapter_starts"], [0, 2])
        self.assertEqual(record["sentences"], example()["text_ref"])

    def test_end_label_interpretation_rejected(self):
        row = example()
        row["target_binary_ref"] = "|=0100"
        with self.assertRaisesRegex(DataValidationError, "binary_marker_start_mismatch"):
            canonicalize_row(row, "train", "revision")

    def test_initial_start_is_not_an_internal_bit(self):
        row = example()
        row["target_binary_ref"] = "|=1010"
        with self.assertRaisesRegex(DataValidationError, "initial_sentence"):
            canonicalize_row(row, "train", "revision")

    def test_inference_is_allowlisted(self):
        record = canonicalize_row(example(), "validation", "revision")
        clean = inference_record(record)
        self.assertEqual(set(clean), {"video_id", "sentences"})
        clean["sentences"].append("Unrelated.")
        self.assertEqual(len(record["sentences"]), 4)
        self.assertNotIn("reference", indexed_transcript(record))
        self.assertEqual(indexed_transcript(record).splitlines()[2], "[2] Beta.")

    def test_target_preserves_start_title_pairs(self):
        record = canonicalize_row(example(), "train", "revision")
        self.assertEqual(outline_target(record), {"chapters": [{"start": 0, "title": "Alpha"},
                                                               {"start": 2, "title": "Beta"}]})

    def test_whitespace_alignment_without_rewriting(self):
        row = example()
        row["text_ref"][0] = "  Alpha.  "
        row["target_text_ref"] = row["target_text_ref"].replace("Alpha. Still", "Alpha.\n Still")
        record = canonicalize_row(row, "train", "revision")
        self.assertEqual(record["sentences"][0], "  Alpha.  ")

    def test_missing_or_changed_sentences_rejected(self):
        for replacement in ["Changed.", ""]:
            row = example()
            row["target_text_ref"] = row["target_text_ref"].replace("Still alpha.", replacement)
            with self.subTest(replacement=replacement), self.assertRaises(DataValidationError):
                canonicalize_row(row, "train", "revision")

    def test_changed_marker_title_rejected(self):
        row = example()
        row["chapter_titles"][1] = "Wrong title"
        with self.assertRaisesRegex(DataValidationError, "marker_title_mismatch"):
            canonicalize_row(row, "train", "revision")

    def test_malformed_binary_fields_rejected(self):
        for bits in ["0010", "|=010", "|=00x0", "|=0000"]:
            row = example()
            row["target_binary_ref"] = bits
            with self.subTest(bits=bits), self.assertRaises(DataValidationError):
                canonicalize_row(row, "train", "revision")

    def test_one_sentence_one_chapter(self):
        row = {"video_id": "one", "text_ref": ["Only sentence."], "target_binary_ref": "|=0",
               "chapter_titles": ["Only"], "target_text_ref": "[CSTART] Only [CEND] Only sentence."}
        self.assertEqual(canonicalize_row(row, "train", "revision")["reference"]["chapter_starts"], [0])

    def test_empty_or_invalid_fields_rejected(self):
        for key, value in [("text_ref", []), ("text_ref", [""]), ("video_id", ""),
                           ("chapter_titles", [" "]), ("target_text_ref", None)]:
            row = copy.deepcopy(example())
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(DataValidationError):
                canonicalize_row(row, "train", "revision")


if __name__ == "__main__":
    unittest.main()
