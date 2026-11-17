import json
import unittest

from chaptering.training import prepare_seq2seq_example
from chaptering.supervision import prepare_causal_example
from chaptering.windowing import build_windows, build_window_prompt, window_training_target
from test_supervision import FakeFastTokenizer


class TinyTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return "User: " + messages[0]["content"] + "\nAnswer:"

    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(c) + 10 for c in text], "attention_mask": [1] * len(text)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.record = {"video_id": "x", "sentences": ["a"] * 8,
                       "reference": {"chapter_starts": [0, 3, 7], "chapter_titles": ["Alpha", "Hidden long title", "Last"]}}
        self.windows = build_windows(self.record, {"x": lambda prompt: 1}, 10,
                                     max_core_sentences=2, left_context=1, right_context=1)

    def test_seq2seq_mask_keeps_unmasked_teacher_forcing_placeholder(self):
        window = self.windows[1]
        target = window_training_target(self.record, window)
        self.assertTrue(target["masked_target_spans"])
        self.assertNotIn("Hidden long title", target["target_json"])
        result = prepare_seq2seq_example(TinyTokenizer(), build_window_prompt(window), target["target_json"],
                                         window=window, masked_target_spans=target["masked_target_spans"])
        self.assertIn(-100, result["labels"])
        self.assertNotIn(-100, result["full_target_ids"])
        self.assertEqual(result["labels"][-1], 1)
        for index, label in enumerate(result["labels"]):
            if label == -100:
                self.assertGreaterEqual(result["full_target_ids"][index], 10)

    def test_causal_global_indices_and_partial_title_mask(self):
        window = self.windows[1]
        target = window_training_target(self.record, window)
        for family in ["qwen", "gemma"]:
            tokenizer = FakeFastTokenizer(family)
            result = prepare_causal_example(tokenizer, build_window_prompt(window), target["target_json"], family,
                                            window=window, masked_target_spans=target["masked_target_spans"])
            self.assertGreater(window["L"], 0)
            self.assertGreater(result["audit"]["masked_partial_title_tokens"], 0)
            self.assertFalse(result["audit"]["fully_supervised_json"])
            self.assertIsNone(result["audit"]["supervised_json_sha256"])
            end = result["audit"]["termination_token_position"]
            self.assertEqual(result["labels"][end], tokenizer.term_id)

    def test_no_truncation_on_excess_input(self):
        window = self.windows[0]
        target = window_training_target(self.record, window)
        with self.assertRaises(ValueError):
            prepare_seq2seq_example(TinyTokenizer(), build_window_prompt(window), target["target_json"],
                                    window=window, masked_target_spans=target["masked_target_spans"], encoder_limit=2)


if __name__ == "__main__":
    unittest.main()
