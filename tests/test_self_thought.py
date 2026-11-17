"""self_thought_v3: the base model's own closed thought is unscored context only."""

import unittest

from chaptering.supervision import (IGNORE_INDEX, SupervisionError, prepare_causal_example,
                                    thought_prefix_from_raw)
from test_supervision import FakeFastTokenizer

TARGET = '{"chapters":[{"start":0,"title":"Café"}]}'
QWEN_RAW = "Thinking Process:\n1. Sentence 0 opens the topic.\n</think>\n\n" + '{"chapters":[]}<|im_end|>'
GEMMA_RAW = "<|channel>thought\n* One topic only.\n<channel|>" + '{"chapters":[]}<turn|>'


class GenerationTokenizer(FakeFastTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if not kwargs.get("add_generation_prompt"):
            return super().apply_chat_template(messages, **kwargs)
        self.template_calls.append((messages, kwargs))
        if self.family == "qwen":
            return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        control = "" if self.fault == "missing_control" else "<|think|>"
        return ("<bos><|turn>system\n" + control + "\n<turn|>\n<|turn>user\n"
                + messages[0]["content"] + "<turn|>\n<|turn>model\n")


class ThoughtPrefixTests(unittest.TestCase):
    def test_extracts_through_native_close_and_following_whitespace(self):
        self.assertEqual(thought_prefix_from_raw(QWEN_RAW, "M1"),
                         "Thinking Process:\n1. Sentence 0 opens the topic.\n</think>\n\n")
        self.assertEqual(thought_prefix_from_raw(GEMMA_RAW, "M2"),
                         "<|channel>thought\n* One topic only.\n<channel|>")

    def test_unusable_generations_give_none(self):
        for raw, family in [("no close here", "M1"), ("</think>\n\n{}", "M1"), (None, "M1"),
                            ("a <think> nested\n</think>\n\n{}", "M1"),
                            ("thought without opener<channel|>{}", "M2"),
                            ("<|channel>thought\n<channel|>{}", "M2"),
                            ("<|channel>thought\nrunaway with no close", "M2")]:
            self.assertIsNone(thought_prefix_from_raw(raw, family))
        with self.assertRaises(SupervisionError):
            thought_prefix_from_raw(QWEN_RAW, "llama")


class SelfThoughtSupervisionTests(unittest.TestCase):
    def prepare(self, family, raw, **kwargs):
        tokenizer = GenerationTokenizer(family, **kwargs)
        prefix = thought_prefix_from_raw(raw, family)
        return tokenizer, prefix, prepare_causal_example(tokenizer, "[0] A transcript.", TARGET, family,
                                                         thought_prefix=prefix)

    def test_only_final_json_and_terminator_are_supervised(self):
        for family, raw in (("qwen", QWEN_RAW), ("gemma", GEMMA_RAW)):
            with self.subTest(family=family):
                tokenizer, prefix, example = self.prepare(family, raw)
                self.assertEqual(tokenizer.template_calls[0][1], {
                    "tokenize": False, "add_generation_prompt": True, "enable_thinking": True})
                self.assertEqual(len(tokenizer.template_calls[0][0]), 1)
                supervised = [token for token in example["labels"] if token != IGNORE_INDEX]
                self.assertEqual(tokenizer.decode(supervised, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False), TARGET + tokenizer.term)
                audit = example["audit"]
                self.assertEqual(audit["final_context"], "self_thought_v3")
                self.assertEqual(audit["supervised_reasoning_tokens"], 0)
                self.assertFalse(audit["native_thought_context_supervised"])
                self.assertTrue(audit["self_generated_thought_context"])
                rendered = tokenizer.decode(example["input_ids"], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                self.assertTrue(rendered.endswith(prefix + TARGET + tokenizer.term + "\n"))
                thought_start = rendered.index(prefix)
                self.assertTrue(all(label == IGNORE_INDEX for label in
                                    example["labels"][:audit["json_token_start"]]))
                self.assertGreater(audit["json_token_start"], thought_start)

    def test_rejects_unclosed_empty_or_framed_thoughts(self):
        cases = [("qwen", "still thinking"), ("qwen", "\n</think>\n\n"), ("qwen", "a\n</think>\n\nprose"),
                 ("qwen", "a<|im_end|>\n</think>\n\n"), ("qwen", "a</think></think>"),
                 ("gemma", "plain text<channel|>"), ("gemma", "<|channel>thought\nx<channel|>tail"),
                 ("gemma", "<|channel>thought\nx<turn|><channel|>")]
        for family, prefix in cases:
            with self.subTest(prefix=prefix), self.assertRaises(SupervisionError):
                prepare_causal_example(GenerationTokenizer(family), "[0] A.", TARGET, family, thought_prefix=prefix)

    def test_cannot_combine_with_other_contexts_or_missing_control(self):
        prefix = thought_prefix_from_raw(GEMMA_RAW, "M2")
        with self.assertRaises(SupervisionError):
            prepare_causal_example(GenerationTokenizer("gemma"), "[0] A.", TARGET, "gemma",
                                   thought_prefix=prefix, gemma_final_context="closed_thought_v2")
        with self.assertRaises(SupervisionError):
            prepare_causal_example(GenerationTokenizer("gemma", fault="missing_control"), "[0] A.", TARGET,
                                   "gemma", thought_prefix=prefix)

    def test_partial_title_masks_still_apply(self):
        tokenizer = GenerationTokenizer("qwen")
        start = TARGET.index("Café")
        example = prepare_causal_example(tokenizer, "[0] A.", TARGET, "qwen",
                                         thought_prefix=thought_prefix_from_raw(QWEN_RAW, "M1"),
                                         masked_target_spans=[[start, start + 4]])
        self.assertEqual(example["audit"]["masked_partial_title_tokens"], 4)


if __name__ == "__main__":
    unittest.main()


class AttachThoughtsTests(unittest.TestCase):
    def test_joins_by_key_and_drops_unusable_generations(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_lora import attach_thoughts

        items = [{"window": {"video_id": "v", "window_index": i}, "target": {}} for i in range(4)]
        rows = [{"video_id": "v", "window_index": 0, "finish_reason": "stop", "raw_text": QWEN_RAW},
                {"video_id": "v", "window_index": 1, "finish_reason": "length", "raw_text": QWEN_RAW},
                {"video_id": "v", "window_index": 2, "finish_reason": "stop", "raw_text": "no close"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "windows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            usable, dropped = attach_thoughts(items, path, "M1")
        self.assertEqual([item["window"]["window_index"] for item in usable], [0])
        self.assertEqual(usable[0]["thought_prefix"], thought_prefix_from_raw(QWEN_RAW, "M1"))
        self.assertNotIn("thought_prefix", items[0])
        self.assertEqual(dropped, {"generation_not_finished": 1, "no_closed_native_thought": 1,
                                   "no_saved_generation": 1})
