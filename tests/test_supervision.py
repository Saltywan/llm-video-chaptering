"""Native-template final-answer mask tests without torch/tokenizer packages."""

import hashlib
import unittest

from chaptering.supervision import GEMMA_CLOSED_THOUGHT, IGNORE_INDEX, SupervisionError, prepare_causal_example


class FakeFastTokenizer:
    """Character tokenizer with native control-token IDs and optional faults."""

    is_fast = True
    pad_token_id = 0
    unk_token_id = 2

    def __init__(self, family="qwen", fault=None, empty_terminator_offset=False,
                 padding=False):
        self.family = family
        self.fault = fault
        self.empty_terminator_offset = empty_terminator_offset
        self.padding = padding
        self.eos_token_id = 248046 if family == "qwen" else 1
        self.eos_token = "<|im_end|>" if family == "qwen" else "<eos>"
        self.term = self.eos_token if family == "qwen" else "<turn|>"
        self.term_id = 248046 if family == "qwen" else 99
        self.tokenization_calls = 0
        self.template_calls = []
        self.encoding = None
        self.id_to_text = {0: "<pad>", 1: "<eos>", 2: "<unk>", self.term_id: self.term}

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        if self.family == "qwen":
            prefix = ("<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
                      + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            if self.fault == "unclosed_thinking":
                prefix = prefix.replace("</think>\n\n", "")
            if self.fault == "nonempty_thinking":
                prefix = prefix.replace("<think>\n\n", "<think>invented reasoning\n")
        else:
            prefix = ("<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n"
                      + messages[0]["content"] + "<turn|>\n<|turn>model\n")
            if self.fault == "unclosed_thinking":
                prefix += "<|channel>thought\n"
            if self.fault == "missing_control":
                prefix = prefix.replace("<|think|>", "")
        target = messages[-1]["content"]
        suffix = self.term + "\n"
        if self.fault == "wrong_suffix":
            suffix = "<eos>\n"
        if self.fault == "prose_suffix":
            suffix += "extra answer"
        self.target_start = len(prefix)
        self.target_end = len(prefix) + len(target)
        self.rendered = prefix + target + suffix
        return self.rendered

    def convert_tokens_to_ids(self, token):
        return self.term_id if token == self.term else self.unk_token_id

    def __call__(self, rendered, **kwargs):
        self.tokenization_calls += 1
        assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True,
                          "return_attention_mask": True, "truncation": False, "padding": False}
        ids, offsets = [], []
        position = 0
        while position < len(rendered):
            if rendered.startswith(self.term, position):
                ids.append(self.term_id)
                offsets.append((0, 0) if self.empty_terminator_offset
                               else (position, position + len(self.term)))
                position += len(self.term)
            else:
                character = rendered[position]
                token_id = ord(character) + 1000
                self.id_to_text[token_id] = character
                ids.append(token_id)
                offsets.append((position, position + 1))
                position += 1
        mask = [1] * len(ids)
        if self.fault == "cross_prefix":
            index = next(i for i, offset in enumerate(offsets) if offset[0] == self.target_start)
            offsets[index] = (self.target_start - 1, offsets[index][1])
        if self.fault == "cross_suffix":
            index = next(i for i, offset in enumerate(offsets) if offset[1] == self.target_end)
            offsets[index] = (offsets[index][0], self.target_end + 1)
        if self.fault == "wrong_id":
            index = next(i for i, offset in enumerate(offsets) if offset == (self.target_end, self.target_end + len(self.term)))
            ids[index] = 1
        if self.fault == "normalization":
            self.id_to_text[ord("é") + 1000] = "e"
        if self.fault == "internal_padding":
            mask[len(mask) // 2] = 0
        if self.padding:
            ids = [self.pad_token_id] + ids + [self.pad_token_id, self.pad_token_id]
            offsets = [(0, 0)] + offsets + [(0, 0), (0, 0)]
            mask = [0] + mask + [0, 0]
        self.encoding = {"input_ids": ids, "offset_mapping": offsets, "attention_mask": mask}
        return self.encoding

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(self.id_to_text[token_id] for token_id in ids)


class SupervisionTests(unittest.TestCase):
    target = '{"chapters":[{"start":0,"title":"Café 日本語"}]}'

    def prepare(self, family="qwen", **kwargs):
        tokenizer = FakeFastTokenizer(family, **kwargs)
        example = prepare_causal_example(tokenizer, "[0] A transcript.", self.target, family)
        return tokenizer, example

    def test_native_template_and_one_full_tokenization(self):
        for family in ("qwen", "gemma"):
            with self.subTest(family=family):
                tokenizer, example = self.prepare(family)
                self.assertEqual(tokenizer.tokenization_calls, 1)
                self.assertTrue(example["audit"]["supervision_gate_passed"])
                self.assertEqual(tokenizer.template_calls[0][1], {
                    "tokenize": False, "add_generation_prompt": False, "enable_thinking": True})
                self.assertEqual(tokenizer.template_calls[0][0][-1]["content"], self.target)
                supervised = [token for token in example["labels"] if token != IGNORE_INDEX]
                self.assertEqual(tokenizer.decode(supervised, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False),
                                 self.target + tokenizer.term)

    def test_unicode_decode_hash_and_only_one_terminator_supervised(self):
        for family in ("qwen", "gemma"):
            tokenizer, example = self.prepare(family)
            audit = example["audit"]
            self.assertEqual(audit["supervised_json_sha256"],
                             hashlib.sha256(self.target.encode("utf-8")).hexdigest())
            self.assertEqual(sum(token == tokenizer.term_id for token in example["labels"]), 1)
            self.assertEqual(audit["supervised_tokens"], audit["json_token_count"] + 1)
            self.assertEqual(audit["supervised_reasoning_tokens"], 0)
            self.assertTrue(all(label == IGNORE_INDEX for label in example["labels"][:audit["json_token_start"]]))
            self.assertTrue(all(label == IGNORE_INDEX for label in example["labels"][audit["termination_token_position"] + 1:]))

    def test_native_empty_thought_is_unscored_context_only(self):
        _, example = self.prepare("qwen")
        self.assertTrue(example["audit"]["native_empty_thought_context"])
        self.assertFalse(example["audit"]["native_thought_context_supervised"])
        self.assertEqual(example["audit"]["termination_token_id"], 248046)

    def test_gemma_uses_turn_end_instead_of_ordinary_eos(self):
        tokenizer, example = self.prepare("gemma")
        self.assertEqual(tokenizer.eos_token_id, 1)
        self.assertEqual(example["audit"]["termination_token"], "<turn|>")
        self.assertNotEqual(example["audit"]["termination_token_id"], tokenizer.eos_token_id)
        self.assertNotIn(tokenizer.eos_token_id, example["labels"])

    def test_gemma_v2_closed_thought_block_has_no_prediction_labels(self):
        tokenizer = FakeFastTokenizer("gemma")
        result = prepare_causal_example(tokenizer, "[0] A transcript.", self.target, "M2",
                                        gemma_final_context="closed_thought_v2")
        audit = result["audit"]
        before = tokenizer.decode(result["input_ids"][:audit["json_token_start"]],
                                  skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertTrue(before.endswith("<|turn>model\n" + GEMMA_CLOSED_THOUGHT))
        self.assertTrue(audit["native_block_explicitly_inserted"])
        self.assertTrue(all(label == IGNORE_INDEX for label in result["labels"][:audit["json_token_start"]]))
        scored = [label for label in result["labels"] if label != IGNORE_INDEX]
        self.assertEqual(tokenizer.decode(scored, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False), self.target + "<turn|>")

    def test_gemma_v2_keeps_partial_title_masks_and_real_unicode_title(self):
        target = '{"chapters":[{"start":0,"title":"Chapter"},{"start":1,"title":"日本語"}]}'
        start = target.index("Chapter")
        tokenizer = FakeFastTokenizer("gemma")
        result = prepare_causal_example(tokenizer, "[0] First. [1] Second.", target, "M2",
                 gemma_final_context="closed_thought_v2", masked_target_spans=[[start, start + 7]])
        scored = tokenizer.decode([label for label in result["labels"] if label != IGNORE_INDEX],
                                   skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertNotIn("Chapter", scored)
        self.assertIn("日本語", scored)
        self.assertTrue(scored.endswith("<turn|>"))

    def test_gemma_context_override_rejects_other_model_or_unknown_version(self):
        for family, version in [("qwen", "closed_thought_v2"), ("gemma", "unknown")]:
            with self.assertRaises(SupervisionError):
                prepare_causal_example(FakeFastTokenizer(family), "prompt", self.target, family,
                                        gemma_final_context=version)

    def test_empty_special_token_offsets_are_supported(self):
        for family in ("qwen", "gemma"):
            _, example = self.prepare(family, empty_terminator_offset=True)
            self.assertTrue(example["audit"]["termination_has_empty_offset"])

    def test_left_and_right_padding_stay_masked(self):
        for family in ("qwen", "gemma"):
            _, example = self.prepare(family, padding=True)
            self.assertEqual(example["audit"]["padding_tokens"], 3)
            self.assertEqual([label for label, bit in zip(example["labels"], example["attention_mask"])
                              if bit == 0], [IGNORE_INDEX] * 3)

    def test_repeated_target_in_prompt_only_supervises_last_assistant_target(self):
        tokenizer = FakeFastTokenizer()
        example = prepare_causal_example(tokenizer, "Example to discuss: " + self.target,
                                         self.target, "M1")
        self.assertEqual(example["audit"]["target_char_span"][0], tokenizer.rendered.rfind(self.target))
        self.assertEqual(example["audit"]["json_token_count"], len(self.target))

    def test_fail_closed_on_boundary_thinking_termination_or_padding_faults(self):
        for family, fault in (("qwen", "cross_prefix"), ("gemma", "cross_suffix"),
                              ("qwen", "unclosed_thinking"), ("gemma", "unclosed_thinking"),
                              ("qwen", "nonempty_thinking"), ("gemma", "missing_control"),
                              ("gemma", "wrong_suffix"), ("qwen", "prose_suffix"),
                              ("gemma", "wrong_id"), ("qwen", "normalization"),
                              ("qwen", "internal_padding")):
            with self.subTest(family=family, fault=fault), self.assertRaises(SupervisionError):
                self.prepare(family, fault=fault)

    def test_invalid_outline_family_or_slow_tokenizer(self):
        bad_targets = ["{}", "```json\n{}\n```", " " + self.target,
                       '{"chapters":[{"start":true,"title":"X"}]}',
                       '{"chapters":[{"start":0,"title":"X"}],"chapters":[]}',
                       '{"chapters":[{"start":0,"title":""}]}']
        for target in bad_targets:
            with self.subTest(target=target), self.assertRaises(SupervisionError):
                prepare_causal_example(FakeFastTokenizer(), "prompt", target, "qwen")
        with self.assertRaises(SupervisionError):
            prepare_causal_example(FakeFastTokenizer(), "prompt", self.target, "other")
        tokenizer = FakeFastTokenizer()
        tokenizer.is_fast = False
        with self.assertRaises(SupervisionError):
            prepare_causal_example(tokenizer, "prompt", self.target, "qwen")

    def test_pinned_qwen_termination_id_cannot_drift(self):
        tokenizer = FakeFastTokenizer()
        tokenizer.term_id = 123
        tokenizer.id_to_text[123] = tokenizer.term
        with self.assertRaises(SupervisionError):
            prepare_causal_example(tokenizer, "prompt", self.target, "qwen")


if __name__ == "__main__":
    unittest.main()
