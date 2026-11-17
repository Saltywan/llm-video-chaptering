"""Native channel budget checks with minimal tensor fakes, without torch."""

import math
import unittest

from chaptering.reasoning import NativeThinkingBudget, ReasoningBudgetError, make_thinking_budget_processor


class FakeIds:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class FakeScores:
    def __init__(self, vocabulary=100, batch=1):
        self.shape = (batch, vocabulary)
        self.values = [[0.5] * vocabulary for _ in range(batch)]

    def fill_(self, value):
        for row in self.values:
            row[:] = [value] * len(row)
        return self

    def __setitem__(self, key, value):
        row, column = key
        self.values[row][column] = value


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"</think>\n\n": [8, 9], "</think>": [8],
                "<|channel>thought\n": [5, 6], "<channel|>": [7]}[text]


class ReasoningTests(unittest.TestCase):
    def apply(self, processor, tail, prompt=(1, 2), vocabulary=100):
        scores = FakeScores(vocabulary)
        result = processor(FakeIds([list(prompt) + list(tail)]), scores)
        self.assertIs(result, scores)
        return scores

    def assert_forced(self, scores, expected):
        self.assertEqual(scores.values[0][expected], 0.0)
        self.assertTrue(all(math.isinf(value) and value < 0
                            for index, value in enumerate(scores.values[0]) if index != expected))

    def test_prompt_tokens_do_not_count_and_before_limit_is_unchanged(self):
        processor = NativeThinkingBudget(2, 3, [8, 9])
        scores = self.apply(processor, [3, 4])
        self.assertEqual(scores.values[0], [0.5] * 100)
        self.assertFalse(processor.audit["force_triggered"])
        self.assert_forced(self.apply(processor, [3, 4, 5]), 8)

    def test_multi_token_close_one_per_step_then_final_is_unmodified(self):
        processor = NativeThinkingBudget(2, 2, [8, 9])
        self.assert_forced(self.apply(processor, [3, 4]), 8)
        self.assert_forced(self.apply(processor, [3, 4, 8]), 9)
        scores = self.apply(processor, [3, 4, 8, 9])
        self.assertEqual(scores.values[0], [0.5] * 100)
        self.assertEqual(self.apply(processor, [3, 4, 8, 9, 10, 11]).values[0], [0.5] * 100)
        self.assertEqual(processor.audit["forced_token_count"], 2)
        self.assertTrue(processor.audit["closing_sequence_complete"])

    def test_natural_early_full_close_is_never_forced(self):
        processor = NativeThinkingBudget(2, 2, [8, 9])
        self.assertEqual(self.apply(processor, [8, 9, 3, 4, 5]).values[0], [0.5] * 100)
        self.assertTrue(processor.audit["closed_naturally_before_forcing"])
        self.assertFalse(processor.audit["force_triggered"])

    def test_semantic_close_without_newline_prevents_duplicate_qwen_close(self):
        processor = NativeThinkingBudget(2, 2, [8, 9], closed_marker_token_ids=[8])
        self.assertEqual(self.apply(processor, [8, 3, 4, 5]).values[0], [0.5] * 100)
        self.assertFalse(processor.audit["force_triggered"])
        self.assertTrue(processor.audit["reasoning_channel_closed"])
        self.assertFalse(processor.audit["closing_sequence_complete"])

    def test_forced_sequence_finishes_even_after_semantic_close_marker(self):
        processor = NativeThinkingBudget(2, 2, [8, 9], closed_marker_token_ids=[8])
        self.assert_forced(self.apply(processor, [3, 4]), 8)
        self.assert_forced(self.apply(processor, [3, 4, 8]), 9)
        self.assertFalse(processor.audit["closing_sequence_complete"])
        self.apply(processor, [3, 4, 8, 9])
        self.assertTrue(processor.audit["closing_sequence_complete"])

    def test_partial_close_at_limit_forces_only_remaining_tokens(self):
        processor = NativeThinkingBudget(2, 3, [8, 9, 10])
        self.assert_forced(self.apply(processor, [3, 8, 9]), 10)
        self.apply(processor, [3, 8, 9, 10])
        self.assertEqual(processor.audit["forced_token_count"], 1)

    def test_direct_gemma_final_without_native_opener_is_never_forced(self):
        processor = NativeThinkingBudget(2, 2, [7], [5, 6])
        self.assertEqual(self.apply(processor, [20] * 20).values[0], [0.5] * 100)
        self.assertFalse(processor.audit["reasoning_channel_observed"])
        self.assertFalse(processor.audit["force_triggered"])

    def test_gemma_opener_counts_in_generated_budget(self):
        processor = NativeThinkingBudget(2, 3, [7], [5, 6])
        self.assertEqual(self.apply(processor, [5, 6]).values[0], [0.5] * 100)
        self.assert_forced(self.apply(processor, [5, 6, 3]), 7)
        self.assertTrue(processor.audit["reasoning_channel_observed"])
        self.assertEqual(processor.audit["generated_tokens_at_first_force"], 3)

    def test_gemma_naturally_closed_channel_is_unmodified(self):
        processor = NativeThinkingBudget(2, 3, [7], [5, 6])
        self.assertEqual(self.apply(processor, [5, 6, 3, 7, 20, 21]).values[0], [0.5] * 100)
        self.assertFalse(processor.audit["force_triggered"])

    def test_repeated_logits_calls_do_not_inflate_forced_token_count(self):
        processor = NativeThinkingBudget(2, 2, [8, 9])
        self.apply(processor, [3, 4])
        self.apply(processor, [3, 4])
        self.assertEqual(processor.audit["forced_token_count"], 1)

    def test_invalid_budget_markers_batch_and_request_reuse_are_rejected(self):
        for kwargs in ({"reasoning_budget": 0}, {"reasoning_budget": True},
                       {"prompt_length": -1}, {"close_token_ids": []},
                       {"close_token_ids": [True]}, {"open_token_ids": [8, 9]},
                       {"closed_marker_token_ids": [7]}):
            arguments = {"prompt_length": 2, "reasoning_budget": 2, "close_token_ids": [8, 9]}
            arguments.update(kwargs)
            with self.subTest(arguments=arguments), self.assertRaises(ReasoningBudgetError):
                NativeThinkingBudget(**arguments)
        processor = NativeThinkingBudget(2, 2, [8, 9])
        with self.assertRaises(ReasoningBudgetError):
            processor(FakeIds([[1, 2], [1, 2]]), FakeScores(batch=2))
        with self.assertRaises(ReasoningBudgetError):
            processor(FakeIds([[1, -1]]), FakeScores())
        self.apply(processor, [3])
        with self.assertRaises(ReasoningBudgetError):
            self.apply(processor, [4])

    def test_factory_encodes_native_markers_without_extra_special_tokens(self):
        tokenizer = FakeTokenizer()
        qwen = make_thinking_budget_processor(tokenizer, "M1", 2, 3)
        self.assertEqual(qwen.audit["family"], "qwen3.5")
        self.assertEqual(qwen.audit["closed_marker_token_ids"], [8])
        gemma = make_thinking_budget_processor(tokenizer, "M2", 2, 3)
        self.assertEqual(gemma.audit["family"], "gemma4")
        self.assertFalse(gemma.audit["reasoning_channel_observed"])
        self.assertTrue(all(kwargs == {"add_special_tokens": False} for _, kwargs in tokenizer.calls))


if __name__ == "__main__":
    unittest.main()
