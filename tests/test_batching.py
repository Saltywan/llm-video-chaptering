import unittest

from chaptering.batching import plan_batches, split_generated


class PlanBatchesTests(unittest.TestCase):
    def test_every_index_once_and_longest_first(self):
        lengths = [10, 300, 20, 300, 5, 100]
        batches = plan_batches(lengths, max_batch=2, square_budget=10**9)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(lengths))))
        self.assertEqual(batches, [[1, 3], [5, 2], [0, 4]])

    def test_square_budget_shrinks_long_batches_but_never_below_one(self):
        lengths = [8000, 8000, 8000, 1000, 1000, 1000, 1000]
        batches = plan_batches(lengths, max_batch=64, square_budget=2 * 8000 * 8000)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 3])
        for batch in batches:
            longest = max(lengths[i] for i in batch)
            self.assertTrue(len(batch) == 1 or len(batch) * longest * longest <= 2 * 8000 * 8000)
        self.assertEqual(plan_batches([9000], max_batch=8, square_budget=1), [[0]])

    def test_rejects_bad_arguments(self):
        for kwargs in [{"max_batch": 0, "square_budget": 1}, {"max_batch": 1, "square_budget": 0}]:
            with self.assertRaises(ValueError):
                plan_batches([1], **kwargs)
        with self.assertRaises(ValueError):
            plan_batches([0], max_batch=1, square_budget=1)
        self.assertEqual(plan_batches([], max_batch=4, square_budget=4), [])


class SplitGeneratedTests(unittest.TestCase):
    def test_finished_row_keeps_end_token_and_drops_padding(self):
        self.assertEqual(split_generated([5, 6, 1, 0, 0], [1]), ([5, 6, 1], True))
        self.assertEqual(split_generated([5, 107, 1, 0], [1, 107]), ([5, 107], True))

    def test_unfinished_row_is_returned_whole(self):
        self.assertEqual(split_generated([5, 6, 7], [1]), ([5, 6, 7], False))
        self.assertEqual(split_generated([], [1]), ([], False))

    def test_end_token_on_last_position_counts_as_finished(self):
        self.assertEqual(split_generated([5, 6, 1], [1]), ([5, 6, 1], True))


if __name__ == "__main__":
    unittest.main()
