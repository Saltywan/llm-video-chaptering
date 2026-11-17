import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora import collate, learning_rate_at, plan_token_batches


def item(index, tokens, thought=""):
    row = {"window": {"video_id": f"v{index}", "window_index": 0, "input_token_counts": {"M1": tokens}}}
    if thought:
        row["thought_prefix"] = thought
    return row


class PlanTokenBatchesTests(unittest.TestCase):
    def test_zero_budget_keeps_one_window_per_step_in_order(self):
        order = [item(i, 100 * (i + 1)) for i in range(5)]
        self.assertEqual(plan_token_batches(order, "M1", 0, 1), [[x] for x in order])

    def test_every_window_once_and_budget_respected(self):
        order = [item(i, 500 + 37 * (i * 7 % 50)) for i in range(200)] + [item(999, 9000)]
        batches = plan_token_batches(order, "M1", 8000, 393)
        flat = [x["window"]["video_id"] for batch in batches for x in batch]
        self.assertEqual(sorted(flat), sorted(x["window"]["video_id"] for x in order))
        for batch in batches:
            longest = max(x["window"]["input_token_counts"]["M1"] + 256 for x in batch)
            self.assertTrue(len(batch) == 1 or len(batch) * longest <= 8000)
        self.assertGreater(max(len(batch) for batch in batches), 1)
        self.assertEqual(batches, plan_token_batches(order, "M1", 8000, 393))
        capped = plan_token_batches(order, "M1", 8000, 393, max_rows=3)
        self.assertEqual(max(len(batch) for batch in capped), 3)
        self.assertEqual(sorted(x["window"]["video_id"] for batch in capped for x in batch), sorted(flat))
        self.assertNotEqual(batches, plan_token_batches(order, "M1", 8000, 394))

    def test_thought_length_counts_toward_the_estimate(self):
        order = [item(0, 1000, "x" * 30000), item(1, 1000, "x" * 30000)]
        self.assertEqual([len(b) for b in plan_token_batches(order, "M1", 12000, 1)], [1, 1])


class LearningRateTests(unittest.TestCase):
    def test_constant_warmup_and_cosine(self):
        self.assertEqual(learning_rate_at(5, 100, 1e-4, 0, "constant"), 1e-4)
        self.assertAlmostEqual(learning_rate_at(0, 100, 1e-4, 10, "cosine"), 1e-5)
        self.assertAlmostEqual(learning_rate_at(9, 100, 1e-4, 10, "cosine"), 1e-4)
        self.assertAlmostEqual(learning_rate_at(10, 100, 1e-4, 10, "cosine"), 1e-4)
        self.assertAlmostEqual(learning_rate_at(55, 100, 1e-4, 10, "cosine"), 5e-5)
        self.assertLess(learning_rate_at(99, 100, 1e-4, 10, "cosine"), 1e-6)


class CollateTests(unittest.TestCase):
    def test_right_padding_has_no_attention_or_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed here")
        def one(ids, labels):
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    "labels": torch.tensor([labels])}
        batch = collate(torch, [one([5, 6, 7], [-100, 6, 7]), one([8], [8])], 0, False)
        self.assertEqual(batch["input_ids"].tolist(), [[5, 6, 7], [8, 0, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 6, 7], [8, -100, -100]])


if __name__ == "__main__":
    unittest.main()


class SelectiveLogitTests(unittest.TestCase):
    def test_tail_covers_every_supervised_label_and_matches_full_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from train_lora import selective_causal_loss, supervised_tail

        torch.manual_seed(0)
        vocab = 11
        table = torch.nn.Embedding(vocab, 6)
        head = torch.nn.Linear(6, vocab)

        class Out:
            def __init__(self, logits):
                self.logits = logits

        def model(input_ids, attention_mask, logits_to_keep=0):
            hidden = table(input_ids).cumsum(dim=1)
            hidden = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
            return Out(head(hidden))

        ids = torch.randint(0, vocab, (2, 9))
        labels = torch.full((2, 9), -100)
        labels[0, 5:7] = ids[0, 5:7]   # shorter row, right padded after index 6
        labels[1, 6:9] = ids[1, 6:9]
        mask = torch.ones_like(ids)
        mask[0, 7:] = 0
        self.assertEqual(supervised_tail(labels), 5)
        logits = model(ids, mask).logits
        full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1),
                                                 ignore_index=-100)
        tail = selective_causal_loss(torch, model, {"input_ids": ids, "attention_mask": mask, "labels": labels})
        self.assertTrue(torch.allclose(full, tail))

    def test_unsupervised_batch_is_rejected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from train_lora import supervised_tail
        with self.assertRaises(ValueError):
            supervised_tail(torch.full((1, 4), -100))
