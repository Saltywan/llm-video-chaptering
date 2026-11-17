import re
import unittest

from chaptering.envelope import strip_envelope_v1
from chaptering.windowing import build_windows

JSON = '{"chapters":[{"start":0,"title":"A"}]}'


class EnvelopeTests(unittest.TestCase):
    def test_strict_text_is_untouched(self):
        self.assertEqual(strip_envelope_v1(JSON), (JSON, []))
        self.assertEqual(strip_envelope_v1("  " + JSON + "\n"), (JSON, []))

    def test_removes_one_whole_answer_fence(self):
        for fenced in [f"```json\n{JSON}\n```", f"```\n{JSON}\n```", f" ```JSON \n{JSON}```\n"]:
            self.assertEqual(strip_envelope_v1(fenced), (JSON, ["markdown_fence"]))

    def test_removes_trailing_period_only_after_closing_brace(self):
        self.assertEqual(strip_envelope_v1(JSON + "."), (JSON, ["trailing_period"]))
        self.assertEqual(strip_envelope_v1(JSON + " .\n"), (JSON, ["trailing_period"]))
        self.assertEqual(strip_envelope_v1("Done."), ("Done.", []))
        self.assertEqual(strip_envelope_v1(f"```json\n{JSON}.\n```"), (JSON, ["markdown_fence", "trailing_period"]))

    def test_does_not_repair_content(self):
        for text in [f"Here you go:\n```json\n{JSON}\n```", f"```json\n{JSON}\n```\nHope this helps",
                     f"```\n{JSON}\n```\n```\n{JSON}\n```", JSON + " and more", JSON[:-1]]:
            inner, removed = strip_envelope_v1(text)
            self.assertEqual((inner, removed), (text.strip(), []))
        title_with_period = '{"chapters":[{"start":0,"title":"End."}]}'
        self.assertEqual(strip_envelope_v1(title_with_period), (title_with_period, []))

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            strip_envelope_v1(None)


class BisectCoreSearchTests(unittest.TestCase):
    def test_bisect_matches_exhaustive_for_monotone_counters(self):
        def words(prompt):
            return len(prompt.split())

        for n, budget in [(1, 400), (40, 400), (300, 400), (300, 700), (300, 5000)]:
            row = {"schema_version": 1, "video_id": "fixture", "split": "train",
                   "sentences": [f"Sentence number {i} has " + "word " * (i % 7) + "end." for i in range(n)]}
            kwargs = {"max_core_sentences": 100000, "left_context": 4, "right_context": 4}
            exhaustive = build_windows(row, {"w": words}, budget, **kwargs)
            bisect = build_windows(row, {"w": words}, budget, core_search="bisect", **kwargs)
            self.assertEqual(bisect, exhaustive)
            self.assertEqual(bisect[-1]["B"], n)
            self.assertTrue(all(w["input_token_counts"]["w"] <= budget for w in bisect))

    def test_unknown_search_rejected(self):
        row = {"schema_version": 1, "video_id": "fixture", "split": "train", "sentences": ["One."]}
        with self.assertRaises(ValueError):
            build_windows(row, {"w": lambda p: len(re.findall(r"\S+", p))}, 400, core_search="fast")


if __name__ == "__main__":
    unittest.main()
