"""Window ownership, visibility, title masking, and strict failure fixtures."""

import copy
import json
import re
import unittest

from chaptering.evaluation import ValidationError
from chaptering.windowing import (
    WindowingError, build_window_prompt, build_windows, parse_window_outline,
    stitch_window_predictions, validate_window_chapters, window_training_target,
)


def record(n=12, starts=None, titles=None):
    starts = [0, 3, 6, 9] if starts is None and n == 12 else ([0] if starts is None else starts)
    titles = [f"Gold {start}" for start in starts] if titles is None else titles
    return {"schema_version": 1, "video_id": "fixture", "split": "train",
            "sentences": [f"Sentence {index}." for index in range(n)],
            "reference": {"chapter_starts": starts, "chapter_titles": titles}}


def manual_window(source, L, R, A, B, index=0):
    return {"schema_version": 1, "video_id": source["video_id"], "window_index": index,
            "sentence_count": len(source["sentences"]), "L": L, "R": R, "A": A, "B": B,
            "sentences": source["sentences"][L:R], "sentence_indices": list(range(L, R))}


def sentence_counter(prompt):
    return len(re.findall(r"^\[\d+\] ", prompt, flags=re.MULTILINE))


def prediction(chapters, **extra):
    return {"status": "ok", "chapters": chapters, "error": None, "usage": {}, **extra}


class WindowPackingTests(unittest.TestCase):
    def test_disjoint_ownership_covers_every_sentence_once_with_global_indices(self):
        source = record(15)
        windows = build_windows(source, {"model": sentence_counter}, 8,
                                max_core_sentences=4, left_context=2, right_context=2)
        owned = [index for window in windows for index in range(window["A"], window["B"])]
        self.assertEqual(owned, list(range(15)))
        self.assertEqual(windows[0]["L"], 0)
        self.assertEqual(windows[-1]["R"], 15)
        for index, window in enumerate(windows):
            self.assertEqual(window["window_index"], index)
            self.assertEqual(window["sentence_indices"], list(range(window["L"], window["R"])))
            self.assertEqual(window["sentences"], source["sentences"][window["L"]:window["R"]])
            self.assertLessEqual(sentence_counter(build_window_prompt(window)), 8)
            if window["A"]:
                self.assertLess(window["L"], window["A"])

    def test_every_counter_and_its_budget_constrain_same_windows(self):
        windows = build_windows(record(13), {"plain": sentence_counter,
                                             "expanded": lambda prompt: sentence_counter(prompt) * 3},
                                {"plain": 20, "expanded": 21}, max_core_sentences=9,
                                left_context=1, right_context=1)
        for window in windows:
            self.assertLessEqual(window["input_token_counts"]["plain"], 20)
            self.assertLessEqual(window["input_token_counts"]["expanded"], 21)
            self.assertLessEqual(window["B"] - window["A"], 6)

    def test_gold_fields_cannot_change_windows_or_prompt(self):
        first = record()
        second = copy.deepcopy(first)
        second["reference"] = {"chapter_starts": [0, 1, 2], "chapter_titles": ["SECRET"] * 3}
        second["gold_marker"] = "SECRET"
        options = {"token_counters": {"model": sentence_counter}, "max_input_tokens": 7,
                   "max_core_sentences": 5, "left_context": 1, "right_context": 1}
        self.assertEqual(build_windows(first, **options), build_windows(second, **options))
        self.assertNotIn("SECRET", build_window_prompt(build_windows(second, **options)[0]))

    def test_preserves_desired_context_before_preferring_larger_core(self):
        windows = build_windows(record(20), {"model": sentence_counter}, 7,
                                max_core_sentences=10, left_context=2, right_context=2)
        self.assertEqual((windows[0]["A"], windows[0]["B"]), (0, 5))
        self.assertEqual((windows[1]["L"], windows[1]["A"], windows[1]["B"], windows[1]["R"]),
                         (3, 5, 8, 10))
        self.assertFalse(windows[1]["context_shrunk"])

    def test_context_shrinks_only_when_single_core_with_desired_context_fails(self):
        windows = build_windows(record(10), {"model": sentence_counter}, 3,
                                max_core_sentences=4, left_context=3, right_context=3)
        self.assertTrue(windows[0]["context_shrunk"])
        for window in windows:
            self.assertLessEqual(window["R"] - window["L"], 3)
            if window["A"]:
                self.assertGreaterEqual(window["left_context_count"], 1)
            if window["B"] < 10 and not window["right_context_omitted_as_infeasible"]:
                self.assertGreaterEqual(window["right_context_count"], 1)

    def test_last_right_context_is_dropped_only_when_infeasible(self):
        windows = build_windows(record(5), {"model": sentence_counter}, 2,
                                max_core_sentences=1, left_context=1, right_context=1)
        self.assertTrue(windows[1]["right_context_omitted_as_infeasible"])
        self.assertEqual(windows[1]["left_context_count"], 1)
        self.assertEqual(windows[1]["right_context_count"], 0)

    def test_oversized_sentence_or_mandatory_left_neighbor_raises_no_truncation(self):
        source = record(2)
        with self.assertRaisesRegex(WindowingError, "no transcript was truncated"):
            build_windows(source, {"model": sentence_counter}, 1,
                          max_core_sentences=1, left_context=1, right_context=1)
        with self.assertRaisesRegex(WindowingError, "no transcript was truncated"):
            build_windows(record(1), {"model": lambda prompt: 100}, 99)

    def test_descending_search_does_not_assume_counter_monotonicity(self):
        def nonmonotonic(prompt):
            return 1 if "ownership region is [0,4)" in prompt else 100
        windows = build_windows(record(4), {"model": nonmonotonic}, 1, max_core_sentences=4)
        self.assertEqual(windows[0]["B"], 4)

    def test_invalid_budgets_and_counter_values_rejected(self):
        for budget in [0, -1, True, {"wrong": 3}]:
            with self.subTest(budget=budget), self.assertRaises(WindowingError):
                build_windows(record(1), {"model": sentence_counter}, budget)
        with self.assertRaises(WindowingError):
            build_windows(record(1), {"model": lambda prompt: True}, 100)


class WindowTargetTests(unittest.TestCase):
    def test_full_owned_chapters_have_true_titles_and_no_masks(self):
        source = record(6, starts=[0, 3], titles=["First", "Second"])
        window = manual_window(source, 0, 6, 0, 6)
        result = window_training_target(source, window)
        self.assertEqual(json.loads(result["target_json"])["chapters"],
                         [{"start": 0, "title": "First"}, {"start": 3, "title": "Second"}])
        self.assertEqual(result["masked_target_spans"], [])
        self.assertEqual(result["title_supervision_stats"]["supervised_titles"], 2)

    def test_partial_and_unowned_titles_never_enter_decoder_context(self):
        source = record(10, starts=[0, 4, 7], titles=["Outside left secret", "Owned complete", "Outside right secret"])
        window = manual_window(source, 2, 9, 4, 7)
        result = window_training_target(source, window)
        target = result["target_json"]
        self.assertEqual(json.loads(target)["chapters"],
                         [{"start": 2, "title": "Chapter"}, {"start": 4, "title": "Owned complete"},
                          {"start": 7, "title": "Chapter"}])
        self.assertNotIn("secret", target)
        self.assertEqual(len(result["masked_target_spans"]), 2)
        for start, end in result["masked_target_spans"]:
            self.assertEqual(target[start:end], "Chapter")
            self.assertEqual(target[start - 1], '"')
            self.assertEqual(target[end], '"')

    def test_owned_partial_chapter_title_is_masked(self):
        source = record(8, starts=[0, 2], titles=["First", "Unseen chapter ending"])
        result = window_training_target(source, manual_window(source, 0, 5, 0, 4))
        self.assertNotIn("Unseen", result["target_json"])
        self.assertEqual(result["title_supervision_stats"]["supervised_titles"], 1)
        self.assertEqual(result["title_supervision_stats"]["masked_titles"], 1)

    def test_unowned_full_chapter_still_uses_placeholder(self):
        source = record(8, starts=[0, 2, 4, 6])
        result = window_training_target(source, manual_window(source, 1, 8, 3, 6))
        chapters = json.loads(result["target_json"])["chapters"]
        self.assertEqual(chapters, [{"start": 1, "title": "Chapter"}, {"start": 2, "title": "Chapter"},
                                    {"start": 4, "title": "Gold 4"}, {"start": 6, "title": "Chapter"}])

    def test_json_escaping_and_placeholder_literal_do_not_corrupt_spans(self):
        title = 'Chapter "quoted" \\ path\n日本語'
        source = record(6, starts=[0, 2, 4], titles=[title, "Chapter", "Never expose"])
        result = window_training_target(source, manual_window(source, 0, 5, 0, 4))
        target = result["target_json"]
        self.assertEqual(json.loads(target)["chapters"][0]["title"], title)
        self.assertEqual(json.loads(target)["chapters"][1]["title"], "Chapter")
        self.assertEqual(len(result["masked_target_spans"]), 1)
        start, end = result["masked_target_spans"][0]
        self.assertEqual(target[start:end], "Chapter")
        self.assertEqual(result["title_supervision_stats"]["supervised_titles"], 2)

    def test_gold_start_at_context_right_edge_is_not_emitted(self):
        source = record(8, starts=[0, 3, 6])
        result = window_training_target(source, manual_window(source, 1, 6, 2, 5))
        self.assertEqual([chapter["start"] for chapter in json.loads(result["target_json"])["chapters"]], [1, 3])

    def test_target_requires_matching_context_and_record(self):
        source = record(6)
        window = manual_window(source, 0, 6, 0, 6)
        window["sentences"][0] = "Different input."
        with self.assertRaisesRegex(WindowingError, "does not match"):
            window_training_target(source, window)


class WindowStitchingTests(unittest.TestCase):
    def setUp(self):
        self.source = record(8, starts=[0, 3, 6])
        self.windows = [manual_window(self.source, 0, 5, 0, 4, 0),
                        manual_window(self.source, 2, 8, 4, 8, 1)]

    def test_processing_anchors_and_unowned_context_starts_do_not_create_cuts(self):
        predictions = [prediction([{"start": 0, "title": "First"}, {"start": 3, "title": "Second"},
                                   {"start": 4, "title": "Unowned"}]),
                       prediction([{"start": 2, "title": "Anchor"}, {"start": 3, "title": "Unowned"},
                                   {"start": 6, "title": "Third"}])]
        result = stitch_window_predictions(self.windows, predictions, run_id="fixture")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([chapter["start"] for chapter in result["chapters"]], [0, 3, 6])
        self.assertNotIn("Anchor", json.dumps(result["chapters"]))

    def test_strict_parser_keeps_global_anchor_and_rejects_malformed_output(self):
        window = self.windows[1]
        self.assertEqual(parse_window_outline('{"chapters":[{"start":2,"title":"Anchor"}]}', window),
                         [{"start": 2, "title": "Anchor"}])
        bad = ['{"chapters":[{"start":0,"title":"Wrong local index"}]}',
               '{"chapters":[{"start":2,"title":"A"},{"start":2,"title":"B"}]}',
               '{"chapters":[{"start":2,"title":"A"},{"start":8,"title":"Out of range"}]}',
               '{"chapters":[{"start":2,"title":" "}]}',
               '{"chapters":[{"start":2,"title":"A","title":"B"}]}',
               '```json\n{"chapters":[]}\n```', '<think>{"chapters":[]}</think>',
               '{"chapters":[{"start":true,"title":"A"}]}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValidationError):
                parse_window_outline(text, window)

    def test_duplicate_or_unsorted_starts_fail_before_ownership_filtering(self):
        first = prediction([{"start": 0, "title": "First"}])
        for starts in [[2, 6, 3], [2, 3, 3], [2, 1]]:
            second = prediction([{"start": start, "title": "Topic"} for start in starts])
            result = stitch_window_predictions(self.windows, [first, second], run_id="fixture")
            self.assertEqual(result["status"], "invalid")
            self.assertEqual(result["chapters"], [])

    def test_any_failed_window_is_whole_video_failure(self):
        first = prediction([{"start": 0, "title": "First"}])
        for status in ["error", "invalid", "truncated"]:
            result = stitch_window_predictions(self.windows, [first, {"status": status, "chapters": []}], run_id="fixture")
            self.assertEqual(result["status"], status)
            self.assertEqual(result["chapters"], [])
            self.assertEqual(result["failed_window_index"], 1)

    def test_missing_prediction_and_wrong_layout_fail_without_repair(self):
        first = prediction([{"start": 0, "title": "First"}])
        self.assertEqual(stitch_window_predictions(self.windows, [first], run_id="fixture")["status"], "error")
        windows = copy.deepcopy(self.windows)
        windows[1]["A"] = 5
        result = stitch_window_predictions(windows, [first, first], run_id="fixture")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["chapters"], [])
        result = stitch_window_predictions(list(reversed(self.windows)), [first, first], run_id="fixture")
        self.assertEqual(result["status"], "invalid")

    def test_invalid_window_schema_and_boolean_indices_fail_without_repair(self):
        predictions = [prediction([{"start": 0, "title": "First"}]),
                       prediction([{"start": 2, "title": "Anchor"}])]
        for field, value in [("schema_version", 2), ("sentence_indices", [False, 1, 2, 3, 4])]:
            windows = copy.deepcopy(self.windows)
            windows[0][field] = value
            with self.subTest(field=field):
                result = stitch_window_predictions(windows, predictions, run_id="fixture")
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["chapters"], [])

    def test_edge_owned_starts_belong_to_exactly_one_core(self):
        predictions = [prediction([{"start": 0, "title": "First"}, {"start": 4, "title": "Unowned"}]),
                       prediction([{"start": 2, "title": "Anchor"}, {"start": 4, "title": "Owned edge"}])]
        result = stitch_window_predictions(self.windows, predictions, run_id="fixture")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["chapters"], [{"start": 0, "title": "First"}, {"start": 4, "title": "Owned edge"}])


if __name__ == "__main__":
    unittest.main()
