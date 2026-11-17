"""Synthetic contract checks; no model, data downloads or optional packages."""

import itertools
import json
import unittest

from chaptering.evaluation import (
    ValidationError, aggregate_scores, boundary_scores, chapter_spans,
    evaluate_records, match_boundaries, match_chapter_spans, parse_outline,
    score_prediction, starts_to_gaps, structural_errors, validate_chapters,
    validate_prediction,
)


def reference(starts=(0, 3), n=6, video_id="video"):
    return {"schema_version": 1, "video_id": video_id, "split": "validation",
            "sentences": [f"Sentence {i}." for i in range(n)],
            "reference": {"chapter_starts": list(starts),
                          "chapter_titles": [f"Gold {i}" for i in range(len(starts))]}}


def prediction(starts=(0, 3), *, status="ok", video_id="video"):
    return {"schema_version": 1, "video_id": video_id, "run_id": "synthetic",
            "status": status,
            "chapters": [{"start": start, "title": f"Predicted {i}"}
                         for i, start in enumerate(starts)],
            "error": None if status == "ok" else "synthetic failure", "usage": {}}


class OutputValidationTests(unittest.TestCase):
    def test_strict_final_json_preserves_titles(self):
        chapters = [{"start": 0, "title": "  Example  "}]
        self.assertEqual(parse_outline(json.dumps({"chapters": chapters}), 2), chapters)

    def test_no_reasoning_fences_prose_duplicate_or_extra_fields(self):
        valid = '{"chapters":[{"start":0,"title":"Example"}]}'
        bad = [f"<think>{valid}</think>", f"```json\n{valid}\n```", f"Answer: {valid}",
               valid + valid, '{"chapters":[],"chapters":[{"start":0,"title":"X"}]}',
               '{"chapters":[{"start":0,"title":"X"}],"reasoning":"x"}',
               '{"chapters":[{"start":NaN,"title":"X"}]}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValidationError):
                parse_outline(text, 6)

    def test_invalid_chapter_shapes(self):
        bad = [[], [{"start": True, "title": "X"}], [{"start": 0.0, "title": "X"}],
               [{"start": 1, "title": "X"}], [{"start": 0, "title": "   "}],
               [{"start": 0, "title": "X", "extra": 1}],
               [{"start": 0, "title": "X"}, {"start": 0, "title": "Y"}],
               [{"start": 0, "title": "X"}, {"start": 6, "title": "Y"}],
               [{"start": 0, "title": "X"}, {"start": 4, "title": "Y"},
                {"start": 3, "title": "Z"}]]
        for chapters in bad:
            with self.subTest(chapters=chapters), self.assertRaises(ValidationError):
                validate_chapters(chapters, 6)

    def test_prediction_envelope_and_video_identity(self):
        for patch in ({"schema_version": True}, {"video_id": "other"},
                      {"status": "unknown"}, {"usage": []}, {"error": "bad"}):
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                validate_prediction({**prediction(), **patch}, 6, "video")


class BoundaryTests(unittest.TestCase):
    def test_gap_conversion_excludes_edges(self):
        self.assertEqual(starts_to_gaps([0, 2, 5], 6), [0, 1, 0, 0, 1])
        self.assertEqual(starts_to_gaps([0], 1), [])
        for starts in ([1], [0, 6], [0, 0], [0, True]):
            with self.subTest(starts=starts), self.assertRaises(ValidationError):
                starts_to_gaps(starts, 6)

    def test_exact_missing_extra_and_shifted(self):
        self.assertEqual(boundary_scores([3], [3])["f1"], 1.0)
        self.assertEqual(boundary_scores([3], [])["fn"], 1)
        self.assertAlmostEqual(boundary_scores([3], [2, 3])["f1"], 2 / 3)
        self.assertEqual(boundary_scores([3], [4])["f1"], 0.0)
        self.assertEqual(boundary_scores([3], [4], 2)["f1"], 1.0)

    def test_empty_boundary_policy(self):
        self.assertEqual(boundary_scores([], [])["f1"], 1.0)
        self.assertEqual(boundary_scores([], [1])["f1"], 0.0)
        self.assertEqual(boundary_scores([1], [])["f1"], 0.0)

    def test_tolerant_matches_are_one_to_one_and_maximum(self):
        self.assertEqual(len(match_boundaries([3], [2, 4], 2)), 1)
        # Nearest-first matching can lose a match; earliest-feasible cannot.
        self.assertEqual(match_boundaries([1, 3], [3, 4], 2), [(1, 3), (3, 4)])

    def test_tolerant_matching_against_exhaustive_small_dynamic_program(self):
        def optimum(left, right, tolerance):
            if not left or not right:
                return 0
            best = max(optimum(left[1:], right, tolerance),
                       optimum(left, right[1:], tolerance))
            if abs(left[0] - right[0]) <= tolerance:
                best = max(best, 1 + optimum(left[1:], right[1:], tolerance))
            return best

        subsets = [list(combo) for length in range(4)
                   for combo in itertools.combinations(range(5), length)]
        for left in subsets:
            for right in subsets:
                for tolerance in (0, 1, 2):
                    self.assertEqual(len(match_boundaries(left, right, tolerance)),
                                     optimum(left, right, tolerance))


class StructuralTests(unittest.TestCase):
    def test_perfect_and_missing_cut(self):
        perfect = structural_errors([0, 3], [0, 3], 6)
        self.assertEqual(perfect, {"pk": 0.0, "window_diff": 0.0,
                                   "window_size": 2, "positions": 4})
        missing = structural_errors([0, 3], [0], 6)
        self.assertEqual(missing["pk"], 0.5)
        self.assertEqual(missing["window_diff"], 0.5)

    def test_pk_and_windowdiff_distinguish_boundary_counts(self):
        result = structural_errors([0, 3], [0, 2, 3], 6)
        self.assertEqual(result["pk"], 0.25)
        self.assertEqual(result["window_diff"], 0.5)

    def test_single_sentence_and_capped_window(self):
        self.assertEqual(structural_errors([0], [0], 1)["pk"], 0.0)
        self.assertEqual(structural_errors([0], [0, 1], 2)["window_size"], 1)
        self.assertEqual(structural_errors([0], [0, 1], 2)["window_diff"], 1.0)


class SpanTests(unittest.TestCase):
    def test_half_open_spans_and_perfect_matches(self):
        self.assertEqual(chapter_spans([0, 2], 5), [(0, 2), (2, 5)])
        matches = match_chapter_spans([0, 3], [0, 3], 6)
        self.assertEqual([(m["reference_index"], m["prediction_index"], m["iou"])
                          for m in matches], [(0, 0, 1.0), (1, 1, 1.0)])

    def test_threshold_is_inclusive_and_matching_is_one_to_one(self):
        matches = match_chapter_spans([0], [0, 3], 6)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["iou"], 0.5)
        self.assertEqual(match_chapter_spans([0], [0, 2, 4], 6), [])

    def test_span_matching_ignores_title_wording(self):
        ref = reference()
        first = prediction()
        second = prediction()
        for chapter in second["chapters"]:
            chapter["title"] = "completely different wording"
        self.assertEqual(score_prediction(ref, first)["title_alignment"],
                         score_prediction(ref, second)["title_alignment"])

    def test_global_iou_matching_matches_exhaustive_search(self):
        def best_sum(reference_intervals, predicted_intervals, threshold):
            if not reference_intervals or not predicted_intervals:
                return 0.0
            best = max(best_sum(reference_intervals[1:], predicted_intervals, threshold),
                       best_sum(reference_intervals, predicted_intervals[1:], threshold))
            left, right = reference_intervals[0], predicted_intervals[0]
            overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
            union = (left[1] - left[0]) + (right[1] - right[0]) - overlap
            iou = overlap / union
            if iou >= threshold:
                best = max(best, iou + best_sum(reference_intervals[1:],
                                                 predicted_intervals[1:], threshold))
            return best

        starts = [[0] + list(combo) for length in range(3)
                  for combo in itertools.combinations(range(1, 6), length)]
        for gold in starts:
            for pred in starts:
                for threshold in (0.3, 0.5):
                    matched = match_chapter_spans(gold, pred, 6, threshold)
                    self.assertAlmostEqual(sum(match["iou"] for match in matched),
                                           best_sum(chapter_spans(gold, 6),
                                                    chapter_spans(pred, 6), threshold))


class FailureAndAggregationTests(unittest.TestCase):
    def test_failed_no_cut_reference_never_scores_perfectly(self):
        ref = reference((0,), 1)
        for status in ("invalid", "error", "truncated"):
            with self.subTest(status=status):
                score = score_prediction(ref, prediction((0,), status=status))
                self.assertEqual(score["exact"]["f1"], 0.0)
                self.assertEqual(score["structural"]["pk"], 1.0)
                self.assertIsNone(score["exact"]["tp"])
        self.assertEqual(score_prediction(ref, prediction((0,)))["exact"]["f1"], 1.0)

    def test_malformed_ok_prediction_becomes_failure(self):
        malformed = prediction((0, 0))
        result = score_prediction(reference(), malformed)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["exact"]["f1"], 0.0)

    def test_manifest_retains_missing_predictions_and_labels_micro_scope(self):
        refs = [reference(video_id="a"), reference(video_id="b")]
        result = evaluate_records(refs, [prediction(video_id="a")])
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["requested_videos"], 2)
        self.assertEqual(aggregate["macro_exact_f1"], 0.5)
        self.assertEqual(aggregate["failure_rate"], 0.5)
        self.assertIsNone(aggregate["all_requests_micro"])
        self.assertEqual(aggregate["successful_only_micro"]["excluded_failures"], 1)
        self.assertEqual(aggregate["successful_only_micro"]["exact"]["f1"], 1.0)
        self.assertEqual(aggregate["title_span_alignment"]["reference_coverage_all_requests"], 0.5)

    def test_all_failed_micro_is_unknown(self):
        result = evaluate_records([reference()], [prediction(status="truncated")])
        self.assertIsNone(result["aggregate"]["successful_only_micro"]["exact"])
        self.assertEqual(result["aggregate"]["macro_exact_f1"], 0.0)

    def test_metric_settings_checked_even_for_failures_and_not_mixed(self):
        with self.assertRaises(ValidationError):
            score_prediction(reference(), None, tolerance=True)
        with self.assertRaises(ValidationError):
            score_prediction(reference(), None, iou_threshold=float("nan"))
        scores = [score_prediction(reference(video_id="a"), prediction(video_id="a"), 1),
                  score_prediction(reference(video_id="b"), prediction(video_id="b"), 2)]
        with self.assertRaises(ValidationError):
            aggregate_scores(scores)

    def test_duplicates_unrequested_ids_and_invalid_references_are_errors(self):
        for refs, preds in (([reference()], [prediction(), prediction()]),
                            ([reference()], [prediction(video_id="other")]),
                            ([reference(), reference()], [prediction()]),
                            ([reference((0, 6))], [prediction()])):
            with self.subTest(refs=refs, preds=preds), self.assertRaises(ValidationError):
                evaluate_records(refs, preds)
        with self.assertRaises(ValidationError):
            aggregate_scores([])


if __name__ == "__main__":
    unittest.main()
