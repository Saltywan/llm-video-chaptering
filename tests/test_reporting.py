"""Reporting tests inject scorers and never fetch or run metric models."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from chaptering.evaluation import ValidationError, evaluate_records
from chaptering.reporting import (
    MetricConfigurationError, MissingMetricDependency, make_bertscore_scorer,
    make_boundary_similarity_scorer, make_rouge_l_scorer,
    paired_video_bootstrap, report_predictions,
)


def reference(video_id="a", starts=(0, 3), n=6):
    return {"video_id": video_id, "sentences": ["Sentence."] * n,
            "reference": {"chapter_starts": list(starts),
                          "chapter_titles": [f"Topic {i}" for i in range(len(starts))]}}


def prediction(video_id="a", starts=(0, 3), status="ok"):
    return {"schema_version": 1, "video_id": video_id, "run_id": "synthetic",
            "status": status, "chapters": [{"start": start, "title": f"Topic {i}"}
                                            for i, start in enumerate(starts)],
            "error": None if status == "ok" else "failure", "usage": {}}


class StubTextScorer:
    metadata = {"implementation": "test_stub"}

    def __init__(self):
        self.calls = []

    def __call__(self, references, candidates):
        self.calls.append((list(references), list(candidates)))
        return [{"precision": float(ref == cand), "recall": float(ref == cand),
                 "f1": float(ref == cand)} for ref, cand in zip(references, candidates)]


class ReportingTests(unittest.TestCase):
    def test_perfect_titles_preserve_canonical_result(self):
        refs, preds = [reference()], [prediction()]
        text = StubTextScorer()
        result = report_predictions(refs, preds, rouge_scorer=text, semantic_scorer=text,
                                    boundary_scorer=lambda *args: 1.0)
        self.assertEqual(result["canonical"], evaluate_records(refs, preds))
        titles = result["additional_metrics"]["titles"]
        self.assertEqual(titles["reference_coverage_all_requests"], 1.0)
        self.assertEqual(titles["rouge_l"]["matched_only_mean"]["f1"], 1.0)
        self.assertEqual(titles["semantic"]["matched_only_mean"]["f1"], 1.0)
        self.assertEqual(result["additional_metrics"]["boundary_similarity"]["macro_all_requests"], 1.0)
        self.assertIn((["Topic 0\nTopic 1"], ["Topic 0\nTopic 1"]), text.calls)

    def test_failure_stays_in_title_document_and_boundary_denominators(self):
        refs = [reference("a"), reference("b")]
        result = report_predictions(refs, [prediction("a")], rouge_scorer=StubTextScorer(),
                                    boundary_scorer=lambda *args: 1.0)
        titles = result["additional_metrics"]["titles"]
        self.assertEqual(titles["failed_videos"], 1)
        self.assertEqual(titles["reference_chapters_all_requests"], 4)
        self.assertEqual(titles["reference_coverage_all_requests"], 0.5)
        self.assertEqual(titles["rouge_l"]["matched_only_mean"]["f1"], 1.0)
        self.assertEqual(titles["rouge_l"]["reference_chapter_mean_with_unmatched_zero"]["f1"], 0.5)
        self.assertEqual(titles["rouge_l"]["concatenated_document_macro_all_requests"]["f1"], 0.5)
        self.assertEqual(result["additional_metrics"]["boundary_similarity"]["macro_all_requests"], 0.5)

    def test_no_span_matches_has_unknown_matched_quality_and_zero_coverage(self):
        result = report_predictions([reference(starts=(0,))], [prediction(starts=(0, 2, 4))],
                                    rouge_scorer=StubTextScorer(), semantic_scorer=StubTextScorer(),
                                    include_boundary_similarity=False)
        titles = result["additional_metrics"]["titles"]
        self.assertEqual(titles["matched_chapters"], 0)
        self.assertIsNone(titles["rouge_l"]["matched_only_mean"])
        self.assertIsNone(titles["semantic"]["matched_only_mean"])
        self.assertEqual(titles["reference_coverage_all_requests"], 0.0)
        self.assertEqual(titles["rouge_l"]["reference_chapter_mean_with_unmatched_zero"]["f1"], 0.0)

    def test_all_failed_never_calls_title_scorer(self):
        text = StubTextScorer()
        result = report_predictions([reference()], [prediction(status="truncated")],
                                    rouge_scorer=text, include_boundary_similarity=False)
        self.assertEqual(text.calls, [])
        titles = result["additional_metrics"]["titles"]
        self.assertEqual(titles["failed_videos"], 1)
        self.assertIsNone(titles["prediction_coverage_successful_requests"])
        self.assertEqual(titles["rouge_l"]["concatenated_document_macro_all_requests"]["f1"], 0.0)

    def test_bad_metric_outputs_raise(self):
        with self.assertRaises(MetricConfigurationError):
            report_predictions([reference()], [prediction()], rouge_scorer=lambda *args: [],
                               include_boundary_similarity=False)
        with self.assertRaises(MetricConfigurationError):
            report_predictions([reference()], [prediction()], rouge_scorer=StubTextScorer(),
                               boundary_scorer=lambda *args: float("nan"))


class BootstrapTests(unittest.TestCase):
    def scores(self):
        refs = [reference("a"), reference("b"), reference("c")]
        left = evaluate_records(refs, [prediction("a"), prediction("b", status="error"),
                                      prediction("c")])["per_video"]
        right = evaluate_records(refs, [prediction("a", starts=(0,)), prediction("b"),
                                       prediction("c", starts=(0,))])["per_video"]
        return left, right

    def test_reproducible_paired_alignment_includes_failures(self):
        left, right = self.scores()
        first = paired_video_bootstrap(left, right, resamples=200, seed=7)
        second = paired_video_bootstrap(list(reversed(left)), right[1:] + right[:1],
                                       resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], 1 / 3)
        self.assertEqual(first["left_failed_videos"], 1)
        self.assertEqual(first["videos"], 3)

    def test_single_pair_one_resample_is_defined(self):
        left, right = self.scores()
        result = paired_video_bootstrap(left[:1], right[:1], resamples=1, seed=0)
        self.assertEqual(result["confidence_interval"], [1.0, 1.0])

    def test_duplicate_unpaired_empty_and_invalid_failures_are_rejected(self):
        left, right = self.scores()
        for a, b in ((left, right[:-1]), (left + left[:1], right), ([], [])):
            with self.assertRaises(ValidationError):
                paired_video_bootstrap(a, b, resamples=5)
        bad = [dict(row) for row in left]
        bad[1] = {**bad[1], "exact": {"f1": 1.0}}
        with self.assertRaises(ValidationError):
            paired_video_bootstrap(bad, right)
        with self.assertRaises(MetricConfigurationError):
            paired_video_bootstrap(left, right, resamples=0)


class DependencyTests(unittest.TestCase):
    def test_missing_optional_libraries_raise_without_fallback(self):
        with patch.dict(sys.modules, {"rouge_score": None}):
            with self.assertRaises(MissingMetricDependency):
                make_rouge_l_scorer()
        with patch.dict(sys.modules, {"segeval": None}):
            with self.assertRaises(MissingMetricDependency):
                make_boundary_similarity_scorer()

    def test_segeval_receives_segment_masses(self):
        calls = []
        module = types.ModuleType("segeval")
        module.BoundaryFormat = types.SimpleNamespace(mass="MASS_ENUM")
        module.boundary_similarity = lambda *args, **kwargs: calls.append((args, kwargs)) or 0.75
        with patch.dict(sys.modules, {"segeval": module}):
            scorer = make_boundary_similarity_scorer(n_t=2)
            self.assertEqual(scorer([0, 2], [0, 3], 6), 0.75)
        self.assertEqual(calls, [(((2, 4), (3, 3)), {"boundary_format": "MASS_ENUM", "n_t": 2,
                                                      "one_minus": False, "return_parts": False})])

    def test_bertscore_resolves_only_pinned_cached_snapshot_and_explicit_cpu(self):
        revision = "a" * 40
        calls, snapshots = [], []
        bert = types.ModuleType("bert_score")
        bert.score = lambda *args, **kwargs: calls.append((args, kwargs)) or ([0.9], [0.8], [0.85])
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kwargs: snapshots.append(kwargs) or f"/cache/snapshots/{revision}"
        with patch.dict(sys.modules, {"bert_score": bert, "huggingface_hub": hub}):
            scorer = make_bertscore_scorer(model_id="org/model", revision=revision, num_layers=9)
            result = scorer(["reference"], ["candidate"])
        self.assertEqual(result[0]["f1"], 0.85)
        self.assertTrue(snapshots[0]["local_files_only"])
        self.assertEqual(snapshots[0]["revision"], revision)
        self.assertEqual(calls[0][0], (["candidate"], ["reference"]))
        self.assertEqual(calls[0][1]["device"], "cpu")
        self.assertEqual(calls[0][1]["model_type"], f"/cache/snapshots/{revision}")
        self.assertEqual(calls[0][1]["num_layers"], 9)

    def test_bertscore_rejects_unpinned_or_automatic_settings(self):
        for values in ({"revision": "main"}, {"device": "auto"}, {"num_layers": None}):
            kwargs = {"model_id": "org/model", "revision": "a" * 40, "num_layers": 9}
            kwargs.update(values)
            with self.assertRaises(MetricConfigurationError):
                make_bertscore_scorer(**kwargs)


class CliTests(unittest.TestCase):
    def cli(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_predictions.py"
        spec = importlib.util.spec_from_file_location("test_reporting_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_jsonl_cli_retains_missing_video_and_is_deterministic(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs, preds = root / "refs.jsonl", root / "preds.jsonl"
            refs.write_text("\n".join(json.dumps(reference(video)) for video in ("a", "b")) + "\n")
            preds.write_text(json.dumps(prediction("a")) + "\n")
            results = []
            for filename in ("first.json", "second.json"):
                output = root / filename
                with patch.object(cli, "make_rouge_l_scorer", return_value=StubTextScorer()), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["--references", str(refs), "--predictions", str(preds),
                                               "--output", str(output), "--skip-boundary-similarity"]), 0)
                results.append(json.loads(output.read_text()))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["canonical"]["aggregate"]["requested_videos"], 2)
            self.assertEqual(results[0]["canonical"]["aggregate"]["failed_videos"], 1)

    def test_jsonl_cli_rejects_duplicate_keys_without_printing_record_text(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"video_id":"a","video_id":"b"}\n')
            with self.assertRaisesRegex(ValueError, "invalid JSON record"):
                cli._records(path)


if __name__ == "__main__":
    unittest.main()
