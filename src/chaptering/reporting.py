"""Optional title/boundary metrics and paired-video uncertainty reporting.

Uses the official rouge-score, segeval and bert-score implementations without
substitute formulas/tokenizers. BERTScore only resolves an explicitly pinned,
already cached model snapshot. The canonical evaluator remains unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from .evaluation import ValidationError, chapter_spans, evaluate_records


class MissingMetricDependency(ImportError):
    """An explicitly requested optional metric dependency is unavailable."""


class MetricConfigurationError(ValueError):
    """Metric settings, pinned model files, or scorer outputs are invalid."""


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class BatchTextScorer:
    """Callable: (reference texts, candidate texts) -> P/R/F1 dictionaries."""
    function: Callable
    metadata: dict[str, Any]

    def __call__(self, references, candidates):
        return self.function(references, candidates)


def make_rouge_l_scorer(*, use_stemmer: bool = False) -> BatchTextScorer:
    """Google's rougeL scorer; no fallback tokenizer or rougeLsum substitution.

    API: https://github.com/google-research/google-research/tree/master/rouge
    """
    if type(use_stemmer) is not bool:
        raise MetricConfigurationError("use_stemmer must be a boolean")
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise MissingMetricDependency("ROUGE-L requires rouge-score; no alternate tokenizer is used") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    def score(references, candidates):
        if len(references) != len(candidates):
            raise MetricConfigurationError("reference/candidate batches differ in length")
        result = []
        for reference, candidate in zip(references, candidates):
            value = scorer.score(reference, candidate)["rougeL"]
            result.append({"precision": float(value.precision),
                           "recall": float(value.recall), "f1": float(value.fmeasure)})
        return result

    return BatchTextScorer(score, {"name": "rougeL", "implementation": "rouge-score",
                                  "package_version": _version("rouge-score"),
                                  "use_stemmer": use_stemmer, "scale": "0_to_1"})


def make_bertscore_scorer(*, model_id: str, revision: str, num_layers: int,
                          device: str = "cpu", batch_size: int = 16,
                          cache_dir: str | None = None,
                          use_fast_tokenizer: bool = False) -> BatchTextScorer:
    """Use an explicit complete commit SHA from the offline Hugging Face cache.

    bert_score.score has no revision argument: resolve with
    snapshot_download(local_files_only=True), then pass the exact local path as
    model_type plus explicit num_layers. No downloads or automatic CUDA choice.
    Raw, non-rescaled BERTScore is used with idf=False.

    API: https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py
    """
    if not isinstance(model_id, str) or not model_id.strip() or Path(model_id).is_absolute():
        raise MetricConfigurationError("semantic model_id must be an explicit Hugging Face repository ID")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise MetricConfigurationError("semantic revision must be a complete 40-character commit SHA")
    if type(num_layers) is not int or num_layers < 1:
        raise MetricConfigurationError("semantic num_layers must be an explicit positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise MetricConfigurationError("semantic batch_size must be a positive integer")
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
        raise MetricConfigurationError("device must be explicitly cpu, cuda, or cuda:<logical index>")
    if type(use_fast_tokenizer) is not bool:
        raise MetricConfigurationError("use_fast_tokenizer must be a boolean")
    try:
        from bert_score import score as bert_score
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingMetricDependency("semantic scoring requires bert-score and huggingface-hub") from exc
    try:
        snapshot = snapshot_download(repo_id=model_id, revision=revision,
                                     cache_dir=cache_dir, local_files_only=True,
                                     allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                     "tokenizer_config.json", "vocab.json", "merges.txt",
                                                     "special_tokens_map.json"])
    except Exception as exc:
        raise MetricConfigurationError("pinned semantic model is not cached locally; cache the exact revision before scoring") from exc
    if Path(snapshot).name.lower() != revision.lower():
        raise MetricConfigurationError("semantic snapshot path does not identify the requested revision")

    def score(references, candidates):
        if len(references) != len(candidates):
            raise MetricConfigurationError("reference/candidate batches differ in length")
        if not references:
            return []
        values = bert_score(list(candidates), list(references), model_type=str(snapshot),
                            num_layers=num_layers, device=device, batch_size=batch_size,
                            nthreads=1, idf=False, rescale_with_baseline=False,
                            use_fast_tokenizer=use_fast_tokenizer, verbose=False)
        vectors = [value.tolist() if hasattr(value, "tolist") else list(value) for value in values]
        if len(vectors) != 3 or any(len(value) != len(references) for value in vectors):
            raise MetricConfigurationError("BERTScore returned the wrong number of scores")
        return [{"precision": float(p), "recall": float(r), "f1": float(f)}
                for p, r, f in zip(*vectors)]

    return BatchTextScorer(score, {
        "name": "BERTScore", "implementation": "bert-score", "package_version": _version("bert-score"),
        "model_id": model_id, "revision": revision.lower(), "num_layers": num_layers,
        "device": device, "batch_size": batch_size, "idf": False, "rescale_with_baseline": False,
        "use_fast_tokenizer": use_fast_tokenizer, "local_files_only": True,
        "scope": "span-matched individual titles; no concatenated-document semantic score"})


def make_boundary_similarity_scorer(*, n_t: int = 2):
    """Use SegEval's documented mass-format Boundary Similarity API.

    https://segeval.readthedocs.io/en/latest/api/
    """
    if type(n_t) is not int or n_t < 1:
        raise MetricConfigurationError("Boundary Similarity n_t must be positive")
    try:
        import segeval
    except ImportError as exc:
        raise MissingMetricDependency("Boundary Similarity requires segeval; no replacement formula is used") from exc

    def score(reference_starts, predicted_starts, sentence_count):
        reference = tuple(end - start for start, end in chapter_spans(reference_starts, sentence_count))
        predicted = tuple(end - start for start, end in chapter_spans(predicted_starts, sentence_count))
        return float(segeval.boundary_similarity(reference, predicted,
                     boundary_format=segeval.BoundaryFormat.mass, n_t=n_t,
                     one_minus=False, return_parts=False))

    score.metadata = {"name": "Boundary Similarity", "implementation": "segeval",
                      "package_version": _version("segeval"), "boundary_format": "mass",
                      "unit": "sentences", "n_t": n_t, "one_minus": False, "return_parts": False}
    return score


def _numeric(value, label, *, unit_interval=False):
    if isinstance(value, (str, bytes, bool)):
        raise MetricConfigurationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetricConfigurationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (unit_interval and not 0 <= result <= 1):
        raise MetricConfigurationError(f"{label} is outside the metric's finite range")
    return result


def _text_scores(scorer, references, candidates, *, unit_interval):
    if not references:
        return []
    values = list(scorer(references, candidates))
    if len(values) != len(references):
        raise MetricConfigurationError("text scorer returned the wrong number of scores")
    result = []
    for value in values:
        if not isinstance(value, dict) or not {"precision", "recall", "f1"}.issubset(value):
            raise MetricConfigurationError("text scorer must return precision/recall/f1 dictionaries")
        result.append({key: _numeric(value[key], key, unit_interval=unit_interval)
                       for key in ("precision", "recall", "f1")})
    return result


def _mean(values):
    return ({key: fmean(value[key] for value in values)
             for key in ("precision", "recall", "f1")} if values else None)


def _zero_mean(values, denominator):
    return {key: sum(value[key] for value in values) / denominator
            for key in ("precision", "recall", "f1")}


def _metadata(scorer):
    value = getattr(scorer, "metadata", None)
    return dict(value) if value else {"implementation": "injected_callable"}


def report_predictions(references, predictions, *, rouge_scorer=None,
                       semantic_scorer=None, boundary_scorer=None,
                       include_boundary_similarity=True, tolerance=2,
                       iou_threshold=0.5):
    """Join every requested video and append metrics separately from canonical.

    Inject text scorers with (references, candidates)->P/R/F1 dictionaries.
    Otherwise ROUGE-L requires rouge-score. A semantic scorer must be explicitly
    supplied; omission is reported as not_requested. Boundary Similarity is
    enabled by default and requires segeval unless a scorer is injected.
    Missing/malformed/failed requests contribute zero document title quality
    and zero chapter coverage, without being passed to a metric model.
    """
    references, predictions = list(references), list(predictions)
    canonical = evaluate_records(references, predictions, tolerance, iou_threshold)
    rouge_scorer = rouge_scorer if rouge_scorer is not None else make_rouge_l_scorer()
    if include_boundary_similarity and boundary_scorer is None:
        boundary_scorer = make_boundary_similarity_scorer()
    by_reference = {record["video_id"]: record for record in references}
    by_prediction = {record["video_id"]: record for record in predictions}
    per_video = []
    pair_refs, pair_preds, pair_rows = [], [], []
    doc_refs, doc_preds, doc_rows = [], [], []
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for score in canonical["per_video"]:
        video_id = score["video_id"]
        reference = by_reference[video_id]
        gold_titles = reference["reference"]["chapter_titles"]
        alignment = score["title_alignment"]
        row = {"video_id": video_id, "status": score["status"],
               "reference_chapters": len(gold_titles), "predicted_chapters": alignment["prediction_count"],
               "matched_chapters": alignment["matched_count"], "reference_coverage": alignment["reference_coverage"],
               "prediction_coverage": alignment["prediction_coverage"], "matches": [],
               "concatenated_title_rouge_l": dict(zero), "boundary_similarity": None}
        per_video.append(row)
        if score["status"] != "ok":
            if include_boundary_similarity:
                row["boundary_similarity"] = 0.0
            continue
        prediction = by_prediction[video_id]
        predicted_titles = [chapter["title"] for chapter in prediction["chapters"]]
        for match in alignment["matches"]:
            match_row = dict(match)
            row["matches"].append(match_row)
            pair_refs.append(gold_titles[match["reference_index"]])
            pair_preds.append(predicted_titles[match["prediction_index"]])
            pair_rows.append(match_row)
        doc_refs.append("\n".join(gold_titles))
        doc_preds.append("\n".join(predicted_titles))
        doc_rows.append(row)
        if include_boundary_similarity:
            row["boundary_similarity"] = _numeric(boundary_scorer(
                reference["reference"]["chapter_starts"],
                [chapter["start"] for chapter in prediction["chapters"]], len(reference["sentences"])),
                "Boundary Similarity", unit_interval=True)
    rouge_values = _text_scores(rouge_scorer, pair_refs, pair_preds, unit_interval=True)
    for match, value in zip(pair_rows, rouge_values):
        match["rouge_l"] = value
    for row, value in zip(doc_rows, _text_scores(rouge_scorer, doc_refs, doc_preds, unit_interval=True)):
        row["concatenated_title_rouge_l"] = value
    semantic_values = None
    if semantic_scorer is not None:
        semantic_values = _text_scores(semantic_scorer, pair_refs, pair_preds, unit_interval=False)
        for match, value in zip(pair_rows, semantic_values):
            match["semantic"] = value
    reference_count = sum(row["reference_chapters"] for row in per_video)
    predicted_count = sum(row["predicted_chapters"] for row in per_video if row["status"] == "ok")
    matched_count = len(pair_rows)
    titles = {
        "requested_videos": len(per_video), "failed_videos": canonical["aggregate"]["failed_videos"],
        "reference_chapters_all_requests": reference_count,
        "predicted_chapters_successful_requests": predicted_count, "matched_chapters": matched_count,
        "reference_coverage_all_requests": matched_count / reference_count,
        "prediction_coverage_successful_requests": matched_count / predicted_count if predicted_count else None,
        "matching_policy": canonical["aggregate"]["metric_policy"],
        "rouge_l": {"metadata": _metadata(rouge_scorer), "matched_only_mean": _mean(rouge_values),
                    "reference_chapter_mean_with_unmatched_zero": _zero_mean(rouge_values, reference_count),
                    "concatenated_document_macro_all_requests": _mean([row["concatenated_title_rouge_l"] for row in per_video]),
                    "concatenation_separator": "\n", "failed_document_score": 0.0},
        "semantic": ({"metadata": _metadata(semantic_scorer), "matched_only_mean": _mean(semantic_values),
                      "reference_chapter_mean_with_unmatched_zero": _zero_mean(semantic_values, reference_count),
                      "zero_fill_is_reporting_convention_not_semantic_lower_bound": True,
                      "scope": "span-matched individual chapter titles"}
                     if semantic_values is not None else {"status": "not_requested"}),
        "per_video": per_video}
    boundary = ({"metadata": _metadata(boundary_scorer),
                 "macro_all_requests": fmean(row["boundary_similarity"] for row in per_video),
                 "failed_request_score": 0.0,
                 "per_video": [{"video_id": row["video_id"], "status": row["status"],
                                "score": row["boundary_similarity"]} for row in per_video]}
                if include_boundary_similarity else {"status": "not_requested"})
    return {"canonical": canonical, "additional_metrics": {"titles": titles, "boundary_similarity": boundary}}


def _percentile(sorted_values, probability):
    index = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_video_bootstrap(left_scores, right_scores, *, resamples=10000, seed=20261116):
    """95% percentile CI for mean video exact-F1 delta (left minus right).

    Align complete equal ID sets, including failures at zero. Resample pairs
    with replacement, never sentences/separate arms or an ID intersection.
    """
    if type(resamples) is not int or resamples < 1:
        raise MetricConfigurationError("bootstrap resamples must be a positive integer")
    if type(seed) is not int:
        raise MetricConfigurationError("bootstrap seed must be an integer")

    def index(scores):
        indexed = {}
        for score in scores:
            if not isinstance(score, dict) or not isinstance(score.get("video_id"), str) or not score["video_id"]:
                raise ValidationError("bootstrap scores need a nonempty video_id")
            video_id = score["video_id"]
            if video_id in indexed:
                raise ValidationError("duplicate video ID in bootstrap arm")
            if score.get("status") not in {"ok", "invalid", "error", "truncated"}:
                raise ValidationError("bootstrap scores need explicit prediction statuses")
            try:
                value = _numeric(score["exact"]["f1"], "exact F1", unit_interval=True)
            except (KeyError, TypeError) as exc:
                raise ValidationError("bootstrap scores need exact F1") from exc
            if score["status"] != "ok" and value != 0:
                raise ValidationError("failed bootstrap requests must retain exact F1 zero")
            indexed[video_id] = (value, score["status"], score.get("metric_policy"))
        return indexed

    left, right = index(left_scores), index(right_scores)
    if not left or left.keys() != right.keys():
        raise ValidationError("bootstrap arms must contain the same nonempty requested video ID set")
    ids = sorted(left)
    policies = [left[video_id][2] for video_id in ids] + [right[video_id][2] for video_id in ids]
    if any(policy != policies[0] for policy in policies):
        raise ValidationError("bootstrap arms use different metric policies")
    deltas = [left[video_id][0] - right[video_id][0] for video_id in ids]
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return {"estimator": "mean_paired_video_exact_f1_difference", "difference": "left_minus_right",
            "mean_difference": fmean(deltas), "confidence": 0.95,
            "interval_method": "percentile_linear_interpolation",
            "confidence_interval": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
            "videos": n, "resamples": resamples, "seed": seed,
            "left_failed_videos": sum(left[video_id][1] != "ok" for video_id in ids),
            "right_failed_videos": sum(right[video_id][1] != "ok" for video_id in ids),
            "paired_video_ids_sha256": hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()}
