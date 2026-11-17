"""Strict, sentence-space evaluation for the compatibility stage.

This module uses only the standard library. It deliberately does not extract
JSON from reasoning, repair malformed predictions, or implement title-text
quality metrics. Span matching uses text-independent sentence intervals.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Sequence


class ValidationError(ValueError):
    """An output violates the frozen chapter-outline contract."""


STATUSES = frozenset({"ok", "invalid", "error", "truncated"})


def _sentence_count(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValidationError("sentence_count must be a positive integer")


def _matching_policy(tolerance: int, threshold: float) -> dict[str, Any]:
    if type(tolerance) is not int or tolerance < 0:
        raise ValidationError("tolerance must be a nonnegative integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValidationError("IoU threshold must be numeric")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValidationError("IoU threshold must be in (0, 1]")
    return {"version": 1, "boundary_tolerance": tolerance,
            "span_iou_threshold": threshold,
            "structural_window": "reference_half_mean_half_up_N_minus_k"}


def _ordered_indices(values: Sequence[int], *, require_zero: bool = False,
                     sentence_count: int | None = None) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError("indices must be a list or tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValidationError("indices must be nonnegative integers, not booleans")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValidationError("indices must be strictly increasing")
    if require_zero and (not values or values[0] != 0):
        raise ValidationError("whole-document chapter starts must begin at zero")
    if sentence_count is not None:
        _sentence_count(sentence_count)
        if any(value >= sentence_count for value in values):
            raise ValidationError("chapter start is outside the document")
    return list(values)


def validate_chapters(chapters: Any, sentence_count: int) -> list[dict[str, Any]]:
    """Validate and copy chapter objects without sorting or changing titles."""
    _sentence_count(sentence_count)
    if not isinstance(chapters, list) or not chapters:
        raise ValidationError("chapters must be a nonempty list")
    for chapter in chapters:
        if not isinstance(chapter, dict) or set(chapter) != {"start", "title"}:
            raise ValidationError("each chapter must contain only start and title")
        if not isinstance(chapter["title"], str) or not chapter["title"].strip():
            raise ValidationError("chapter titles must be nonempty strings")
    _ordered_indices([chapter["start"] for chapter in chapters],
                     require_zero=True, sentence_count=sentence_count)
    return [dict(chapter) for chapter in chapters]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"nonstandard JSON constant: {value}")


def parse_outline(text: str, sentence_count: int) -> list[dict[str, Any]]:
    """Parse the final-answer field as one strict JSON object.

    Leading/trailing JSON whitespace is allowed. Markdown fences, reasoning
    tags, prose, duplicate keys and additional root fields are rejected. Callers
    must supply an actual final-answer field, never concatenate reasoning into
    it or search reasoning for an apparent answer.
    """
    if not isinstance(text, str):
        raise ValidationError("final answer must be a string")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("final answer is not a single valid JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"chapters"}:
        raise ValidationError("final answer must contain only chapters")
    return validate_chapters(value["chapters"], sentence_count)


def validate_prediction(record: Any, sentence_count: int,
                        expected_video_id: str | None = None) -> dict[str, Any]:
    """Validate a prediction envelope. Extra logging metadata is permitted."""
    if not isinstance(record, dict):
        raise ValidationError("prediction must be an object")
    required = {"schema_version", "video_id", "run_id", "status", "chapters",
                "error", "usage"}
    if not required.issubset(record):
        raise ValidationError("prediction is missing required envelope fields")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValidationError("unsupported prediction schema")
    for field in ("video_id", "run_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValidationError(f"{field} must be a nonempty string")
    if expected_video_id is not None and record["video_id"] != expected_video_id:
        raise ValidationError("prediction video_id differs from requested video")
    if not isinstance(record["status"], str) or record["status"] not in STATUSES:
        raise ValidationError("unknown prediction status")
    if not isinstance(record["usage"], dict):
        raise ValidationError("usage must be an object")
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ValidationError("error must be null or a string")
    result = dict(record)
    if record["status"] == "ok":
        if record["error"] is not None:
            raise ValidationError("successful prediction must have null error")
        result["chapters"] = validate_chapters(record["chapters"], sentence_count)
    elif record["chapters"] is not None and not isinstance(record["chapters"], list):
        raise ValidationError("failed prediction chapters must be null or a list")
    return result


def starts_to_gaps(starts: Sequence[int], sentence_count: int) -> list[int]:
    """Convert starts j>0 to N-1 gap bits at j-1, excluding document edges."""
    starts = _ordered_indices(starts, require_zero=True, sentence_count=sentence_count)
    gaps = [0] * (sentence_count - 1)
    for start in starts[1:]:
        gaps[start - 1] = 1
    return gaps


def match_boundaries(reference: Sequence[int], predicted: Sequence[int],
                     tolerance: int = 0) -> list[tuple[int, int]]:
    """Maximum-cardinality, one-to-one matching within +/- tolerance.

    Sorted points on a line admit earliest-feasible greedy matching: a point
    strictly too early for the other sequence can never match a later point.
    Matching the earliest feasible pair leaves the widest remaining suffix.
    Returned pairs contain boundary positions, not sequence indices.
    """
    reference = _ordered_indices(reference)
    predicted = _ordered_indices(predicted)
    if type(tolerance) is not int or tolerance < 0:
        raise ValidationError("tolerance must be a nonnegative integer")
    i = j = 0
    matches: list[tuple[int, int]] = []
    while i < len(reference) and j < len(predicted):
        if reference[i] < predicted[j] - tolerance:
            i += 1
        elif predicted[j] < reference[i] - tolerance:
            j += 1
        else:
            matches.append((reference[i], predicted[j]))
            i += 1
            j += 1
    return matches


def _scores_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    both_empty = tp + fp + fn == 0
    precision = tp / (tp + fp) if tp + fp else float(both_empty)
    recall = tp / (tp + fn) if tp + fn else float(both_empty)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


def boundary_scores(reference: Sequence[int], predicted: Sequence[int],
                     tolerance: int = 0) -> dict[str, Any]:
    """Score internal boundary positions. Do not include mandatory start zero."""
    matches = match_boundaries(reference, predicted, tolerance)
    return {**_scores_from_counts(len(matches), len(predicted) - len(matches),
                                 len(reference) - len(matches)),
            "tolerance": tolerance}


def structural_errors(reference_starts: Sequence[int], predicted_starts: Sequence[int],
                      sentence_count: int) -> dict[str, Any]:
    """Sentence-space Pk/WindowDiff under the execution contract's convention.

    k is half the reference mean chapter length, rounded half up, at least one
    and at most N-1. At each i in [0, N-k), compare sentences i and i+k. Their
    intervening boundaries are exactly gaps[i:i+k]. Pk compares whether each
    interval contains any cut; WindowDiff compares its exact count of cuts.
    """
    reference = starts_to_gaps(reference_starts, sentence_count)
    predicted = starts_to_gaps(predicted_starts, sentence_count)
    if sentence_count < 2:
        return {"pk": 0.0, "window_diff": 0.0, "window_size": 0,
                "positions": 0}
    k = min(sentence_count - 1,
            max(1, math.floor(sentence_count / len(reference_starts) / 2 + 0.5)))
    ref_prefix = [0]
    pred_prefix = [0]
    for ref_bit, pred_bit in zip(reference, predicted):
        ref_prefix.append(ref_prefix[-1] + ref_bit)
        pred_prefix.append(pred_prefix[-1] + pred_bit)
    pk_errors = wd_errors = 0
    for i in range(sentence_count - k):
        ref_count = ref_prefix[i + k] - ref_prefix[i]
        pred_count = pred_prefix[i + k] - pred_prefix[i]
        pk_errors += bool(ref_count) != bool(pred_count)
        wd_errors += ref_count != pred_count
    positions = sentence_count - k
    return {"pk": pk_errors / positions, "window_diff": wd_errors / positions,
            "window_size": k, "positions": positions}


def chapter_spans(starts: Sequence[int], sentence_count: int) -> list[tuple[int, int]]:
    """Produce half-open sentence intervals [start, end)."""
    starts = _ordered_indices(starts, require_zero=True, sentence_count=sentence_count)
    return list(zip(starts, starts[1:] + [sentence_count]))


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union


def match_chapter_spans(reference_starts: Sequence[int], predicted_starts: Sequence[int],
                        sentence_count: int, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Monotone one-to-one matches maximizing summed span IoU.

    Eligibility is IoU >= threshold. Ties prefer more matches, then a matched
    diagonal, then skipping a reference, then skipping a prediction. Titles
    are never read when choosing matches.
    """
    _matching_policy(0, threshold)
    reference = chapter_spans(reference_starts, sentence_count)
    predicted = chapter_spans(predicted_starts, sentence_count)
    rows, columns = len(reference), len(predicted)
    values = [[(0.0, 0) for _ in range(columns + 1)] for _ in range(rows + 1)]
    choices = [["" for _ in range(columns + 1)] for _ in range(rows + 1)]
    for r in range(1, rows + 1):
        for p in range(1, columns + 1):
            # Prefer up over left when their scores/counts are identical.
            best, choice = values[r - 1][p], "up"
            candidate = values[r][p - 1]
            if candidate > best:
                best, choice = candidate, "left"
            overlap = _iou(reference[r - 1], predicted[p - 1])
            if overlap >= threshold:
                previous = values[r - 1][p - 1]
                candidate = (previous[0] + overlap, previous[1] + 1)
                if candidate >= best:
                    best, choice = candidate, "match"
            values[r][p], choices[r][p] = best, choice
    result: list[dict[str, Any]] = []
    r, p = rows, columns
    while r and p:
        choice = choices[r][p]
        if choice == "match":
            result.append({"reference_index": r - 1, "prediction_index": p - 1,
                           "reference_span": list(reference[r - 1]),
                           "prediction_span": list(predicted[p - 1]),
                           "iou": _iou(reference[r - 1], predicted[p - 1])})
            r, p = r - 1, p - 1
        elif choice == "up":
            r -= 1
        else:
            p -= 1
    return list(reversed(result))


def _reference(record: Any) -> tuple[str, int, list[dict[str, Any]]]:
    if not isinstance(record, dict):
        raise ValidationError("reference record must be an object")
    video_id = record.get("video_id")
    sentences = record.get("sentences")
    reference = record.get("reference")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValidationError("reference video_id must be a nonempty string")
    if not isinstance(sentences, list) or not sentences:
        raise ValidationError("reference sentences must be a nonempty list")
    if any(not isinstance(text, str) or not text.strip() for text in sentences):
        raise ValidationError("reference sentences must be nonempty strings")
    if not isinstance(reference, dict):
        raise ValidationError("reference labels must be an object")
    starts, titles = reference.get("chapter_starts"), reference.get("chapter_titles")
    if not isinstance(starts, list) or not isinstance(titles, list) or len(starts) != len(titles):
        raise ValidationError("reference starts/titles must be aligned lists")
    chapters = validate_chapters([{"start": start, "title": title}
                                  for start, title in zip(starts, titles)], len(sentences))
    return video_id, len(sentences), chapters


def score_prediction(reference_record: dict[str, Any], prediction_record: Any,
                     tolerance: int = 2, iou_threshold: float = 0.5) -> dict[str, Any]:
    """Score one requested video; invalid/missing predictions are explicit failures.

    Invalid references raise ValidationError instead of being dropped or
    mislabeled as model failures. Failed requests receive zero primary exact
    and tolerant F1, and worst-case structural errors, regardless of the gold.
    Their count-based micro contribution is undefined, not fabricated.
    """
    policy = _matching_policy(tolerance, iou_threshold)
    video_id, n, gold = _reference(reference_record)
    reference_starts = [chapter["start"] for chapter in gold]
    try:
        if prediction_record is None:
            prediction = None
            status, error = "error", "missing prediction for requested video"
        else:
            prediction = validate_prediction(prediction_record, n, video_id)
            status, error = prediction["status"], prediction["error"]
    except ValidationError as exc:
        prediction = None
        status, error = "invalid", str(exc)
    if status != "ok":
        failed_scores = {"tp": None, "fp": None, "fn": None,
                         "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"video_id": video_id, "status": status, "error": error,
                "metric_policy": policy,
                "sentence_count": n, "exact": {**failed_scores, "tolerance": 0},
                "tolerant": {**failed_scores, "tolerance": tolerance},
                "structural": {"pk": 1.0, "window_diff": 1.0,
                               "window_size": None, "positions": None},
                "title_alignment": {"matches": [], "matched_count": 0,
                                    "reference_count": len(gold), "prediction_count": None,
                                    "reference_coverage": 0.0, "prediction_coverage": None}}
    assert prediction is not None
    predicted_starts = [chapter["start"] for chapter in prediction["chapters"]]
    matches = match_chapter_spans(reference_starts, predicted_starts, n, iou_threshold)
    return {"video_id": video_id, "status": "ok", "error": None,
            "metric_policy": policy,
            "sentence_count": n,
            "exact": boundary_scores(reference_starts[1:], predicted_starts[1:]),
            "tolerant": boundary_scores(reference_starts[1:], predicted_starts[1:], tolerance),
            "structural": structural_errors(reference_starts, predicted_starts, n),
            "title_alignment": {"matches": matches, "matched_count": len(matches),
                                "reference_count": len(gold),
                                "prediction_count": len(predicted_starts),
                                "reference_coverage": len(matches) / len(gold),
                                "prediction_coverage": len(matches) / len(predicted_starts)}}


def aggregate_scores(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate requested-video results without hiding failures.

    The caller must supply one score per requested ID. Duplicate IDs raise an
    error. Use evaluate_records() to enforce completeness against a manifest.
    Successful-only micro metrics are always labeled and report exclusions;
    all-request micro metrics are present only when every request succeeded.
    """
    if not scores:
        raise ValidationError("cannot aggregate an empty request set")
    ids = [score["video_id"] for score in scores]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate video IDs in scored requests")
    policy = scores[0]["metric_policy"]
    if any(score["metric_policy"] != policy for score in scores):
        raise ValidationError("cannot aggregate scores from different metric policies")
    valid = [score for score in scores if score["status"] == "ok"]
    failures = len(scores) - len(valid)
    micro: dict[str, Any] = {}
    for field in ("exact", "tolerant"):
        micro[field] = (_scores_from_counts(
            sum(score[field]["tp"] for score in valid),
            sum(score[field]["fp"] for score in valid),
            sum(score[field]["fn"] for score in valid)) if valid else None)
    matched = sum(score["title_alignment"]["matched_count"] for score in scores)
    reference_count = sum(score["title_alignment"]["reference_count"] for score in scores)
    predicted_count = sum(score["title_alignment"]["prediction_count"] for score in valid)
    return {
        "metric_policy": dict(policy),
        "requested_videos": len(scores), "successful_videos": len(valid),
        "failed_videos": failures, "failure_rate": failures / len(scores),
        "status_counts": dict(Counter(score["status"] for score in scores)),
        "macro_exact_f1": sum(score["exact"]["f1"] for score in scores) / len(scores),
        "macro_tolerant_f1": sum(score["tolerant"]["f1"] for score in scores) / len(scores),
        "macro_pk": sum(score["structural"]["pk"] for score in scores) / len(scores),
        "macro_window_diff": sum(score["structural"]["window_diff"] for score in scores) / len(scores),
        "successful_only_micro": {"videos": len(valid), "excluded_failures": failures, **micro},
        "all_requests_micro": micro if failures == 0 else None,
        "title_span_alignment": {
            "matched_chapters": matched, "reference_chapters_all_requests": reference_count,
            "predicted_chapters_successful_requests": predicted_count,
            "reference_coverage_all_requests": matched / reference_count,
            "prediction_coverage_successful_requests": matched / predicted_count if predicted_count else None,
        },
    }


def evaluate_records(references: Sequence[dict[str, Any]],
                     predictions: Sequence[dict[str, Any]], tolerance: int = 2,
                     iou_threshold: float = 0.5) -> dict[str, Any]:
    """Join a requested reference manifest to predictions and score every ID.

    Missing predictions are failures. Duplicate or unrequested prediction IDs
    raise errors so results cannot silently select favorable duplicate runs.
    Malformed envelopes with an identifiable requested video ID are scored as
    invalid predictions; records without an identifiable ID cannot be joined.
    """
    reference_ids = [_reference(record)[0] for record in references]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValidationError("duplicate reference video IDs")
    requested = set(reference_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for record in predictions:
        if not isinstance(record, dict) or not isinstance(record.get("video_id"), str):
            raise ValidationError("prediction without a joinable video_id")
        video_id = record["video_id"]
        if video_id not in requested:
            raise ValidationError("prediction video_id was not requested")
        if video_id in by_id:
            raise ValidationError("duplicate prediction video_id")
        by_id[video_id] = record
    per_video = [score_prediction(record, by_id.get(video_id), tolerance, iou_threshold)
                 for record, video_id in zip(references, reference_ids)]
    return {"aggregate": aggregate_scores(per_video), "per_video": per_video}
