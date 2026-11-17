"""Gold-independent sentence windows, safe title targets, and strict stitching.

Each [A,B) ownership core is disjoint. [L,R) supplies context, and L is a
processing anchor in the model output, not necessarily a true chapter cut.
Only starts owned by [A,B) survive stitching. No tokenizer/model dependency is
required: callers supply counters for the actual rendered model prompts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from typing import Any

from .data import inference_record
from .evaluation import STATUSES, ValidationError, validate_chapters

WINDOW_PROMPT_VERSION = "owned-window-outline-v1"
MASKED_TITLE_PLACEHOLDER = "Chapter"


class WindowingError(ValueError):
    """Invalid configuration or an input that cannot fit without truncation."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WindowingError(f"{name} must be an integer >= {minimum}")
    return value


def validate_window(window: Any) -> dict[str, Any]:
    """Validate global bounds and context content, without reading references."""
    if not isinstance(window, dict):
        raise WindowingError("window must be an object")
    if type(window.get("schema_version")) is not int or window["schema_version"] != 1:
        raise WindowingError("window must use schema_version 1")
    for key in ["L", "R", "A", "B", "sentence_count", "window_index"]:
        if key not in window:
            raise WindowingError(f"window is missing {key}")
        _integer(window[key], key, 1 if key == "sentence_count" else 0)
    L, R, A, B, n = (window[key] for key in ["L", "R", "A", "B", "sentence_count"])
    if not 0 <= L <= A < B <= R <= n:
        raise WindowingError("window must satisfy 0 <= L <= A < B <= R <= sentence_count")
    if (A == 0 and L != 0) or (A > 0 and L >= A):
        raise WindowingError("nonfirst ownership cores require at least one left-context sentence")
    if not isinstance(window.get("video_id"), str) or not window["video_id"].strip():
        raise WindowingError("window video_id must be a nonempty string")
    sentences = window.get("sentences")
    if not isinstance(sentences, list) or len(sentences) != R - L or any(
        not isinstance(sentence, str) or not sentence.strip() for sentence in sentences
    ):
        raise WindowingError("window sentences must exactly cover [L,R)")
    if "sentence_indices" in window:
        indices = window["sentence_indices"]
        if (not isinstance(indices, list) or any(type(index) is not int for index in indices)
                or indices != list(range(L, R))):
            raise WindowingError("window sentence_indices must be the contiguous global indices")
    return window


def build_window_prompt(window: dict[str, Any]) -> str:
    """Render only visible text and global bounds; never consult gold fields."""
    validate_window(window)
    L, R, A, B = (window[key] for key in ["L", "R", "A", "B"])
    indexed = "\n".join(f"[{index}] {sentence}"
                        for index, sentence in enumerate(window["sentences"], L))
    return (
        "Divide the visible transcript into contiguous topical chapters and give each a concise, "
        "specific title supported only by visible text. Do not rewrite the transcript. "
        "Bracketed numbers are GLOBAL sentence indices. A chapter start is the index of its first sentence. "
        f"Visible context is [{L},{R}); the ownership region is [{A},{B}). "
        f"Always include start {L} as the first item: it is a processing anchor, not evidence of a real cut. "
        "Include other chapter starts throughout visible context when its content changes. "
        "Ownership edges alone do not create chapter boundaries. Only starts inside the ownership region "
        "will be retained after processing all windows. Choose chapter count from visible content. "
        f"Use strictly increasing integer starts from {L} through {R - 1}, with nonempty titles. "
        "In your final answer return only one JSON object with exactly this structure: "
        f'{{"chapters":[{{"start":{L},"title":"Short title"}}]}}. '
        "Do not put commentary or a Markdown fence around the final JSON.\n\n"
        f"TRANSCRIPT\n{indexed}"
    )


def build_windows(
    record: Mapping[str, Any],
    token_counters: Mapping[str, Callable[[str], int]],
    max_input_tokens: int | Mapping[str, int],
    *,
    max_core_sentences: int = 128,
    left_context: int = 16,
    right_context: int = 16,
    core_search: str = "exhaustive",
) -> list[dict[str, Any]]:
    """Pack common windows under every counter's actual rendered-input budget.

    Preserve desired context first, testing core sizes in descending order.
    Only if a one-sentence core cannot fit with desired context, shrink context
    deterministically, then maximize the core with that reduced context. The
    mandatory left neighbor is retained. A right neighbor is retained whenever
    it fits; dropping that last right sentence is the final permitted fallback.
    No monotonicity assumption is made about counters or tokenization with
    the default ``core_search="exhaustive"``. ``core_search="bisect"`` is for
    large core limits: it bisects on the core end assuming token counts do not
    decrease as sentences are added, then re-tests the eight next larger cores.
    """
    clean = inference_record(record)
    sentences, video_id = clean["sentences"], clean["video_id"]
    n = len(sentences)
    _integer(max_core_sentences, "max_core_sentences", 1)
    _integer(left_context, "left_context", 1)
    _integer(right_context, "right_context", 1)
    if core_search not in {"exhaustive", "bisect"}:
        raise WindowingError("core_search must be exhaustive or bisect")
    if not isinstance(token_counters, Mapping) or not token_counters or any(
        not isinstance(alias, str) or not alias.strip() or not callable(counter)
        for alias, counter in token_counters.items()
    ):
        raise WindowingError("provide at least one named token-counter callable")
    if isinstance(max_input_tokens, Mapping):
        if set(max_input_tokens) != set(token_counters):
            raise WindowingError("budget aliases must exactly match token-counter aliases")
        budgets = dict(max_input_tokens)
        for alias, budget in budgets.items():
            _integer(budget, f"max_input_tokens[{alias}]", 1)
    else:
        _integer(max_input_tokens, "max_input_tokens", 1)
        budgets = {alias: max_input_tokens for alias in token_counters}

    windows = []
    A = 0
    while A < n:
        cache: dict[tuple[int, int, int, int], tuple[dict[str, Any], bool]] = {}

        def candidate(B: int, left: int, right: int) -> tuple[dict[str, Any], bool]:
            L, R = max(0, A - left), min(n, B + right)
            key = (L, R, A, B)
            if key in cache:
                return cache[key]
            window = {
                "schema_version": 1, "video_id": video_id, "window_index": len(windows),
                "sentence_count": n, "L": L, "R": R, "A": A, "B": B,
                "sentences": list(sentences[L:R]), "sentence_indices": list(range(L, R)),
                "prompt_version": WINDOW_PROMPT_VERSION,
            }
            prompt = build_window_prompt(window)
            counts = {}
            for alias, counter in token_counters.items():
                try:
                    value = counter(prompt)
                except Exception as exc:
                    raise WindowingError(f"token counter failed: {alias}") from exc
                counts[alias] = _integer(value, f"token count from {alias}")
            window["input_token_counts"] = counts
            window["input_token_budgets"] = dict(budgets)
            result = (window, all(counts[alias] <= budgets[alias] for alias in counts))
            cache[key] = result
            return result

        def largest_core(left: int, right: int) -> dict[str, Any] | None:
            upper = min(n, A + max_core_sentences)
            if core_search == "bisect":
                window, fits = candidate(upper, left, right)
                if fits:
                    return dict(window)
                low, high = A, upper  # low fits (or is the empty core); high does not
                while high - low > 1:
                    middle = (low + high) // 2
                    if candidate(middle, left, right)[1]:
                        low = middle
                    else:
                        high = middle
                upper = min(upper, low + 8)
            for B in range(upper, A, -1):
                window, fits = candidate(B, left, right)
                if fits:
                    return dict(window)
            return None

        chosen = largest_core(left_context, right_context)
        shrunk = chosen is None
        selected_left, selected_right = left_context, right_context
        right_omitted = False
        if shrunk:
            # Minimal context is tried only after every desired-context core
            # size failed. Reduce the side with the most excess context; ties
            # reduce right first, preserving more evidence before a boundary.
            selected_left = min(left_context, A)
            selected_right = min(right_context, n - A - 1)
            minimum_left = int(A > 0)
            minimum_right = int(A + 1 < n)
            while True:
                _, fits = candidate(A + 1, selected_left, selected_right)
                if fits:
                    break
                left_excess = selected_left - minimum_left
                right_excess = selected_right - minimum_right
                if right_excess > 0 and right_excess >= left_excess:
                    selected_right -= 1
                elif left_excess > 0:
                    selected_left -= 1
                elif right_excess > 0:
                    selected_right -= 1
                elif selected_right > 0:
                    selected_right = 0
                    right_omitted = True
                else:
                    raise WindowingError(
                        f"sentence {A} and mandatory left context cannot fit every input budget; "
                        "no transcript was truncated"
                    )
            chosen = largest_core(selected_left, selected_right)
            assert chosen is not None
        chosen.update({
            "desired_left_context": left_context, "desired_right_context": right_context,
            "left_context_count": chosen["A"] - chosen["L"],
            "right_context_count": chosen["R"] - chosen["B"],
            "context_shrunk": shrunk,
            "right_context_omitted_as_infeasible": right_omitted,
        })
        windows.append(chosen)
        A = chosen["B"]
    return windows


def validate_window_chapters(chapters: Any, window: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate all context starts before ownership filtering; never sort."""
    validate_window(window)
    L, R = window["L"], window["R"]
    if not isinstance(chapters, list) or not chapters:
        raise ValidationError("window chapters must be a nonempty list")
    translated = []
    for chapter in chapters:
        if not isinstance(chapter, dict) or set(chapter) != {"start", "title"}:
            raise ValidationError("each window chapter must contain only start and title")
        if type(chapter["start"]) is not int:
            raise ValidationError("window starts must be integer global indices")
        translated.append({"start": chapter["start"] - L, "title": chapter["title"]})
    valid = validate_chapters(translated, R - L)
    return [{"start": chapter["start"] + L, "title": chapter["title"]} for chapter in valid]


def parse_window_outline(text: str, window: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse one final JSON object; reject reasoning, prose and duplicate keys."""
    if not isinstance(text, str):
        raise ValidationError("window final answer must be a string")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError("duplicate JSON key in window outline")
            value[key] = item
        return value

    def constant(value: str) -> None:
        raise ValidationError("nonstandard JSON constant in window outline")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("window final answer must be one valid JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"chapters"}:
        raise ValidationError("window final answer must contain only chapters")
    return validate_window_chapters(value["chapters"], window)


def window_training_target(record: Mapping[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    """Supervise starts/structure while withholding unsupported gold titles.

    Title content is supervised only for an owned true chapter whose entire
    gold span is visible. Otherwise the literal placeholder Chapter is decoder
    context, and its content-only character interval must receive no loss.
    Quotes and all JSON structure remain supervised. Offsets index target_json
    as a Python Unicode string, including any JSON escape characters.
    """
    validate_window(window)
    clean = inference_record(record)
    n = len(clean["sentences"])
    L, R, A, B = (window[key] for key in ["L", "R", "A", "B"])
    if (clean["video_id"] != window["video_id"] or n != window["sentence_count"]
            or clean["sentences"][L:R] != window["sentences"]):
        raise WindowingError("training record does not match its context window")
    reference = record.get("reference")
    if not isinstance(reference, dict):
        raise WindowingError("training record requires a canonical reference")
    starts, titles = reference.get("chapter_starts"), reference.get("chapter_titles")
    if not isinstance(starts, list) or not isinstance(titles, list) or len(starts) != len(titles):
        raise WindowingError("training starts and titles must be aligned lists")
    chapters = validate_chapters([{"start": start, "title": title}
                                  for start, title in zip(starts, titles)], n)
    ends = starts[1:] + [n]
    gold = {chapter["start"]: (chapter["title"], end) for chapter, end in zip(chapters, ends)}
    emitted_starts = [L] + [start for start in starts if L < start < R]
    pieces: list[str] = []
    position = 0
    masked_spans = []
    stats = {"emitted_chapters": len(emitted_starts), "supervised_titles": 0,
             "masked_titles": 0, "masked_processing_anchor_titles": 0,
             "masked_unowned_titles": 0, "masked_partial_or_synthetic_titles": 0,
             "processing_anchor_is_true_chapter_start": L in gold}

    def append(value: str) -> None:
        nonlocal position
        pieces.append(value)
        position += len(value)

    append('{"chapters":[')
    for index, start in enumerate(emitted_starts):
        if index:
            append(",")
        owned = A <= start < B
        complete = start in gold and L <= start and gold[start][1] <= R
        supervised = owned and complete
        title = gold[start][0] if supervised else MASKED_TITLE_PLACEHOLDER
        if supervised:
            stats["supervised_titles"] += 1
        else:
            stats["masked_titles"] += 1
            stats["masked_processing_anchor_titles"] += int(index == 0)
            stats["masked_unowned_titles"] += int(not owned)
            stats["masked_partial_or_synthetic_titles"] += int(not complete)
        append(f'{{"start":{start},"title":')
        literal = json.dumps(title, ensure_ascii=False)
        content_start = position + 1
        append(literal)
        if not supervised:
            masked_spans.append([content_start, position - 1])
        append("}")
    append("]}")
    target = "".join(pieces)
    # Validate the actual serialized training output, not a parallel object.
    parse_window_outline(target, window)
    return {"target_json": target, "masked_target_spans": masked_spans,
            "title_supervision_stats": stats}


def stitch_window_predictions(
    windows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], *, run_id: str
) -> dict[str, Any]:
    """Keep owned starts in order; any missing/failed/malformed window fails video.

    Each prediction needs status and chapters, with optional usage/error and
    video_id/window_index checks. This function never sorts windows or starts,
    invents a boundary, merges duplicate titles, or returns partial success.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise WindowingError("run_id must be a nonempty string")
    if (not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows
            or not isinstance(windows[0], dict) or not isinstance(windows[0].get("video_id"), str)):
        raise WindowingError("stitching requires identifiable nonempty windows")
    video_id = windows[0]["video_id"]
    result = {"schema_version": 1, "video_id": video_id, "run_id": run_id,
              "status": "error", "chapters": [], "error": None,
              "usage": {"window_count": len(windows), "per_window": []},
              "prompt_version": WINDOW_PROMPT_VERSION}

    def fail(status: str, error: str, index: int | None = None) -> dict[str, Any]:
        result.update(status=status, chapters=[], error=error)
        if index is not None:
            result["failed_window_index"] = index
        return result

    # Account for every executed request before any early validation/failure return.
    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
        result["usage"]["per_window"] = [
            dict(item.get("usage", {})) if isinstance(item, dict) and isinstance(item.get("usage", {}), dict) else {}
            for item in predictions]
        result["usage"]["executed_window_count"] = sum(not item.get("skipped", False)
            for item in predictions if isinstance(item, dict))
        for field in ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]:
            values = [usage.get(field) for usage in result["usage"]["per_window"]]
            result["usage"][field] = (sum(values) if values and all(type(value) in {int, float}
                and math.isfinite(value) and value >= 0 for value in values) else None)

    try:
        n = windows[0]["sentence_count"]
        next_core = 0
        for index, window in enumerate(windows):
            validate_window(window)
            if (window["video_id"] != video_id or window["sentence_count"] != n
                    or window["window_index"] != index or window["A"] != next_core):
                return fail("invalid", "window layout is not an ordered, disjoint ownership partition", index)
            next_core = window["B"]
        if next_core != n:
            return fail("invalid", "window ownership does not cover the complete transcript")
    except (WindowingError, KeyError, TypeError):
        return fail("invalid", "malformed context/ownership window")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        return fail("invalid", "window predictions must be an ordered sequence")
    if len(predictions) != len(windows):
        return fail("error", "missing or extra window prediction")
    stitched = []
    for index, (window, prediction) in enumerate(zip(windows, predictions)):
        if not isinstance(prediction, dict):
            return fail("invalid", "window prediction must be an object", index)
        status = prediction.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            return fail("invalid", "unknown or missing window prediction status", index)
        if ("video_id" in prediction and prediction["video_id"] != video_id) or (
            "window_index" in prediction and prediction["window_index"] != index
        ):
            return fail("invalid", "window prediction identity mismatch", index)
        usage = prediction.get("usage", {})
        if not isinstance(usage, dict):
            return fail("invalid", "window usage must be an object", index)
        if status != "ok":
            return fail(status, "one or more windows failed; no partial outline was accepted", index)
        if prediction.get("error") is not None:
            return fail("invalid", "successful window has an error field", index)
        try:
            chapters = validate_window_chapters(prediction.get("chapters"), window)
        except (ValidationError, WindowingError):
            return fail("invalid", "malformed window chapters", index)
        stitched.extend(chapter for chapter in chapters if window["A"] <= chapter["start"] < window["B"])
    try:
        validate_chapters(stitched, n)
    except ValidationError:
        return fail("invalid", "owned starts do not form a valid whole-document outline")
    result.update(status="ok", chapters=stitched, error=None)
    return result
