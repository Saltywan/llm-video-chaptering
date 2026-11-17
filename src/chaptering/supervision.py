"""Auditable final-outline loss masks for the pinned native causal templates.

Only final JSON tokens and one native assistant termination token receive
loss. Native thinking context is preserved as unscored input; this module
neither creates reasoning traces nor creates empty reasoning targets.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .evaluation import parse_outline


IGNORE_INDEX = -100
GEMMA_CLOSED_THOUGHT = "<|channel>thought\n<channel|>"


class SupervisionError(ValueError):
    """Native formatting or token offsets cannot support a safe loss mask."""


_FAMILIES = {
    "qwen": "qwen3.5", "qwen3.5": "qwen3.5", "qwen3_5": "qwen3.5",
    "m1": "qwen3.5", "qwen/qwen3.5-9b": "qwen3.5",
    "gemma": "gemma4", "gemma4": "gemma4", "gemma-4": "gemma4",
    "m2": "gemma4", "google/gemma-4-12b-it": "gemma4",
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_target(target: str) -> None:
    if not isinstance(target, str) or not target.strip():
        raise SupervisionError("target_json must be a nonempty JSON outline string")
    # Infer only an upper bound for structural validation. Actual document
    # bounds remain the canonical data loader's responsibility.
    try:
        value = json.loads(target)
        chapters = value.get("chapters") if isinstance(value, dict) else None
        starts = [chapter.get("start") for chapter in chapters
                  if isinstance(chapter, dict)] if isinstance(chapters, list) else []
        upper_bound = max([start for start in starts if type(start) is int] + [0]) + 1
        parse_outline(target, upper_bound)
        target.encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SupervisionError("target_json violates the chapter-outline contract") from exc
    # Whitespace outside the JSON would make the supervised span include
    # formatting rather than only the requested final JSON object.
    if target != target.strip():
        raise SupervisionError("target_json must not have outer whitespace")


def _decode(tokenizer: Any, ids: list[int]) -> str:
    value = tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
    if not isinstance(value, str):
        raise SupervisionError("tokenizer.decode must return text")
    return value


def _native_context(prefix: str, family: str, gemma_final_context: str = "direct_v1") -> dict[str, Any]:
    if family == "qwen3.5":
        marker = "<|im_start|>assistant\n"
        position = prefix.rfind(marker)
        if position < 0:
            raise SupervisionError("missing native final Qwen assistant marker")
        context = prefix[position + len(marker):]
        if not context.startswith("<think>") or "</think>" not in context:
            raise SupervisionError("Qwen final JSON would be inside unclosed thinking")
        body, after = context[len("<think>"):].split("</think>", 1)
        if body.strip() or after.strip() or context.count("<think>") != 1:
            raise SupervisionError("unexpected Qwen final-answer context; template review required")
        return {"assistant_marker_char_start": position,
                "native_empty_thought_context": True,
                "native_thinking_control_present": True,
                "native_thought_context_supervised": False,
                "native_thought_context_characters": len(context)}
    marker = "<|turn>model\n"
    position = prefix.rfind(marker)
    if position < 0:
        raise SupervisionError("missing native final Gemma model marker")
    context = prefix[position + len(marker):]
    expected = GEMMA_CLOSED_THOUGHT if gemma_final_context == "closed_thought_v2" else ""
    if context != expected:
        raise SupervisionError("unexpected or unclosed Gemma final-answer channel")
    if "<|think|>" not in prefix[:position]:
        raise SupervisionError("native Gemma thinking control missing despite enable_thinking=True")
    return {"assistant_marker_char_start": position,
            "native_empty_thought_context": bool(expected),
            "native_thinking_control_present": True,
            "native_thought_context_supervised": False,
            "native_thought_context_characters": len(context),
            "gemma_final_context": gemma_final_context,
            "native_block_explicitly_inserted": bool(expected)}


def thought_prefix_from_raw(raw_text: str, family: str) -> str | None:
    """Cut a saved native generation down to its closed thought, or return None.

    The result is exactly what the base model emitted from the first generated
    token through the native thought close and the whitespace after it. It is
    used only as unscored conditioning context (``self_thought_v3``); a
    generation with no closed thought is unusable and gives None.
    """
    if not isinstance(family, str) or family.lower() not in _FAMILIES:
        raise SupervisionError("unsupported pinned causal-model family")
    close = "</think>" if _FAMILIES[family.lower()] == "qwen3.5" else "<channel|>"
    if not isinstance(raw_text, str) or close not in raw_text:
        return None
    end = raw_text.index(close) + len(close)
    while end < len(raw_text) and raw_text[end] in " \t\r\n":
        end += 1
    prefix = raw_text[:end]
    try:
        _self_thought_context(prefix, _FAMILIES[family.lower()])
    except SupervisionError:
        return None
    return prefix


def _self_thought_context(prefix: str, family: str) -> None:
    """Require one complete native thought and nothing after its close."""
    if family == "qwen3.5":
        close, forbidden = "</think>", ("<think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>")
        body = prefix.split(close)[0]
    else:
        close, forbidden = "<channel|>", ("<turn|>", "<|turn>", "<eos>", "<bos>")
        opener = "<|channel>thought\n"
        if not prefix.startswith(opener) or prefix.count("<|channel>") != 1:
            raise SupervisionError("Gemma self thought must open exactly one native thought channel")
        body = prefix[len(opener):].split(close)[0]
    if prefix.count(close) != 1 or prefix.split(close, 1)[1].strip() or not body.strip():
        raise SupervisionError("self thought must be one nonempty closed native thought")
    if any(token in prefix for token in forbidden):
        raise SupervisionError("self thought contains native turn framing")


def _termination(tokenizer: Any, family: str) -> tuple[str, int]:
    token = "<|im_end|>" if family == "qwen3.5" else "<turn|>"
    token_id = tokenizer.convert_tokens_to_ids(token)
    if type(token_id) is not int or token_id < 0:
        raise SupervisionError("native assistant terminator has no integer token ID")
    if token_id == getattr(tokenizer, "unk_token_id", None):
        raise SupervisionError("native assistant terminator resolves to unknown token")
    if _decode(tokenizer, [token_id]) != token:
        raise SupervisionError("native assistant terminator token does not decode exactly")
    if family == "qwen3.5":
        if token_id != 248046:
            raise SupervisionError("Qwen terminator differs from the inspected pinned token ID")
    else:
        if getattr(tokenizer, "eos_token_id", None) != 1:
            raise SupervisionError("Gemma EOS differs from the inspected pinned tokenizer")
        if getattr(tokenizer, "eos_token", None) != "<eos>" or token_id == 1:
            raise SupervisionError("Gemma requires native <turn|>, not ordinary EOS")
    return token, token_id


def prepare_causal_example(tokenizer: Any, prompt: str, target_json: str,
                           family: str, *, window: dict[str, Any] | None = None,
                           masked_target_spans: list[list[int]] | None = None,
                           gemma_final_context: str = "direct_v1",
                           thought_prefix: str | None = None) -> dict[str, Any]:
    """Render once, tokenize once, and mask to final JSON plus native turn end.

    With ``thought_prefix`` (recipe ``self_thought_v3``) the sequence is the
    native generation prompt, then the base model's own saved closed thought,
    then the final JSON and terminator. Every thought token is unscored context.

    Accepted family names include qwen3.5/M1 and gemma4/M2 (or their complete
    checkpoint names). The tokenizer must be fast and must match the inspected
    native template. No truncation or padding is requested. Any existing masked
    padding returned by a tokenizer is retained with label -100.

    Returned input_ids/attention_mask/labels are ordinary integer lists; audit
    is metadata and must not be passed as a model.forward argument. The causal
    model performs its usual internal next-token label shift. The caller must
    separately check target starts against the true document sentence count.
    """
    if not isinstance(family, str) or family.lower() not in _FAMILIES:
        raise SupervisionError("unsupported pinned causal-model family")
    family = _FAMILIES[family.lower()]
    if gemma_final_context not in {"direct_v1", "closed_thought_v2"}:
        raise SupervisionError("unsupported Gemma final-context version")
    if family != "gemma4" and gemma_final_context != "direct_v1":
        raise SupervisionError("Gemma final-context override is only valid for Gemma")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SupervisionError("prompt must be a nonempty string")
    if not getattr(tokenizer, "is_fast", False):
        raise SupervisionError("a fast tokenizer with character offsets is required")
    if window is None:
        _validate_target(target_json)
    else:
        from .windowing import parse_window_outline
        parse_window_outline(target_json, window)
        if target_json != target_json.strip():
            raise SupervisionError("target_json must not have outer whitespace")
    end_token, end_id = _termination(tokenizer, family)
    if thought_prefix is not None:
        if gemma_final_context != "direct_v1":
            raise SupervisionError("self thought cannot be combined with another final context")
        if not isinstance(thought_prefix, str):
            raise SupervisionError("thought_prefix must be a string")
        _self_thought_context(thought_prefix, family)
        generation = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        marker = "<|im_start|>assistant\n" if family == "qwen3.5" else "<|turn>model\n"
        if not isinstance(generation, str) or marker not in generation:
            raise SupervisionError("native generation prompt has no assistant marker")
        if family == "gemma4" and "<|think|>" not in generation:
            raise SupervisionError("native Gemma thinking control missing despite enable_thinking=True")
        rendered = generation + thought_prefix + target_json + end_token + "\n"
        target_start = len(generation) + len(thought_prefix)
        target_end = target_start + len(target_json)
        prefix, suffix = rendered[:target_start], rendered[target_end:]
        context_audit = {"assistant_marker_char_start": generation.rfind(marker),
                         "native_empty_thought_context": False,
                         "native_thinking_control_present": True,
                         "native_thought_context_supervised": False,
                         "native_thought_context_characters": len(prefix) - generation.rfind(marker) - len(marker),
                         "final_context": "self_thought_v3",
                         "self_generated_thought_context": True,
                         "self_thought_sha256": _hash(thought_prefix)}
    else:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_json}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        if not isinstance(rendered, str):
            raise SupervisionError("native template must render a string")
        target_start = rendered.rfind(target_json)
        if target_start < 0:
            raise SupervisionError("native template did not preserve the exact final JSON")
        if family == "gemma4" and gemma_final_context == "closed_thought_v2":
            # The official content-only template puts JSON directly after the open
            # model header. Move its supervision to the native final-answer phase.
            # This explicit empty block is conditioning only: every token is -100.
            _native_context(rendered[:target_start], family)
            rendered = rendered[:target_start] + GEMMA_CLOSED_THOUGHT + rendered[target_start:]
            target_start += len(GEMMA_CLOSED_THOUGHT)
        target_end = target_start + len(target_json)
        prefix, suffix = rendered[:target_start], rendered[target_end:]
        context_audit = _native_context(prefix, family, gemma_final_context)
    if target_json in prefix[context_audit["assistant_marker_char_start"]:]:
        raise SupervisionError("final assistant JSON is not unique within its message")
    if not suffix.startswith(end_token) or suffix[len(end_token):].strip():
        raise SupervisionError("JSON must be followed immediately by the native assistant terminator")
    termination_end = target_end + len(end_token)

    # One tokenization of the complete rendered string avoids differing
    # prefix/target tokenization at boundaries (including Unicode boundaries).
    encoded = tokenizer(rendered, add_special_tokens=False,
                        return_offsets_mapping=True, return_attention_mask=True,
                        truncation=False, padding=False)
    try:
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        offsets = list(encoded["offset_mapping"])
    except (KeyError, TypeError) as exc:
        raise SupervisionError("tokenizer omitted required flat IDs/mask/offsets") from exc
    if not input_ids or not (len(input_ids) == len(attention_mask) == len(offsets)):
        raise SupervisionError("token IDs, attention mask and offsets must have equal nonzero lengths")
    if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
        raise SupervisionError("token IDs must be nonnegative integers")
    if any(type(bit) is not int or bit not in (0, 1) for bit in attention_mask):
        raise SupervisionError("attention mask must contain integer zeros and ones")
    active = [index for index, bit in enumerate(attention_mask) if bit]
    if not active or active != list(range(active[0], active[-1] + 1)):
        raise SupervisionError("attention mask contains no tokens or internal padding")
    json_positions: list[int] = []
    for index, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise SupervisionError("each token offset must be a character interval")
        start, end = offset
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(rendered):
            raise SupervisionError("token offset is outside rendered text")
        if not attention_mask[index]:
            if start != end or input_ids[index] != getattr(tokenizer, "pad_token_id", None):
                raise SupervisionError("masked padding has text offsets or a non-pad token ID")
            continue
        if start == end:
            continue
        if start < target_start < end or start < target_end < end:
            raise SupervisionError("token crosses a final JSON supervision boundary")
        if target_start <= start < end <= target_end:
            json_positions.append(index)
    if not json_positions or json_positions != list(range(json_positions[0], json_positions[-1] + 1)):
        raise SupervisionError("final JSON tokens are missing or not contiguous")
    if _decode(tokenizer, [input_ids[index] for index in json_positions]) != target_json:
        raise SupervisionError("supervised JSON tokens do not decode to the exact target")
    termination_position = json_positions[-1] + 1
    if (termination_position >= len(input_ids) or not attention_mask[termination_position]
            or input_ids[termination_position] != end_id):
        raise SupervisionError("native assistant terminator is not the token immediately after JSON")
    term_offset = tuple(offsets[termination_position])
    if term_offset[0] != term_offset[1] and term_offset != (target_end, termination_end):
        raise SupervisionError("assistant terminator offsets do not align with the rendered suffix")
    supervised_positions = json_positions + [termination_position]
    labels = [IGNORE_INDEX] * len(input_ids)
    for position in supervised_positions:
        labels[position] = input_ids[position]
    partial_mask = []
    for span in masked_target_spans or []:
        if (not isinstance(span, (tuple, list)) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] <= len(target_json)):
            raise SupervisionError("invalid masked target character span")
        partial_mask.append((target_start + span[0], target_start + span[1]))
    removed = 0
    span_coverage = [sum(offsets[position][0] < right and offsets[position][1] > left
                         for position in json_positions) for left, right in partial_mask]
    if any(count == 0 for count in span_coverage):
        raise SupervisionError("Partial-title span has no masked tokenizer position")
    for position in json_positions:
        a, b = offsets[position]
        if any(a < right and b > left for left, right in partial_mask):
            labels[position] = IGNORE_INDEX
            removed += 1
    if labels[termination_position] != end_id or not any(labels[i] != IGNORE_INDEX for i in json_positions):
        raise SupervisionError("partial-title mask removed all final supervision or its terminator")
    actual_supervised = sum(label != IGNORE_INDEX for label in labels)
    if any(label != IGNORE_INDEX and (not attention_mask[index] or label != input_ids[index])
           for index, label in enumerate(labels)):
        raise SupervisionError("labels and attention mask are inconsistent")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "audit": {"version": 1, "family": family, "enable_thinking": True,
                      "supervision_gate_passed": True,
                      "rendered_sha256": _hash(rendered), "prompt_sha256": _hash(prompt),
                      "target_sha256": _hash(target_json),
                      "supervised_json_sha256": (_hash(_decode(
                          tokenizer, [input_ids[index] for index in json_positions])) if not removed else None),
                      "target_char_span": [target_start, target_end],
                      "json_token_start": json_positions[0],
                      "json_token_end_exclusive": json_positions[-1] + 1,
                      "json_token_count": len(json_positions),
                      "termination_token": end_token, "termination_token_id": end_id,
                      "termination_token_position": termination_position,
                      "termination_char_span": [target_end, termination_end],
                      "termination_has_empty_offset": term_offset[0] == term_offset[1],
                      "total_tokens": len(input_ids),
                      "padding_tokens": len(input_ids) - len(active),
                      "supervised_tokens": actual_supervised,
                      "masked_tokens": len(input_ids) - actual_supervised,
                      "masked_partial_title_tokens": removed,
                      "masked_target_spans": masked_target_spans or [],
                      "masked_span_token_coverage": span_coverage,
                      "fully_supervised_json": not bool(partial_mask),
                      "supervised_reasoning_tokens": 0,
                      "trailing_suffix_characters_masked": len(rendered) - termination_end,
                      **context_audit}}
