"""Teacher-forcing preparation with explicit loss masks for partial chapters."""

from __future__ import annotations

from typing import Any

from .supervision import IGNORE_INDEX, SupervisionError


def prepare_seq2seq_example(tokenizer: Any, prompt: str, target_json: str, *,
                            window: dict[str, Any], masked_target_spans: list[list[int]],
                            encoder_limit: int = 8192, decoder_limit: int = 8192) -> dict[str, Any]:
    from .windowing import parse_window_outline
    parse_window_outline(target_json, window)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=True,
                        truncation=False, padding=False)
    target = tokenizer(target_json, add_special_tokens=False, return_offsets_mapping=True,
                       truncation=False, padding=False)
    ids = list(target["input_ids"])
    offsets = list(target["offset_mapping"])
    if not ids or len(ids) != len(offsets):
        raise SupervisionError("seq2seq target IDs/offsets missing")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target_json:
        raise SupervisionError("seq2seq target does not decode exactly")
    if type(tokenizer.eos_token_id) is not int:
        raise SupervisionError("seq2seq EOS must be defined")
    ids.append(tokenizer.eos_token_id)
    labels = list(ids)
    coverage = []
    for span in masked_target_spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] <= len(target_json)):
            raise SupervisionError("invalid partial-title span")
        covered = 0
        for index, (a, b) in enumerate(offsets):
            if a < span[1] and b > span[0]:
                labels[index] = IGNORE_INDEX
                covered += 1
        if not covered:
            raise SupervisionError("Partial-title span has no masked tokenizer position")
        coverage.append(covered)
    if len(encoded["input_ids"]) > encoder_limit or len(ids) > decoder_limit:
        raise SupervisionError("example exceeds supported context; truncation forbidden")
    return {"input_ids": list(encoded["input_ids"]), "attention_mask": list(encoded["attention_mask"]),
            "labels": labels, "full_target_ids": ids,
            "audit": {"supervision_gate_passed": True, "encoder_tokens": len(encoded["input_ids"]),
                      "target_tokens": len(ids), "masked_partial_title_tokens": labels.count(IGNORE_INDEX),
                      "masked_span_token_coverage": coverage,
                      "decoder_input_policy": "shift full_target_ids, not masked labels; placeholder context contains no unsupported gold title"}}
