"""Per-request native thinking budgets for batch-one HF generation.

This policy forces the model's own channel-closing tokens at a generated-token
budget. It does not disable native thinking or truncate total generation. The
caller must reserve a separate, larger max_new_tokens budget for the final JSON.
No torch import, model access or GPU selection occurs here.
"""

from __future__ import annotations

from typing import Any, Sequence


class ReasoningBudgetError(ValueError):
    """Budget configuration or generation tensor shape is unsupported."""


def _tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ReasoningBudgetError(f"{name} must be a nonempty list or tuple")
    if any(type(token) is not int or token < 0 for token in values):
        raise ReasoningBudgetError(f"{name} must contain nonnegative integer token IDs")
    return tuple(values)


def _find(sequence, marker):
    for position in range(len(sequence) - len(marker) + 1):
        if tuple(sequence[position:position + len(marker)]) == marker:
            return position
    return None


def _partial_close(sequence, marker):
    for length in range(min(len(sequence), len(marker) - 1), 0, -1):
        if tuple(sequence[-length:]) == marker[:length]:
            return length
    return 0


class NativeThinkingBudget:
    """HF-compatible logits processor for one request and batch size one.

    open_token_ids=None means the prompt has already opened its native thought
    channel (Qwen). Otherwise a complete generated native opener must be seen
    before the budget can act (Gemma). The budget counts *generated* tail tokens,
    including a generated opener, and never counts prompt tokens.

    close_token_ids is the exact sequence to force one token at a time. An
    optional shorter closed_marker_token_ids detects semantic native closure
    even when whitespace after that marker differs. Once forcing starts, the
    full requested closing sequence is completed. Naturally closed channels
    are never reopened or closed a second time.

    Create a new object for every generation request. Beam/batched generation
    is explicitly unsupported; changing a request prefix is rejected.
    """

    def __init__(self, prompt_length: int, reasoning_budget: int,
                 close_token_ids: Sequence[int],
                 open_token_ids: Sequence[int] | None = None,
                 *, closed_marker_token_ids: Sequence[int] | None = None):
        if type(prompt_length) is not int or prompt_length < 0:
            raise ReasoningBudgetError("prompt_length must be a nonnegative integer")
        if type(reasoning_budget) is not int or reasoning_budget < 1:
            raise ReasoningBudgetError("reasoning_budget must be a positive integer")
        self.prompt_length = prompt_length
        self.reasoning_budget = reasoning_budget
        self.close_token_ids = _tokens(close_token_ids, "close_token_ids")
        self.open_token_ids = (_tokens(open_token_ids, "open_token_ids")
                               if open_token_ids is not None else None)
        self.closed_marker_token_ids = (_tokens(closed_marker_token_ids, "closed_marker_token_ids")
                                        if closed_marker_token_ids is not None else self.close_token_ids)
        if self.close_token_ids[:len(self.closed_marker_token_ids)] != self.closed_marker_token_ids:
            raise ReasoningBudgetError("semantic close marker must prefix the native closing sequence")
        if self.open_token_ids == self.close_token_ids:
            raise ReasoningBudgetError("native open and close sequences must differ")
        self._previous_tail: tuple[int, ...] = ()
        self._prefix: tuple[int, ...] | None = None
        self._reasoning_observed = self.open_token_ids is None
        self._closed = False
        self._closed_naturally = False
        self._close_start: int | None = None
        self._first_force: int | None = None
        self._forced: dict[int, int] = {}
        self._family: str | None = None

    @property
    def audit(self) -> dict[str, Any]:
        return {"policy": "native_channel_generated_token_budget_v1",
                "family": self._family, "prompt_length": self.prompt_length,
                "reasoning_budget": self.reasoning_budget,
                "budget_counts_generated_opener": True,
                "force_triggered": self._first_force is not None,
                "forced_token_count": len(self._forced),
                "forced_token_positions": sorted(self._forced),
                "forced_token_ids": [self._forced[position] for position in sorted(self._forced)],
                "reasoning_channel_observed": self._reasoning_observed,
                "reasoning_channel_closed": self._closed,
                "closing_sequence_complete": _find(self._previous_tail, self.close_token_ids) is not None,
                "closed_naturally_before_forcing": self._closed_naturally,
                "generated_tokens_at_first_force": self._first_force,
                "generated_tokens_seen": len(self._previous_tail),
                "close_token_ids": list(self.close_token_ids),
                "closed_marker_token_ids": list(self.closed_marker_token_ids),
                "open_token_ids": list(self.open_token_ids) if self.open_token_ids is not None else None}

    def __call__(self, input_ids, scores):
        try:
            ids = input_ids.tolist()
            shape = tuple(scores.shape)
        except (AttributeError, TypeError) as exc:
            raise ReasoningBudgetError("HF-style input IDs and score tensors are required") from exc
        if (not isinstance(ids, list) or len(ids) != 1
                or not isinstance(ids[0], list) or len(shape) != 2 or shape[0] != 1):
            raise ReasoningBudgetError("thinking-budget policy supports batch size one only")
        if type(shape[1]) is not int or shape[1] < 1:
            raise ReasoningBudgetError("scores must have a nonempty vocabulary dimension")
        vocabulary = shape[1]
        tokens = ids[0]
        if any(type(token) is not int or not 0 <= token < vocabulary for token in tokens):
            raise ReasoningBudgetError("input IDs must be nonnegative integers within the score vocabulary")
        markers = self.close_token_ids + self.closed_marker_token_ids + (self.open_token_ids or ())
        if any(token >= vocabulary for token in markers):
            raise ReasoningBudgetError("native channel token ID exceeds the score vocabulary")
        if len(tokens) < self.prompt_length:
            raise ReasoningBudgetError("input sequence is shorter than its recorded prompt")
        prefix = tuple(tokens[:self.prompt_length])
        tail = tuple(tokens[self.prompt_length:])
        if self._prefix is not None and self._prefix != prefix:
            raise ReasoningBudgetError("thinking-budget processor was reused for a different prompt")
        if len(tail) < len(self._previous_tail) or tail[:len(self._previous_tail)] != self._previous_tail:
            raise ReasoningBudgetError("generation tail changed or moved backwards within a request")
        self._prefix, self._previous_tail = prefix, tail
        if self._closed:
            return scores

        if self._close_start is not None:
            observed = tail[self._close_start:self._close_start + len(self.close_token_ids)]
            if observed != self.close_token_ids[:len(observed)]:
                raise ReasoningBudgetError("generation did not honor its forced closing sequence")
            if len(observed) == len(self.close_token_ids):
                self._closed = True
                return scores
            next_token = self.close_token_ids[len(observed)]
        else:
            # Only a native channel that really opened can consume a thought
            # budget. A direct Gemma final answer with no opener is untouched.
            open_position = (_find(tail, self.open_token_ids)
                             if self.open_token_ids is not None else -1)
            if open_position is not None:
                self._reasoning_observed = True
            if not self._reasoning_observed:
                return scores
            close_position = _find(tail, self.closed_marker_token_ids)
            if close_position is not None:
                # A close before the first generated opener cannot close a
                # later channel; avoid interpreting ordinary earlier content.
                if self.open_token_ids is None or close_position >= open_position + len(self.open_token_ids):
                    self._closed = self._closed_naturally = True
                    return scores
            if len(tail) < self.reasoning_budget:
                return scores
            partial = _partial_close(tail, self.close_token_ids)
            self._close_start = len(tail) - partial
            next_token = self.close_token_ids[partial]

        try:
            scores.fill_(float("-inf"))
            scores[0, next_token] = 0.0
        except (AttributeError, IndexError, TypeError) as exc:
            raise ReasoningBudgetError("scores do not support HF-style in-place logits masking") from exc
        self._first_force = len(tail) if self._first_force is None else self._first_force
        self._forced[len(tail)] = next_token
        return scores


def make_thinking_budget_processor(tokenizer, alias: str, prompt_length: int,
                                   budget: int) -> NativeThinkingBudget:
    """Encode inspected native markers for Qwen/M1 or Gemma/M2.

    Qwen's semantic close is </think>; its forced sequence additionally contains
    the official double newline. Recognizing the marker separately prevents a
    second close being inserted into a naturally final JSON answer.
    """
    if not isinstance(alias, str):
        raise ReasoningBudgetError("model alias must identify Qwen/M1 or Gemma/M2")
    family = alias.lower()
    if family in {"m1", "qwen", "qwen3.5", "qwen3_5", "qwen3.5-9b", "qwen/qwen3.5-9b"}:
        close = tokenizer.encode("</think>\n\n", add_special_tokens=False)
        marker = tokenizer.encode("</think>", add_special_tokens=False)
        processor = NativeThinkingBudget(prompt_length, budget, close,
                                         closed_marker_token_ids=marker)
        processor._family = "qwen3.5"
    elif family in {"m2", "gemma", "gemma4", "gemma-4", "gemma-4-12b-it", "google/gemma-4-12b-it"}:
        opener = tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
        close = tokenizer.encode("<channel|>", add_special_tokens=False)
        processor = NativeThinkingBudget(prompt_length, budget, close, opener)
        processor._family = "gemma4"
    else:
        raise ReasoningBudgetError("model alias must identify Qwen/M1 or Gemma/M2")
    return processor
