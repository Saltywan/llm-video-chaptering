"""Declared lenient reading ``envelope-v1`` of a final answer.

It removes only wrapping that carries no outline content: one Markdown code fence
around the whole answer, and trailing periods/whitespace after the last closing
brace. The inner text must then pass the unchanged strict parser. Nothing is
repaired: duplicate or extra keys, bad indices, prose and truncations stay failures.
Strict parsing remains the primary result; this is reported beside it.
"""

from __future__ import annotations

import re

ENVELOPE_VERSION = "envelope-v1"
_FENCE = re.compile(r"\A```[A-Za-z]*[ \t]*\r?\n(.*?)\r?\n?```\Z", re.DOTALL)


def strip_envelope_v1(text: str) -> tuple[str, list[str]]:
    """Return ``(inner_text, removed)`` where ``removed`` names what was stripped."""
    if not isinstance(text, str):
        raise TypeError("final answer must be a string")
    removed = []
    inner = text.strip()
    match = _FENCE.match(inner)
    if match and "```" not in match.group(1):
        inner = match.group(1).strip()
        removed.append("markdown_fence")
    trimmed = inner.rstrip(". \t\r\n")
    if trimmed != inner and trimmed.endswith("}"):
        inner = trimmed
        removed.append("trailing_period")
    return inner, removed
