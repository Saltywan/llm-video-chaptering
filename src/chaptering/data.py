"""Strict, dependency-free YTSeg reference-transcript canonicalization.

Verified on untruncated public train rows 0--2 on 2026-11-16: a one in
``target_binary_ref`` marks the sentence *starting* an internal chapter.
The initial chapter starts at zero but its binary bit is zero. Each row is
checked independently against its marked transcript; mismatches are errors.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

SCHEMA_VERSION = 1
DATASET = "retkowski/ytseg"
CONFIG = "text"
LABEL_SEMANTICS = "internal_chapter_start_sentence_index"
_MARKER = re.compile(r"\[CSTART\](.*?)\[CEND\]", re.DOTALL)


class DataValidationError(ValueError):
    """A source row cannot be represented without changing its labels."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        # Codes intentionally avoid putting source transcript text in logs.
        raise DataValidationError(code)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _string_list(value: Any, code: str) -> list[str]:
    _require(isinstance(value, (list, tuple)) and bool(value), code)
    _require(all(isinstance(item, str) and bool(item.strip()) for item in value), code)
    return list(value)


def marker_chapter_starts(
    sentences: list[str], marked_text: str, titles: list[str]
) -> list[int]:
    """Recover starts from markers while checking all sentence/title alignment.

    Only whitespace is normalized for this comparison. Sentence content and
    the canonical record remain verbatim. No sentence splitting is performed.
    """
    _require(isinstance(marked_text, str), "marked_transcript_not_string")
    markers = list(_MARKER.finditer(marked_text))
    _require(len(markers) == len(titles), "marker_title_count_mismatch")
    _require(bool(markers), "missing_chapter_markers")
    _require(not marked_text[: markers[0].start()].strip(), "text_before_first_marker")
    normalized_sentences = [_normalized(sentence) for sentence in sentences]
    starts: list[int] = []
    cursor = 0
    for index, marker in enumerate(markers):
        _require(_normalized(marker.group(1)) == _normalized(titles[index]), "marker_title_mismatch")
        next_marker = markers[index + 1].start() if index + 1 < len(markers) else len(marked_text)
        body = _normalized(marked_text[marker.end() : next_marker])
        _require(bool(body), "empty_marked_chapter")
        starts.append(cursor)
        consumed = 0
        while body:
            _require(cursor < len(sentences), "marked_text_exceeds_sentences")
            sentence = normalized_sentences[cursor]
            _require(body.startswith(sentence), "marked_sentence_content_mismatch")
            remainder = body[len(sentence) :]
            _require(not remainder or remainder[0].isspace(), "marked_sentence_boundary_mismatch")
            body = remainder.lstrip()
            cursor += 1
            consumed += 1
        _require(consumed > 0, "empty_marked_chapter")
    _require(cursor == len(sentences), "marked_text_missing_sentences")
    return starts


def canonicalize_row(
    row: Mapping[str, Any], split: str, revision: str
) -> dict[str, Any]:
    """Validate and project one official text-config row into schema version 1."""
    _require(split in {"train", "validation", "test"}, "unknown_split")
    _require(isinstance(revision, str) and bool(revision), "missing_revision")
    video_id = row.get("video_id")
    _require(isinstance(video_id, str) and bool(video_id.strip()), "invalid_video_id")
    sentences = _string_list(row.get("text_ref"), "invalid_sentences")
    titles = _string_list(row.get("chapter_titles"), "invalid_chapter_titles")
    binary = row.get("target_binary_ref")
    _require(isinstance(binary, str) and binary.startswith("|="), "invalid_binary_prefix")
    bits = binary[2:]
    _require(len(bits) == len(sentences), "binary_sentence_length_mismatch")
    _require(set(bits) <= {"0", "1"}, "nonbinary_label")
    _require(bits[0] == "0", "initial_sentence_must_not_be_internal_boundary")
    starts = [0] + [index for index, bit in enumerate(bits) if bit == "1"]
    _require(len(starts) == len(titles), "binary_title_count_mismatch")
    marker_starts = marker_chapter_starts(sentences, row.get("target_text_ref"), titles)
    _require(starts == marker_starts, "binary_marker_start_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "split": split,
        "sentences": sentences,
        "reference": {"chapter_starts": starts, "chapter_titles": titles},
        "provenance": {
            "dataset": DATASET,
            "config": CONFIG,
            "revision": revision,
            "label_semantics": LABEL_SEMANTICS,
            "alignment_check": "all_marked_sentences_and_titles",
        },
    }


def inference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist the only two fields that may reach generation code."""
    video_id = record.get("video_id")
    _require(isinstance(video_id, str) and bool(video_id.strip()), "invalid_video_id")
    sentences = _string_list(record.get("sentences"), "invalid_sentences")
    return {"video_id": video_id, "sentences": sentences}


def indexed_transcript(record: Mapping[str, Any]) -> str:
    """Format sentence indices without consulting labels or metadata."""
    clean = inference_record(record)
    return "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(clean["sentences"]))


def outline_target(record: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the supervised outline; never call this for inference input."""
    reference = record["reference"]
    starts = reference["chapter_starts"]
    titles = reference["chapter_titles"]
    _require(len(starts) == len(titles), "canonical_start_title_count_mismatch")
    return {"chapters": [{"start": start, "title": title} for start, title in zip(starts, titles)]}
