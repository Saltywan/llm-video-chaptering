#!/usr/bin/env python
"""Prepare a pinned, CPU-only common-window pilot pack or full corpus census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaptering.windowing import (  # noqa: E402
    WINDOW_PROMPT_VERSION, WindowingError, build_windows, window_training_target,
)
from chaptering.evaluation import ValidationError  # noqa: E402

ALIASES = ("M1", "M2", "E1", "E2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_length_stratified(
    records: list[dict[str, Any]], count: int, seed: int = 393, bins: int = 4
) -> list[dict[str, Any]]:
    """Select from four sentence-length bins using only length and video ID.

    Bins are contiguous in (sentence_count, video_id) order and differ in size
    by at most one. Quotas are allocated round-robin, then IDs within each bin
    are ranked by seeded SHA256. No title, boundary, or chapter count is read.
    """
    if type(count) is not int or count < 0 or count > len(records):
        raise ValueError("selection count must be between zero and available records")
    if type(seed) is not int or type(bins) is not int or bins < 1:
        raise ValueError("seed must be integer and bins must be positive integer")
    ids = [row["video_id"] for row in records]
    if any(not isinstance(video_id, str) or not video_id.strip() for video_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("selection requires unique nonempty video IDs")
    if any(not isinstance(row["sentences"], list) or not row["sentences"] for row in records):
        raise ValueError("selection requires nonempty sentence lists")
    if not count:
        return []
    ordered = sorted(records, key=lambda row: (len(row["sentences"]), row["video_id"]))
    groups = [ordered[index * len(ordered) // bins : (index + 1) * len(ordered) // bins]
              for index in range(bins)]
    ranked = [sorted(group, key=lambda row: (
        hashlib.sha256(f"{seed}\0{row['video_id']}".encode("utf-8")).hexdigest(), row["video_id"]))
        for group in groups]
    quotas = [0] * bins
    remaining = count
    while remaining:
        for index, group in enumerate(ranked):
            if remaining and quotas[index] < len(group):
                quotas[index] += 1
                remaining -= 1
    return [row for group, quota in zip(ranked, quotas) for row in group[:quota]]


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "quantiles": {}}
    values = sorted(values)
    return {"count": len(values), "total": sum(values), "mean": sum(values) / len(values),
            "quantiles": {str(q): values[int((len(values) - 1) * q + .5)]
                          for q in [0, .1, .5, .9, .95, .99, 1]}}


def read_canonical(path: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                raise ValueError(f"invalid canonical JSON in {split} line {line_index}") from None
            if not isinstance(row, dict) or row.get("schema_version") != 1 or row.get("split") != split:
                raise ValueError(f"invalid canonical schema/split in {split} line {line_index}")
            rows.append(row)
    return rows


def load_tokenizers(root: Path, input_tokens: int):
    """Only local tokenizer files are opened; model weights are never loaded."""
    from transformers import AutoTokenizer

    config_path = root / "configs/models.json"
    inventory_path = root / "artifacts/runtime/model_inventory.json"
    config = json.loads(config_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    tokenizers, counters, metadata = {}, {}, {}
    tokenizer_assets = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "chat_template.jinja", "added_tokens.json", "tokenizer.model", "spiece.model",
                        "sentencepiece.model", "config.json", "PINNED_REVISION")
    for alias in ALIASES:
        item, planned = inventory["models"][alias], config["models"][alias]
        revision = item["revision"]
        if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or revision != planned["revision"] or item["model_id"] != planned["model_id"]):
            raise ValueError(f"inventory/config model pin mismatch for {alias}")
        causal = planned["architecture"] == "causal"
        if planned["architecture"] not in {"causal", "seq2seq"}:
            raise ValueError(f"unknown architecture for {alias}")
        if causal and planned["enable_thinking"] is not True:
            raise ValueError(f"native thinking must remain enabled for {alias}")
        directory = Path(item["directory"]) if alias in {"M1", "M2"} else root / "models" / alias
        marker = directory / "PINNED_REVISION"
        if marker.exists() and marker.read_text().strip() != revision:
            raise ValueError(f"local model revision marker mismatch for {alias}")
        if not causal and not marker.is_file():
            raise ValueError(f"missing local pinned revision marker for {alias}")
        context_limit = item["text_max_position_embeddings"] if causal else item["encoder_max_position_embeddings"]
        if type(context_limit) is not int or input_tokens > context_limit:
            raise ValueError(f"input budget exceeds pinned context support for {alias}")
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        tokenizers[alias] = tokenizer

        def counter(prompt, tokenizer=tokenizer, causal=causal):
            kwargs = {"enable_thinking": True} if causal else {}
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=False, add_generation_prompt=True, **kwargs)
            if not isinstance(rendered, str):
                raise ValueError("official template did not produce text")
            return len(tokenizer.encode(rendered, add_special_tokens=False))

        counters[alias] = counter
        metadata[alias] = {"model_id": item["model_id"], "revision": revision,
                           "architecture": planned["architecture"], "directory": str(directory),
                           "context_limit": context_limit, "enable_thinking": True if causal else None,
                           "tokenizer_class": type(tokenizer).__name__,
                           "tokenizer_and_config_sha256": {name: sha256(directory / name)
                               for name in tokenizer_assets if (directory / name).is_file()},
                           "pin_source": "matching inventory/config plus local marker where available"}
    return tokenizers, counters, metadata, {"models.json": sha256(config_path),
                                          "model_inventory.json": sha256(inventory_path)}


def new_statistics() -> dict[str, Any]:
    return {"attempted_videos": 0, "successful_videos": 0, "failed_videos": 0, "windows": 0,
            "owned_sentences": 0, "context_shrunk_windows": 0, "right_context_omitted_windows": 0,
            "title_supervision": {}, "input_tokens": {alias: [] for alias in ALIASES},
            "target_json_tokens_no_template": {alias: [] for alias in ALIASES}}


def add_statistics(stats, windows, targets, target_counts):
    stats["successful_videos"] += 1
    stats["windows"] += len(windows)
    stats["owned_sentences"] += sum(window["B"] - window["A"] for window in windows)
    stats["context_shrunk_windows"] += sum(window["context_shrunk"] for window in windows)
    stats["right_context_omitted_windows"] += sum(window["right_context_omitted_as_infeasible"] for window in windows)
    for window, target, counts in zip(windows, targets, target_counts):
        for alias in ALIASES:
            stats["input_tokens"][alias].append(window["input_token_counts"][alias])
            stats["target_json_tokens_no_template"][alias].append(counts[alias])
        for key, count in target["title_supervision_stats"].items():
            stats["title_supervision"][key] = stats["title_supervision"].get(key, 0) + int(count)


def finish_statistics(stats):
    result = dict(stats)
    for key in ["input_tokens", "target_json_tokens_no_template"]:
        result[key] = {alias: distribution(values) for alias, values in stats[key].items()}
    supervised = stats["title_supervision"].get("supervised_titles", 0)
    masked = stats["title_supervision"].get("masked_titles", 0)
    result["title_supervision_fraction"] = supervised / (supervised + masked) if supervised + masked else None
    return result


def prepare(args, *, root: Path = ROOT) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("prepare_windows requires an explicitly empty CUDA_VISIBLE_DEVICES mask")
    if args.output.exists():
        raise ValueError("output must be a new directory; refusing to overwrite")
    for name in ["input_tokens", "core_sentences", "context_sentences"]:
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive integer")
    for name in ["train_videos", "validation_videos"]:
        if type(getattr(args, name)) is not int or getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative integer")
    if not args.census_all and (args.train_videos == 0 or args.validation_videos == 0):
        raise ValueError("zero selected videos is allowed only with --census-all")
    resolution_path = args.data / "resolution.json"
    resolution = json.loads(resolution_path.read_text())
    if resolution.get("resolved_integrity_gate_passed") is not True:
        raise ValueError("resolved input integrity gate must pass before window preparation")
    source_hashes = {f"{split}.jsonl": sha256(args.data / f"{split}.jsonl")
                     for split in ["train", "validation"]}
    if any(resolution.get("output_files_sha256", {}).get(name) != digest for name, digest in source_hashes.items()):
        raise ValueError("canonical source files do not match resolved audit hashes")
    records = {split: read_canonical(args.data / f"{split}.jsonl", split) for split in ["train", "validation"]}
    selected = {"train": select_length_stratified(records["train"], args.train_videos, args.seed),
                "validation": select_length_stratified(records["validation"], args.validation_videos, args.seed)}
    all_ids = [{row["video_id"] for row in records[split]} for split in ["train", "validation"]]
    if all_ids[0] & all_ids[1]:
        raise ValueError("resolved input unexpectedly has train/validation overlap")
    tokenizers, counters, model_metadata, config_hashes = load_tokenizers(root, args.input_tokens)
    args.output.mkdir(parents=True, exist_ok=False)
    statistics, failures = {}, []
    for split in ["train", "validation"]:
        selected_ids = {row["video_id"] for row in selected[split]}
        processing = selected[split] + ([row for row in records[split] if row["video_id"] not in selected_ids]
                                      if args.census_all else [])
        with (args.output / f"{split}_records.jsonl").open("x", encoding="utf-8") as handle:
            for row in selected[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        scope_stats, selected_stats = new_statistics(), new_statistics()
        with (args.output / f"{split}_windows.jsonl").open("x", encoding="utf-8") as handle:
            for index, row in enumerate(processing):
                chosen = row["video_id"] in selected_ids
                scope_stats["attempted_videos"] += 1
                if chosen:
                    selected_stats["attempted_videos"] += 1
                try:
                    windows = build_windows(row, counters, args.input_tokens,
                                            max_core_sentences=args.core_sentences,
                                            left_context=args.context_sentences,
                                            right_context=args.context_sentences,
                                            core_search=getattr(args, "core_search", "exhaustive"))
                    targets = [window_training_target(row, window) for window in windows]
                    target_counts = [{alias: len(tokenizers[alias].encode(target["target_json"], add_special_tokens=False))
                                      for alias in ALIASES} for target in targets]
                except Exception as exc:
                    # Never silently omit an impossible video or print its transcript.
                    message = str(exc) if isinstance(exc, (WindowingError, ValidationError)) else "unexpected preparation failure"
                    failures.append({"video_id": row["video_id"], "split": split, "selected": chosen,
                                     "error_type": type(exc).__name__, "error": message})
                    scope_stats["failed_videos"] += 1
                    if chosen:
                        selected_stats["failed_videos"] += 1
                    continue
                add_statistics(scope_stats, windows, targets, target_counts)
                if chosen:
                    add_statistics(selected_stats, windows, targets, target_counts)
                    for window, target in zip(windows, targets):
                        packet = {"window": window}
                        if split == "train":
                            packet["target"] = target
                        handle.write(json.dumps(packet, ensure_ascii=False) + "\n")
                if (index + 1) % 100 == 0:
                    print(json.dumps({"event": "window_preparation_progress", "split": split,
                                      "processed_videos": index + 1, "failed_videos": scope_stats["failed_videos"]}), flush=True)
        statistics[split] = {"processed_scope": "entire_split" if args.census_all else "selected_videos_only",
                             "processed": finish_statistics(scope_stats), "selected": finish_statistics(selected_stats)}
    with (args.output / "failures.jsonl").open("x", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    source_unchanged = all(sha256(args.data / name) == digest for name, digest in source_hashes.items())
    manifest = {
        "schema_version": 1, "purpose": "common-window corpus census" if args.census_all else "bounded common-window pilot pack",
        "prompt_version": WINDOW_PROMPT_VERSION, "data_directory": str(args.data.resolve()),
        "output_directory": str(args.output.resolve()), "dataset_revision": resolution.get("revision"),
        "source_files_sha256": {**source_hashes, "resolution.json": sha256(resolution_path)},
        "source_files_unchanged": source_unchanged, "models": model_metadata,
        "config_files_sha256": config_hashes,
        "code_files_sha256": {str(path.relative_to(root)): sha256(path) for path in
            [Path(__file__).resolve(), root / "src/chaptering/windowing.py", root / "src/chaptering/data.py",
             root / "src/chaptering/evaluation.py"]},
        "settings": {name: getattr(args, name) for name in ["train_videos", "validation_videos", "input_tokens",
                                                               "core_sentences", "context_sentences", "seed", "census_all"]}
                    | {"core_search": getattr(args, "core_search", "exhaustive")},
        "selection_policy": "four contiguous bins sorted(sentence_count,video_id); round-robin quotas; seeded SHA256 video-ID rank within each bin; no gold used",
        "selected_video_ids": {split: [row["video_id"] for row in selected[split]] for split in selected},
        "input_token_measure": "official user prompt template, add_generation_prompt=True; native thinking enabled M1/M2; encode(add_special_tokens=False)",
        "target_token_measure": "raw final JSON only, without template or assistant terminator; not a complete training sequence length",
        "statistics": statistics, "failure_count": len(failures),
        "preparation_gate_passed": not failures and source_unchanged,
        "full_corpus_census_passed": bool(args.census_all and not failures and source_unchanged),
        "test_data_read": False,
        "files_sha256": {path.name: sha256(path) for path in sorted(args.output.glob("*.jsonl"))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-videos", type=int, default=32)
    parser.add_argument("--validation-videos", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--core-sentences", type=int, default=128)
    parser.add_argument("--context-sentences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=393)
    parser.add_argument("--census-all", action="store_true")
    parser.add_argument("--core-search", choices=["exhaustive", "bisect"], default="exhaustive")
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps({"output": manifest["output_directory"], "preparation_gate_passed": manifest["preparation_gate_passed"],
                      "full_corpus_census_passed": manifest["full_corpus_census_passed"], "failure_count": manifest["failure_count"],
                      "selected_videos": {split: len(ids) for split, ids in manifest["selected_video_ids"].items()},
                      "windows": {split: stats["selected"]["windows"] for split, stats in manifest["statistics"].items()}}, indent=2))
    return 0 if manifest["preparation_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
