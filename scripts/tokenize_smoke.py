#!/usr/bin/env python
"""Exact official-template token counts for fixed train/validation smoke inputs."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.data import outline_target
from chaptering.inference import build_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Tokenizer audit must be CPU-only")
    from transformers import AutoConfig, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "artifacts/runtime/model_inventory.json").read_text())
    config = json.loads((root / "configs/models.json").read_text())
    rows = []
    for split in ["train", "validation"]:
        rows.extend(json.loads(line) for line in (args.data / f"smoke_{split}.jsonl").read_text().splitlines())
    report = {"purpose": "fixed smoke inputs only, not full corpus census",
              "input_measure": "official rendered template, no duplicate special tokens",
              "target_measure": "final JSON tokens plus EOS; excludes unsupervised reasoning",
              "models": {}, "common_encoder_budget_candidate": 8192}
    for alias in ["E2", "E1", "M1", "M2"]:
        item = inventory["models"][alias]
        directory = Path(item["directory"]) if "directory" in item else root / "models" / alias
        revision = item["revision"]
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        model_config = AutoConfig.from_pretrained(str(directory), local_files_only=True)
        seq2seq = config["models"][alias]["architecture"] == "seq2seq"
        limit = model_config.encoder.max_position_embeddings if seq2seq else model_config.text_config.max_position_embeddings
        entries = []
        for row in rows:
            kwargs = {} if seq2seq else {"enable_thinking": True}
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(row)}],
                                                     tokenize=False, add_generation_prompt=True, **kwargs)
            input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            target = json.dumps(outline_target(row), ensure_ascii=False, separators=(",", ":"))
            target_tokens = len(tokenizer.encode(target, add_special_tokens=False)) + 1
            entries.append({"video_id": row["video_id"], "split": row["split"],
                            "sentence_count": len(row["sentences"]), "input_tokens": input_tokens,
                            "final_target_tokens": target_tokens,
                            "fits_8192_input": input_tokens <= 8192,
                            "fits_pinned_context_with_8192_generation": input_tokens <= limit if seq2seq else input_tokens + 8192 <= limit})
        report["models"][alias] = {"model_id": item["model_id"], "revision": revision,
                                   "directory": str(directory), "context_limit": limit,
                                   "enable_thinking": None if seq2seq else True,
                                   "examples": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"output": str(args.output), "models": {
        alias: {"max_smoke_input_tokens": max(row["input_tokens"] for row in item["examples"]),
                "over_8192_input": sum(not row["fits_8192_input"] for row in item["examples"])}
        for alias, item in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
