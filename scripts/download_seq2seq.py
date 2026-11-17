#!/usr/bin/env python
"""CPU-only download of the two pinned, selected T5Gemma checkpoints."""

import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run downloads with CUDA_VISIBLE_DEVICES empty")

root = Path(__file__).resolve().parents[1]
inventory = json.loads((root / "artifacts/runtime/model_inventory.json").read_text())
for alias in ["E2", "E1"]:
    entry = inventory["models"][alias]
    if not entry.get("config_access"):
        raise RuntimeError(f"Access not established for {alias}")
    directory = root / "models" / alias
    print(json.dumps({"event": "download_start", "alias": alias, "revision": entry["revision"]}), flush=True)
    snapshot_download(entry["model_id"], revision=entry["revision"], local_dir=directory,
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer.model", "*.txt"])
    (directory / "PINNED_REVISION").write_text(entry["revision"] + "\n")
    print(json.dumps({"event": "download_complete", "alias": alias, "directory": str(directory)}), flush=True)
