#!/usr/bin/env python
"""CPU integration checks on the fixed, cleaned train/validation manifests."""

import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.data import inference_record, outline_target
from chaptering.inference import build_prompt
from chaptering.evaluation import evaluate_records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Contract integration check must be CPU-only")
rows = []
for split in ["train", "validation"]:
    rows.extend(json.loads(line) for line in (args.data / f"smoke_{split}.jsonl").read_text().splitlines())
predictions = []
for row in rows:
    clean = inference_record(row)
    if set(clean) != {"video_id", "sentences"} or build_prompt(clean) != build_prompt(row):
        raise RuntimeError("Reference-bearing record changed inference input")
    predictions.append({"schema_version": 1, "video_id": row["video_id"], "run_id": "gold-contract-check",
                        "status": "ok", "chapters": outline_target(row)["chapters"], "error": None, "usage": {}})
result = evaluate_records(rows, predictions)
aggregate = result["aggregate"]
if (aggregate["macro_exact_f1"] != 1.0 or aggregate["macro_pk"] != 0.0
        or aggregate["macro_window_diff"] != 0.0 or aggregate["failure_rate"] != 0.0
        or aggregate["title_span_alignment"]["reference_coverage_all_requests"] != 1.0):
    raise RuntimeError("Reference self-scoring integration failed")
result["purpose"] = "gold self-check only; not model performance"
result["inference_projection_gate_passed"] = True
result["packages"] = {}
for name in ["torch", "transformers", "datasets", "huggingface-hub", "peft", "accelerate", "trl"]:
    result["packages"][name] = importlib.metadata.version(name)
root = Path(__file__).resolve().parents[1]
result["source_sha256_at_check"] = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in ["src", "scripts", "configs"]
    for path in sorted((root / folder).rglob("*"))
    if path.is_file() and path.suffix in {".py", ".json"} and not path.name.startswith("._")}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps({"requested_videos": aggregate["requested_videos"],
                  "macro_exact_f1": aggregate["macro_exact_f1"],
                  "macro_pk": aggregate["macro_pk"], "macro_window_diff": aggregate["macro_window_diff"],
                  "inference_projection_gate_passed": True}))
