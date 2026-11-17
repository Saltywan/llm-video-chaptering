#!/usr/bin/env python
"""Versioned structural parser correction; preserve the original pilot report."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.inference import strip_qwen_terminal_tokens
from chaptering.evaluation import parse_outline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sentence-count", type=int, required=True)
args = parser.parse_args()
source = args.source.read_bytes()
original = json.loads(source)
if original["alias"] != "M1":
    raise ValueError("This correction is limited to native Qwen framing")
report = copy.deepcopy(original)
for phase in ["unadapted_generation", "adapted_generation"]:
    value = report[phase]
    value["original_status"] = value["status"]
    if value["status"] == "truncated" or "final_text" not in value:
        continue
    value["original_framed_final_text"] = value["final_text"]
    value["final_text"] = strip_qwen_terminal_tokens(value["final_text"])
    try:
        value["chapters"] = parse_outline(value["final_text"], args.sentence_count)
        value["status"] = "ok"
        value.pop("error_type", None)
    except ValueError as exc:
        value["status"] = "invalid"
        value["error_type"] = type(exc).__name__
report["native_parser_version"] = "native-final-v2"
report["parser_correction"] = {
    "source_report": str(args.source), "source_sha256": hashlib.sha256(source).hexdigest(),
    "policy": "Remove only trailing native im_end/endoftext tokens from every saved final response, then apply unchanged strict JSON validation; no regeneration or semantic repair.",
    "all_saved_full_generation_phases_reparsed": True}
report["final_output_gate_passed"] = all(report[phase]["status"] == "ok"
                                         for phase in ["unadapted_generation", "adapted_generation"])
with args.output.open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({"output": str(args.output), "training_status": report["status"],
                  "final_output_gate_passed": report["final_output_gate_passed"]}))
