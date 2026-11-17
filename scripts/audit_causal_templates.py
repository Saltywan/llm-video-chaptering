#!/usr/bin/env python
"""CPU gate for actual pinned loss masks and reasoning/final channel parsing."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.supervision import prepare_causal_example
from chaptering.evaluation import parse_outline

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Template audit must be CPU-only")
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
inventory = json.loads((root / "artifacts/runtime/model_inventory.json").read_text())
report = {"purpose": "actual pinned native template/mask and final-parser gate", "models": {}}
for alias in ["M1", "M2"]:
    item = inventory["models"][alias]
    tokenizer = AutoTokenizer.from_pretrained(item["directory"], local_files_only=True)
    cases = []
    prompt = "Make topical transcript chapters. [0] We bake bread. [1] We clean a bicycle."
    for title in ["Bread", "Bread 日本語 ภาษาไทย"]:
        target = json.dumps({"chapters": [{"start": 0, "title": title}]}, ensure_ascii=False, separators=(",", ":"))
        prepared = prepare_causal_example(tokenizer, prompt, target, alias)
        if not prepared["audit"].get("supervision_gate_passed"):
            raise RuntimeError("Supervision helper did not pass its gate")
        labeled = [token for token in prepared["labels"] if token != -100]
        decoded = tokenizer.decode(labeled, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != target + prepared["audit"]["termination_token"]:
            raise RuntimeError("Actual decoded labeled tokens differ from final JSON + terminator")
        cases.append({"target": target, "decoded_supervised_tokens": decoded, "audit": prepared["audit"]})
    parser_cases = []
    if alias == "M2":
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        target = '{"chapters":[{"start":0,"title":"Bread"}]}'
        fixtures = [
            ("unclosed_thinking_contains_json", "<|channel>thought\n" + target, False),
            ("closed_thinking_without_final", "<|channel>thought\n" + target + "<channel|><turn|>", False),
            ("thinking_then_final", "<|channel>thought\nA short synthetic thought.<channel|>" + target + "<turn|>", True),
            ("native_final_only", target + "<turn|>", True),
        ]
        for name, raw, expected in fixtures:
            accepted = False
            try:
                parsed = tokenizer.parse_response(raw, prefix=prefix)
                if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
                    parse_outline(parsed["content"].strip(), 2)
                    accepted = True
            except (ValueError, TypeError, KeyError):
                pass
            if accepted != expected:
                raise RuntimeError("Pinned Gemma final parser failed fixture: " + name)
            parser_cases.append({"name": name, "accepted_final_outline": accepted})
    report["models"][alias] = {"model_id": item["model_id"], "revision": item["revision"],
                                "supervision_gate_passed": True, "mask_cases": cases,
                                "parser_gate_passed": True, "parser_cases": parser_cases}
path = root / "artifacts/runtime/causal_supervision.json"
with path.open("x") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print(json.dumps({"output": str(path), "models": {alias: {
    "supervision_gate_passed": item["supervision_gate_passed"],
    "parser_gate_passed": item["parser_gate_passed"]} for alias, item in report["models"].items()}}, indent=2))
