#!/usr/bin/env python
"""Audit actual four-model partial-title masks without loading model weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.modeling import model_entry
from chaptering.windowing import build_window_prompt
from chaptering.training import prepare_seq2seq_example
from chaptering.supervision import GEMMA_CLOSED_THOUGHT, prepare_causal_example

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--windows", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--aliases", nargs="+", choices=["E2", "E1", "M1", "M2"], default=["E2", "E1", "M1", "M2"])
parser.add_argument("--gemma-final-context", choices=["direct_v1", "closed_thought_v2"], default="direct_v1")
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Supervision audit is CPU-only")
from transformers import AutoTokenizer
root = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in args.windows.read_text().splitlines()]
report = {"passed": True, "window_count": len(rows), "models": {},
          "windows_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
          "gemma_final_context": args.gemma_final_context}
for alias in args.aliases:
    entry = model_entry(root, alias)
    tokenizer = AutoTokenizer.from_pretrained(entry["directory"], local_files_only=True)
    info = {"revision": entry["revision"], "windows": 0, "masked_title_windows": 0,
            "masked_title_tokens": 0, "max_full_training_tokens": 0, "max_target_tokens": 0}
    if alias == "M2" and args.gemma_final_context == "closed_thought_v2":
        info["native_phase_cases"] = []
        for title in ["Topic", "Topic 日本語 ภาษาไทย"]:
            target_json = json.dumps({"chapters": [{"start": 0, "title": title}]}, ensure_ascii=False, separators=(",", ":"))
            prepared = prepare_causal_example(tokenizer, "[0] A sentence.", target_json, alias,
                                              gemma_final_context=args.gemma_final_context)
            audit = prepared["audit"]
            scored = [token for token in prepared["labels"] if token != -100]
            if tokenizer.decode(scored, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target_json + "<turn|>":
                raise RuntimeError("Native final-phase labels differ from JSON plus turn end")
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": "[0] A sentence."}],
                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
            parsed = tokenizer.parse_response(GEMMA_CLOSED_THOUGHT + target_json + "<turn|>", prefix=prefix)
            if parsed.get("content", "").strip() != target_json:
                raise RuntimeError("Pinned parser did not accept the empty closed native thought context")
            if any(label != -100 for label in prepared["labels"][:audit["json_token_start"]]):
                raise RuntimeError("Native context received prediction labels")
            info["native_phase_cases"].append({"title": title, "parser_passed": True, "audit": audit})
    for item in rows:
        window, target = item["window"], item["target"]
        kwargs = {"window": window, "masked_target_spans": target["masked_target_spans"]}
        prompt = build_window_prompt(window)
        if alias.startswith("E"):
            prepared = prepare_seq2seq_example(tokenizer, prompt, target["target_json"], **kwargs)
            if prepared["labels"][-1] != tokenizer.eos_token_id:
                raise RuntimeError("Seq2seq EOS not supervised")
            info["max_target_tokens"] = max(info["max_target_tokens"], len(prepared["labels"]))
        else:
            if alias == "M2":
                kwargs["gemma_final_context"] = args.gemma_final_context
            prepared = prepare_causal_example(tokenizer, prompt, target["target_json"], alias, **kwargs)
            end = prepared["audit"]["termination_token_position"]
            if prepared["labels"][end] != prepared["audit"]["termination_token_id"]:
                raise RuntimeError("Causal native terminator not supervised")
            if any(label != -100 for label in prepared["labels"][:prepared["audit"]["json_token_start"]]):
                raise RuntimeError("Causal context received prediction labels")
        audit = prepared["audit"]
        if not audit["supervision_gate_passed"] or any(count < 1 for count in audit["masked_span_token_coverage"]):
            raise RuntimeError("Actual partial-title token-mask gate failed")
        info["windows"] += 1
        info["masked_title_windows"] += bool(target["masked_target_spans"])
        info["masked_title_tokens"] += audit["masked_partial_title_tokens"]
        info["max_full_training_tokens"] = max(info["max_full_training_tokens"], len(prepared["input_ids"]))
    report["models"][alias] = info
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
