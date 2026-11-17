#!/usr/bin/env python
"""Write configs/frozen_test.json: the eight test configurations, frozen before any test data is read.

Records, per backbone: pinned model revision, adapter path and SHA-256, training report settings,
window pack and prompt version, the exact dev inference run_config used as the decoding template,
driver script hashes and format policy, plus the dev selection evidence. --verify re-checks every
recorded hash (used by the test queue before each run).
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S5 = "artifacts/stage5-v1"
BACKBONES = {
    "M1": {"adapter_run": f"{S5}/M1-direct-safe", "dev_run": f"{S5}/M1-dev/FT", "dev_zs_run": f"{S5}/M1-dev/ZS",
           "dev_paired": f"{S5}/M1-dev-paired.json", "driver": "scripts/infer_vllm_pair.py",
           "driver_args": ["--max-num-seqs", "100", "--batch-size", "100", "--max-new-tokens", "32768"]},
    "M2": {"adapter_run": f"{S5}/M2-direct-safe", "dev_run": f"{S5}/M2-dev/FT", "dev_zs_run": f"{S5}/M2-dev/ZS",
           "dev_paired": f"{S5}/M2-dev-paired.json", "driver": "scripts/infer_vllm_pair.py",
           "driver_args": ["--max-num-seqs", "100", "--batch-size", "100", "--max-new-tokens", "32768"]},
    "E2": {"adapter_run": f"{S5}/E2-native-safe", "dev_run": f"{S5}/E2-dev-FT", "dev_zs_run": f"{S5}/E2-dev-ZS",
           "dev_paired": f"{S5}/E2-dev-paired.json", "driver": "scripts/infer_seq2seq_batched.py",
           "driver_args": ["--square-budget", str(16 * 8192 * 8192)]},
    "E1": {"adapter_run": f"{S5}/E1-native-safe", "dev_run": f"{S5}/E1-dev-FT", "dev_zs_run": f"{S5}/E1-dev-ZS",
           "dev_paired": f"{S5}/E1-dev-paired.json", "driver": "scripts/infer_seq2seq_batched.py",
           "driver_args": ["--square-budget", str(8 * 8192 * 8192)]},
}
SHARED_CODE = ["src/chaptering/windowing.py", "src/chaptering/modeling.py", "src/chaptering/batching.py",
               "src/chaptering/inference.py", "src/chaptering/evaluation.py", "src/chaptering/data.py",
               "scripts/evaluate_predictions.py", "scripts/rescore_envelope.py", "src/chaptering/envelope.py",
               "scripts/prepare_test.py", "configs/models.json"]


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def maybe_json(path):
    full = ROOT / path
    return json.loads(full.read_text()) if full.is_file() else None


def build():
    pack = "artifacts/windows/common-v2"
    manifest = json.loads((ROOT / pack / "manifest.json").read_text())
    config = json.loads((ROOT / "configs/models.json").read_text())
    frozen = {"status": "frozen", "purpose": "final test configurations; no test data read before this file",
              "primary_metric": "exact boundary macro F1 over all test videos, failed video = 0",
              "format_policy": {"primary": "strict parser (no repair)",
                                "secondary": "envelope-v1 lenient reading, reported separately"},
              "window_pack_template": {"path": pack, "manifest_sha256": digest(f"{pack}/manifest.json"),
                                       "settings": manifest["settings"], "prompt_version": manifest["prompt_version"]},
              "seed": 393, "shared_code_sha256": {path: digest(path) for path in SHARED_CODE},
              "training_note": ("LoRA adapters are the stage-5 one-pass runs on 1,000 training videos "
                                "(1,306 windows); full-data training did not fit the project deadline"),
              "configurations": {}}
    for alias, spec in BACKBONES.items():
        adapter = Path(spec["adapter_run"]) / "adapter"
        report = maybe_json(f"{spec['adapter_run']}/report.json") or {}
        paired = maybe_json(spec["dev_paired"])
        evidence = None
        if paired:
            boot = paired["comparison"]["paired_exact_f1_bootstrap"]
            evidence = {"dev_videos": 200, "ft_exact_f1": paired["canonical"]["aggregate"]["macro_exact_f1"],
                        "zs_exact_f1": paired["comparison"]["right_arm"]["canonical"]["aggregate"]["macro_exact_f1"],
                        "ft_minus_zs": boot["mean_difference"], "ci95": boot.get("confidence_interval"),
                        "file": spec["dev_paired"], "file_sha256": digest(spec["dev_paired"])}
        common = {"model_id": config["models"][alias]["model_id"], "model_revision": config["models"][alias]["revision"],
                  "driver": spec["driver"], "driver_sha256": digest(spec["driver"]), "driver_args": spec["driver_args"]}
        frozen["configurations"][f"{alias}-ZS"] = common | {
            "arm": "ZS", "dev_template_run_config": f"{spec['dev_zs_run']}/run_config.json"
            if (ROOT / spec["dev_zs_run"] / "run_config.json").is_file() else None}
        frozen["configurations"][f"{alias}-FT"] = common | {
            "arm": "FT", "adapter": str(adapter),
            "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": digest(adapter / "adapter_config.json"),
            "training": {key: report.get(key) for key in [
                "status", "rank", "lora_alpha", "lora_dropout", "learning_rate", "lr_schedule", "warmup_steps",
                "grad_accum", "target_modules", "final_context", "planned_training_videos", "planned_steps",
                "windows_sha256", "training_seconds", "seed"]},
            "dev_selection_evidence": evidence,
            "selection": "only adapted candidate for this backbone; kept regardless of dev sign (both ZS and LoRA are reported for every backbone)"}
    return frozen


def verify(frozen):
    bad = [path for path, value in frozen["shared_code_sha256"].items() if digest(path) != value]
    for name, item in frozen["configurations"].items():
        if digest(item["driver"]) != item["driver_sha256"]:
            bad.append(item["driver"])
        if item["arm"] == "FT" and digest(Path(item["adapter"]) / "adapter_model.safetensors") != item["adapter_sha256"]:
            bad.append(item["adapter"])
    if bad:
        raise SystemExit("Frozen inputs changed: " + ", ".join(sorted(set(bad))))
    print("frozen configuration verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/frozen_test.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.output.read_text()))
        return
    if args.output.exists():
        raise SystemExit("Freeze already exists; never overwrite it")
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
