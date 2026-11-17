"""Shared helpers for the experiment notebooks.

The notebooks run from ``notebooks/``. Light steps (loading data, windows, predictions,
recomputing metrics) run anywhere on CPU. Heavy steps (LoRA training, GPU inference) are
shown as the exact commands that produced the saved artifacts and only execute when
``RUN_GPU = True`` on a machine with the models and an authorized GPU.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path


def _find_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "chaptering").is_dir():
            return candidate
    raise RuntimeError("run the notebooks from inside the project folder")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "src"))
# Full run outputs live in artifacts/ (not in git). A repo clone holds the small subset in results/ with the same
# layout: summaries, predictions, metrics and training logs. Cells that need transcripts or windows need artifacts/.
A = ROOT / "artifacts" if (ROOT / "artifacts").is_dir() else ROOT / "results"
S5, TEST, RFT1, RFT2 = A / "stage5-v1", A / "heldout-test-v1", A / "rft-v1", A / "rft-v2"
DEV_PACK, TEST_PACK = A / "windows/dev-v2", A / "windows/test-v1"
DEV_RECORDS, TEST_RECORDS = DEV_PACK / "validation_records.jsonl", TEST_PACK / "test_records.jsonl"
DEV_WINDOWS, TEST_WINDOWS = DEV_PACK / "validation_windows.jsonl", TEST_PACK / "test_windows.jsonl"
TRAIN_WINDOWS = DEV_PACK / "train_windows.jsonl"
SEMANTIC = ["--semantic-model", "FacebookAI/roberta-large",
            "--semantic-revision", "722cf37b1afa9454edce342e7895e588b6ff1d59",
            "--semantic-layers", "17", "--semantic-device", "cpu"]


def _arm_dirs(alias, dev_zs, dev_ft, test_zs, test_ft):
    return {("dev", "ZS"): dev_zs, ("dev", "FT"): dev_ft, ("test", "ZS"): test_zs, ("test", "FT"): test_ft}


BACKBONES = {
    "E1": {"name": "T5Gemma 9B-2B (UL2, instruction-tuned)", "model_id": "google/t5gemma-9b-2b-ul2-it",
           "revision": "99b9b9da34b8e66e2290c5c6d17306231c5b2566", "family": "encoder-decoder",
           "engine": "seq2seq", "train_dir": S5 / "E1-native-safe", "square_budget": 8 * 8192 * 8192,
           "runs": _arm_dirs("E1", S5 / "E1-dev-ZS", S5 / "E1-dev-FT", TEST / "E1-ZS-gpu2", TEST / "E1-FT")},
    "E2": {"name": "T5Gemma 2B-2B (UL2, instruction-tuned)", "model_id": "google/t5gemma-2b-2b-ul2-it",
           "revision": "6a196a58ce2786cd139a1217a85d1f7b9984e330", "family": "encoder-decoder",
           "engine": "seq2seq", "train_dir": S5 / "E2-native-safe", "square_budget": 16 * 8192 * 8192,
           "runs": _arm_dirs("E2", S5 / "E2-dev-ZS", S5 / "E2-dev-FT", TEST / "E2-ZS", TEST / "E2-FT")},
    "M1": {"name": "Qwen3.5-9B (native thinking)", "model_id": "Qwen/Qwen3.5-9B",
           "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "family": "decoder-only (hybrid attention)",
           "engine": "vllm", "train_dir": S5 / "M1-direct-safe",
           "runs": _arm_dirs("M1", S5 / "M1-dev/ZS", S5 / "M1-dev/FT", TEST / "M1/ZS", TEST / "M1/FT")},
    "M2": {"name": "Gemma 4 12B (native thinking)", "model_id": "google/gemma-4-12B-it",
           "revision": "43c9d014b231474fa7212dbb48a762566c2b255d", "family": "decoder-only",
           "engine": "vllm", "train_dir": S5 / "M2-direct-safe",
           "runs": _arm_dirs("M2", S5 / "M2-dev/ZS", S5 / "M2-dev/FT", TEST / "M2/ZS", TEST / "M2/FT")},
}


def rel(path) -> str:
    return str(Path(path).relative_to(ROOT))


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path, limit=None):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def run(cmd, enabled, env=None):
    """Print a pipeline command; execute it from the project root only when enabled."""
    shown = " ".join(str(part) if str(part).startswith("$") else shlex.quote(str(part)) for part in cmd)
    note = f"   # {env}" if env else ""
    print(("$ " if enabled else "[not run here; RUN_GPU=False]  $ ") + shown + note)
    if enabled:
        subprocess.run(shown, shell=True, cwd=ROOT, check=True)


def train_command(alias):
    spec = BACKBONES[alias]
    return ["python", "scripts/train_lora.py", "--alias", alias, "--windows", rel(TRAIN_WINDOWS),
            "--max-steps", "0", "--target-modules", "q_proj,k_proj,v_proj,o_proj", "--learning-rate", "2e-4",
            "--lr-schedule", "cosine", "--checkpoint-every", "100", "--grad-accum", "8", "--warmup-steps", "10",
            "--output", rel(spec["train_dir"])]


def infer_command(alias, split, arm):
    """The exact inference call behind each saved run (see that run's run_config.json)."""
    spec = BACKBONES[alias]
    windows = DEV_WINDOWS if split == "dev" else TEST_WINDOWS
    out = spec["runs"][(split, arm)]
    adapter = ["--adapter", rel(spec["train_dir"] / "adapter")] if arm == "FT" else []
    if spec["engine"] == "seq2seq":
        return ["python", "scripts/infer_seq2seq_batched.py", "--alias", alias, *adapter, "--windows", rel(windows),
                "--output", rel(out), "--square-budget", str(spec["square_budget"])]
    # vLLM runs zero-shot and LoRA arms in one engine; output/<ARM>/ holds each arm.
    return ["$VLLM_PYTHON", "scripts/infer_vllm_pair.py", "--alias", alias, "--arms", "ZS", "FT",
            "--adapter", rel(spec["train_dir"] / "adapter"), "--windows", rel(windows), "--output", rel(out.parent),
            "--max-num-seqs", "100", "--batch-size", "100", "--max-new-tokens", "32768"]


def evaluate_command(records, left, output, right=None):
    cmd = ["python", "scripts/evaluate_predictions.py", "--references", rel(records), "--predictions", rel(left)]
    if right:
        cmd += ["--compare-predictions", rel(right)]
    return cmd + ["--output", rel(output), *SEMANTIC]


def metrics(block):
    """Flatten one evaluate_predictions arm into the headline metrics."""
    if not block:
        return None
    agg = block["canonical"]["aggregate"]
    extra = block.get("additional_metrics", {})
    titles = extra.get("titles", {})
    align = agg.get("title_span_alignment") or {}
    return {"videos": agg["requested_videos"], "failed": agg["failed_videos"],
            "exact_f1": agg["macro_exact_f1"], "tolerant_f1": agg["macro_tolerant_f1"],
            "pk": agg["macro_pk"], "window_diff": agg["macro_window_diff"],
            "boundary_similarity": extra.get("boundary_similarity", {}).get("macro_all_requests"),
            "title_rouge_l": titles.get("rouge_l", {}).get("concatenated_document_macro_all_requests", {}).get("f1"),
            "title_bertscore": titles.get("semantic", {}).get("reference_chapter_mean_with_unmatched_zero", {}).get("f1"),
            "pred_per_ref": (align["predicted_chapters_successful_requests"] / align["reference_chapters_all_requests"]
                             if align else None)}


def paired_file(alias, split):
    if split == "dev":
        return S5 / f"{alias}-dev-paired.json"
    return TEST / f"{alias}-test-paired.json"


def paired_metrics(path):
    """(LoRA metrics, zero-shot metrics, bootstrap) from a paired file (LoRA left, zero-shot right)."""
    data = load_json(path)
    return metrics(data), metrics(data["comparison"]["right_arm"]), data["comparison"]["paired_exact_f1_bootstrap"]


def recompute(records_path, predictions_path):
    """Re-score saved predictions with the project evaluator (stdlib only)."""
    from chaptering.evaluation import evaluate_records
    return evaluate_records(load_jsonl(records_path), load_jsonl(predictions_path))


def bootstrap(left_result, right_result):
    from chaptering.reporting import paired_video_bootstrap
    return paired_video_bootstrap(left_result["per_video"], right_result["per_video"])


def metric_table(rows):
    import pandas as pd
    frame = pd.DataFrame([{"configuration": label, **(m or {})} for label, m in rows]).set_index("configuration")
    return frame.round(3)


def chapters_side_by_side(record, predictions, max_rows=30):
    """Gold vs predicted chapters (start sentence and title) for one video."""
    import pandas as pd
    columns = {"gold": dict(zip(record["reference"]["chapter_starts"], record["reference"]["chapter_titles"]))}
    for label, prediction in predictions.items():
        columns[label] = ({c["start"]: c["title"] for c in prediction["chapters"]}
                          if prediction and prediction["status"] == "ok" else {})
    starts = sorted(set().union(*[set(c) for c in columns.values()]))[:max_rows]
    return pd.DataFrame({label: [c.get(s, "") for s in starts] for label, c in columns.items()},
                        index=pd.Index(starts, name="start sentence"))


def by_id(path):
    return {row["video_id"]: row for row in load_jsonl(path)}


def plot_loss(steps_path, title, ax=None, window=50):
    import matplotlib.pyplot as plt
    steps = load_jsonl(steps_path)
    loss = [s["loss"] for s in steps]
    smooth = [sum(loss[max(0, i - window + 1):i + 1]) / len(loss[max(0, i - window + 1):i + 1]) for i in range(len(loss))]
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    ax.plot(loss, color="#c5ccd6", linewidth=0.8, label="per window")
    ax.plot(smooth, color="#2f5fd0", linewidth=2, label=f"rolling mean ({window})")
    ax.set_xlabel("training window (one forward pass)")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax
