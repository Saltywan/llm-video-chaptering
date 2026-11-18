#!/usr/bin/env python
"""Generate the experiment notebooks in notebooks/ (one per experiment).

  python scripts/build_notebooks.py            # write the .ipynb files
  jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

Every notebook runs on CPU from the saved artifacts. GPU steps (LoRA training, model
inference) are shown as the exact commands that produced those artifacts and run only
with RUN_GPU = True on a machine that has the models.
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


SETUP = """
%matplotlib inline
import json, statistics
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from nbcommon import *

pd.set_option("display.max_colwidth", 80)
RUN_GPU = False   # True only on the training/inference machine (needs the models and a GPU)
print("project:", ROOT.name, "| run outputs read from:", A.relative_to(ROOT))
"""


def save(name, title, cells):
    nb = new_notebook(cells=[md(title), code(SETUP), *cells])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    path = OUT / name
    nbformat.write(nb, path)
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------------------------------------
def data_notebook():
    save("01_data_loading_and_audit.ipynb", """
# 01 · Data loading and audit

**Dataset:** [`retkowski/ytseg`](https://huggingface.co/datasets/retkowski/ytseg), `text` configuration, pinned to revision
`4ee1c1fccc5a9e118ff0918f3df1e61aadca8970`. Each row is one YouTube video: its transcript as a list of sentences plus
the creator's chapters (start sentence and title).

This notebook covers:
1. Loading raw rows and converting them to the project's canonical record format.
2. Checking the label semantics.
3. Auditing the full splits, which found duplicate IDs and train/validation overlap.
4. Building the deduplicated variant used by every experiment.
5. Auditing the test split, done once and only after all configurations were frozen.
6. Corpus statistics, and the dev and pilot subsets used for LoRA training and model selection.
""", [
        md("""
## 1. Raw rows → canonical records

Raw rows carry the sentences (`text_ref`), a binary string marking the sentence that *starts* each internal chapter
(`target_binary_ref`), a marked copy of the transcript (`target_text_ref`) and the titles (`chapter_titles`).
`chaptering.data.canonicalize_row` converts a row into the canonical record. It also cross-checks the binary labels against
the `[CSTART]title[CEND]` markers and raises `DataValidationError` on any mismatch, so no label is ever guessed.

This cell streams a few training rows from the Hugging Face Hub, which needs internet. Offline, it falls back to the saved
canonical records.
"""),
        code("""
from chaptering.data import DATASET, CONFIG, canonicalize_row, indexed_transcript, outline_target
REVISION = "4ee1c1fccc5a9e118ff0918f3df1e61aadca8970"
STREAM_FROM_HF = True

raw_rows = []
if STREAM_FROM_HF:
    try:
        from datasets import load_dataset
        stream = load_dataset(DATASET, CONFIG, revision=REVISION, split="train", streaming=True)
        for row in stream:
            raw_rows.append(row)
            if len(raw_rows) == 3:
                break
        print("raw fields:", sorted(raw_rows[0].keys()))
    except Exception as exc:  # offline or hub unavailable
        print("streaming unavailable, using saved canonical records:", type(exc).__name__)

if raw_rows:
    row = raw_rows[0]
    print("video:", row["video_id"], "| sentences:", len(row["text_ref"]), "| chapters:", len(row["chapter_titles"]))
    print("binary labels (first 60):", row["target_binary_ref"][:62])
    record = canonicalize_row(row, "train", REVISION)
else:
    record = load_jsonl(A / "data_audit/2026-11-16-resolved-v1/train.jsonl", limit=1)[0]
print(json.dumps({k: record[k] for k in ["schema_version", "video_id", "split", "reference", "provenance"]}, indent=1)[:1500])
"""),
        code("""
# The model only ever sees the numbered sentences; the reference stays outside every generation payload.
print(indexed_transcript(record)[:800], "...")
print()
print("training target:", json.dumps(outline_target(record))[:400])
"""),
        md("""
## 2. Full-split audit (`scripts/audit_data.py`)

This is a CPU-only pass over every train and validation row at the pinned revision. Test rows were only counted by ID; their content was not read. The audit found:
* **Duplicate video IDs**: 17 in train, 1 in validation. Most are exact copies; a few have *conflicting* content.
* **Train/validation overlap**: some validation videos also appear in train.
"""),
        code("""
run(["python", "scripts/audit_data.py", "--output", "artifacts/data_audit/source-audit", "--revision", REVISION], RUN_GPU)
audit = load_json(A / "data_audit/source-audit/audit.json")
rows = []
for split in ["train", "validation"]:
    s = audit["splits"][split]
    rows.append({"split": split, "rows": s["rows_seen"], "valid rows": s["valid_rows"], "unique ids": s["unique_ids"],
                 "duplicate ids": len(s["duplicate_ids"]),
                 "median sentences": s["sentence_counts"]["quantiles"]["0.5"],
                 "max sentences": s["sentence_counts"]["quantiles"]["1"],
                 "mean chapters": round(s["chapter_counts"]["mean"], 2)})
display(pd.DataFrame(rows).set_index("split"))
print("split overlaps:", json.dumps(audit["split_overlaps"], indent=1)[:600])
print("integrity gate passed:", audit["full_integrity_gate_passed"])
"""),
        md("""
## 3. Deduplicated / quarantined variant (`scripts/resolve_duplicates.py`)

The original files are kept untouched. A declared variant is derived from them:
* Every training row whose ID also appears in validation is removed.
* For exact duplicates, the first copy is kept.
* IDs with conflicting copies are quarantined, never resolved by picking one.
"""),
        code("""
run(["python", "scripts/resolve_duplicates.py", "--source", "artifacts/data_audit/source-audit",
     "--output", "artifacts/data_audit/2026-11-16-resolved-v1"], RUN_GPU)
res = load_json(A / "data_audit/2026-11-16-resolved-v1/resolution.json")
for key, text in res["policy"].items():
    print(f"- {key}: {text}")
counts = res["original_and_resolved_counts"]
display(pd.DataFrame({split: {k: v for k, v in c.items() if isinstance(v, int)} for split, c in counts.items()}))
print("resolved gate passed:", res["resolved_integrity_gate_passed"])
"""),
        md("""
## 4. Test split (read once, after the freeze)

`scripts/prepare_test.py` canonicalizes the test split only after `configs/frozen_test.json` was written. It confirms that every row is valid and unique and that none overlaps the resolved train/validation sets.
"""),
        code("""
test_audit = load_json(A / "data_audit/2026-11-17-test-v1/test_audit.json")
print({k: test_audit[k] for k in ["rows_seen", "valid_rows", "invalid_rows", "unique_ids", "duplicate_ids",
                                   "overlap_with_resolved_train_validation", "passed"]})
"""),
        md("## 5. Corpus statistics (canonical records)"),
        code("""
validation = load_jsonl(A / "data_audit/2026-11-16-resolved-v1/validation.jsonl")
test = load_jsonl(TEST_RECORDS)
train = load_jsonl(A / "data_audit/2026-11-16-resolved-v1/train.jsonl")

def stats(records):
    sentences = [len(r["sentences"]) for r in records]
    chapters = [len(r["reference"]["chapter_starts"]) for r in records]
    lengths = [n / c for n, c in zip(sentences, chapters)]
    return {"videos": len(records), "median sentences": statistics.median(sentences), "max sentences": max(sentences),
            "median chapters": statistics.median(chapters), "mean chapters": round(statistics.mean(chapters), 2),
            "median sentences / chapter": round(statistics.median(lengths), 1)}

display(pd.DataFrame({"train (resolved)": stats(train), "validation (resolved)": stats(validation), "test": stats(test)}).T)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for name, records in [("train", train), ("validation", validation), ("test", test)]:
    axes[0].hist([min(len(r["sentences"]), 1500) for r in records], bins=60, histtype="step", density=True, label=name)
    axes[1].hist([min(len(r["reference"]["chapter_starts"]), 40) for r in records], bins=40, histtype="step", density=True, label=name)
axes[0].set_xlabel("sentences per video (clipped at 1,500)"); axes[1].set_xlabel("chapters per video (clipped at 40)")
for ax in axes: ax.legend(frameon=False)
plt.tight_layout()
"""),
        md("""
## 6. Dev and pilot subsets (`scripts/prepare_windows.py`)

* **LoRA training set:** 1,000 videos drawn from the resolved training split. These are the *pilot-scale* adapters.
* **Dev set:** 200 validation videos, used for model selection.

Selection is gold-free: videos are binned by length and then drawn with a seeded hash. The full validation set (1,446 videos) is windowed separately for the long-transcript analysis.
"""),
        code("""
manifest = load_json(DEV_PACK / "manifest.json")
print(manifest["selection_policy"])
print(json.dumps(manifest["settings"], indent=1))
for split in ["train", "validation"]:
    p = manifest["statistics"][split]["processed"]
    print(split, {k: p[k] for k in ["attempted_videos", "successful_videos", "windows", "owned_sentences"]})
"""),
    ])


# ---------------------------------------------------------------------------------------------
def window_notebook():
    save("02_windowing_and_prompt.ipynb", """
# 02 · Windows, prompt, training targets and stitching

Some transcripts are longer than the 8,192-token input budget. Rather than truncate them, every video is cut into **windows**:
* Each window *owns* a disjoint block of sentences `[A, B)`.
* Each window also sees up to 16 sentences of context on each side, `[L, R)`.
* The model must list `L` first; it is a processing anchor, not a claimed boundary.
* After generation, only the chapter starts inside `[A, B)` are kept.

The owned blocks tile the transcript exactly, so stitching is a plain concatenation. The windows are identical for all four backbones because each window is sized to fit the budget under all four tokenizers at once.
""", [
        md("## 1. One multi-window video from the dev pack"),
        code("""
from chaptering.windowing import build_window_prompt, parse_window_outline, stitch_window_predictions

windows = defaultdict(list)
for row in load_jsonl(DEV_WINDOWS):
    windows[row["window"]["video_id"]].append(row["window"])
records = by_id(DEV_RECORDS)
video = sorted(v for v, ws in windows.items() if len(ws) == 3)[0]
ws = windows[video]
n = len(records[video]["sentences"])
display(pd.DataFrame([{k: w[k] for k in ["window_index", "L", "A", "B", "R", "left_context_count", "right_context_count"]}
                      | {"tokens (M1/M2/E1/E2)": "/".join(str(w["input_token_counts"][m]) for m in ["M1", "M2", "E1", "E2"])}
                      for w in ws]).set_index("window_index"))
cores = [(w["A"], w["B"]) for w in ws]
assert cores[0][0] == 0 and cores[-1][1] == n and all(a[1] == b[0] for a, b in zip(cores, cores[1:]))
print(f"video {video}: {n} sentences; owned blocks tile [0, {n}) exactly:", cores)
"""),
        md("## 2. The prompt every model receives (window 2)"),
        code("""
prompt = build_window_prompt(ws[1])
print(prompt[:1800], "\\n...")
"""),
        md("""
## 3. Training targets

A training window's target is the gold chapter JSON inside its visible range. Titles that the window cannot fully support are replaced by the placeholder `Chapter` and **masked out of the loss**. This applies to the processing anchor, chapters it does not own, and chapters cut by the window edge.
"""),
        code("""
train_row = next(r for r in load_jsonl(TRAIN_WINDOWS) if r["window"]["window_index"] == 1)
print("window", {k: train_row["window"][k] for k in ["video_id", "L", "A", "B", "R"]})
print("target:", train_row["target"]["target_json"][:500])
print("masked character spans:", train_row["target"]["masked_target_spans"][:6])
print("title stats:", train_row["target"]["title_supervision_stats"])
"""),
        md("## 4. Stitching: gold window outlines → whole-video chapters"),
        code("""
# Use each window's gold window target as a stand-in for a model answer. Stitching keeps only owned starts, so the
# result must equal the gold chapter starts of the whole video.
from chaptering.windowing import window_training_target
answers = []
for w in ws:
    target_json = window_training_target(records[video], w)["target_json"]
    answers.append({"video_id": video, "window_index": w["window_index"], "status": "ok",
                    "chapters": parse_window_outline(target_json, w)})
stitched = stitch_window_predictions(ws, answers, run_id="gold-demo")
print("status:", stitched["status"])
print("stitched starts:", [c["start"] for c in stitched["chapters"]])
print("gold starts:    ", records[video]["reference"]["chapter_starts"])
assert [c["start"] for c in stitched["chapters"]] == records[video]["reference"]["chapter_starts"]
"""),
        md("## 5. Window packs used in the experiments"),
        code("""
rows = []
for name, path in [("pilot train (LoRA)", TRAIN_WINDOWS), ("dev (200 validation)", DEV_WINDOWS), ("test", TEST_WINDOWS)]:
    count = Counter(json.loads(line)["window"]["video_id"] for line in open(path))
    dist = Counter(min(c, 4) for c in count.values())
    rows.append({"pack": name, "videos": len(count), "windows": sum(count.values()), "1 window": dist[1], "2": dist[2],
                 "3": dist[3], "4+": dist[4], "max windows": max(count.values())})
display(pd.DataFrame(rows).set_index("pack"))
tokens = [json.loads(line)["window"]["input_token_counts"]["M1"] for line in open(TEST_WINDOWS)]
plt.figure(figsize=(7, 2.8)); plt.hist(tokens, bins=50, color="#2f5fd0")
plt.axvline(8192, color="#b3261e", linestyle="--", label="budget 8,192")
plt.xlabel("prompt tokens per test window (Qwen3.5 tokenizer)"); plt.legend(frameon=False); plt.tight_layout()
"""),
        code("""
# The test pack copies every setting from the validation pack and is built once, after the freeze.
test_manifest = load_json(TEST_PACK / "manifest.json")
print(test_manifest["purpose"]); print(test_manifest["settings"])
run(["python", "scripts/prepare_test.py"], RUN_GPU)
"""),
    ])


# ---------------------------------------------------------------------------------------------
def metrics_notebook():
    save("03_evaluation_metrics.ipynb", """
# 03 · Evaluation metrics

All scores are computed in sentence space against the creator's chapters and macro-averaged over **every requested
video**. A video whose output fails strict parsing stays in the average with the worst score: 0 for F1/similarity, 1 for Pk/WindowDiff.

| Metric | Meaning |
|---|---|
| **Exact F1** (primary) | Boundary = first sentence of every chapter after the first. F1 counts only exact index matches. |
| ±2 F1 | Same, but a predicted boundary within 2 sentences of an unmatched reference boundary also counts (one-to-one). |
| Pk ↓ | Probe of k sentences (half the mean reference chapter length). Error when reference and prediction disagree on whether both ends are in one chapter. |
| WindowDiff ↓ | Same probe; error when the *number* of boundaries inside it differs. |
| Boundary Similarity | segeval edit-distance similarity; near misses get partial credit. |
| Title ROUGE-L | ROUGE-L F1 between the concatenated predicted and reference titles of each video. |
| Title BERTScore | Chapters paired by span (IoU ≥ 0.5, titles never read for pairing); RoBERTa-large layer-17 F1 per pair; unmatched reference chapters count 0. |
| Paired bootstrap | Per-video exact-F1 difference between two systems; 10,000 video resamples; 95% percentile interval. |
""", [
        md("## 1. A toy example"),
        code("""
from chaptering.evaluation import score_prediction
reference = {"video_id": "toy", "sentences": [f"s{i}" for i in range(20)],
             "reference": {"chapter_starts": [0, 5, 12], "chapter_titles": ["Intro", "Setup", "Results"]}}
prediction = {"video_id": "toy", "status": "ok",
              "chapters": [{"start": 0, "title": "Intro"}, {"start": 6, "title": "Setting up"}, {"start": 12, "title": "Results"}]}
s = score_prediction(reference, prediction)
print("exact   :", {k: s["exact"][k] for k in ["tp", "fp", "fn", "f1"]})
print("±2      :", {k: s["tolerant"][k] for k in ["tp", "fp", "fn", "f1"]})
print("Pk / WD :", s["structural"])
failed = score_prediction(reference, {"video_id": "toy", "status": "invalid", "chapters": []})
print("failed  :", failed["exact"]["f1"], failed["structural"]["pk"])
"""),
        md("## 2. Re-scoring a saved run and checking it against the saved evaluation"),
        code("""
# E2 LoRA on the 200 dev videos: re-score the saved predictions and compare with the saved paired evaluation file.
ft = recompute(DEV_RECORDS, BACKBONES["E2"]["runs"][("dev", "FT")] / "predictions.jsonl")
zs = recompute(DEV_RECORDS, BACKBONES["E2"]["runs"][("dev", "ZS")] / "predictions.jsonl")
saved_ft, saved_zs, saved_boot = paired_metrics(paired_file("E2", "dev"))
print("recomputed exact F1  LoRA %.4f  zero-shot %.4f" % (ft["aggregate"]["macro_exact_f1"], zs["aggregate"]["macro_exact_f1"]))
print("saved      exact F1  LoRA %.4f  zero-shot %.4f" % (saved_ft["exact_f1"], saved_zs["exact_f1"]))
assert abs(ft["aggregate"]["macro_exact_f1"] - saved_ft["exact_f1"]) < 1e-12
"""),
        code("""
boot = bootstrap(ft, zs)
print("paired bootstrap (LoRA − zero-shot): %+.4f  95%% CI [%+.4f, %+.4f]" % (boot["mean_difference"], *boot["confidence_interval"]))
print("saved:                               %+.4f  95%% CI [%+.4f, %+.4f]" % (saved_boot["mean_difference"], *saved_boot["confidence_interval"]))
"""),
        md("""
## 3. Full evaluation command

The title metrics use `rouge-score`, `segeval` and `bert-score`, with RoBERTa-large pinned to a commit. All of them run in `scripts/evaluate_predictions.py`, which writes the paired files used throughout the report:
"""),
        code("""
run(evaluate_command(DEV_RECORDS, BACKBONES["E2"]["runs"][("dev", "FT")] / "predictions.jsonl", S5 / "E2-dev-paired.json",
                     BACKBONES["E2"]["runs"][("dev", "ZS")] / "predictions.jsonl"), RUN_GPU)
display(metric_table([("E2 zero-shot (dev)", saved_zs), ("E2 LoRA (dev)", saved_ft)]))
"""),
    ])


# ---------------------------------------------------------------------------------------------
TAKEAWAYS = {
    "E1": "Zero-shot E1 almost never produces the required JSON: on test, 13 of 1,448 videos are valid and 129 are cut off at the 4,096-token cap. "
          "Pilot-scale LoRA fixes the format and reaches exact F1 0.484 on test (+0.481). It is still slightly below the much smaller E2 (−0.017, post-hoc).",
    "E2": "Zero-shot E2 fails strict parsing on a third of the test videos (477). LoRA reduces that to 4 and reaches the best test result of the study, "
          "exact F1 0.500 (±2 F1 0.679, Pk 0.229). The gain over zero-shot is +0.352.",
    "M1": "Zero-shot M1 is strong (0.403 test) because it reasons at length before answering. Answer-only LoRA halves that reasoning "
          "(median 7,192 → 3,328 tokens on dev), under-segments, and fails more often: 0.365 on test (−0.038). Its q/k/v/o adapter reaches only the 8 "
          "full-attention layers of this hybrid model. Notebook 08 shows that RFT recovers and then exceeds zero-shot on dev.",
    "M2": "Zero-shot M2 is the best decoder-only system (0.451 test). Answer-only LoRA makes it stop thinking entirely (median 4,907 → 0 tokens) "
          "and under-segment (predicted/reference chapters 1.48 → 0.81): 0.344 on test (−0.108). RFT restores thinking but stays at zero-shot level.",
}


def backbone_notebook(index, alias, slug):
    spec_name = {"E1": "T5Gemma 9B-2B", "E2": "T5Gemma 2B-2B", "M1": "Qwen3.5-9B", "M2": "Gemma 4 12B"}[alias]
    seq2seq = alias.startswith("E")
    cells = [
        md("## 0. Configuration"),
        code(f"""
ALIAS = "{alias}"
spec = BACKBONES[ALIAS]
frozen = load_json(ROOT / "configs/frozen_test.json")
display(pd.Series({{"model": spec["model_id"], "revision": spec["revision"], "family": spec["family"],
                   "inference engine": "HF batched greedy" if spec["engine"] == "seq2seq" else "vLLM, native thinking",
                   "LoRA adapter": rel(spec["train_dir"] / "adapter")}}))
print("frozen entries for this backbone:", [c for c in frozen["configurations"] if ALIAS in c] if isinstance(frozen.get("configurations"), dict) else "see configs/frozen_test.json")
"""),
        md("""
## 1. Data

All runs use the same window packs (notebook 02):
* **Training:** 1,000 training videos (1,306 windows), used for LoRA only.
* **Dev:** 200 validation videos (270 windows), used to select settings before the freeze.
* **Test:** 1,448 test videos (1,905 windows), run once.
"""),
        code("""
dev_records, test_records = load_jsonl(DEV_RECORDS), load_jsonl(TEST_RECORDS)
print("dev:", len(dev_records), "videos,", sum(1 for _ in open(DEV_WINDOWS)), "windows")
print("test:", len(test_records), "videos,", sum(1 for _ in open(TEST_WINDOWS)), "windows")
print("train:", sum(1 for _ in open(TRAIN_WINDOWS)), "windows")
"""),
        md("""
## 2. Zero-shot inference

The base model receives the window prompt, and its answer is parsed strictly. Inference settings are identical for zero-shot and LoRA.
""" + ("""
Encoder-decoder models use **greedy decoding** in length tiers. Each window starts with a 384-token output cap; any window that hits the cap moves to the next tier (1,536, then 4,096 tokens).
""" if seq2seq else """
Decoder-only models run in **vLLM** with native thinking enabled and not forced:
* Model-card sampling: temperature 1.0, top-p 0.95, top-k 20 for Qwen (presence penalty 1.5) or 64 for Gemma.
* A per-window seed.
* Up to 32,768 new tokens.

The final JSON is taken after the model's native end-of-thinking marker.
""")),
        code("""
for split in ["dev", "test"]:
    run(infer_command(ALIAS, split, "ZS"), RUN_GPU)
"""),
        code(f"""
def summary(split, arm):
    d = spec["runs"][(split, arm)]
    return load_json(d / "summary.json")

rows = []
for split in ["dev", "test"]:
    for arm in ["ZS", "FT"]:
        s = summary(split, arm)
        rows.append({{"split": split, "arm": arm, "videos": s["requested_videos"], "windows": s["requested_windows"],
                     **{{f"videos {{k}}": v for k, v in sorted(s["status_counts"].items())}},
                     "generation min": round(s["generation_seconds"] / 60, 1), "output tokens": s["completion_tokens"]}})
display(pd.DataFrame(rows).fillna(0).set_index(["split", "arm"]))
"""),
        code(f"""
# One raw zero-shot answer for a test window. {"Encoder-decoder output is the JSON itself." if seq2seq else "Decoder-only output is a thought, then the final JSON."}
raw = load_jsonl(spec["runs"][("test", "ZS")] / "windows.jsonl")
print("window statuses:", dict(Counter(r["status"] for r in raw)))
for wanted in ["ok", "failed"]:
    example = next((r for r in raw if (r["status"] == "ok") == (wanted == "ok")), None)
    if example is None:
        continue
    text = example.get("raw_text") or ""
    print(f"\\n=== first {{wanted}} window: {{example['video_id']}} #{{example['window_index']}} | status {{example['status']}} | error {{example.get('error')}}")
    print("raw output:", len(text), "characters | reasoning tokens:", (example.get("usage") or {{}}).get("reasoning_tokens"))
    print(text[:600] + (" ..." if len(text) > 600 else ""))
    if example.get("final_text") and example["final_text"] != text:
        print("final answer:", example["final_text"][:400])
"""),
        md("""
## 3. LoRA fine-tuning

* **Adapter:** LoRA rank 16, alpha 32, dropout 0.05, on q/k/v/o projections; the base model stays frozen (bf16).
* **Optimizer:** AdamW at 2e-4 with cosine decay and 10 warmup steps; gradient clipping 1.0.
* **Batching:** one window per forward pass, gradient accumulation 8, one pass over the 1,306 training windows.
""" + ("""
The loss is decoder cross-entropy on the window's gold JSON. Titles that a window cannot support are masked out.
""" if seq2seq else """
The loss covers **only the final JSON answer** plus the native turn-end token, rendered with the model's chat template. The prompt is unscored, and no reasoning text is invented or supervised.
""")),
        code("""
run(train_command(ALIAS), RUN_GPU)
report = load_json(spec["train_dir"] / "report.json")
keep = ["final_context", "rank", "lora_alpha", "lora_dropout", "target_modules", "learning_rate", "lr_schedule", "warmup_steps",
        "grad_accum", "actual_steps", "planned_optimizer_steps", "actual_training_videos", "trainable_parameters",
        "teacher_forcing_tokens", "first_loss", "last_loss", "mean_loss", "training_seconds", "peak_allocated_bytes",
        "optimizer_update_verified", "save_reload_adapter_tensors_equal"]
display(pd.Series({k: report.get(k) for k in keep}))
"""),
        code(f"""
plot_loss(spec["train_dir"] / "steps.jsonl", "{alias} LoRA training loss (1,306 windows, one pass)")
plt.tight_layout()
adapter = spec["train_dir"] / "adapter"
print({{p.name: f"{{p.stat().st_size / 2**20:.1f}} MiB" for p in adapter.iterdir()}} if adapter.exists() else "adapter not downloaded")
"""),
        md("## 4. LoRA inference (same engine and settings, adapter loaded)"),
        code("""
for split in ["dev", "test"]:
    if spec["engine"] == "seq2seq":
        run(infer_command(ALIAS, split, "FT"), RUN_GPU)
    else:
        print(f"{split}: the LoRA arm ran in the same vLLM call as zero-shot (--arms ZS FT), shown above")
"""),
        md("""
## 5. Evaluation

The saved predictions are re-scored with the project evaluator and checked against the saved evaluation files. The full metric set and the paired bootstrap come from `scripts/evaluate_predictions.py`.
"""),
        code("""
recomputed, results = {}, []
for split, records in [("dev", DEV_RECORDS), ("test", TEST_RECORDS)]:
    for arm in ["ZS", "FT"]:
        recomputed[(split, arm)] = recompute(records, spec["runs"][(split, arm)] / "predictions.jsonl")
    saved_ft, saved_zs, saved_boot = paired_metrics(paired_file(ALIAS, split))
    assert abs(recomputed[(split, "FT")]["aggregate"]["macro_exact_f1"] - saved_ft["exact_f1"]) < 1e-12
    assert abs(recomputed[(split, "ZS")]["aggregate"]["macro_exact_f1"] - saved_zs["exact_f1"]) < 1e-12
    results += [(f"{split} zero-shot", saved_zs), (f"{split} LoRA", saved_ft)]
    print(f"{split}: LoRA − zero-shot exact F1 = {saved_boot['mean_difference']:+.3f} "
          f"[{saved_boot['confidence_interval'][0]:+.3f}, {saved_boot['confidence_interval'][1]:+.3f}] "
          f"(paired bootstrap, {saved_boot['videos']} videos)")
print("re-scored exact F1 matches the saved evaluation for all four runs")
display(metric_table(results))
"""),
        code("""
lenient = {arm: metrics(load_json(TEST / f"envelope-{ALIAS}-{arm}-metrics.json")) for arm in ["ZS", "FT"]}
print("Secondary lenient reading on test (one Markdown fence / trailing period stripped):")
display(metric_table([("test zero-shot (lenient)", lenient["ZS"]), ("test LoRA (lenient)", lenient["FT"])]))
"""),
        md("## 6. One test video: gold vs zero-shot vs LoRA"),
        code("""
zs_pred, ft_pred = by_id(spec["runs"][("test", "ZS")] / "predictions.jsonl"), by_id(spec["runs"][("test", "FT")] / "predictions.jsonl")
window_count = Counter(json.loads(line)["window"]["video_id"] for line in open(TEST_WINDOWS))
# first single-window test video of moderate length (display only; nothing is selected or tuned on test)
example = next(r for r in test_records if window_count[r["video_id"]] == 1 and 60 <= len(r["sentences"]) <= 120
               and 5 <= len(r["reference"]["chapter_starts"]) <= 9)
vid = example["video_id"]
print(vid, "|", len(example["sentences"]), "sentences | ZS status:", zs_pred[vid]["status"], "| LoRA status:", ft_pred[vid]["status"])
display(chapters_side_by_side(example, {"zero-shot": zs_pred[vid], "LoRA": ft_pred[vid]}))
"""),
        md("## 7. Error analysis"),
        code("""
# Exact F1 on test by number of windows per video.
rows = []
for arm in ["ZS", "FT"]:
    buckets = defaultdict(list)
    for score in recomputed[("test", arm)]["per_video"]:
        buckets[min(window_count[score["video_id"]], 3)].append(score["exact"]["f1"])
    rows.append({"arm": arm, **{("3+ windows" if k == 3 else f"{k} window" + ("s" if k > 1 else "")): round(statistics.mean(v), 3)
                               for k, v in sorted(buckets.items())},
                 **{f"status {k}": v for k, v in recomputed[("test", arm)]["aggregate"]["status_counts"].items()}})
display(pd.DataFrame(rows).fillna(0).set_index("arm"))
"""),
    ]
    if not seq2seq:
        cells.append(code("""
# Native thinking: reasoning tokens per dev window, zero-shot vs LoRA.
fig, ax = plt.subplots(figsize=(8, 3))
for arm, color in [("ZS", "#8a97a8"), ("FT", "#2f5fd0")]:
    tokens = [(r.get("usage") or {}).get("reasoning_tokens") or 0 for r in load_jsonl(spec["runs"][("dev", arm)] / "windows.jsonl")]
    ax.hist(tokens, bins=40, alpha=.7, color=color, label=f"{arm} (median {int(statistics.median(tokens))})")
ax.set_xlabel("reasoning tokens per window (dev)"); ax.legend(frameon=False); plt.tight_layout()
"""))
    else:
        cells.append(code("""
# Where zero-shot fails: window-level statuses and the output tier reached.
for arm in ["ZS", "FT"]:
    s = load_json(spec["runs"][("test", arm)] / "summary.json")
    print(arm, "window statuses:", s.get("window_status_counts"), "| tiers:",
          [(t["cap"], t["windows"], t["carried_to_next"]) for t in s.get("tiers", [])])
"""))
    cells.append(md(f"## 8. Takeaway\n\n{TAKEAWAYS[alias]}"))
    save(f"0{index}_{alias}_{slug}.ipynb", f"""
# 0{index} · {alias}: {spec_name}, zero-shot vs LoRA

End-to-end experiment for one backbone: data → zero-shot inference → LoRA fine-tuning → LoRA inference → evaluation on dev
(200 videos) and the held-out test set (1,448 videos, run once with the frozen configuration in `configs/frozen_test.json`).
""", cells)


# ---------------------------------------------------------------------------------------------
def rft_notebook():
    save("08_RFT_M1_M2.ipynb", """
# 08 · Rejection-sampling fine-tuning (RFT) for M1 and M2

Answer-only LoRA teaches the decoder-only models to skip their reasoning (notebooks 06–07). **RFT**, a simple
reinforcement-style recipe (expert iteration with a binary reward), trains on the model's *own* best attempts, reasoning included:

1. **Sample.** The base model answers every training window 4 times, using the zero-shot sampling settings.
2. **Score.** Each finished, strictly valid sample gets the exact F1 of its starts inside the window's owned block, against gold.
3. **Select.** Keep the best sample if F1 ≥ 0.5; otherwise drop the window. Gold only *chooses* a sample; it never enters the text.
4. **Train.** One-pass LoRA on prompt → own thought → own JSON → turn end.

**v1** used 300 training windows. **v2** keeps every setting and covers all 1,306 windows of the same 1,000 videos used by the direct LoRA. RFT is a dev-only follow-up after the freeze; it was never run on test.
""", [
        md("## 1. Training-window manifests (gold-free)"),
        code("""
run(["python", "scripts/rft_build.py", "manifest", "--train-windows", rel(TRAIN_WINDOWS), "--max-windows", "300",
     "--output", "artifacts/rft-v1/train_manifest.jsonl"], RUN_GPU)
run(["python", "scripts/rft_build.py", "manifest", "--train-windows", rel(TRAIN_WINDOWS), "--max-windows", "100000",
     "--exclude", "artifacts/rft-v1/train_manifest.jsonl", "--output", "artifacts/rft-v2/train_manifest.jsonl"], RUN_GPU)
for name, path in [("v1", RFT1 / "train_manifest.jsonl"), ("v2 new windows", RFT2 / "train_manifest.jsonl"), ("v1+v2", RFT2 / "union_manifest.jsonl")]:
    rows = load_jsonl(path)
    print(f"{name}: {len(rows)} windows from {len({r['window']['video_id'] for r in rows})} videos")
"""),
        md("""
## 2. Sampling

* **v1:** sampled through the existing vLLM services (OpenAI-compatible endpoints; the API key is read from stdin and never stored).
* **v2:** sampled with the local vLLM driver, using identical sampling settings and the same per-window seed scheme.
"""),
        code("""
for alias in ["M1", "M2"]:
    run(["python", "scripts/infer_endpoint_zs.py", "--alias", alias, "--windows", "artifacts/rft-v1/train_manifest.jsonl",
         "--output", f"artifacts/rft-v1/samples/{alias}-seed1001", "--base-url", "$ENDPOINT_URL", "--concurrency", "16",
         "--seed", "1001"], RUN_GPU, env="API key read from stdin")
    run(["$VLLM_PYTHON", "scripts/infer_vllm_pair.py", "--alias", alias, "--arms", "ZS",
         "--windows", "artifacts/rft-v2/train_manifest.jsonl", "--output", f"artifacts/rft-v2/samples/{alias}-seed1001",
         "--seed", "1001", "--max-num-seqs", "256", "--batch-size", "1006", "--max-new-tokens", "32768"], RUN_GPU)
print("(repeated for seeds 2002, 3003, 4004)")
rows = []
for version, base, sub in [("v1", RFT1, ""), ("v2", RFT2, "ZS/")]:
    for alias in ["M1", "M2"]:
        for seed in [1001, 2002, 3003, 4004]:
            s = load_json(base / f"samples/{alias}-seed{seed}/{sub}summary.json")
            seconds = s.get("generation_seconds", s.get("this_invocation_seconds"))
            rows.append({"version": version, "model": alias, "seed": seed, "windows": s["requested_windows"],
                         "ok videos": s["status_counts"].get("ok", 0), "minutes": round(seconds / 60, 1),
                         "output tokens": s["completion_tokens"]})
display(pd.DataFrame(rows).set_index(["version", "model", "seed"]))
"""),
        md("## 3. Selection (best of N, owned-start F1 ≥ 0.5)"),
        code("""
run(["python", "scripts/rft_build.py", "select", "--train-windows", rel(TRAIN_WINDOWS),
     "--manifest", "artifacts/rft-v2/union_manifest.jsonl", "--samples", "<8 sample windows.jsonl files>",
     "--min-f1", "0.5", "--output", "artifacts/windows/dev-v2/rft_M1_v2.jsonl"], RUN_GPU)
rows = []
for version in ["v1", "v2"]:
    for alias in ["M1", "M2"]:
        s = load_json(DEV_PACK / f"rft_{alias}_{version}.summary.json")
        rows.append({"version": version, "model": alias, "windows sampled": s["manifest_windows"], "samples": s["samples_read"],
                     "kept windows": s["kept_windows"], "mean F1 of valid samples": round(s["mean_valid_sample_f1"], 3),
                     "mean F1 of kept samples": round(s["mean_kept_f1"], 3), **s["drop_counts"]})
display(pd.DataFrame(rows).fillna(0).set_index(["version", "model"]))
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, key, label in [(axes[0], "window_f1", "owned-start F1 of the kept sample"), (axes[1], "reasoning_tokens", "reasoning tokens of the kept sample")]:
    for alias, color in [("M1", "#2f5fd0"), ("M2", "#c98a00")]:
        kept = load_jsonl(DEV_PACK / f"rft_{alias}_v2.jsonl")
        ax.hist([r["rft_meta"][key] or 0 for r in kept], bins=30, alpha=.6, color=color, label=f"{alias} ({len(kept)} windows)")
    ax.set_xlabel(label); ax.legend(frameon=False)
plt.tight_layout()
"""),
        md("## 4. Training (same LoRA settings as direct LoRA; warmup 3; loss on everything after the prompt)"),
        code("""
for alias in ["M1", "M2"]:
    run(["python", "scripts/train_lora.py", "--alias", alias, "--windows", f"artifacts/windows/dev-v2/rft_{alias}_v2.jsonl",
         "--max-steps", "0", "--target-modules", "q_proj,k_proj,v_proj,o_proj", "--learning-rate", "2e-4", "--lr-schedule", "cosine",
         "--warmup-steps", "3", "--grad-accum", "8", "--checkpoint-every", "100", "--output", f"artifacts/rft-v2/{alias}-rft-lora"], RUN_GPU)
rows = []
for version, base in [("v1", RFT1), ("v2", RFT2)]:
    for alias in ["M1", "M2"]:
        r = load_json(base / f"{alias}-rft-lora/report.json")
        rows.append({"version": version, "model": alias, "windows": r["actual_steps"], "optimizer steps": r["planned_optimizer_steps"],
                     "tokens processed": r["teacher_forcing_tokens"], "mean loss": round(r["mean_loss"], 3),
                     "minutes": round(r["training_seconds"] / 60, 1), "final context": r["final_context"]})
display(pd.DataFrame(rows).set_index(["version", "model"]))
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, alias in zip(axes, ["M1", "M2"]):
    plot_loss(RFT2 / f"{alias}-rft-lora/steps.jsonl", f"{alias} RFT v2 loss", ax=ax, window=25)
plt.tight_layout()
"""),
        md("## 5. Dev evaluation (200 videos; same inference settings as zero-shot)"),
        code("""
def think(path):
    return int(statistics.median((r.get("usage") or {}).get("reasoning_tokens") or 0 for r in load_jsonl(path)))

rows = []
for alias in ["M1", "M2"]:
    direct_ft, zs, direct_boot = paired_metrics(paired_file(alias, "dev"))
    v1, _, v1_boot = paired_metrics(RFT1 / f"{alias}-rft-dev-paired.json")
    v2, _, v2_boot = paired_metrics(RFT2 / f"{alias}-rft-dev-paired.json")
    _, _, v2v1 = paired_metrics(RFT2 / f"{alias}-v2-vs-v1-paired.json")
    check = recompute(DEV_RECORDS, RFT2 / f"{alias}-dev/FT/predictions.jsonl")["aggregate"]["macro_exact_f1"]
    assert abs(check - v2["exact_f1"]) < 1e-12
    for label, m, boot, path in [("zero-shot", zs, None, S5 / f"{alias}-dev/ZS/windows.jsonl"),
                                 ("direct LoRA", direct_ft, direct_boot, S5 / f"{alias}-dev/FT/windows.jsonl"),
                                 ("RFT v1", v1, v1_boot, RFT1 / f"{alias}-dev/FT/windows.jsonl"),
                                 ("RFT v2", v2, v2_boot, RFT2 / f"{alias}-dev/FT/windows.jsonl")]:
        rows.append({"model": alias, "config": label, "exact F1": round(m["exact_f1"], 3),
                     "− zero-shot [95% CI]": "" if boot is None else "%+.3f [%+.3f, %+.3f]" % (boot["mean_difference"], *boot["confidence_interval"]),
                     "failed": m["failed"], "pred/ref chapters": round(m["pred_per_ref"], 2), "median thinking": think(path)})
    print(f"{alias}: RFT v2 − RFT v1 = {v2v1['mean_difference']:+.3f} [{v2v1['confidence_interval'][0]:+.3f}, {v2v1['confidence_interval'][1]:+.3f}]")
display(pd.DataFrame(rows).set_index(["model", "config"]))
"""),
        md("""
## 6. Takeaway

* **RFT repairs the direct-LoRA damage:** thinking returns, failures drop, and the chapter count returns to zero-shot level.
* **M1 gains with more data.** RFT v2 reaches **0.440** dev exact F1. That is significantly above RFT v1 (+0.031 [+0.010, +0.052]) and trends above zero-shot (+0.019 [−0.003, +0.041]).
* **M2 does not gain.** It stays at zero-shot level, 0.464 vs 0.459.
* **Not tested on the held-out set.** A test run would need a new, declared configuration and freeze.
"""),
    ])


# ---------------------------------------------------------------------------------------------
def summary_notebook():
    save("09_results_summary.ipynb", """
# 09 · Results summary (all eight test configurations)

All numbers come from the compiled results file `reports/results.json` (`scripts/compile_report.py`), which reads each run's evaluation output. The headline exact F1 values are re-checked below against the saved predictions.
""", [
        code("""
R = load_json(ROOT / "reports/results.json")
order = ["E1", "E2", "M1", "M2"]
rows = []
for alias in order:
    t = R["test"][alias]
    rows += [(f"{alias} zero-shot", t["ZS"]), (f"{alias} LoRA", t["FT"])]
display(metric_table(rows))
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 3.4))
x = range(len(order))
zs = [R["test"][a]["ZS"]["exact_f1"] for a in order]
ft = [R["test"][a]["FT"]["exact_f1"] for a in order]
ax.bar([i - .2 for i in x], zs, .4, color="#8a97a8", label="zero-shot")
ax.bar([i + .2 for i in x], ft, .4, color="#2f5fd0", label="LoRA")
for i, (a, b) in enumerate(zip(zs, ft)):
    ax.text(i - .2, a + .01, f"{a:.3f}", ha="center", fontsize=9); ax.text(i + .2, b + .01, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(list(x), [f"{a}\\n{BACKBONES[a]['model_id'].split('/')[1]}" for a in order])
ax.set_ylabel("exact boundary F1 (test)"); ax.set_ylim(0, .6); ax.legend(frameon=False); plt.tight_layout()
"""),
        md("## LoRA effect (paired bootstrap) and dev → test consistency"),
        code("""
rows = []
for alias in order:
    t, d = R["test"][alias]["ft_minus_zs_exact_f1"], R["dev"][alias]["ft_minus_zs_exact_f1"]
    rows.append({"backbone": alias, "dev LoRA − ZS": "%+.3f [%+.3f, %+.3f]" % (d["mean_difference"], *d["confidence_interval"]),
                 "test LoRA − ZS": "%+.3f [%+.3f, %+.3f]" % (t["mean_difference"], *t["confidence_interval"]),
                 "failed ZS / LoRA (test)": f'{R["test"][alias]["ZS"]["failed"]} / {R["test"][alias]["FT"]["failed"]}'})
display(pd.DataFrame(rows).set_index("backbone"))
print("Post-hoc cross-backbone comparisons on test:")
for p in R["posthoc_test_comparisons"]:
    print(f"  {p['left']} − {p['right']}: {p['mean_difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]")
"""),
        md("## Re-check headline numbers from the saved predictions"),
        code("""
if TEST_RECORDS.exists():
    for alias in order:
        for arm in ["ZS", "FT"]:
            value = recompute(TEST_RECORDS, BACKBONES[alias]["runs"][("test", arm)] / "predictions.jsonl")["aggregate"]["macro_exact_f1"]
            assert abs(value - R["test"][alias][arm]["exact_f1"]) < 1e-12, (alias, arm)
    print("all eight test exact-F1 values reproduce from saved predictions")
else:
    print("skipped: rebuild the test records first (scripts/prepare_test.py); transcripts are not redistributed")
"""),
        md("## Rejection-sampling fine-tuning (dev only)"),
        code("""
rows = []
for alias in ["M1", "M2"]:
    rows += [(f"{alias} zero-shot", R["dev"][alias]["ZS"]), (f"{alias} direct LoRA", R["dev"][alias]["FT"]),
             (f"{alias} RFT v1", R["rft"][alias]["RFT"]), (f"{alias} RFT v2", R["rft_v2"][alias]["RFT"])]
display(metric_table(rows))
"""),
        md("## Compute and time"),
        code("""
train = pd.DataFrame([{"recipe": r["recipe"], "model": r["alias"], "windows": r.get("actual_steps"),
                       "minutes": round(r["training_seconds"] / 60, 1) if r.get("training_seconds") else None,
                       "peak GiB": round(r["peak_allocated_bytes"] / 2**30) if r.get("peak_allocated_bytes") else None}
                      for r in R["training_runs"]]).set_index(["recipe", "model"])
display(train)
timing = pd.DataFrame(R["timing"])
test_timing = timing[timing.stage.str.startswith("Test")].assign(minutes=lambda f: (f.generation_seconds / 60).round(1))
display(test_timing[["model", "arm", "device", "engine", "minutes", "seconds_per_video", "output_tokens", "failed_videos"]].set_index(["model", "arm"]))
"""),
        md("""
## Failure analysis and chapter counts

From `reports/failure-analysis.json` (`scripts/failure_analysis.py`, which needs the raw outputs in `artifacts/`). Every failed window gets exactly one cause; a failed video is attributed to its first failed window. Chapter counts use parsed videos only on both sides.
"""),
        code("""
FA = load_json(ROOT / "reports/failure-analysis.json")
groups = list(FA["groups"].items())
causes = pd.DataFrame({t["config"]: {(g, c): t["video_causes"].get(c, 0) for g, cs in groups for c in cs} for t in FA["test"]})
causes = causes[(causes > 0).any(axis=1)]
causes.loc[("Total", "failed videos"), :] = [t["failed_videos"] for t in FA["test"]]
display(causes.astype(int).replace(0, ""))
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
colors = ["#8a97a8", "#c98a00", "#b3261e", "#7b4fc9", "#2f5fd0", "#18794e"]
configs = [t["config"] for t in FA["test"]][::-1]
left = [0] * len(configs)
for (g, cs), col in zip(groups, colors):
    vals = [sum(t["video_causes"].get(c, 0) for c in cs) for t in FA["test"]][::-1]
    ax.barh(configs, vals, left=left, color=col, label=g)
    left = [a + b for a, b in zip(left, vals)]
for i, v in enumerate(left):
    ax.text(v + 10, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("failed test videos (of 1,448)"); ax.legend(frameon=False, fontsize=8, loc="lower right"); plt.tight_layout()
"""),
        code("""
counts = pd.DataFrame([{"config": t["config"], **{k: t["chapter_counts"][k] for k in
    ["ok_videos", "gold_mean", "pred_mean", "pred_per_gold", "under_pct", "exact_count_pct", "over_pct", "within_1_pct",
     "spearman_pred_vs_gold", "one_sentence_chapter_runs", "single_chapter_for_3plus_gold"]}} for t in FA["test"]]).set_index("config")
display(counts.round(2))
length = pd.DataFrame({t["config"]: {(b, m): t["by_windows"][b][m] for b in ["1 window", "2 windows", "3+ windows"]
                                     for m in ["exact_f1", "failed_pct", "pred_per_gold"]} for t in FA["test"]}).T
display(length.round(3))
"""),
        code("""
for t in FA["test"]:
    if t["failed_videos"] >= 10:
        cause, n = next(iter(t["window_causes"].items()))
        print(f"{t['config']}: most common window failure = {cause} ({n} windows)\\n   {t['examples'].get(cause, '')[:160]!r}")
"""),
        md("""
## Conclusions

* **Best system: E2 (T5Gemma 2B-2B) + LoRA**, exact F1 0.500 on test. The smaller encoder-decoder beats the 9B-2B one and every decoder-only setup.
* **LoRA effect depends on architecture.** It is essential for the encoder-decoders, whose zero-shot output mostly fails the format. Answer-only LoRA *hurts* the reasoning decoder-only models (M1 −0.038, M2 −0.108) because it suppresses their native thinking.
* **RFT on the model's own reasoning** repairs that; with 1,306 windows, M1 reaches 0.440 on dev, above zero-shot (dev only).
* **Failures and counts:** zero-shot encoder-decoders fail on format (fences, chat text, runaway lists); answer-only LoRA adds fences and invalid starts to M1/M2. Zero-shot M1/M2 over-segment (1.25×, 1.48×), mostly on multi-window videos.
* **Limitations:** pilot-scale adapters (1,000 videos, one pass, one seed); M1's adapter covers only 8 of 32 layers; multi-window videos remain much harder.
"""),
    ])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data_notebook()
    window_notebook()
    metrics_notebook()
    for i, (alias, slug) in enumerate([("E1", "t5gemma_9b_2b"), ("E2", "t5gemma_2b_2b"), ("M1", "qwen3_5_9b"), ("M2", "gemma4_12b")], start=4):
        backbone_notebook(i, alias, slug)
    rft_notebook()
    summary_notebook()
