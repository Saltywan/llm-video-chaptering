# LLM video chaptering

This project segments a YouTube transcript into chapters and gives each chapter a title. It compares four open backbones, each run **zero-shot** and with **LoRA fine-tuning**. The dataset is [YTSeg](https://huggingface.co/datasets/retkowski/ytseg) (`text` configuration, revision `4ee1c1f`).

| Alias | Backbone | Type | Test exact F1, zero-shot → LoRA |
|---|---|---|---|
| E1 | `google/t5gemma-9b-2b-ul2-it` | encoder-decoder | 0.003 → 0.484 |
| E2 | `google/t5gemma-2b-2b-ul2-it` | encoder-decoder | 0.149 → **0.500** |
| M1 | `Qwen/Qwen3.5-9B` | decoder-only, native thinking | 0.403 → 0.365 |
| M2 | `google/gemma-4-12B-it` | decoder-only, native thinking | 0.451 → 0.344 |

The test set has 1,448 videos, each configuration ran once after the freeze in `configs/frozen_test.json`, and every difference comes with a paired bootstrap interval. A follow-up, rejection-sampling fine-tuning on each model's own reasoning (RFT), was evaluated on dev only. With it, M1 reaches 0.440 against 0.420 zero-shot.

## Start here

| Notebook | Experiment |
|---|---|
| [01 Data loading and audit](notebooks/01_data_loading_and_audit.ipynb) | YTSeg → canonical records, duplicate/overlap audit, deduplicated variant, dev/pilot subsets |
| [02 Windows and prompt](notebooks/02_windowing_and_prompt.ipynb) | 8,192-token owned-core windows, prompt, masked training targets, stitching |
| [03 Evaluation metrics](notebooks/03_evaluation_metrics.ipynb) | Exact/±2 F1, Pk, WindowDiff, boundary similarity, title ROUGE-L/BERTScore, paired bootstrap |
| [04 E1 T5Gemma 9B-2B](notebooks/04_E1_t5gemma_9b_2b.ipynb) · [05 E2 T5Gemma 2B-2B](notebooks/05_E2_t5gemma_2b_2b.ipynb) · [06 M1 Qwen3.5-9B](notebooks/06_M1_qwen3_5_9b.ipynb) · [07 M2 Gemma 4 12B](notebooks/07_M2_gemma4_12b.ipynb) | Per backbone: data → zero-shot → LoRA training → LoRA inference → dev/test evaluation → examples and errors |
| [08 RFT for M1/M2](notebooks/08_RFT_M1_M2.ipynb) | Sample ×4 → select best (F1 ≥ 0.5) → train on own reasoning → dev evaluation |
| [09 Results summary](notebooks/09_results_summary.ipynb) | All eight test configurations, LoRA effects, post-hoc comparisons, compute and time |

The notebooks are committed with their outputs. They were executed against the full run outputs. GPU steps print the exact command that produced each saved result and run only with `RUN_GPU = True`. The full write-up is [reports/results.md](reports/results.md), and [reports/results.html](reports/results.html) is the rendered page.

## Layout

```
src/chaptering/   data loading, windowing, prompts, supervision masks, evaluation, reporting (stdlib core)
scripts/          pipeline entry points: audit, windows, training, inference (HF / vLLM), evaluation, freeze, report
tests/            unit tests (python -m unittest discover -s tests, with PYTHONPATH=src)
configs/          pinned model revisions and run settings (models.json); frozen_test.json = the frozen test configuration
notebooks/        one notebook per experiment (built by scripts/build_notebooks.py)
results/          small run outputs: predictions, summaries, run configs, aggregate metrics, training logs
reports/          compiled results (Markdown, JSON, HTML) with a method appendix
```

## Reproducing

1. **Environment:**
   - Python 3.11+ with `torch`, `transformers`, `peft`, `datasets`, `safetensors`.
   - `vllm` 0.33 for M1/M2 inference.
   - `rouge-score`, `segeval` and `bert-score` for the full metric set.
   - Gemma models require accepting their licenses on Hugging Face.
2. **Data and windows:**
   - Run `scripts/audit_data.py`, then `scripts/resolve_duplicates.py`, then `scripts/prepare_windows.py`; run `scripts/prepare_test.py` only after freezing.
   - Transcripts are not redistributed here; they are rebuilt from the pinned dataset revision.
3. **Training and inference:** see the commands printed in notebooks 04–08 (`scripts/train_lora.py`, `scripts/infer_seq2seq_batched.py`, `scripts/infer_vllm_pair.py`). GPU selection is an allow-list of device UUIDs in `src/chaptering/modeling.py` and `scripts/pilot_seq2seq.py`; replace it with your own device UUIDs.
4. **Scoring:** `scripts/evaluate_predictions.py --references … --predictions … [--compare-predictions …]`.

Endpoint URLs in `configs/models.json` are placeholders (`${CHAPTERING_ENDPOINT_URL}`); API keys are only read from environment variables or stdin. The frozen configuration records SHA-256 hashes of the files exactly as they ran; this copy has cleaned paths and file names, so `scripts/freeze.py --verify` reports a mismatch here by design.

## Not in this repository

* **LoRA adapters (~540 MB for 8 adapters):** kept out of git; distributed separately as a GitHub Release asset or a Hugging Face model repository.
* **Full run outputs (~2.6 GB):** transcripts and windows, raw generations including thinking, and per-video metric details. Per-video scores can be recomputed from `results/**/predictions.jsonl` and the rebuilt records.
