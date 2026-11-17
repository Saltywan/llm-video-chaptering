# LLM video chaptering

This project segments a YouTube transcript into chapters and gives each chapter a title. It compares four open backbones, each run **zero-shot** and with **LoRA fine-tuning**. The dataset is [YTSeg](https://huggingface.co/datasets/retkowski/ytseg) (`text` configuration, revision `4ee1c1f`).

| Alias | Backbone | Type | Test exact F1, zero-shot → LoRA |
|---|---|---|---|
| E1 | `google/t5gemma-9b-2b-ul2-it` | encoder-decoder | 0.003 → 0.484 |
| E2 | `google/t5gemma-2b-2b-ul2-it` | encoder-decoder | 0.149 → **0.500** |
| M1 | `Qwen/Qwen3.5-9B` | decoder-only, native thinking | 0.403 → 0.365 |
| M2 | `google/gemma-4-12B-it` | decoder-only, native thinking | 0.451 → 0.344 |

The test set has 1,448 videos, each configuration ran once after the freeze in `configs/frozen_test.json`, and every difference comes with a paired bootstrap interval. The LoRA adapters are pilot-scale: 1,000 training videos, one pass.

## Comparison with published results

The test split here is the official YTSeg test split (1,448 videos), and boundaries are counted at sentence level on the dataset's own sentences. The original YTSeg paper uses that same setup. It reports sentence-level boundary precision, recall and F1, which we take to be pooled over all test boundaries (micro).

The headline numbers above are **macro** averages over videos, so the table below recomputes the same predictions as **micro** scores. Every gold internal boundary counts; failed videos contribute only misses.

| System | Approach | P | R | F1 | Joint titles |
|---|---|---|---|---|---|
| MiniSeg, Wiki-727K → YTSeg (Retkowski & Waibel, 2024; best published) | supervised segmentation model | 48.30 | 43.56 | 45.81 | no |
| MiniSeg, YTSeg only (same paper) | supervised segmentation model | 45.44 | 41.48 | 43.37 | no |
| **E2 T5Gemma 2B-2B + LoRA (ours)** | generative, JSON outline | **53.15** | **50.87** | **51.99** | yes |
| E1 T5Gemma 9B-2B + LoRA (ours) | generative, JSON outline | 52.01 | 48.89 | 50.40 | yes |
| M2 Gemma 4 12B zero-shot (ours) | generative, native thinking | 35.18 | 54.03 | 42.61 | yes |
| M1 Qwen3.5-9B zero-shot (ours) | generative, native thinking | 35.32 | 45.07 | 39.61 | yes |

Caveats:
* **Metric details.** The published paper does not spell out every detail: micro vs macro averaging, and the Pk window. The comparison is therefore indicative, not a controlled re-evaluation.
* **Model size.** Our models are much larger than MiniSeg, and they also generate the titles.
* **Titles are not comparable.** The paper scores titles given the gold segments (BART: ROUGE-L 42.45). Our title scores come from end-to-end predicted chapters.
* **Different protocol in later work.** The 2026 follow-up, [*Beyond Transcripts*](https://aclanthology.org/2026.acl-long.396/) (Retkowski et al., ACL 2026), scores on a 6-second time grid, so its numbers can't be compared with ours directly. Its results are listed below.

**2026 follow-up results (F1@6s, B@6s, Pk@6s on a 6-second time grid):**

| System (Retkowski et al., 2026) | Input | F1@6s | B@6s | Pk@6s ↓ |
|---|---|---|---|---|
| AudioSeg, Whisper Large encoder (best overall) | audio | 45.52 | 36.17 | 28.89 |
| Qwen3-Omni, in-context learning (videos < 30 min only) | audio + text | 41.30 | 35.22 | 33.00 |
| MiniSeg + acoustic features | text + audio features | 40.30 | 33.48 | 30.35 |
| MiniSeg, trained on reference + ASR transcripts (best text-only) | text | 40.01 | 33.24 | 29.84 |
| LLaMA 3.1 8B, zero-shot with constrained decoding | text | 25.92 | 20.69 | 39.98 |

* **Why there's no row for us.** Placing our predictions on that grid needs sentence-level timestamps. YTSeg provides chapter timestamps only, and the paper gets sentence times by forcing an alignment against the audio, which we did not do.
* **No YTSeg v2.** The 2026 paper adds ASR transcripts, duration and speaker annotations, and audio to the same dataset repository. The text and labels used here (revision `4ee1c1f`) are that current version.

## Preliminary: rejection-sampling fine-tuning (dev only)

Answer-only LoRA made the two reasoning models skip most of their thinking and score lower. As a follow-up, we tried **rejection-sampling fine-tuning (RFT)**:
1. The base model answers each training window 4 times.
2. The best answer is kept if its boundary F1 against gold is at least 0.5. Gold only selects; it is never written into the text.
3. LoRA is trained on the model's *own* reasoning plus its answer.

* **v1:** 300 training windows.
* **v2:** all 1,306 windows from the same 1,000 videos; it kept 599 windows for M1 and 573 for M2.

Results on the 200-video dev set:

| Model | Configuration | Dev exact F1 | − zero-shot [95% CI] | Failed videos | Pred / gold chapters | Median thinking tokens |
|---|---|---|---|---|---|---|
| M1 Qwen3.5-9B | zero-shot | 0.420 | | 1 | 1.17 | 7,191 |
| | direct LoRA | 0.379 | −0.042 [−0.072, −0.012] | 10 | 0.99 | 3,327 |
| | RFT v1 | 0.409 | −0.012 [−0.035, +0.012] | 0 | 1.17 | 6,280 |
| | **RFT v2** | **0.440** | +0.019 [−0.003, +0.041] | 4 | 1.09 | 6,071 |
| M2 Gemma 4 12B | zero-shot | 0.459 | | 0 | 1.46 | 4,903 |
| | direct LoRA | 0.346 | −0.114 [−0.144, −0.084] | 8 | 0.79 | 0 |
| | RFT v1 | 0.467 | +0.007 [−0.007, +0.021] | 0 | 1.48 | 4,596 |
| | RFT v2 | 0.464 | +0.005 [−0.010, +0.020] | 0 | 1.47 | 4,630 |

* **Thinking returns.** RFT restores native thinking and removes the damage done by direct LoRA.
* **M1 gains with more data.** RFT v2 beats RFT v1 by +0.031 [+0.010, +0.052] and trends above zero-shot.
* **M2 does not gain.** It stays at zero-shot level; v2 − v1 is −0.002 [−0.018, +0.013].
* **Preliminary.** RFT was run after the freeze and never on the test set, so these numbers are not comparable to the test table above.

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

## Licenses and attribution

* **Code:** [MIT](LICENSE). This covers `src/`, `scripts/`, `tests/`, `configs/` and the code cells of the notebooks.
* **Dataset:** [YTSeg](https://huggingface.co/datasets/retkowski/ytseg) by Retkowski & Waibel is licensed **CC BY-NC-SA 4.0**. It was introduced in *From Text Segmentation to Smart Chaptering: A Novel Benchmark for Structuring Video Transcriptions* (EACL 2024). This repository does not redistribute the transcripts. Some material here is derived from the dataset:
  * notebook outputs that quote transcript sentences and gold chapter titles;
  * the model predictions and metrics in `results/` and `reports/`.

  That material is shared under the same **CC BY-NC-SA 4.0** terms, for non-commercial use only.
* **Models (not redistributed here):**

  | Model | License |
  |---|---|
  | T5Gemma 9B-2B / 2B-2B | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (gated) |
  | Qwen3.5-9B | Apache-2.0 |
  | Gemma 4 12B | [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |
  | RoBERTa-large (used for BERTScore) | MIT |

  LoRA adapters are derivatives of their base models. If the T5Gemma adapters are published, they must carry the Gemma Terms of Use and its use restrictions.
* **Libraries:** used as installed dependencies, not vendored: PyTorch, Transformers, PEFT, vLLM, datasets, rouge-score, segeval and bert-score.
