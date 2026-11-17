#!/usr/bin/env python
"""Compile every result into one Markdown report plus a JSON table (CPU, read-only).

Missing inputs are shown as "pending" so the report can be compiled at any time.
Run from the project root on the server:
  python scripts/compile_report.py --output reports/results
"""

import argparse
import collections
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
S5, TEST, RFT, RFT2 = A / "stage5-v1", A / "heldout-test-v1", A / "rft-v1", A / "rft-v2"
BACKBONES = [("E1", "T5Gemma 9B-2B (encoder-decoder)"), ("E2", "T5Gemma 2B-2B (encoder-decoder)"),
             ("M1", "Qwen3.5-9B (causal, native thinking)"), ("M2", "Gemma 4 12B (causal, native thinking)")]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def metrics(block):
    """Flatten one evaluate_predictions arm (top level or comparison.right_arm)."""
    if not block:
        return None
    agg = block["canonical"]["aggregate"]
    extra = block.get("additional_metrics", {})
    titles = extra.get("titles", {})
    rouge = titles.get("rouge_l", {}).get("concatenated_document_macro_all_requests", {}).get("f1")
    bert = titles.get("semantic", {}).get("reference_chapter_mean_with_unmatched_zero", {}).get("f1")
    return {"videos": agg["requested_videos"], "failed": agg["failed_videos"],
            "exact_f1": agg["macro_exact_f1"], "tolerant_f1": agg["macro_tolerant_f1"],
            "pk": agg["macro_pk"], "window_diff": agg["macro_window_diff"],
            "boundary_similarity": extra.get("boundary_similarity", {}).get("macro_all_requests"),
            "title_rouge_l": rouge, "title_bertscore": bert,
            "pred_per_ref_chapters": (agg["title_span_alignment"]["predicted_chapters_successful_requests"]
                                      / agg["title_span_alignment"]["reference_chapters_all_requests"]
                                      if agg.get("title_span_alignment") else None)}


def paired(path):
    """(FT metrics, ZS metrics, bootstrap) from a paired file where FT is left and ZS is right."""
    data = load(path)
    if not data:
        return None, None, None
    return metrics(data), metrics(data["comparison"]["right_arm"]), data["comparison"]["paired_exact_f1_bootstrap"]


def fmt(value, digits=3):
    return "pending" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def ci(boot):
    if not boot:
        return "pending"
    low, high = boot["confidence_interval"]
    return f"{boot['mean_difference']:+.3f} [{low:+.3f}, {high:+.3f}]"


METRIC_COLUMNS = [("exact_f1", "Exact F1"), ("tolerant_f1", "±2 F1"), ("pk", "Pk ↓"), ("window_diff", "WD ↓"),
                  ("boundary_similarity", "B-sim"), ("title_rouge_l", "Title R-L"),
                  ("title_bertscore", "Title BERTScore"), ("failed", "Failed videos")]


def table(rows):
    lines = ["| Config | " + " | ".join(name for _, name in METRIC_COLUMNS) + " |",
             "|---|" + "---|" * len(METRIC_COLUMNS)]
    for label, m in rows:
        cells = [fmt(m.get(key) if m else None) for key, _ in METRIC_COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def status_lines(path):
    try:
        return [line.split("\t") for line in Path(path).read_text().splitlines() if line.strip()]
    except OSError:
        return []


def timing_rows():
    """Wall-clock generation time of every inference run, from each run's own summary.json."""
    vllm, hf = "vLLM, max 100 concurrent", "HF batched greedy"
    specs = []  # (stage, model, arm, summary path, load-time path or None, device, engine)
    for alias in ["E1", "E2"]:
        for arm in ["ZS", "FT"]:
            specs.append(("Dev (200 videos)", alias, arm, S5 / f"{alias}-dev-{arm}/summary.json", None, "GPU0", hf))
    for alias in ["M1", "M2"]:
        for arm in ["ZS", "FT"]:
            specs.append(("Dev (200 videos)", alias, arm, S5 / f"{alias}-dev/{arm}/summary.json", S5 / f"{alias}-dev/summary.json", "GPU0", vllm))
        specs.append(("Dev (200 videos)", alias, "RFT v1", RFT / f"{alias}-dev/FT/summary.json", RFT / f"{alias}-dev/summary.json", "GPU2", vllm))
        specs.append(("Dev (200 videos)", alias, "RFT v2", RFT2 / f"{alias}-dev/FT/summary.json", RFT2 / f"{alias}-dev/summary.json",
                      "GPU0" if alias == "M1" else "GPU2", vllm))
    for alias, arm, sub, device in [("E1", "ZS", "E1-ZS-gpu2", "GPU2"), ("E1", "FT", "E1-FT", "GPU0"),
                                    ("E2", "ZS", "E2-ZS", "GPU0"), ("E2", "FT", "E2-FT", "GPU0")]:
        specs.append(("Test (1,448 videos)", alias, arm, TEST / f"{sub}/summary.json", None, device, hf))
    for alias in ["M1", "M2"]:
        for arm in ["ZS", "FT"]:
            specs.append(("Test (1,448 videos)", alias, arm, TEST / f"{alias}/{arm}/summary.json", TEST / f"{alias}/summary.json", "GPU0", vllm))
    for alias in ["M1", "M2"]:
        for seed in [1001, 2002, 3003, 4004]:
            specs.append(("RFT v1 sampling (300 train windows)", alias, f"seed {seed}", RFT / f"samples/{alias}-seed{seed}/summary.json",
                          None, "GPU1 service", "vLLM endpoint, 16 concurrent; 4 seeds in parallel"))
    for alias in ["M1", "M2"]:
        for seed in [1001, 2002, 3003, 4004]:
            specs.append(("RFT v2 sampling (1,006 train windows)", alias, f"seed {seed}", RFT2 / f"samples/{alias}-seed{seed}/ZS/summary.json",
                          RFT2 / f"samples/{alias}-seed{seed}/summary.json", "GPU0" if alias == "M1" else "GPU2", "vLLM, max 256 concurrent"))
    rows = []
    for stage, alias, arm, path, load_path, device, engine in specs:
        s = load(path) or {}
        seconds = s.get("generation_seconds", s.get("this_invocation_seconds"))
        load_seconds = s.get("load_seconds", (load(load_path) or {}).get("loaded_seconds") if load_path else None)
        if "endpoint" in engine:
            load_seconds = "n/a"  # long-running shared service; no load inside the run
        counts = s.get("status_counts") or {}
        videos, windows, tokens = s.get("requested_videos"), s.get("requested_windows"), s.get("completion_tokens")
        rows.append({"stage": stage, "model": alias, "arm": arm, "device": device, "engine": engine,
                     "videos": videos, "windows": windows, "generation_seconds": seconds, "load_seconds": load_seconds,
                     "seconds_per_video": seconds / videos if seconds and videos else None,
                     "output_tokens": tokens, "output_tokens_per_second": tokens / seconds if tokens and seconds else None,
                     "failed_videos": sum(v for k, v in counts.items() if k != "ok") if counts else None,
                     "source": str(path.relative_to(ROOT))})
    return rows


def method_markdown(out):
    """Appendix: windows, metric definitions and training parameters (same text as the HTML page)."""
    from render_report_html import METRIC_DEFS  # scripts/ is on sys.path when run as a script
    md = ["", "## Appendix A. Method", "",
          "**Task and output.** Input is a transcript split into numbered sentences. The model returns one JSON object "
          '`{"chapters":[{"start":<sentence index>,"title":"..."}]}`. Every system gets the same prompt (`owned-window-outline-v1`), '
          "and the output is parsed strictly.", "",
          "**Windows.** At most 8,192 input tokens per window, counted with the rendered prompt under all four tokenizers, so the "
          "windows are identical for every backbone. A longer transcript is cut into consecutive windows. Each window owns a "
          "disjoint block of sentences [A,B), as large as fits, and also shows up to 16 sentences of context on each side, [L,R). "
          "The model must list L first as a processing anchor. Only starts inside each window's owned block are kept, and the "
          "blocks are concatenated in order. Stitching never merges, sorts or invents anything. Any failed window fails the "
          "video. No transcript is truncated.", "",
          "| Split | Videos | Windows | 1 window | 2 | 3 | 4+ | Max windows | Median sentences | Max sentences |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, w in (out.get("windows") or {}).items():
        d = w["videos_by_window_count"]
        md.append(f"| {name} | {w['videos']} | {w['windows']} | {d['1']} | {d['2']} | {d['3']} | {d['4']} | "
                  f"{w['max_windows_per_video']} | {w['median_sentences']:.0f} | {w['max_sentences']} |")
    md += ["", "**Metrics.** Sentence space, macro over every requested video; failed videos stay in the average.", ""]
    md += [f"- **{term}.** {text}" for term, text in METRIC_DEFS]
    md += ["", "**Training parameters (all LoRA runs).**", "",
           "- Adapter: LoRA rank 16, alpha 32, dropout 0.05, on attention q/k/v/o; base frozen; bfloat16. M1 (Qwen3.5-9B, "
           "hybrid) has full attention only in every 4th layer, so its adapter covers 8 of 32 layers (3.9 M parameters); "
           "the 24 linear-attention layers are not adapted.",
           "- Optimizer: AdamW, LR 2e-4, cosine decay, gradient clipping 1.0.",
           "- Batching: one window per forward pass, gradient accumulation 8 (effective batch 8 windows), gradient checkpointing.",
           "- Schedule: one pass; warmup 10 steps (direct) or 3 (RFT); seed 393.",
           "- Direct LoRA data: 1,000 training videos, 1,306 windows, gold chapter JSON target, with titles cut by a window edge masked out of the loss.",
           "- Direct LoRA loss: E1/E2 decoder cross-entropy on the JSON; M1/M2 causal loss on final JSON plus turn-end only. "
           "Prompt unscored; no reasoning text written or supervised.",
           "- RFT LoRA loss: the model's own selected sample (thought + JSON + turn end), fully supervised after the unscored prompt.",
           "- Inference: M1/M2 vLLM, native thinking unforced, temperature 1.0, top-p 0.95, top-k 20 (M1, presence penalty 1.5) or 64 (M2), "
           "per-window seed, up to 32,768 new tokens. E1/E2 greedy, batched by length tier, up to 4,096 new tokens. ZS and LoRA share path and settings.",
           "", "| Recipe | Model | Windows | Optimizer steps | Warmup | Trainable params | Tokens processed | Loss first → last (mean) | Minutes | Peak GiB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out.get("training_runs") or []:
        if r.get("training_seconds") is None:
            md.append(f"| {r['recipe']} | {r.get('alias')} | pending | | | | | | | |")
            continue
        md.append(f"| {r['recipe']} | {r['alias']} | {r['actual_steps']} | {r['planned_optimizer_steps']} | {r['warmup_steps']} | "
                  f"{r['trainable_parameters'] / 1e6:.1f} M | {r['teacher_forcing_tokens'] / 1e6:.2f} M | "
                  f"{r['first_loss']:.2f} → {r['last_loss']:.2f} ({r['mean_loss']:.2f}) | {r['training_seconds'] / 60:.0f} | "
                  f"{r['peak_allocated_bytes'] / 2**30:.0f} |")
    md += ["", "**RFT recipe (`rft_self_samples_v1`).** (1) Sample each training window 4 times from the base model (seeds "
           "1001-4004, the zero-shot sampling settings). (2) Score each finished, strictly valid sample by exact F1 of its starts "
           "inside the owned block, anchor excluded, against gold. (3) Keep the best sample if F1 ≥ 0.5. (4) One-pass LoRA on "
           "prompt → own thought → own JSON → turn end. v1: 300 windows via the GPU1 services. v2: all 1,306 pilot windows "
           "(the other 1,006 sampled locally with identical settings); same recipe and hyperparameters.", ""]
    md += ["", "## Appendix B. Compute and time (one NVIDIA B200 per run)", "",
           "**Training** (one pass; seconds per window = training time / windows):", "",
           "| Recipe | Model | Device | Windows | Minutes | Seconds / window | Peak GiB |", "|---|---|---|---|---|---|---|"]
    for r in out.get("training_runs") or []:
        device = "GPU2" if r["recipe"] == "RFT v1 LoRA" or (r["recipe"] == "RFT v2 LoRA" and r.get("alias") == "M2") else "GPU0"
        if r.get("training_seconds") is None:
            md.append(f"| {r['recipe']} | {r.get('alias')} | {device} | pending | | | |")
        else:
            md.append(f"| {r['recipe']} | {r['alias']} | {device} | {r['actual_steps']} | {r['training_seconds'] / 60:.1f} | "
                      f"{r['training_seconds'] / r['actual_steps']:.2f} | {r['peak_allocated_bytes'] / 2**30:.0f} |")
    md += ["", "**Inference** (generation wall-clock; model load excluded and listed separately):", "",
           "| Stage | Model | Arm | Device | Engine | Videos | Windows | Generation min | Seconds / video | Output tokens | Tokens / s | Load s | Failed videos |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in out.get("timing") or []:
        g = t["generation_seconds"]
        md.append(f"| {t['stage']} | {t['model']} | {t['arm']} | {t['device']} | {t['engine']} | {fmt(t['videos'])} | "
                  f"{fmt(t['windows'])} | {fmt(g and g / 60, 1)} | {fmt(t['seconds_per_video'], 2)} | {fmt(t['output_tokens'])} | "
                  f"{fmt(t['output_tokens_per_second'], 0)} | {fmt(t['load_seconds'], 0)} | {fmt(t['failed_videos'])} |")
    posthoc = out.get("posthoc_test_comparisons") or []
    if posthoc:
        md += ["**Post-hoc cross-backbone test comparisons** (paired bootstrap, 10,000 resamples):", ""]
        md += [f"- {p['left']} − {p['right']}: {p['mean_difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
               for p in posthoc]
    return md


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="path stem; writes .md and .json")
    args = parser.parse_args()
    out = {"test": {}, "dev": {}, "rft": {}, "validation_endpoint": {}}
    out["test_pack"] = None
    md = ["# Results: YouTube chaptering with four backbones, zero-shot and LoRA", ""]
    freeze = load(ROOT / "configs/frozen_test.json") or {}
    test_manifest = load(A / "windows/test-v1/manifest.json") or {}
    out["test_pack"] = {"videos": test_manifest.get("videos"), "windows": test_manifest.get("windows")}
    md += ["Generated by `scripts/compile_report.py` from the server artifacts. Numbers are machine-filled; "
           "items still running show as *pending*.", ""]

    # 1. Final test results.
    md += ["## 1. Held-out test (frozen configurations, run once)", "",
           f"Test set: {test_manifest.get('videos', 'pending')} videos, {test_manifest.get('windows', 'pending')} windows "
           "(retkowski/ytseg test split, pinned revision; 8,192-token window pack identical in settings to validation). "
           "Exact boundary macro F1 is primary; a failed video scores 0. Strict parsing, no repair.", ""]
    rows, effects = [], []
    for alias, name in BACKBONES:
        path = TEST / f"{alias}-test-paired.json"
        ft, zs, boot = paired(path)
        if ft is None:  # fall back to single-arm files while the pair is pending
            ft = None
        out["test"][alias] = {"FT": ft, "ZS": zs, "ft_minus_zs_exact_f1": boot}
        rows += [(f"{alias} zero-shot", zs), (f"{alias} LoRA", ft)]
        effects.append(f"| {alias} | {name} | {fmt(zs and zs['exact_f1'])} | {fmt(ft and ft['exact_f1'])} | {ci(boot)} |")
    md += [table(rows), "", "**LoRA effect on test (paired by video, 10,000 bootstrap resamples):**", "",
           "| Backbone | Model | ZS exact F1 | LoRA exact F1 | LoRA − ZS [95% CI] |", "|---|---|---|---|---|", *effects, ""]

    # 1b. Lenient secondary on test.
    md += ["### Secondary: lenient envelope-v1 reading on test", "",
           "Strips one Markdown code fence or a trailing period around otherwise valid JSON; nothing else is repaired.", ""]
    lenient = []
    for alias, _ in BACKBONES:
        for arm, tag in [("ZS", "zero-shot"), ("FT", "LoRA")]:
            m = metrics(load(TEST / f"envelope-{alias}-{arm}-metrics.json"))
            out["test"].setdefault(alias, {})[f"{arm}_lenient"] = m
            lenient.append((f"{alias} {tag} (lenient)", m))
    md += [table(lenient), ""]

    # 2. Dev pilots.
    md += ["## 2. Development evidence (200 validation videos, 270 windows)", "",
           "LoRA here is the frozen adapter: one pass over 1,000 training videos (1,306 windows), rank 16, alpha 32, "
           "q/k/v/o projections, LR 2e-4 cosine, effective batch 8 windows. Same inference path for both arms.", ""]
    rows, effects = [], []
    for alias, name in BACKBONES:
        ft, zs, boot = paired(S5 / f"{alias}-dev-paired.json")
        out["dev"][alias] = {"FT": ft, "ZS": zs, "ft_minus_zs_exact_f1": boot}
        rows += [(f"{alias} zero-shot", zs), (f"{alias} LoRA", ft)]
        effects.append(f"| {alias} | {fmt(zs and zs['exact_f1'])} | {fmt(ft and ft['exact_f1'])} | {ci(boot)} | "
                       f"{fmt(zs and zs['pred_per_ref_chapters'], 2)} / {fmt(ft and ft['pred_per_ref_chapters'], 2)} |")
    md += [table(rows), "", "| Backbone | ZS | LoRA | LoRA − ZS [95% CI] | predicted/reference chapters ZS / LoRA |",
           "|---|---|---|---|---|", *effects, ""]
    env = []
    for label in ["dev-M1-zs", "dev-M1-ft", "dev-M2-zs", "dev-M2-ft", "dev-E2-zs", "dev-E2-ft", "dev-E1-zs"]:
        m = metrics(load(A / f"envelope-dev-v1/{label}-metrics.json"))
        env.append((label.replace("dev-", "") + " (lenient)", m))
    md += ["Lenient envelope-v1 on the same dev runs:", "", table(env), ""]

    # Thinking behaviour on dev.
    md += ["Native thinking on dev (median reasoning tokens per window, re-tokenized):", ""]
    think = ["| Model | ZS | LoRA (direct) | RFT |", "|---|---|---|---|"]
    for alias in ["M1", "M2"]:
        cells = []
        for path in [S5 / f"{alias}-dev/ZS/windows.jsonl", S5 / f"{alias}-dev/FT/windows.jsonl",
                     RFT / f"{alias}-dev/FT/windows.jsonl"]:
            try:
                values = sorted((json.loads(line).get("usage") or {}).get("reasoning_tokens") or 0
                                for line in path.read_text().splitlines() if line.strip())
                cells.append(str(values[len(values) // 2]))
            except OSError:
                cells.append("pending")
        think.append(f"| {alias} | " + " | ".join(cells) + " |")
        out.setdefault("dev_thinking_median_tokens", {})[alias] = dict(zip(["ZS", "direct_LoRA", "RFT"], cells))
    md += think + [""]

    # 3. RFT.
    md += ["## 3. Rejection-sampling fine-tuning (RFT) for M1/M2: dev-only follow-up", "",
           "Post-freeze experiment, not part of the eight test configurations. Four sampled completions per "
           "training window (300 windows, 233 videos) from the base models via the GPU1 services. The best "
           "finished, strictly valid sample per window is kept when its owned-start F1 against gold is at least 0.5. "
           "One-pass LoRA trains on the model's own unedited thought and answer.", ""]
    rows = []
    for alias in ["M1", "M2"]:
        summary = load(ROOT / f"artifacts/windows/dev-v2/rft_{alias}_v1.summary.json")
        report = load(RFT / f"{alias}-rft-lora/report.json")
        ft, zs, boot = paired(RFT / f"{alias}-rft-dev-paired.json")
        out["rft"][alias] = {"selection": summary, "RFT": ft, "ZS": zs, "rft_minus_zs_exact_f1": boot,
                             "training_seconds": report and report.get("training_seconds")}
        direct = out["dev"].get(alias, {}).get("FT")
        rows.append(f"| {alias} | {summary['kept_windows'] if summary else 'pending'} | "
                    f"{fmt(summary and summary.get('mean_valid_sample_f1'))} | {fmt(zs and zs['exact_f1'])} | "
                    f"{fmt(direct and direct['exact_f1'])} | {fmt(ft and ft['exact_f1'])} | {ci(boot)} | "
                    f"{fmt(ft and ft['failed'])} |")
    md += ["| Model | Kept windows | Mean sample F1 | Dev ZS | Dev direct LoRA | Dev RFT | RFT − ZS [95% CI] | RFT failed videos |",
           "|---|---|---|---|---|---|---|---|", *rows, ""]

    # 4. Full-validation zero-shot via the GPU1 endpoints.
    md += ["## 4. Full validation zero-shot (1,446 videos, GPU1 vLLM endpoints)", ""]
    windows = collections.Counter(json.loads(line)["window"]["video_id"] for line in
                                  (A / "windows/common-v2/validation_windows.jsonl").read_text().splitlines())
    rows = []
    for alias in ["M1", "M2"]:
        data = load(A / f"endpoint-zs-v1/{alias}-full-metrics.json")
        m = metrics(data)
        buckets = collections.defaultdict(list)
        if data:
            per = data["canonical"]["per_video"]
            for row in (per.values() if isinstance(per, dict) else per):
                buckets[min(windows[row["video_id"]], 3)].append(row["exact"]["f1"])
        by = {k: (len(v), statistics.mean(v)) for k, v in sorted(buckets.items())}
        out["validation_endpoint"][alias] = {"metrics": m, "by_window_count": by}
        rows.append((f"{alias} zero-shot", m))
        md.append(f"- {alias} exact F1 by window count: " + ", ".join(
            f"{'3+' if k == 3 else k} window{'s' if k != 1 else ''}: {v[1]:.3f} (n={v[0]})" for k, v in by.items()))
    md += ["", table(rows), "",
           "On the 200 dev videos (identical windows) the endpoint and the owned GPU0 vLLM agree within noise "
           "(M1 0.412 vs 0.420; M2 0.466 vs 0.459).", ""]

    # 5. Cost.
    md += ["## 5. Compute", "", "| Run | Seconds | Notes |", "|---|---|---|"]
    for alias, stem in [("M1", "M1-direct-safe"), ("M2", "M2-direct-safe"), ("E2", "E2-native-safe"), ("E1", "E1-native-safe")]:
        report = load(S5 / f"{stem}/report.json")
        md.append(f"| {alias} LoRA training (1,306 windows) | {fmt(report and report.get('training_seconds'), 0)} | "
                  "one window per step, gradient checkpointing, accumulation 8 |")
        out.setdefault("compute", []).append([f"{alias} LoRA training (1,306 windows)", report and report.get("training_seconds")])
    for alias in ["M1", "M2"]:
        summary = load(TEST / f"{alias}/ZS/summary.json"), load(TEST / f"{alias}/FT/summary.json")
        out.setdefault("compute", []).append([f"{alias} test generation ZS / LoRA (vLLM)",
                                              [summary[0] and summary[0].get("generation_seconds"),
                                               summary[1] and summary[1].get("generation_seconds")]])
        md.append(f"| {alias} test generation ZS / LoRA | {fmt(summary[0] and summary[0].get('generation_seconds'), 0)} / "
                  f"{fmt(summary[1] and summary[1].get('generation_seconds'), 0)} | vLLM, GPU0 |")
    for alias, zs_dir in [("E2", "E2-ZS"), ("E1", "E1-ZS-gpu2")]:
        zs, ft = load(TEST / f"{zs_dir}/summary.json"), load(TEST / f"{alias}-FT/summary.json")
        out.setdefault("compute", []).append([f"{alias} test generation ZS / LoRA (HF batched greedy)",
                                              [zs and zs.get("generation_seconds"), ft and ft.get("generation_seconds")]])
        md.append(f"| {alias} test generation ZS / LoRA | {fmt(zs and zs.get('generation_seconds'), 0)} / "
                  f"{fmt(ft and ft.get('generation_seconds'), 0)} | HF batched greedy |")
    md.append("")

    # 6. Deviations and caveats.
    md += ["## 6. Deviations, caveats and audit trail", "",
           "- **Adapters are pilot-scale.** Full-data LoRA (16,377 training videos) did not fit the deadline; "
           "every LoRA result uses 1,000 training videos, one pass, single seed.",
           "- **Direct LoRA suppresses thinking in M1/M2.** Thinking stays enabled and unforced at inference, but the "
           "adapted models think less (M1) or not at all (M2), and under-segment. RFT (section 3) tests a fix.",
           "- **E1 zero-shot ran on physical GPU2** (authorized 2026-11-17) with the frozen driver and settings; all "
           "other test runs used GPU0. GPU0's own E1-ZS step was intentionally blocked by a placeholder directory and "
           "appears as a failure in its issue log.",
           "- **Encoder-decoder inference** uses the batched greedy driver for both arms; it is statistically "
           "equivalent but not token-identical to sequential decoding (55% identical raw text on 816 windows).",
           "- **Window packing:** transcripts longer than 8,192 tokens are split into windows with 16-sentence context "
           "and stitched; multi-window videos remain much harder (section 4). No transcript is truncated.",
           "- **Data:** deduplicated/quarantined variant (train 16,377; validation 1,446); test 1,448 videos, audited "
           "unique and disjoint from train/validation; test data was first read after the freeze.", ""]
    # 7. Training runs, window statistics, RFT v2 and post-hoc comparisons (structured for the HTML page).
    keep = ["alias", "final_context", "rank", "lora_alpha", "lora_dropout", "target_modules", "learning_rate",
            "lr_schedule", "warmup_steps", "grad_accum", "planned_optimizer_steps", "actual_steps",
            "actual_training_videos", "trainable_parameters", "teacher_forcing_tokens", "first_loss", "last_loss",
            "mean_loss", "training_seconds", "peak_allocated_bytes", "seed", "gradient_checkpointing", "attention"]
    runs = [("Direct LoRA", S5 / f"{stem}/report.json") for stem in
            ["E1-native-safe", "E2-native-safe", "M1-direct-safe", "M2-direct-safe"]]
    runs += [("RFT v1 LoRA", RFT / f"{alias}-rft-lora/report.json") for alias in ["M1", "M2"]]
    runs += [("RFT v2 LoRA", RFT2 / f"{alias}-rft-lora/report.json") for alias in ["M1", "M2"]]
    out["training_runs"] = []
    for recipe, path in runs:
        report = load(path)
        out["training_runs"].append({"recipe": recipe, "path": str(path.relative_to(ROOT)),
                                     **({k: report.get(k) for k in keep} if report else {"alias": path.parent.name[:2]})})
    out["windows"] = {}
    for name, path in [("Pilot train (LoRA)", A / "windows/dev-v2/train_windows.jsonl"),
                       ("Dev (200 validation videos)", A / "windows/dev-v2/validation_windows.jsonl"),
                       ("Test", A / "windows/test-v1/test_windows.jsonl")]:
        per_video, longest = collections.Counter(), {}
        for line in path.read_text().splitlines():
            window = json.loads(line)["window"]
            per_video[window["video_id"]] += 1
            longest[window["video_id"]] = max(longest.get(window["video_id"], 0), window["sentence_count"])
        dist = collections.Counter(min(count, 4) for count in per_video.values())
        out["windows"][name] = {"videos": len(per_video), "windows": sum(per_video.values()),
                                "videos_by_window_count": {str(k): dist.get(k, 0) for k in (1, 2, 3, 4)},
                                "max_windows_per_video": max(per_video.values()),
                                "median_sentences": statistics.median(longest.values()),
                                "max_sentences": max(longest.values())}
    out["rft_v2"] = {}
    for alias in ["M1", "M2"]:
        ft, zs, boot = paired(RFT2 / f"{alias}-rft-dev-paired.json")
        _, v1_ft, v1_boot = paired(RFT2 / f"{alias}-v2-vs-v1-paired.json")
        think = None
        try:
            values = sorted((json.loads(line).get("usage") or {}).get("reasoning_tokens") or 0 for line in
                            (RFT2 / f"{alias}-dev/FT/windows.jsonl").read_text().splitlines() if line.strip())
            think = values[len(values) // 2]
        except OSError:
            pass
        out["rft_v2"][alias] = {"selection": load(ROOT / f"artifacts/windows/dev-v2/rft_{alias}_v2.summary.json"),
                                "RFT": ft, "ZS": zs, "rft_minus_zs_exact_f1": boot, "v1": v1_ft,
                                "v2_minus_v1_exact_f1": v1_boot, "median_thinking_tokens": think,
                                "status": (status_lines(RFT2 / f"status-{alias}.tsv") or [[None]])[-1]}
    out["posthoc_test_comparisons"] = load(TEST / "posthoc/cross_backbone_bootstrap.json") or []
    out["timing"] = timing_rows()
    md7 = ["", "## 7. RFT v2 (same recipe, all 1,306 pilot training windows; dev only)", "",
           "| Model | Kept windows | Dev RFT v2 | RFT v2 − ZS [95% CI] | RFT v2 − RFT v1 [95% CI] | Failed | Median thinking |",
           "|---|---|---|---|---|---|---|"]
    for alias, r in out["rft_v2"].items():
        sel, ft = r["selection"], r["RFT"]
        md7.append(f"| {alias} | {sel['kept_windows'] if sel else 'pending'} | {fmt(ft and ft['exact_f1'])} | "
                   f"{ci(r['rft_minus_zs_exact_f1'])} | {ci(r['v2_minus_v1_exact_f1'])} | {fmt(ft and ft['failed'])} | "
                   f"{fmt(r['median_thinking_tokens'])} |")

    out["amendments"] = freeze.get("amendments", [])
    out["issues"] = []
    for item in freeze.get("amendments", []):
        md.append(f"- Freeze amendment {item['time_utc']}: `{item['file']}`. {item['reason']}")
    for path in [TEST / "issues.txt", RFT / "issues.txt"]:
        try:
            for line in path.read_text().splitlines():
                md.append(f"- Logged issue ({path.parent.name}): {line}")
                out["issues"].append(f"{path.parent.name}: {line}")
        except OSError:
            pass
    md += ["", "### Queue status at compile time", ""]
    for name, path in [("GPU0 test queue", TEST / "status.tsv"), ("GPU2 queue", RFT / "status.tsv")]:
        lines = status_lines(path)
        md.append(f"- {name}: " + (" | ".join(lines[-1]) if lines else "no status"))
        out.setdefault("queue_status", {})[name] = lines[-1] if lines else None
    md += md7 + method_markdown(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
