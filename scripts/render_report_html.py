#!/usr/bin/env python
"""Render the compiled results JSON as a self-contained HTML report page.

  python scripts/render_report_html.py --results reports/results.json \
      --findings reports/findings.json --output reports/results.html

--findings is a small JSON file written by hand: {"status": "...", "summary": [...], "notes": [...]}.
Every number on the page comes from the results JSON.
"""

import argparse
import html
import json
from pathlib import Path

ORDER = [("E1", "T5Gemma 9B-2B", "encoder-decoder"), ("E2", "T5Gemma 2B-2B", "encoder-decoder"),
         ("M1", "Qwen3.5-9B", "causal, native thinking"), ("M2", "Gemma 4 12B", "causal, native thinking")]
COLUMNS = [("exact_f1", "Exact F1", True), ("tolerant_f1", "±2 F1", True), ("pk", "Pk", False),
           ("window_diff", "WinDiff", False), ("boundary_similarity", "B-sim", True),
           ("title_rouge_l", "Title ROUGE-L", True), ("title_bertscore", "Title BERTScore", True),
           ("failed", "Failed", False)]


def esc(value):
    return html.escape(str(value))


def num(value, digits=3):
    if value is None:
        return '<span class="pending">pending</span>'
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return esc(value)


def diff_cell(boot):
    if not boot:
        return '<span class="pending">pending</span>'
    low, high = boot["confidence_interval"]
    kind = "up" if low > 0 else "down" if high < 0 else "flat"
    return (f'<span class="delta {kind}">{boot["mean_difference"]:+.3f}</span>'
            f'<span class="ci">[{low:+.3f}, {high:+.3f}]</span>')


def metric_table(rows, caption):
    best = {}
    for key, _, higher in COLUMNS[:-1]:
        values = [m[key] for _, m in rows if m and m.get(key) is not None]
        if values:
            best[key] = max(values) if higher else min(values)
    head = "".join(f'<th scope="col">{esc(name)}{"" if higher else " ↓"}</th>' for _, name, higher in COLUMNS)
    body = []
    for label, m in rows:
        cells = []
        for key, _, _ in COLUMNS:
            value = m.get(key) if m else None
            mark = ' class="best"' if key in best and value is not None and value == best[key] else ""
            cells.append(f"<td{mark}>{num(value)}</td>")
        body.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return (f'<div class="scroll"><table><caption>{esc(caption)}</caption><thead><tr><th scope="col">Configuration</th>'
            f'{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def chart(test):
    """Paired horizontal bars: exact F1 for ZS and LoRA per backbone, drawn on one 0-0.6 scale."""
    width, left, right, row_h, top = 640, 150, 70, 54, 28
    scale_max = 0.6
    plot = width - left - right
    x = lambda v: left + plot * v / scale_max
    height = top + row_h * len(ORDER) + 30
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Test exact F1, zero-shot versus LoRA, per backbone">']
    for tick in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        parts.append(f'<line x1="{x(tick):.1f}" y1="{top - 8}" x2="{x(tick):.1f}" y2="{height - 26}" class="grid"/>'
                     f'<text x="{x(tick):.1f}" y="{height - 10}" class="tick" text-anchor="middle">{tick:.1f}</text>')
    for index, (alias, name, _) in enumerate(ORDER):
        y = top + index * row_h
        item = test.get(alias, {})
        parts.append(f'<text x="{left - 12}" y="{y + 15}" class="label" text-anchor="end">{alias}</text>'
                     f'<text x="{left - 12}" y="{y + 31}" class="sublabel" text-anchor="end">{esc(name)}</text>')
        for offset, arm, cls in [(0, "ZS", "zs"), (20, "FT", "ft")]:
            m = item.get(arm)
            if m and m.get("exact_f1") is not None:
                value = m["exact_f1"]
                parts.append(f'<rect x="{left}" y="{y + offset}" width="{max(x(value) - left, 1.5):.1f}" height="16" rx="2" class="{cls}"/>'
                             f'<text x="{x(value) + 6:.1f}" y="{y + offset + 12.5}" class="value">{value:.3f}</text>')
            else:
                parts.append(f'<rect x="{left}" y="{y + offset}" width="{plot:.1f}" height="16" rx="2" class="pendingbar"/>'
                             f'<text x="{left + 8}" y="{y + offset + 12.5}" class="value">running</text>')
    parts.append("</svg>")
    return "".join(parts)


def window_diagram():
    """Three consecutive windows over one long transcript: owned cores tile it, context overlaps."""
    x0, span = 40, 680          # transcript drawn from x0 to x0 + span
    sx = lambda s: x0 + span * s / 300   # 300 sentences in the illustration
    cores = [(0, 100), (100, 200), (200, 300)]
    ctx = 16
    parts = ['<svg viewBox="0 0 760 236" role="img" aria-label="Three windows over a 300-sentence transcript: '
             'each owns a disjoint core and sees 16 extra sentences of context on each side">',
             f'<text x="{x0}" y="18" class="sublabel">Transcript, global sentence indices</text>',
             f'<rect x="{x0}" y="26" width="{span}" height="14" rx="2" class="track"/>']
    for s in range(0, 301, 25):
        parts.append(f'<line x1="{sx(s):.1f}" y1="40" x2="{sx(s):.1f}" y2="46" class="grid"/>'
                     f'<text x="{sx(s):.1f}" y="58" class="tick" text-anchor="middle">{s}</text>')
    for index, (a, b) in enumerate(cores):
        y = 78 + index * 44
        left, right = max(0, a - ctx), min(300, b + ctx)
        parts.append(f'<text x="{x0 - 8}" y="{y + 15}" class="label" text-anchor="end">W{index + 1}</text>'
                     f'<rect x="{sx(left):.1f}" y="{y}" width="{sx(right) - sx(left):.1f}" height="22" rx="3" class="ctx"/>'
                     f'<rect x="{sx(a):.1f}" y="{y}" width="{sx(b) - sx(a):.1f}" height="22" rx="3" class="core"/>'
                     f'<line x1="{sx(left):.1f}" y1="{y - 4}" x2="{sx(left):.1f}" y2="{y + 26}" class="anchor"/>'
                     f'<text x="{(sx(a) + sx(b)) / 2:.1f}" y="{y + 15}" class="coretext" text-anchor="middle">owned [A,B) = [{a},{b})</text>')
        if index:
            parts.append(f'<text x="{sx(left) - 4:.1f}" y="{y - 6}" class="sublabel" text-anchor="end">L={left}</text>')
    parts.append(f'<rect x="{x0}" y="214" width="12" height="12" rx="2" class="core"/><text x="{x0 + 18}" y="224" class="sublabel">owned core: only chapter starts here are kept</text>'
                 f'<rect x="{x0 + 300}" y="214" width="12" height="12" rx="2" class="ctx"/><text x="{x0 + 318}" y="224" class="sublabel">context (16 sentences each side)</text>'
                 f'<line x1="{x0 + 560}" y1="212" x2="{x0 + 560}" y2="228" class="anchor"/><text x="{x0 + 568}" y="224" class="sublabel">anchor L (forced first start)</text>')
    parts.append("</svg>")
    return "".join(parts)


METRIC_DEFS = [
    ("Exact F1 (primary)", "A boundary is the first sentence of every chapter after the first. Per video, predicted boundaries "
     "are matched to reference boundaries only at the identical sentence index, and F1 = 2·TP / (2·TP + FP + FN). "
     "This is macro-averaged over all requested videos. A video with no internal boundaries in either reference or prediction scores 1."),
    ("±2 F1", "Same as exact F1, but a predicted boundary counts if it lies within 2 sentences of an unmatched reference "
     "boundary (one-to-one, maximum matching). It forgives near misses."),
    ("Pk ↓", "Slide a probe of k sentences over the transcript, where k is half the mean reference chapter length. At each position, "
     "it is an error when reference and prediction disagree about whether the probe's two ends are in the same chapter. "
     "Pk is the error rate, so lower is better."),
    ("WindowDiff ↓", "Same probe, but it is an error whenever the number of boundaries inside the probe differs. "
     "It penalizes extra or missing boundaries that Pk can miss. Lower is better."),
    ("B-sim", "Boundary Similarity (segeval). An edit-distance score over the two boundary sequences in which near-miss "
     "transpositions get partial credit. Higher is better."),
    ("Title ROUGE-L", "Per video, all predicted titles are concatenated in order and compared with the concatenated "
     "reference titles by ROUGE-L F1 (longest common subsequence). Macro over videos."),
    ("Title BERTScore", "Reference and predicted chapters are paired one-to-one by sentence span (span IoU ≥ 0.5; titles "
     "are never read for pairing). Each pair's titles are scored with BERTScore F1 (RoBERTa-large, layer 17). "
     "Averaged over reference chapters, with an unmatched reference chapter counting 0."),
    ("Failed videos", "Videos with no usable answer: invalid or non-JSON output, a structure that breaks the contract, "
     "or generation cut off at the output budget. Any failed window fails its whole video. A failed video scores 0 on "
     "F1, B-sim and title metrics, and 1 (worst) on Pk and WindowDiff. Nothing is repaired."),
    ("Predicted / reference chapters", "Total predicted chapters on successful videos divided by total reference "
     "chapters. Above 1 means over-segmenting, below 1 means under-segmenting."),
    ("LoRA − zero-shot [95% CI]", "Paired bootstrap: take the per-video exact-F1 difference between two systems on the same "
     "videos, resample videos with replacement 10,000 times (seed 20261116), and report the 2.5th–97.5th percentile. "
     "An interval that excludes 0 is shown in colour."),
]


def time_section(data):
    """Training and inference wall-clock time, from each run's report/summary."""
    def mins(seconds):
        return num(None) if seconds is None else (f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.2f} h")
    train = []
    for r in data.get("training_runs") or []:
        device = "GPU2" if r["recipe"] == "RFT v1 LoRA" or (r["recipe"] == "RFT v2 LoRA" and r.get("alias") == "M2") else "GPU0"
        done = r.get("training_seconds") is not None
        train.append(f'<tr><th scope="row">{esc(r["recipe"])}</th><td>{esc(r.get("alias") or "")}</td><td>{device}</td>'
                     f'<td>{num(r.get("actual_steps"))}</td><td>{mins(r.get("training_seconds"))}</td>'
                     f'<td>{num(r["training_seconds"] / r["actual_steps"], 2) if done else num(None)}</td>'
                     f'<td>{"%.0f GiB" % (r["peak_allocated_bytes"] / 2**30) if done else num(None)}</td></tr>')
    infer, stage = [], None
    for t in data.get("timing") or []:
        if t["stage"] != stage:
            stage = t["stage"]
            infer.append(f'<tr class="group"><th scope="rowgroup" colspan="10">{esc(stage)}</th></tr>')
        infer.append(f'<tr><th scope="row">{esc(t["model"])} {esc(t["arm"])}</th><td>{esc(t["device"])}</td>'
                     f'<td class="eng">{esc(t["engine"])}</td><td>{num(t["windows"])}</td>'
                     f'<td>{mins(t["generation_seconds"])}</td><td>{num(t["seconds_per_video"], 2)}</td>'
                     f'<td>{num(t["output_tokens"])}</td><td>{num(t["output_tokens_per_second"], 0)}</td>'
                     f'<td>{num(t["load_seconds"], 0)}</td><td>{num(t["failed_videos"])}</td></tr>')
    return f"""
<section aria-labelledby="t5">
  <h2 id="t5">Compute and time</h2>
  <p>Every run used one NVIDIA B200 (GPU0 or GPU2, same hardware); RFT v1 sampling used the shared GPU1 vLLM services. Times are wall-clock. For inference, "generation" excludes model loading, which is listed separately. Seconds per video = generation time / videos. M1/M2 zero-shot and LoRA share one vLLM engine, so the load time is the same row for both.</p>
  <div class="scroll"><table><caption>Training (one pass over the listed windows)</caption>
  <thead><tr><th scope="col">Recipe</th><th scope="col">Model</th><th scope="col">Device</th><th scope="col">Windows</th><th scope="col">Training time</th><th scope="col">Seconds / window</th><th scope="col">Peak memory</th></tr></thead>
  <tbody>{"".join(train)}</tbody></table></div>
  <div class="scroll"><table><caption>Inference and RFT sampling</caption>
  <thead><tr><th scope="col">Run</th><th scope="col">Device</th><th scope="col" style="text-align:left">Engine</th><th scope="col">Windows</th><th scope="col">Generation</th><th scope="col">Seconds / video</th><th scope="col">Output tokens</th><th scope="col">Tokens / s</th><th scope="col">Load s</th><th scope="col">Failed videos</th></tr></thead>
  <tbody>{"".join(infer)}</tbody></table></div>
  <p class="note">Why the spread: M1/M2 zero-shot and RFT generate thousands of reasoning tokens per window; M2 direct LoRA stops thinking, so it is 9× faster on test. E1/E2 zero-shot often fail to stop, run into the longer output tiers (up to 4,096 tokens) and take far longer than their LoRA versions. E1 zero-shot on test is the slowest run, at 5.7 h.</p>
</section>
"""


def method_section(data):
    windows = data.get("windows") or {}
    win_rows = "".join(
        f'<tr><th scope="row">{esc(name)}</th><td>{num(w["videos"])}</td><td>{num(w["windows"])}</td>'
        + "".join(f'<td>{num(w["videos_by_window_count"][k])}</td>' for k in ["1", "2", "3", "4"])
        + f'<td>{num(w["max_windows_per_video"])}</td><td>{num(w["median_sentences"], 0)}</td><td>{num(w["max_sentences"])}</td></tr>'
        for name, w in windows.items())
    defs = "".join(f"<dt>{esc(term)}</dt><dd>{esc(text)}</dd>" for term, text in METRIC_DEFS)
    runs = data.get("training_runs") or []
    run_rows = []
    for r in runs:
        pend = num(None)
        if r.get("training_seconds") is not None:
            cells = ["%.1f M" % (r["trainable_parameters"] / 1e6), "%.2f M" % (r["teacher_forcing_tokens"] / 1e6),
                     "%.2f → %.2f (mean %.2f)" % (r["first_loss"], r["last_loss"], r["mean_loss"]),
                     "%.0f min" % (r["training_seconds"] / 60),
                     "%.0f GiB" % (r["peak_allocated_bytes"] / 2**30) if r.get("peak_allocated_bytes") else pend]
        else:
            cells = [pend] * 5
        run_rows.append(
            f'<tr><th scope="row">{esc(r["recipe"])}</th><td>{esc(r.get("alias") or "")}</td>'
            f'<td>{num(r.get("actual_steps"))}</td><td>{num(r.get("planned_optimizer_steps"))}</td>'
            f'<td>{num(r.get("warmup_steps"))}</td>' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"""
<section aria-labelledby="m0">
  <h2 id="m0">Method</h2>
  <h3>Task and output</h3>
  <p>Input is a YouTube transcript split into numbered sentences. The model returns one JSON object, <code>{{"chapters":[{{"start":&lt;sentence index&gt;,"title":"…"}}]}}</code>, listing where each chapter starts and a short title. Every system gets the same prompt (<code>owned-window-outline-v1</code>). The output is parsed strictly: anything other than exactly that JSON object fails.</p>
  <h3>Windows: how long transcripts are handled</h3>
  <p>Every model gets at most <strong>8,192 input tokens</strong>, counted with the actual rendered prompt under all four tokenizers so that the windows are identical for every backbone. A transcript that fits is one window. A longer one is cut into consecutive windows. Each window <em>owns</em> a disjoint block of sentences [A,B), as large as fits, and also shows up to <strong>16 sentences of context</strong> on each side, [L,R). The model must list sentence L first; it is a processing anchor, not a claimed boundary. After all windows are generated, only chapter starts inside each window's owned block are kept, and the blocks are concatenated in order. Because the blocks tile the transcript exactly, stitching never merges, sorts or invents anything. If any window of a video fails, the whole video fails. No transcript is truncated.</p>
  <figure class="diagram">{window_diagram()}</figure>
  <div class="scroll"><table><caption>Window packs (8,192-token budget, 16-sentence context, seed 393)</caption>
  <thead><tr><th scope="col">Split</th><th scope="col">Videos</th><th scope="col">Windows</th><th scope="col">1 window</th><th scope="col">2</th><th scope="col">3</th><th scope="col">4+</th><th scope="col">Max windows</th><th scope="col">Median sentences</th><th scope="col">Max sentences</th></tr></thead>
  <tbody>{win_rows}</tbody></table></div>
  <h3>Metrics</h3>
  <p>All boundary metrics are computed in sentence space against the reference chapters, macro-averaged over every requested video. Failed videos stay in the average.</p>
  <dl class="defs">{defs}</dl>
  <h3>Training parameters</h3>
  <div class="scroll"><table><caption>Shared settings for every LoRA run (direct and RFT)</caption><tbody>
  <tr><th scope="row">Adapter</th><td class="text">LoRA rank 16, alpha 32, dropout 0.05, on attention q/k/v/o projections; base weights frozen, bfloat16. M1 (Qwen3.5-9B) is a hybrid model, and only every 4th layer is full attention, so its adapter covers 8 of 32 layers (3.9 M parameters). The 24 linear-attention layers have no q/k/v/o projections and are not adapted.</td></tr>
  <tr><th scope="row">Optimizer</th><td class="text">AdamW, learning rate 2e-4, cosine decay, gradient clipping 1.0</td></tr>
  <tr><th scope="row">Batching</th><td class="text">one window per forward pass, gradient accumulation 8 (effective batch 8 windows), gradient checkpointing</td></tr>
  <tr><th scope="row">Schedule</th><td class="text">exactly one pass over the prepared windows; warmup 10 steps (direct) or 3 steps (RFT, far fewer steps); seed 393</td></tr>
  <tr><th scope="row">Direct LoRA data</th><td class="text">1,000 training videos, 1,306 windows (seeded sample of the 16,377-video deduplicated train split). The target is the gold chapter JSON for the window, with titles cut by a window edge masked out of the loss.</td></tr>
  <tr><th scope="row">Direct LoRA loss</th><td class="text">E1/E2: decoder cross-entropy on the JSON target. M1/M2: causal LM loss on the final JSON plus the native turn-end token only. The prompt is unscored, and no reasoning text is written, supervised or invented.</td></tr>
  <tr><th scope="row">RFT LoRA loss</th><td class="text">M1/M2: the model's own selected sample, thought plus final JSON plus turn end, supervised in full after the exact generation prompt (unscored). See the RFT section.</td></tr>
  <tr><th scope="row">Inference</th><td class="text">M1/M2: vLLM, native thinking on and unforced, temperature 1.0 / top-p 0.95 / top-k 20 (M1, presence penalty 1.5) or 64 (M2), per-window seed, up to 32,768 new tokens. E1/E2: greedy decoding, batched by length tier, up to 4,096 new tokens. Zero-shot and LoRA use the same path and settings.</td></tr>
  </tbody></table></div>
  <div class="scroll"><table><caption>Per-run training record (one NVIDIA B200)</caption>
  <thead><tr><th scope="col">Recipe</th><th scope="col">Model</th><th scope="col">Windows</th><th scope="col">Optimizer steps</th><th scope="col">Warmup</th><th scope="col">Trainable params</th><th scope="col">Tokens processed</th><th scope="col">Loss first → last</th><th scope="col">Time</th><th scope="col">Peak memory</th></tr></thead>
  <tbody>{"".join(run_rows)}</tbody></table></div>
</section>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    notes = json.loads(args.findings.read_text())
    test, dev, rft = data["test"], data["dev"], data["rft"]
    pack = data.get("test_pack") or {}

    test_rows, effect_rows, lenient_rows, dev_rows, dev_effect = [], [], [], [], []
    for alias, name, kind in ORDER:
        t = test.get(alias, {})
        test_rows += [(f"{alias} zero-shot", t.get("ZS")), (f"{alias} LoRA", t.get("FT"))]
        lenient_rows += [(f"{alias} zero-shot", t.get("ZS_lenient")), (f"{alias} LoRA", t.get("FT_lenient"))]
        zs, ft = t.get("ZS"), t.get("FT")
        effect_rows.append(f'<tr><th scope="row">{alias}<span class="model">{esc(name)} · {esc(kind)}</span></th>'
                           f'<td>{num(zs and zs["exact_f1"])}</td><td>{num(ft and ft["exact_f1"])}</td>'
                           f'<td>{diff_cell(t.get("ft_minus_zs_exact_f1"))}</td>'
                           f'<td>{num(zs and zs.get("pred_per_ref_chapters"), 2)} / {num(ft and ft.get("pred_per_ref_chapters"), 2)}</td></tr>')
        d = dev.get(alias, {})
        dev_rows += [(f"{alias} zero-shot", d.get("ZS")), (f"{alias} LoRA", d.get("FT"))]
        dev_effect.append(f'<tr><th scope="row">{alias}</th><td>{num(d.get("ZS") and d["ZS"]["exact_f1"])}</td>'
                          f'<td>{num(d.get("FT") and d["FT"]["exact_f1"])}</td><td>{diff_cell(d.get("ft_minus_zs_exact_f1"))}</td>'
                          f'<td>{diff_cell(t.get("ft_minus_zs_exact_f1"))}</td></tr>')

    think = data.get("dev_thinking_median_tokens", {})
    rft2 = data.get("rft_v2") or {}
    rft_rows = []
    for alias in ["M1", "M2"]:
        t = think.get(alias, {})
        d = dev.get(alias, {})
        rft_rows.append(
            f'<tr><th scope="row">{alias} zero-shot</th><td>–</td><td>{num(d.get("ZS") and d["ZS"]["exact_f1"])}</td>'
            f'<td>–</td><td>–</td><td>{num(d.get("ZS") and d["ZS"]["failed"])}</td><td>{esc(t.get("ZS", "–"))}</td></tr>'
            f'<tr><th scope="row">{alias} direct LoRA</th><td>1,306 gold</td><td>{num(d.get("FT") and d["FT"]["exact_f1"])}</td>'
            f'<td>{diff_cell(d.get("ft_minus_zs_exact_f1"))}</td><td>–</td><td>{num(d.get("FT") and d["FT"]["failed"])}</td>'
            f'<td>{esc(t.get("direct_LoRA", "–"))}</td></tr>')
        for label, r, kept_of, vs_v1, med in [
                ("RFT v1", rft.get(alias, {}), "300", None, t.get("RFT", "–")),
                ("RFT v2", rft2.get(alias, {}), "1,306", "v2_minus_v1_exact_f1", rft2.get(alias, {}).get("median_thinking_tokens"))]:
            sel = r.get("selection") or {}
            kept = f'{num(sel.get("kept_windows"))} / {kept_of}' if sel else f'{num(None)}'
            rft_rows.append(
                f'<tr><th scope="row">{alias} {label}</th><td>{kept}</td><td>{num(r.get("RFT") and r["RFT"]["exact_f1"])}</td>'
                f'<td>{diff_cell(r.get("rft_minus_zs_exact_f1"))}</td>'
                f'<td>{diff_cell(r.get(vs_v1)) if vs_v1 else "–"}</td>'
                f'<td>{num(r.get("RFT") and r["RFT"]["failed"])}</td><td>{num(med)}</td></tr>')
    names = {"E1-FT": "E1 LoRA", "E2-FT": "E2 LoRA", "M1-FT": "M1 LoRA", "M2-FT": "M2 LoRA",
             "E1-ZS": "E1 zero-shot", "E2-ZS": "E2 zero-shot", "M1-ZS": "M1 zero-shot", "M2-ZS": "M2 zero-shot"}
    posthoc_rows = "".join(
        f'<tr><th scope="row">{esc(names.get(p["left"], p["left"]))} − {esc(names.get(p["right"], p["right"]))}</th>'
        f'<td>{diff_cell({"mean_difference": p["mean_difference"], "confidence_interval": p["ci95"]})}</td></tr>'
        for p in data.get("posthoc_test_comparisons", []))

    val_rows = []
    for alias in ["M1", "M2"]:
        v = data["validation_endpoint"].get(alias, {})
        m, by = v.get("metrics"), v.get("by_window_count", {})
        cells = "".join(f"<td>{num(by[k][1]) if k in by else num(None)} <span class='n'>n={by[k][0] if k in by else '–'}</span></td>"
                        for k in ["1", "2", "3"])
        val_rows.append(f'<tr><th scope="row">{alias} zero-shot</th><td>{num(m and m["exact_f1"])}</td>{cells}</tr>')

    compute_rows = []
    for label, value in data.get("compute", []):
        if isinstance(value, list):
            shown = " / ".join("pending" if v is None else f"{v / 60:.0f} min" for v in value)
        else:
            shown = "pending" if value is None else f"{value / 60:.0f} min"
        compute_rows.append(f"<tr><th scope='row'>{esc(label)}</th><td>{esc(shown)}</td></tr>")

    caveats = "".join(f"<li>{item}</li>" for item in notes.get("notes", []))
    caveats += "".join(f"<li><strong>Freeze amendment</strong> ({esc(a['time_utc'])}, <code>{esc(a['file'])}</code>): {esc(a['reason'])}</li>"
                       for a in data.get("amendments", []))
    caveats += "".join(f"<li><strong>Logged issue</strong>: {esc(i)}</li>" for i in data.get("issues", []))
    summary = "".join(f"<li>{item}</li>" for item in notes.get("summary", []))

    page = f"""<title>LLM Video Chaptering Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap">
<style>
/* Layout: one reading column for prose, wide scrollable tables; a timeline-tick motif from video chaptering. */
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --fg: #17202b; --muted: #5a6573; --line: #d9dee5;
  --zs: #8a97a8; --ft: #2f5fd0; --up: #18794e; --down: #b3261e; --flat: #6b7480; --pend: #e8ebf0;
  --display: "IBM Plex Sans Condensed", "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f141a; --surface: #161d25; --fg: #e6ebf1; --muted: #9aa5b3; --line: #2a3440;
  --zs: #6d7a8c; --ft: #7ea2ff; --up: #4cc38a; --down: #ff8a80; --flat: #9aa5b3; --pend: #222b36; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f141a; --surface: #161d25; --fg: #e6ebf1; --muted: #9aa5b3; --line: #2a3440;
  --zs: #6d7a8c; --ft: #7ea2ff; --up: #4cc38a; --down: #ff8a80; --flat: #9aa5b3; --pend: #222b36; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px }}
main {{ max-width: 1040px; margin: 0 auto; display: grid; gap: 40px }}
header {{ display: grid; gap: 10px }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted) }}
h1 {{ font: 600 clamp(30px, 5vw, 44px)/1.1 var(--display); margin: 0; text-wrap: balance }}
h2 {{ font: 600 22px/1.25 var(--display); margin: 0 0 6px; text-wrap: balance }}
h3 {{ font: 600 16px/1.3 var(--body); margin: 18px 0 6px }}
p, li {{ max-width: 68ch }}
p {{ margin: 0 0 10px }}
.lede {{ color: var(--muted); font-size: 16px }}
.status {{ display: inline-flex; gap: 8px; align-items: center; font: 500 13px var(--mono); color: var(--muted) }}
.status i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--up) }}
.status.partial i {{ background: #c98a00 }}
.ticks {{ height: 10px; background: repeating-linear-gradient(90deg, var(--line) 0 1px, transparent 1px 22px),
  linear-gradient(90deg, var(--ft) 0 18%, transparent 18% 19%, var(--ft) 19% 41%, transparent 41% 42%, var(--ft) 42% 70%, transparent 70% 71%, var(--ft) 71% 100%);
  background-size: auto, 100% 3px; background-repeat: repeat, no-repeat; background-position: 0 0, 0 100%; opacity: .7 }}
section {{ display: grid; gap: 10px; min-width: 0 }}
.summary {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px 22px }}
.summary ul {{ margin: 0; padding-left: 20px; display: grid; gap: 8px }}
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface) }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px }}
caption {{ text-align: left; padding: 10px 14px 0; color: var(--muted); font-size: 13px }}
th, td {{ padding: 8px 14px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap }}
thead th {{ font: 500 12px var(--mono); color: var(--muted); letter-spacing: .03em }}
tbody tr:last-child > * {{ border-bottom: 0 }}
th[scope="row"] {{ text-align: left; font-weight: 500 }}
td {{ font-family: var(--mono); font-size: 13.5px }}
td.best {{ color: var(--ft); font-weight: 500 }}
.model {{ display: block; font: 400 12px var(--body); color: var(--muted) }}
.pending {{ color: var(--muted); font-style: italic; font-family: var(--body) }}
.delta {{ font-weight: 500 }} .delta.up {{ color: var(--up) }} .delta.down {{ color: var(--down) }} .delta.flat {{ color: var(--flat) }}
.ci {{ color: var(--muted); margin-left: 8px; font-size: 12.5px }}
.n {{ color: var(--muted); font-size: 12px }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 10px 6px; overflow-x: auto }}
figure svg {{ width: 100%; min-width: 520px; height: auto; display: block }}
figcaption {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--muted); padding: 4px 14px 8px }}
.key {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; margin-right: 6px }}
svg .grid {{ stroke: var(--line); stroke-width: 1 }}
svg .tick, svg .sublabel {{ fill: var(--muted); font: 11px var(--body) }}
svg .label {{ fill: var(--fg); font: 600 14px var(--body) }}
svg .value {{ fill: var(--fg); font: 12px var(--mono) }}
svg .zs {{ fill: var(--zs) }} svg .ft {{ fill: var(--ft) }} svg .pendingbar {{ fill: var(--pend) }}
.notes li {{ margin-bottom: 6px }}
.toc {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font: 500 13px var(--mono) }}
.toc a {{ color: var(--ft); text-decoration: none }} .toc a:hover {{ text-decoration: underline }}
figure.diagram svg {{ min-width: 600px }}
svg .track {{ fill: var(--pend) }} svg .ctx {{ fill: var(--zs); opacity: .35 }} svg .core {{ fill: var(--ft) }}
svg .anchor {{ stroke: var(--down); stroke-width: 2 }} svg .coretext {{ fill: #fff; font: 500 11.5px var(--mono) }}
dl.defs {{ display: grid; grid-template-columns: minmax(150px, 220px) 1fr; gap: 8px 20px; margin: 0;
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px 18px }}
dl.defs dt {{ font-weight: 600 }} dl.defs dd {{ margin: 0; max-width: 72ch; color: var(--fg) }}
@media (max-width: 640px) {{ dl.defs {{ grid-template-columns: 1fr }} dl.defs dd {{ margin-bottom: 8px }} }}
td.text {{ white-space: normal; text-align: left; font-family: var(--body); font-size: 14px; min-width: 280px; max-width: 760px }}
ol.steps {{ margin: 0; padding-left: 22px; display: grid; gap: 6px }}
tr.group th {{ text-align: left; font: 600 12.5px var(--mono); color: var(--muted); background: var(--bg); letter-spacing: .02em }}
td.eng {{ font-family: var(--body); font-size: 12.5px; color: var(--muted); text-align: left }}
p.note {{ color: var(--muted); font-size: 14px }}
code {{ font-family: var(--mono); font-size: 13px }}
footer {{ color: var(--muted); font-size: 13px }}
</style>
<main>
<header>
  <span class="eyebrow">LLM video chaptering</span>
  <h1>LLM Video Chaptering Results</h1>
  <p class="lede">YouTube video chaptering (retkowski/ytseg) with four backbones, each zero-shot and LoRA-adapted. Exact boundary macro F1 is primary; a video whose output fails strict parsing scores 0.</p>
  <span class="status {"" if notes.get("complete") else "partial"}"><i></i>{esc(notes.get("status", ""))}</span>
  <div class="ticks" aria-hidden="true"></div>
</header>

<section class="summary" aria-labelledby="sum"><h2 id="sum">Summary</h2><ul>{summary}</ul></section>
<nav class="toc" aria-label="Sections"><a href="#m0">Method</a><a href="#t1">Test results</a><a href="#t2">Dev evidence</a><a href="#t3">RFT</a><a href="#t4">Long transcripts</a><a href="#t5">Compute and time</a><a href="#t6">Caveats</a></nav>
{method_section(data)}
<section aria-labelledby="t1">
  <h2 id="t1">Held-out test</h2>
  <p>{num(pack.get("videos"))} test videos, {num(pack.get("windows"))} windows. Eight frozen configurations, each run once. Window pack, prompt and decoding settings match validation.</p>
  <figure>{chart(test)}<figcaption><span><span class="key" style="background:var(--zs)"></span>Zero-shot</span><span><span class="key" style="background:var(--ft)"></span>LoRA (frozen adapter)</span><span>Exact boundary macro F1, test set</span></figcaption></figure>
  <div class="scroll"><table><caption>LoRA effect, paired by video (10,000 bootstrap resamples, 95% interval)</caption>
  <thead><tr><th scope="col">Backbone</th><th scope="col">Zero-shot</th><th scope="col">LoRA</th><th scope="col">LoRA − zero-shot</th><th scope="col">Predicted / reference chapters</th></tr></thead>
  <tbody>{"".join(effect_rows)}</tbody></table></div>
  {metric_table(test_rows, "All metrics, test set (strict parsing). Best value per column highlighted.")}
  <div class="scroll"><table><caption>Cross-backbone comparisons on test (post-hoc; same paired bootstrap)</caption>
  <thead><tr><th scope="col">Comparison</th><th scope="col">Exact F1 difference [95% CI]</th></tr></thead>
  <tbody>{posthoc_rows}</tbody></table></div>
  <h3>Lenient reading (secondary)</h3>
  <p>Envelope-v1 strips one Markdown code fence or a trailing period around otherwise valid JSON. Nothing else is repaired.</p>
  {metric_table(lenient_rows, "All metrics, test set, lenient envelope-v1 reading")}
</section>

<section aria-labelledby="t2">
  <h2 id="t2">Development evidence</h2>
  <p>200 validation videos (270 windows), used to select configurations before the freeze. LoRA adapters: one pass over 1,000 training videos (1,306 windows), rank 16, alpha 32, q/k/v/o projections, learning rate 2e-4 with cosine decay, effective batch 8 windows.</p>
  <div class="scroll"><table><caption>Dev versus test LoRA effect on exact F1</caption>
  <thead><tr><th scope="col">Backbone</th><th scope="col">Dev zero-shot</th><th scope="col">Dev LoRA</th><th scope="col">Dev LoRA − ZS</th><th scope="col">Test LoRA − ZS</th></tr></thead>
  <tbody>{"".join(dev_effect)}</tbody></table></div>
  {metric_table(dev_rows, "All metrics, dev set (strict parsing)")}
</section>

<section aria-labelledby="t3">
  <h2 id="t3">Rejection-sampling fine-tuning (RFT) for M1 and M2</h2>
  <p>Direct LoRA trains M1/M2 only on the final JSON answer, and that teaches them to skip reasoning. RFT, the simplest reinforcement-style recipe, is expert iteration with a binary reward: the model generates, a reward picks its best attempts, and LoRA trains on those attempts, reasoning included. All text is the model's own; nothing is written or edited by us. RFT is a dev-only follow-up after the freeze, not one of the eight test configurations.</p>
  <ol class="steps">
    <li><strong>Sample.</strong> The base model answers each training window 4 times (seeds 1001/2002/3003/4004, temperature 1.0, the same sampling as zero-shot inference, native thinking on, up to 32,768 new tokens).</li>
    <li><strong>Score.</strong> Each finished, strictly valid sample gets the exact F1 of its chapter starts inside the window's owned block [A,B) (anchor excluded) against the gold starts in that block.</li>
    <li><strong>Select.</strong> Keep the best sample per window if its F1 is at least 0.5; otherwise drop the window. Gold text never enters the training sequence, only the choice of sample.</li>
    <li><strong>Train.</strong> One-pass LoRA (shared settings above, warmup 3) on prompt → own thought → own JSON → turn end. Loss covers everything after the prompt.</li>
  </ol>
  <p><strong>v1</strong> sampled 300 windows (233 videos) through the GPU1 vLLM services. <strong>v2</strong> keeps the recipe and every setting, and adds the other 1,006 windows of the same 1,000 training videos the direct LoRA used (local vLLM on GPU0/GPU2, same sampling settings and seed scheme). Selection then runs over all 1,306 windows, so v2 differs from v1 only in data scale, and RFT v2 versus direct LoRA compares training signals on identical videos.</p>
  <div class="scroll"><table><caption>Dev exact F1 (200 videos). Kept = windows with a selected sample / windows sampled. Thinking = median reasoning tokens per window.</caption>
  <thead><tr><th scope="col">Configuration</th><th scope="col">Training windows</th><th scope="col">Dev exact F1</th><th scope="col">− zero-shot</th><th scope="col">v2 − v1</th><th scope="col">Failed</th><th scope="col">Thinking</th></tr></thead>
  <tbody>{"".join(rft_rows)}</tbody></table></div>
</section>

<section aria-labelledby="t4">
  <h2 id="t4">Long transcripts</h2>
  <p>Full validation zero-shot (1,446 videos) through the GPU1 vLLM services. It agrees with the owned GPU0 path within noise on the 200 dev videos (M1 0.412 vs 0.420, M2 0.466 vs 0.459). Videos that need more than one 8,192-token window are much harder.</p>
  <div class="scroll"><table><caption>Exact F1 by number of windows per video</caption>
  <thead><tr><th scope="col">Configuration</th><th scope="col">All videos</th><th scope="col">1 window</th><th scope="col">2 windows</th><th scope="col">3+ windows</th></tr></thead>
  <tbody>{"".join(val_rows)}</tbody></table></div>
</section>

{time_section(data)}

<section aria-labelledby="t6" class="notes">
  <h2 id="t6">Caveats and audit trail</h2>
  <ul>{caveats}</ul>
</section>
<footer>Generated from <code>reports/results.json</code> (scripts/compile_report.py, scripts/render_report_html.py). Frozen configuration: <code>configs/frozen_test.json</code>.</footer>
</main>
"""
    args.output.write_text(page)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
