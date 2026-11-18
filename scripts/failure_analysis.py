#!/usr/bin/env python
"""Failure analysis and chapter-count analysis for the eight test configurations (+ dev RFT runs).

  python scripts/failure_analysis.py      # writes reports/failure-analysis.{md,json}

CPU only, read-only over artifacts/. Every failed window is put into exactly one cause; a failed video is
attributed to its first failed window. Chapter counts are measured on successfully parsed videos only.
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chaptering.evaluation import evaluate_records  # noqa: E402

A = ROOT / "artifacts"
TEST = {"E1-ZS": "heldout-test-v1/E1-ZS-gpu2", "E1-FT": "heldout-test-v1/E1-FT", "E2-ZS": "heldout-test-v1/E2-ZS", "E2-FT": "heldout-test-v1/E2-FT",
        "M1-ZS": "heldout-test-v1/M1/ZS", "M1-FT": "heldout-test-v1/M1/FT", "M2-ZS": "heldout-test-v1/M2/ZS", "M2-FT": "heldout-test-v1/M2/FT"}
DEV = {"M1-ZS": "stage5-v1/M1-dev/ZS", "M1-FT": "stage5-v1/M1-dev/FT", "M1-RFTv2": "rft-v2/M1-dev/FT",
       "M2-ZS": "stage5-v1/M2-dev/ZS", "M2-FT": "stage5-v1/M2-dev/FT", "M2-RFTv2": "rft-v2/M2-dev/FT"}
LABEL = {"ZS": "zero-shot", "FT": "LoRA", "RFTv2": "RFT v2"}

# Cause groups, in display order. "Recoverable" = the lenient reading (one fence / trailing period) would accept it.
GROUPS = {
    "Format wrapper (recoverable)": ["Markdown fence around JSON", "trailing period"],
    "Extra text": ["text before JSON", "text after JSON", "text around JSON", "several JSON objects", "prose, no JSON",
                   "empty answer"],
    "Broken JSON": ["incomplete JSON", "missing \"title\" key", "unescaped quote in a title",
                    "timestamps instead of sentence indices", "invalid JSON syntax"],
    "Wrong structure": ["bare list instead of {chapters: ...}", "wrong JSON schema"],
    "Invalid chapter starts": ["start past window end", "start before window", "starts repeated or out of order",
                               "empty title"],
    "Did not finish": ["cut off: runaway chapter list", "cut off: echoing the transcript", "cut off: repetition loop",
                       "cut off: still thinking", "cut off: long output", "thinking never closed"],
}
CAUSE_GROUP = {cause: group for group, causes in GROUPS.items() for cause in causes}
DEC = json.JSONDecoder()


def jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def unfence(text):
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```\s*$", text, re.S)
    return (m.group(1), True) if m else (text, False)


def first_json(text):
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, end = DEC.raw_decode(text, i)
                return value, i, end
            except ValueError:
                continue
    return None, -1, -1


def repeating(text):
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= 5


def classify(row, window):
    err = row.get("error") or ""
    raw = row.get("raw_text") or ""
    text = (row.get("final_text") if row.get("final_text") is not None else raw) or ""
    s = text.strip()
    if row["status"] == "truncated":
        if row.get("thinking_protocol") and not (row.get("final_text") or "").strip():
            return "cut off: still thinking"
        if len(re.findall(r'"start"', raw)) >= 30:
            return "cut off: runaway chapter list"
        if len(re.findall(r"\[\d+\]\s", raw)) >= 5:
            return "cut off: echoing the transcript"
        if repeating(raw):
            return "cut off: repetition loop"
        return "cut off: long output"
    if "No native final-answer delimiter" in err:
        return "thinking never closed"
    if "outside the document" in err:
        value, _, _ = first_json(unfence(s)[0])
        try:
            starts = [c["start"] for c in value["chapters"]]
            return "start before window" if any(x < window["L"] for x in starts) else "start past window end"
        except Exception:
            return "start past window end"
    if "strictly increasing" in err:
        return "starts repeated or out of order"
    if "titles must be" in err:
        return "empty title"
    if any(k in err for k in ["duplicate JSON key", "only start and title", "integer global", "contain only chapters"]):
        inner = unfence(s)[0].strip()
        if inner.startswith("["):
            return "bare list instead of {chapters: ...}"
        return "wrong JSON schema"
    # "must be one valid JSON object"
    if not s:
        return "empty answer"
    inner, fenced = unfence(s)
    inner = inner.strip()
    for candidate, cause in [(inner, "Markdown fence around JSON"), (inner[:-1], "trailing period")]:
        if cause == "trailing period" and not inner.endswith("."):
            continue
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, list):
            return "bare list instead of {chapters: ...}"
        if fenced and cause == "trailing period":
            return "Markdown fence around JSON"
        return cause if (fenced or cause == "trailing period") else "wrong JSON schema"
    if "{" not in s and "[" not in s:
        return "prose, no JSON"
    if inner.startswith(("{", "[")):
        try:
            _, end = DEC.raw_decode(inner, 0)
            rest = inner[end:].strip()
            return "several JSON objects" if rest.startswith(("{", "[")) else "text after JSON"
        except ValueError:
            if re.search(r'"start"\s*:\s*\d+:\d', inner):
                return "timestamps instead of sentence indices"
            if re.search(r'\{\s*"start"\s*:\s*\d+\s*,\s*"[^"]*"\s*\}', inner):
                return "missing \"title\" key"
            if re.search(r'"title"\s*:\s*"[^"]*"[^,}\s]', inner):
                return "unescaped quote in a title"
            if inner.count("{") > inner.count("}") or inner.count("[") > inner.count("]"):
                return "incomplete JSON"
            return "invalid JSON syntax"
    value, i, end = first_json(inner)
    if value is None:
        return "prose, no JSON"
    pre, post = inner[:i].strip(), inner[end:].strip()
    if post.startswith(("{", "[")):
        return "several JSON objects"
    return "text before JSON" if pre and not post else "text around JSON"


def analyse(name, run, records, windows):
    rows = jsonl(A / run / "windows.jsonl")
    preds = {p["video_id"]: p for p in jsonl(A / run / "predictions.jsonl")}
    scores = {s["video_id"]: s for s in evaluate_records(records, list(preds.values()))["per_video"]}
    window_cause, video_cause = Counter(), Counter()
    first_fail = {}
    examples = {}
    for row in sorted(rows, key=lambda r: (r["video_id"], r["window_index"])):
        if row["status"] == "ok":
            continue
        cause = classify(row, windows[(row["video_id"], row["window_index"])])
        window_cause[cause] += 1
        first_fail.setdefault(row["video_id"], cause)
        if cause not in examples:
            text = (row.get("final_text") or row.get("raw_text") or "").strip().replace("\n", " ")
            examples[cause] = text[:220]
    for vid, cause in first_fail.items():
        if preds[vid]["status"] != "ok":
            video_cause[cause] += 1

    window_count = Counter(vid for vid, _ in windows)
    gold = {r["video_id"]: r for r in records}
    ok = [vid for vid, p in preds.items() if p["status"] == "ok"]
    pred_n = {vid: len(preds[vid]["chapters"]) for vid in ok}
    gold_n = {vid: len(gold[vid]["reference"]["chapter_starts"]) for vid in gold}
    diffs = [pred_n[v] - gold_n[v] for v in ok]

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rp, rg = ranks([pred_n[v] for v in ok]), ranks([gold_n[v] for v in ok])
    spearman = statistics.correlation(rp, rg) if len(ok) > 2 else None
    consecutive = sum(1 for v in ok if sum(1 for a, b in zip(preds[v]["chapters"], preds[v]["chapters"][1:])
                                           if b["start"] - a["start"] == 1) >= 5)
    single = sum(1 for v in ok if pred_n[v] == 1 and gold_n[v] >= 3)

    def bucket_of(vid):
        return {1: "1 window", 2: "2 windows"}.get(window_count[vid], "3+ windows")

    by_bucket = {}
    for b in ["1 window", "2 windows", "3+ windows"]:
        vids = [v for v in gold if bucket_of(v) == b]
        okb = [v for v in vids if v in pred_n]
        by_bucket[b] = {"videos": len(vids), "failed_pct": 100 * (1 - len(okb) / len(vids)),
                        "exact_f1": statistics.mean(scores[v]["exact"]["f1"] or 0 for v in vids),
                        "gold_chapters_mean": statistics.mean(gold_n[v] for v in vids),
                        "pred_chapters_mean": statistics.mean(pred_n[v] for v in okb) if okb else None,
                        "pred_per_gold": (sum(pred_n[v] for v in okb) / sum(gold_n[v] for v in okb)) if okb else None}
    return {
        "config": name, "videos": len(gold), "windows": len(rows),
        "failed_videos": len(gold) - len(ok), "failed_windows": sum(window_cause.values()),
        "window_causes": dict(window_cause.most_common()), "video_causes": dict(video_cause.most_common()),
        "examples": examples,
        "chapter_counts": {
            "ok_videos": len(ok),
            "gold_mean": statistics.mean(gold_n[v] for v in ok) if ok else None,
            "pred_mean": statistics.mean(pred_n[v] for v in ok) if ok else None,
            "gold_median": statistics.median(gold_n[v] for v in ok) if ok else None,
            "pred_median": statistics.median(pred_n[v] for v in ok) if ok else None,
            "pred_per_gold": sum(pred_n.values()) / sum(gold_n[v] for v in ok) if ok else None,
            "under_pct": 100 * sum(d < 0 for d in diffs) / len(ok) if ok else None,
            "exact_count_pct": 100 * sum(d == 0 for d in diffs) / len(ok) if ok else None,
            "over_pct": 100 * sum(d > 0 for d in diffs) / len(ok) if ok else None,
            "within_1_pct": 100 * sum(abs(d) <= 1 for d in diffs) / len(ok) if ok else None,
            "double_or_more_pct": 100 * sum(pred_n[v] >= 2 * gold_n[v] for v in ok) / len(ok) if ok else None,
            "half_or_less_pct": 100 * sum(2 * pred_n[v] <= gold_n[v] for v in ok) / len(ok) if ok else None,
            "spearman_pred_vs_gold": spearman,
            "one_sentence_chapter_runs": consecutive, "single_chapter_for_3plus_gold": single,
            "pred_minus_gold_hist": {str(k): v for k, v in sorted(Counter(max(-6, min(6, d)) for d in diffs).items())},
        },
        "by_windows": by_bucket,
    }


def pct(x):
    return "—" if x is None else f"{x:.0f}%"


def num(x, d=1):
    return "—" if x is None else f"{x:.{d}f}"


def markdown(test, dev):
    md = ["# Failure and chapter-count analysis", "",
          "* **Test set:** 1,448 videos and 1,905 windows per configuration.",
          "* **Dev set (RFT section only):** 200 videos.",
          "* **Generated by** `scripts/failure_analysis.py` from the saved raw outputs.",
          "* **How failures are counted:**",
          "  * Each failed window is assigned exactly one cause.",
          "  * A failed video is attributed to its first failed window, because any failed window fails the whole video.",
          "  * A failed video scores 0 F1 and stays in every average.", ""]

    md += ["## 1. Failed videos by cause (test)", "",
           "| Cause group | " + " | ".join(t["config"] for t in test) + " |",
           "|---|" + "---|" * len(test)]
    for group, causes in GROUPS.items():
        md.append(f"| **{group}** | " + " | ".join(str(sum(t["video_causes"].get(c, 0) for c in causes)) or "0" for t in test) + " |")
        for cause in causes:
            values = [t["video_causes"].get(cause, 0) for t in test]
            if any(values):
                md.append(f"| &nbsp;&nbsp;{cause} | " + " | ".join(str(v) if v else "·" for v in values) + " |")
    md.append("| **Failed videos** | " + " | ".join(f"**{t['failed_videos']}**" for t in test) + " |")
    md.append("| Failed windows | " + " | ".join(str(t["failed_windows"]) for t in test) + " |")
    md += ["",
           "* **Format wrapper (recoverable):** the JSON itself is valid, but it is wrapped in a Markdown fence or followed by a period. The secondary *lenient* reading in the results report accepts exactly these cases.",
           "* **Extra text:** commentary, headings or a summary around the JSON, or no JSON at all.",
           "* **Broken JSON:** for example a missing closing bracket, or `\"start\": 4:59` timestamps in place of sentence numbers.",
           "* **Wrong structure:** a bare list in place of `{\"chapters\": [...]}`, extra keys, or duplicate keys.",
           "* **Invalid chapter starts:** a start outside the window's visible range, starts out of order, or an empty title.",
           "* **Did not finish:** the output hit its token cap (4,096 for the encoder-decoders, 32,768 for M1/M2), or the thinking block never closed.", ""]

    md += ["## 2. What the failures look like", ""]
    for t in test:
        if t["failed_videos"] >= 10:
            top = list(t["window_causes"].items())[:3]
            md.append(f"**{t['config']}** ({t['failed_windows']} failed windows):")
            for cause, n in top:
                ex = t["examples"].get(cause, "").replace("|", "\\|").replace("`", "'")
                md.append(f"- *{cause}* ({n}): `{ex[:160]}`")
            md.append("")

    md += ["## 3. Predicted vs gold chapter counts (test, successfully parsed videos)", "",
           "| Config | Parsed videos | Gold mean | Predicted mean | Pred / gold | Under | Exact count | Over | Within ±1 | ≥ 2× gold | ≤ ½ gold | Spearman ρ |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in test:
        c = t["chapter_counts"]
        md.append(f"| {t['config']} | {c['ok_videos']} | {num(c['gold_mean'])} | {num(c['pred_mean'])} | {num(c['pred_per_gold'], 2)} | "
                  f"{pct(c['under_pct'])} | {pct(c['exact_count_pct'])} | {pct(c['over_pct'])} | {pct(c['within_1_pct'])} | "
                  f"{pct(c['double_or_more_pct'])} | {pct(c['half_or_less_pct'])} | {num(c['spearman_pred_vs_gold'], 2)} |")
    md += ["",
           "Notes on the columns:",
           "* **Under / Exact count / Over:** the share of videos with fewer, exactly as many, or more predicted chapters than gold.",
           "* **Spearman ρ:** whether longer or richer videos also get more chapters.",
           "* **E1 zero-shot:** only 13 videos parsed, so its count statistics are not meaningful.", "",
           "Degenerate but valid outputs (test):", "",
           "| Config | Runs of one-sentence chapters (≥5 in a row) | Single chapter where gold has ≥3 |", "|---|---|---|"]
    for t in test:
        c = t["chapter_counts"]
        md.append(f"| {t['config']} | {c['one_sentence_chapter_runs']} | {c['single_chapter_for_3plus_gold']} |")

    md += ["", "## 4. By video length (number of 8,192-token windows, test)", "",
           "| Config | " + " | ".join(f"{b}: F1 / failed / pred÷gold" for b in ["1 window", "2 windows", "3+ windows"]) + " |",
           "|---|---|---|---|"]
    for t in test:
        cells = []
        for b in ["1 window", "2 windows", "3+ windows"]:
            x = t["by_windows"][b]
            cells.append(f"{x['exact_f1']:.3f} / {x['failed_pct']:.0f}% / {num(x['pred_per_gold'], 2)}")
        md.append(f"| {t['config']} | " + " | ".join(cells) + " |")
    v = test[0]["by_windows"]
    md += ["", f"Video counts per bucket:"]
    md += [f"* **{b}:** {v[b]['videos']} videos, {v[b]['gold_chapters_mean']:.1f} gold chapters on average." for b in v]
    md += ["", "## 5. Dev: zero-shot vs answer-only LoRA vs RFT v2 (M1/M2, 200 videos)", "",
           "| Config | Failed videos | Main causes | Pred / gold | Under | Over | Within ±1 |", "|---|---|---|---|---|---|---|"]
    for t in dev:
        c = t["chapter_counts"]
        causes = ", ".join(f"{k} {n}" for k, n in list(t["video_causes"].items())[:3]) or "—"
        md.append(f"| {t['config']} | {t['failed_videos']} | {causes} | {num(c['pred_per_gold'], 2)} | {pct(c['under_pct'])} | "
                  f"{pct(c['over_pct'])} | {pct(c['within_1_pct'])} |")
    md += ["", "## Summary", ""] + SUMMARY + [""]
    return "\n".join(md)


SUMMARY = []


def summarize(test, dev):
    t = {x["config"]: x for x in test}

    def share(cfg, group):
        return sum(t[cfg]["video_causes"].get(c, 0) for c in GROUPS[group])

    def cause(cfg, name):
        return t[cfg]["video_causes"].get(name, 0)

    lines = [
        f"* **E1 zero-shot** fails on {t['E1-ZS']['failed_videos']:,} of 1,448 videos. In {share('E1-ZS', 'Format wrapper (recoverable)'):,} of them the JSON is valid but wrapped "
        f"in a Markdown fence, a habit of the instruction-tuned model. {share('E1-ZS', 'Did not finish')} hit the 4,096-token cap, "
        f"{cause('E1-ZS', 'cut off: runaway chapter list')} of them by listing a new \"chapter\" almost every sentence. "
        f"{share('E1-ZS', 'Broken JSON')} have broken JSON, and {share('E1-ZS', 'Extra text')} add other text.",
        f"* **E2 zero-shot** fails on {t['E2-ZS']['failed_videos']} videos, mostly by talking instead of answering: {share('E2-ZS', 'Extra text')} cases of extra text, "
        f"including {cause('E2-ZS', 'prose, no JSON')} with no JSON at all (summaries, or chat replies such as \"You are correct. I apologize…\"). "
        f"There are also {share('E2-ZS', 'Format wrapper (recoverable)')} fence or period wrappers and {share('E2-ZS', 'Did not finish')} cut-offs, "
        f"{cause('E2-ZS', 'cut off: echoing the transcript')} of them echoing the transcript.",
        f"* **LoRA fixes the format for the encoder-decoders.** Failed videos drop to {t['E1-FT']['failed_videos']} (E1) and {t['E2-FT']['failed_videos']} (E2). "
        f"E1's remaining failures are mostly titles that quote something without escaping the quotes "
        f"({cause('E1-FT', 'unescaped quote in a title')} of {t['E1-FT']['failed_videos']}, e.g. `\"Victor Frankl's \"Man Search for Meaning\"\"`).",
        f"* **M1/M2 zero-shot almost never fail** ({t['M1-ZS']['failed_videos']} and {t['M2-ZS']['failed_videos']} videos). Their native thinking ends in clean JSON.",
        f"* **Answer-only LoRA makes M1/M2 sloppier.**",
        f"  * M1 LoRA fails on {t['M1-FT']['failed_videos']} videos. {share('M1-FT', 'Format wrapper (recoverable)')} add Markdown fences that zero-shot never produced, "
        f"{share('M1-FT', 'Invalid chapter starts')} have invalid starts, and {share('M1-FT', 'Did not finish')} never finish thinking.",
        f"  * M2 LoRA fails on {t['M2-FT']['failed_videos']}, mainly from invalid starts ({share('M2-FT', 'Invalid chapter starts')}) and broken JSON "
        f"({share('M2-FT', 'Broken JSON')}, mostly unescaped quotes in titles).",
        f"* **Chapter count.**",
        f"  * Zero-shot M1/M2 over-segment: pred/gold {t['M1-ZS']['chapter_counts']['pred_per_gold']:.2f} and {t['M2-ZS']['chapter_counts']['pred_per_gold']:.2f}. "
        f"Most of that comes from long videos (section 4).",
        f"  * Their LoRA versions predict fewer chapters ({t['M1-FT']['chapter_counts']['pred_per_gold']:.2f} and {t['M2-FT']['chapter_counts']['pred_per_gold']:.2f}). "
        f"M2 LoRA under-segments {t['M2-FT']['chapter_counts']['under_pct']:.0f}% of videos.",
        f"  * The encoder-decoder LoRAs are closest to gold: {t['E1-FT']['chapter_counts']['pred_per_gold']:.2f} and {t['E2-FT']['chapter_counts']['pred_per_gold']:.2f}, "
        f"with {t['E2-FT']['chapter_counts']['within_1_pct']:.0f}% of E2 videos within ±1 chapter, and the highest rank correlation with gold counts.",
        f"  * E2 zero-shot's counts barely track gold (ρ {t['E2-ZS']['chapter_counts']['spearman_pred_vs_gold']:.2f}). "
        f"It gives one chapter per sentence in {t['E2-ZS']['chapter_counts']['one_sentence_chapter_runs']} videos and a single chapter in "
        f"{t['E2-ZS']['chapter_counts']['single_chapter_for_3plus_gold']}.",
        "* **Long videos are the main remaining weakness.** Every configuration loses F1 sharply from 1 window to 3+ windows (section 4), and M1/M2 zero-shot nearly double the chapter count there.",
    ]
    SUMMARY.extend(lines)


def main():
    windows = {}
    for row in jsonl(A / "windows/test-v1/test_windows.jsonl"):
        w = row["window"]
        windows[(w["video_id"], w["window_index"])] = w
    dev_windows = {}
    for row in jsonl(A / "windows/dev-v2/validation_windows.jsonl"):
        w = row["window"]
        dev_windows[(w["video_id"], w["window_index"])] = w
    test_records = jsonl(A / "windows/test-v1/test_records.jsonl")
    dev_records = jsonl(A / "windows/dev-v2/validation_records.jsonl")
    test = [analyse(n, r, test_records, windows) for n, r in TEST.items()]
    dev = [analyse(n, r, dev_records, dev_windows) for n, r in DEV.items()]
    summarize(test, dev)
    out = {"groups": GROUPS, "test": test, "dev": dev, "summary": SUMMARY}
    (ROOT / "reports/failure-analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (ROOT / "reports/failure-analysis.md").write_text(markdown(test, dev) + "\n")
    for t in test:
        assert sum(t["video_causes"].values()) == t["failed_videos"], t["config"]
    print("wrote reports/failure-analysis.{md,json}")


if __name__ == "__main__":
    main()
