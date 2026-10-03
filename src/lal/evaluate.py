"""Final evaluation on the human gold set (window level) and the full test match (event level).

Methods: keyword baseline, zero-shot laya-multilingual, fine-tuned laya-multilingual (val-calibrated), and the
Claude teacher labels as an LLM reference. Every threshold comes from train/val, never from the test match.

Window level (gold, 150 windows, unsure items excluded): P/R/F1 on the raw sample and Horvitz-Thompson
estimates reweighted by each stratum's inclusion weight; AP for probabilistic methods; intensity accuracy/MAE.
Event level (whole test match): positive windows merge into triggers (gap from configs/pipeline.yaml); an event
counts as caught if a trigger lies within its tolerance. Ground truth: StatsBomb goals (+-30 s) and cards
(-10 s .. +120 s: the card is announced after the foul StatsBomb timestamps); big_chance and controversy use
the moments you confirmed in the pooled review when data/gold/review.jsonl exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from lal.baselines import keywords
from lal.calibrate import apply_platt, fit_val, load_labels, load_preds
from lal.config import load_matches, repo_path
from lal.metrics import average_precision, ece, event_match, merge_triggers, prf
from lal.schema import INTENSITY_LEVELS, NOUL_QUESTIONS
from lal.teacher import load_teacher
from lal.weak_labels import audio_time, load_events, project
from lal.windows import load_windows, pipeline_cfg

EVENT_TOL = {"goal": (-30, 30), "card": (-10, 120), "big_chance": (-30, 30), "controversy": (-30, 30)}


def test_match() -> str:
    return load_matches(split="test")[0]["id"]


def load_gold() -> list[dict]:
    sample = {json.loads(l)["idx"]: json.loads(l) for l in repo_path("data/gold/sample.jsonl").read_text().splitlines()}
    rows = []
    for line in repo_path("data/gold/gold.jsonl").read_text().splitlines():
        r = json.loads(line)
        s = sample[r["idx"]]
        rows.append({"wid": s["wid"], "text": s["text"], "weight": s["weight"], "stratum": s["stratum"],
                     "unsure": r["labels"].get("unsure", False),
                     "labels": {q: r["labels"][q] for q in (*NOUL_QUESTIONS, "intensity")}})
    return rows


# ---------- per-method window scores on the test match: {wid: {q: (score, pred)}} ----------

def keyword_scores(windows: list[dict]) -> dict:
    train_val = []
    for split in ("train", "val"):
        labels = load_labels(split)
        for m in load_matches(split=split):
            train_val += [(w["text"], labels[w["wid"]]) for w in load_windows(m["id"])]
    th = keywords.tune([t for t, _ in train_val], [lab for _, lab in train_val])
    out = {}
    for w in windows:
        p = keywords.predict(w["text"], th)
        out[w["wid"]] = {q: (p[q]["score"], p[q]["pred"]) for q in (*NOUL_QUESTIONS, "intensity")}
    return out, {"thresholds": th}


def laya_scores(tag: str, platt: bool, windows: list[dict]) -> tuple[dict, dict]:
    cal = fit_val(tag, platt)
    preds = load_preds(tag, "test")
    out = {}
    for w in windows:
        a = preds[w["wid"]]
        row = {}
        for q in NOUL_QUESTIONS:
            p = a[q]["noul"]
            if platt:
                p = apply_platt(p, cal["platt"][q])
            row[q] = (p, int(p >= cal["thresholds"][q]))
        probs = a["intensity"]["probabilities"]
        row["intensity"] = (a["intensity"]["score"], max(range(len(INTENSITY_LEVELS)), key=lambda i: probs[str(i)]))
        out[w["wid"]] = row
    return out, cal


def teacher_scores(match_id: str, windows: list[dict]) -> dict:
    """Teacher spans alone (no event data) projected onto windows; uncertain spans count as positive."""
    spans, intensity, _ = load_teacher(match_id)
    blocks = json.loads(repo_path(f"data/interim/teacher_packs/{match_id}.blocks.json").read_text())
    cfg = pipeline_cfg()["labels"]
    out = {w["wid"]: {} for w in windows}
    for q in NOUL_QUESTIONS:
        proj = project(windows, [s for s in spans if s["question"] == q], cfg["min_overlap_s"], 0)
        for w, y in zip(windows, proj):
            out[w["wid"]][q] = (float(y == 1), int(y == 1))
    for w in windows:
        lv = [INTENSITY_LEVELS.index(intensity[b["block"]]["level"]) for b in blocks
              if b["block"] in intensity and min(w["end"], b["end"]) - max(w["start"], b["start"]) >= 5]
        best = max(lv) if lv else 0
        out[w["wid"]]["intensity"] = (float(best), best)
    return out


# ---------- metrics ----------

def window_metrics(gold: list[dict], scores: dict) -> dict:
    rows = [g for g in gold if not g["unsure"]]
    out = {}
    for q in NOUL_QUESTIONS:
        y = [g["labels"][q] for g in rows]
        pred = [scores[g["wid"]][q][1] for g in rows]
        s = [scores[g["wid"]][q][0] for g in rows]
        raw = prf(pred, y)
        weighted = prf(pred, y, [g["weight"] for g in rows])
        out[q] = {"n": len(rows), "positives": sum(y), "precision": raw["precision"], "recall": raw["recall"],
                  "f1": raw["f1"], "f1_weighted": weighted["f1"], "precision_weighted": weighted["precision"],
                  "recall_weighted": weighted["recall"], "ap": average_precision(s, y)}
    y = [g["labels"]["intensity"] for g in rows]
    p = [scores[g["wid"]]["intensity"][1] for g in rows]
    out["intensity"] = {"n": len(rows), "accuracy": sum(a == b for a, b in zip(p, y)) / len(rows),
                        "mae": sum(abs(a - b) for a, b in zip(p, y)) / len(rows),
                        "confusion": {f"{INTENSITY_LEVELS[t]}->{INTENSITY_LEVELS[q]}": c
                                      for (t, q), c in sorted(Counter(zip(y, p)).items())}}
    return out


def _dedupe(times: list[float], gap: float = 30.0) -> list[float]:
    """Merge confirmations of the same moment (review cards within `gap` s) into one event."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return [sum(c) / len(c) for c in out]


def truth_events(match_id: str, with_ignore: bool = False):
    """Event ground truth. Goals/cards: StatsBomb. big_chance/controversy: confirmed (not unsure) review moments,
    deduplicated within 30 s; moments the reviewer marked unsure go to `ignore` (neither required nor penalised)."""
    align = json.loads(repo_path(f"data/align/{match_id}.json").read_text())
    ev = [e for e in load_events(match_id) if e["period"] != 5]
    out = {"goal": [audio_time(e, align) for e in ev if e["type"] in ("goal", "own_goal")],
           "card": [audio_time(e, align) for e in ev if e["type"] == "card"]}
    ignore: dict[str, list[float]] = {}
    review = repo_path("data/gold/review.jsonl")
    if review.exists():
        items = {json.loads(l)["idx"]: json.loads(l) for l in repo_path("data/gold/review_items.jsonl").read_text().splitlines()}
        yes: dict[str, list[float]] = {}
        for line in review.read_text().splitlines():
            r = json.loads(line)
            it = items[r["idx"]]
            if r["labels"].get("unsure"):
                ignore.setdefault(it["question"], []).append(it["t"])
            elif r["labels"]["verdict"] == "yes":
                yes.setdefault(it["question"], []).append(it["t"])
        for q, ts in yes.items():
            out[q] = _dedupe(ts)
    return (out, ignore) if with_ignore else out


def event_metrics(windows: list[dict], scores: dict, truth: dict, gap: float, ignore: dict | None = None,
                  max_len: float | None = None) -> dict:
    out = {}
    for q in NOUL_QUESTIONS:
        if q not in truth:
            continue
        pos = [(w["start"], w["end"], scores[w["wid"]][q][0]) for w in windows if scores[w["wid"]][q][1]]
        trig = merge_triggers(pos, gap, max_len)
        minutes = sum(t["end"] - t["start"] for t in trig) / 60
        lo, hi = EVENT_TOL[q]
        ign = (ignore or {}).get(q, [])
        if ign:  # triggers that only match an unsure moment count neither way
            hit = lambda t, e: t["start"] + lo <= e <= t["end"] + hi  # noqa: E731
            trig = [t for t in trig if any(hit(t, e) for e in truth[q]) or not any(hit(t, e) for e in ign)]
        # event_match uses a symmetric tolerance; shift events by the asymmetric window's centre
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        out[q] = event_match(trig, [t + centre for t in truth[q]], half) | {"highlight_minutes": round(minutes, 1)}
    return out


def calibration_report(gold: list[dict], scores: dict) -> dict:
    rows = [g for g in gold if not g["unsure"]]
    probs, ys, per_q = [], [], {}
    for q in NOUL_QUESTIONS:
        p = [scores[g["wid"]][q][0] for g in rows]
        y = [g["labels"][q] for g in rows]
        per_q[q] = ece(p, y)[0]
        probs += p
        ys += y
    pooled, table = ece(probs, ys)
    return {"ece_pooled": pooled, "ece_per_question": per_q, "reliability": table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ft", required=True, help="prediction tag of the fine-tuned, val-calibrated model")
    ap.add_argument("--ft-raw", default="", help="prediction tag of the fine-tuned model before calibration")
    ap.add_argument("--zs", default="zeroshot-en")
    ap.add_argument("--platt", action="store_true")
    args = ap.parse_args()

    mid = test_match()
    windows = load_windows(mid)
    gold = load_gold()
    gap, max_len = pipeline_cfg()["events"]["merge_gap_s"], pipeline_cfg()["events"]["max_trigger_s"]
    truth, ignore = truth_events(mid, with_ignore=True)

    methods, meta = {}, {}
    methods["keyword"], meta["keyword"] = keyword_scores(windows)
    methods["zero-shot"], meta["zero-shot"] = laya_scores(args.zs, False, windows)
    methods["fine-tuned"], meta["fine-tuned"] = laya_scores(args.ft, args.platt, windows)
    methods["claude-teacher"] = teacher_scores(mid, windows)

    results = {"test_match": mid, "gold_windows": len(gold), "gold_unsure": sum(g["unsure"] for g in gold),
               "truth_events": {q: len(v) for q, v in truth.items()},
               "ignored_unsure_moments": {q: len(v) for q, v in ignore.items()}, "methods": {}, "thresholds": meta}
    for name, sc in methods.items():
        results["methods"][name] = {"window": window_metrics(gold, sc), "event": event_metrics(windows, sc, truth, gap, ignore, max_len)}
    results["calibration"] = {"fine-tuned": calibration_report(gold, methods["fine-tuned"])}
    if args.ft_raw:
        raw, _ = laya_scores(args.ft_raw, False, windows)
        results["calibration"]["fine-tuned-uncalibrated"] = calibration_report(gold, raw)
    latency = repo_path(f"outputs/latency_{args.ft.split('-cal')[0]}.json")
    if latency.exists():
        results["latency"] = json.loads(latency.read_text())
    out = repo_path("outputs/results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({n: {q: round(m["window"][q]["f1"], 3) for q in NOUL_QUESTIONS}
                      for n, m in results["methods"].items()}, indent=1))


if __name__ == "__main__":
    main()
