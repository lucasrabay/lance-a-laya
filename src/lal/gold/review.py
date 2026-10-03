"""Build the pooled trigger review for big_chance and controversy on the test match (Phase 5).

Candidates = triggers from the keyword baseline, zero-shot and fine-tuned Laya (each at its val threshold)
plus StatsBomb seeds (high-xG/woodwork shots for big_chance, penalty fouls for controversy), merged when within
MERGE_S seconds. Each item shows ~45 s of narration around the moment; the page does not say which method
proposed it. Confirmed moments become event-level ground truth for those two questions (lal.evaluate).
Output: data/gold/review_items.jsonl (narration text: gitignored).
"""

from __future__ import annotations

import argparse
import json
import random

from lal.config import repo_path
from lal.evaluate import keyword_scores, laya_scores, test_match
from lal.metrics import merge_triggers
from lal.teacher import clock
from lal.transcripts import load_words, text_between
from lal.weak_labels import audio_time, load_events
from lal.windows import load_windows, pipeline_cfg

QUESTIONS = ("big_chance", "controversy")
MERGE_S = 30.0


def candidates(windows, methods: dict, seeds: dict, gap: float) -> dict[str, list[dict]]:
    out = {}
    for q in QUESTIONS:
        points = []
        for name, sc in methods.items():
            pos = [(w["start"], w["end"], sc[w["wid"]][q][0]) for w in windows if sc[w["wid"]][q][1]]
            points += [(t["peak"], name) for t in merge_triggers(pos, gap, pipeline_cfg()["events"]["max_trigger_s"])]
        points += [(t, "statsbomb") for t in seeds[q]]
        points.sort()
        merged: list[dict] = []
        for t, src in points:
            if merged and t - merged[-1]["t_last"] <= MERGE_S:
                merged[-1]["sources"].add(src)
                merged[-1]["t_last"] = t
            else:
                merged.append({"t": t, "t_last": t, "sources": {src}})
        out[q] = merged
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--zs", default="zeroshot-en")
    ap.add_argument("--platt", action="store_true")
    args = ap.parse_args()
    mid = test_match()
    path = repo_path("data/gold/review_items.jsonl")
    if path.exists():
        raise SystemExit(f"{path} exists; the review pool is frozen once created")
    windows = load_windows(mid)
    gap = pipeline_cfg()["events"]["merge_gap_s"]
    methods = {"keyword": keyword_scores(windows)[0], "zero-shot": laya_scores(args.zs, False, windows)[0],
               "fine-tuned": laya_scores(args.ft, args.platt, windows)[0]}
    align = json.loads(repo_path(f"data/align/{mid}.json").read_text())
    ev = [e for e in load_events(mid) if e["period"] != 5]
    cfg = pipeline_cfg()["labels"]
    seeds = {"big_chance": [audio_time(e, align) for e in ev if e["type"] == "shot" and e["outcome"] != "Goal"
                            and (e["xg"] >= cfg["big_chance_min_xg"] or e["outcome"] in ("Post", "Saved to Post"))],
             "controversy": [audio_time(e, align) for e in ev if e["type"] == "penalty_foul"]}
    cands = candidates(windows, methods, seeds, gap)
    words = load_words(mid)
    periods = sorted(align["periods"].items(), key=lambda kv: int(kv[0]))
    items = []
    for q, cs in cands.items():
        for c in cs:
            t = (c["t"] + c["t_last"]) / 2
            per = max((p for p in periods if p[1]["audio_start"] - 30 <= t), key=lambda p: p[1]["audio_start"],
                      default=periods[0])
            items.append({"question": q, "t": round(t, 1), "sources": sorted(c["sources"]),
                          "clock": f"P{per[0]} {clock(t - per[1]['offset'])}",
                          "prev_text": text_between(words, t - 40, t - 20), "text": text_between(words, t - 20, t + 25)})
    random.Random(0).shuffle(items)
    for i, it in enumerate(items):
        it["idx"] = i
        it["wid"] = f"review:{it['question']}:{it['t']}"
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print({q: len(cs) for q, cs in cands.items()}, f"-> {len(items)} review items")


if __name__ == "__main__":
    main()
