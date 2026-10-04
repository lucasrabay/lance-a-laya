"""Keyword goal rule at other hit thresholds, event level on the test match -> outputs/keyword_sweep.json.

    uv run python -m lal.keyword_sweep

results.json reports the keyword rule at its tuned threshold ("gol" at least twice in a 20 s window, tuned on
train + val). This shows what a looser or stricter rule would have done on the test match, using the same windows,
merge rule and event tolerance as lal.evaluate. Not used to pick anything: it answers "would counting a single
'gol' have found every goal?".
"""

from __future__ import annotations

import json

from lal.baselines import keywords
from lal.config import repo_path
from lal.evaluate import event_metrics, keyword_scores, test_match, truth_events
from lal.schema import NOUL_QUESTIONS
from lal.windows import load_windows, pipeline_cfg


def main() -> None:
    mid = test_match()
    windows = load_windows(mid)
    truth, ignore = truth_events(mid, with_ignore=True)
    gap, max_len = pipeline_cfg()["events"]["merge_gap_s"], pipeline_cfg()["events"]["max_trigger_s"]
    _, meta = keyword_scores(windows)
    tuned = meta["thresholds"]
    rows = []
    for th in (1, 2, 3):
        t = dict(tuned) | {"goal": th}
        scores = {}
        for w in windows:
            p = keywords.predict(w["text"], t)
            scores[w["wid"]] = {q: (p[q]["score"], p[q]["pred"]) for q in NOUL_QUESTIONS}
        goal = event_metrics(windows, scores, {"goal": truth["goal"]}, gap, ignore, max_len)["goal"]
        rows.append({"min_gol_hits": th, "tuned": th == tuned["goal"]} | goal)
    out = repo_path("outputs/keyword_sweep.json")
    out.write_text(json.dumps({"test_match": mid, "question": "goal", "level": "event", "sweep": rows}, indent=1))
    for r in rows:
        print(f"gol >= {r['min_gol_hits']}: {r['caught']}/{r['events']} goals, {r['triggers']} alerts, "
              f"{r['highlight_minutes']} min{'  (tuned)' if r['tuned'] else ''}")


if __name__ == "__main__":
    main()
