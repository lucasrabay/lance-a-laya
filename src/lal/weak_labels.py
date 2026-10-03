"""Event-data labels: StatsBomb events -> audio-time spans -> per-window labels (1 / 0 / None=ignore).

Spans are in stream seconds (offset[period] + event t). Tolerances come from configs/pipeline.yaml:
- goal:        [t + goal_span]           (shot -> "gol!" call and celebration)
- big_chance:  [t + big_chance_span]     for non-goal shots with xG >= big_chance_min_xg or off the woodwork
- card:        search region [t + card_search]; the teacher pinpoints the announcement inside it, else the
               fallback span [t + card_fallback_span] is used and flagged low-confidence (see teacher.py)
- controversy: [t + penalty_span] around penalty fouls (seeds only; teacher labels add the rest)
Projection: a window is positive if it overlaps a span by >= min_overlap_s, ignored if it touches a span
less than that or lies within ignore_near_s of one, else negative.
"""

from __future__ import annotations

import json

from lal.config import repo_path


def audio_time(event: dict, align: dict) -> float | None:
    per = align["periods"].get(str(event["period"])) or align["periods"].get(event["period"])
    if per is None or event.get("t") is None:
        return None
    return per["offset"] + event["t"]


def event_spans(events: list[dict], align: dict, cfg: dict) -> list[dict]:
    """Spans per question from event data. Each: {question, start, end, source, ...}."""
    out = []
    for e in events:
        if e["period"] == 5:
            continue
        t = audio_time(e, align)
        if t is None:
            continue

        def span(q, rng, **extra):
            out.append({"question": q, "start": round(t + rng[0], 2), "end": round(t + rng[1], 2),
                        "source": "event", "event_type": e["type"], "event_t": round(t, 2), **extra})

        if e["type"] in ("goal", "own_goal"):
            span("goal", cfg["goal_span"])
        elif e["type"] == "shot" and e["outcome"] != "Goal" and (
            e["xg"] >= cfg["big_chance_min_xg"] or e["outcome"] in ("Post", "Saved to Post")
        ):
            span("big_chance", cfg["big_chance_span"], xg=e["xg"], outcome=e["outcome"])
        elif e["type"] == "card":
            span("card_search", cfg["card_search"], card=e["card"])
        elif e["type"] == "penalty_foul":
            span("controversy", cfg["penalty_span"])
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def gap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, max(a0, b0) - min(a1, b1))


def project(windows: list[dict], spans: list[dict], min_overlap: float, ignore_near: float) -> list[int | None]:
    labels: list[int | None] = []
    for w in windows:
        best_ov, best_gap = 0.0, float("inf")
        for s in spans:
            best_ov = max(best_ov, overlap(w["start"], w["end"], s["start"], s["end"]))
            best_gap = min(best_gap, gap(w["start"], w["end"], s["start"], s["end"]))
        if best_ov >= min_overlap:
            labels.append(1)
        elif best_ov > 0 or best_gap < ignore_near:
            labels.append(None)
        else:
            labels.append(0)
    return labels


def load_events(match_id: str) -> list[dict]:
    with open(repo_path(f"data/events/{match_id}.jsonl")) as f:
        return [json.loads(line) for line in f]
