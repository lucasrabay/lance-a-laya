"""Align StatsBomb match clock to audio (stream) time, one offset per period.

audio_time = offset[period] + event.t   (event.t = StatsBomb timestamp, seconds since period kickoff)

For each period we grid-search the offset that minimises a robust cost: for every anchor event, the
distance from its predicted audio time (plus a typical narration lag) to the nearest matching narration
cue, truncated at a cap so a missing cue costs a constant instead of dragging the fit.
Anchors: goals ("gooool"), corners ("escanteio"), period start ("rola a bola", ...) and period end
("termina o primeiro tempo", ...). Residuals are reported leave-one-out: each anchor is scored against
an offset fitted without it. The acceptance gate is a median |goal residual| <= 15 s.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
from dataclasses import dataclass

from lal.config import get_match, load_matches, repo_path
from lal.sources import kickoff_utc, metadata
from lal.transcripts import load_words


@dataclass(frozen=True)
class Cue:
    anchor: str     # which anchor kind this cue evidences
    pattern: re.Pattern
    lag: float      # typical seconds from the event to the cue (negative: cue comes first)
    cap: float      # cost cap / max distance at which a cue still counts as "found"
    weight: float


CUES = {
    # Whisper writes the long call as "gol!". Penalties are often called by name ("Neymar!") and the score
    # change ("faz 2x0", "abre o placar") follows ~20 s later, so it is a second, later cue for goals.
    "goal_call": Cue("goal", re.compile(r"go{2,}l|gol\b|golaço"), lag=2.0, cap=30.0, weight=5.0),
    "goal_score": Cue("goal", re.compile(r"faz (o )?\d ?(x|a) ?\d|abre o placar|amplia|empata o jogo|empatou|"
                                         r"virou o jogo"), lag=20.0, cap=30.0, weight=5.0),
    "corner": Cue("corner", re.compile(r"escanteio"), lag=-12.0, cap=40.0, weight=1.0),
    # Shots are the densest anchor (~25 per match): the narrator calls the finish as it happens.
    "shot": Cue("shot", re.compile(r"chut|bateu|bate (de|pro|para|cruzado|colocado|forte)|finaliz|cabece|"
                                   r"testou|de cabeça|pra fora|para fora|defende|defendeu|na trave|por cima"),
                lag=1.0, cap=12.0, weight=1.0),
    "period_start": Cue(
        "period_start",
        re.compile(r"rola a bola|rolar a bola|rolou a bola|bola rolando|bola est[áa] rolando|simbora|"
                   r"come[çc]a (o|a|a partida|o jogo)|come[çc]ou|apitou|"
                   r"t[áa] valendo|est[áa] valendo|autoriza o árbitro"),
        lag=0.0, cap=60.0, weight=3.0),
    "period_end": Cue(
        "period_end",
        re.compile(r"termina (o|a|o jogo|o primeiro|o segundo)|fim de (jogo|papo|primeiro|segundo)|"
                   r"final de (jogo|primeiro|segundo)|acabou|encerrad|apita o (fim|final)|vamos para o intervalo"),
        lag=5.0, cap=60.0, weight=2.0),
}
ANCHOR_KINDS = ("goal", "shot", "corner", "period_start", "period_end")
# Expected gap from the end of period p-1 to the kickoff of period p (stream seconds).
BREAK_BEFORE = {2: 15 * 60, 3: 5 * 60, 4: 60, 5: 5 * 60}
SEARCH_P1 = 20 * 60
SEARCH_LATER = 10 * 60
ALIGN_DIR = repo_path("data/align")


def cue_hits(words: list[dict]) -> dict[str, list[float]]:
    """Times at which each cue's pattern matches text starting at a word (4-word context)."""
    toks = [w["word"].strip().lower() for w in words]
    hits: dict[str, list[float]] = {k: [] for k in CUES}
    for i, w in enumerate(words):
        ctx = " ".join(toks[i : i + 4])
        for kind, cue in CUES.items():
            if cue.pattern.match(ctx):
                hits[kind].append(w["start"])
    return hits


def anchors_for_period(events: list[dict], period: int) -> list[dict]:
    out = []
    for e in events:
        if e["period"] != period or e.get("t") is None:
            continue
        kind = {"goal": "goal", "own_goal": "goal", "corner": "corner", "shot": "shot",
                "period_start": "period_start", "period_end": "period_end"}.get(e["type"])
        if kind:
            out.append({"kind": kind, "t": e["t"], "minute": e.get("minute")})
    return out


def nearest(hits: list[float], x: float) -> float | None:
    if not hits:
        return None
    i = bisect.bisect_left(hits, x)
    cands = [hits[j] for j in (i - 1, i) if 0 <= j < len(hits)]
    return min(cands, key=lambda h: abs(h - x))


def best_match(anchor: dict, offset: float, hits: dict[str, list[float]]) -> tuple[float, float | None, str]:
    """(weighted capped cost, signed residual or None, cue name) of the best cue for this anchor."""
    best = (float("inf"), None, "")
    for name, cue in CUES.items():
        if cue.anchor != anchor["kind"]:
            continue
        pred = offset + anchor["t"] + cue.lag
        h = nearest(hits[name], pred)
        d = None if h is None or abs(h - pred) > cue.cap else h - pred
        c = cue.weight * (cue.cap if d is None else abs(d))
        if c < best[0]:
            best = (c, d, name)
    return best


def cost(offset: float, anchors: list[dict], hits: dict[str, list[float]]) -> float:
    return sum(best_match(a, offset, hits)[0] for a in anchors)


def fit_offset(anchors: list[dict], hits: dict[str, list[float]], center: float, radius: float,
               step: float = 1.0) -> float:
    """Grid search, then a 0.25 s refinement around the best grid point."""
    n = int(2 * radius / step) + 1
    grid = [center - radius + i * step for i in range(n)]
    best = min(grid, key=lambda o: cost(o, anchors, hits))
    fine = [best - step + 0.25 * i for i in range(9)]
    return min(fine, key=lambda o: cost(o, anchors, hits))


def residual(anchor: dict, offset: float, hits: dict[str, list[float]]) -> float | None:
    d = best_match(anchor, offset, hits)[1]
    return None if d is None else round(d, 2)


def kickoff_hint(match: dict) -> float:
    meta = metadata(match["youtube"])
    ko = kickoff_utc(match)
    if ko and meta.get("live_status") == "was_live" and meta.get("release_timestamp"):
        return ko.timestamp() - meta["release_timestamp"]
    return 0.0


def align_match(match_id: str, transcript_name: str | None = None) -> dict:
    match = get_match(match_id)
    with open(repo_path(f"data/events/{match_id}.jsonl")) as f:
        events = [json.loads(line) for line in f]
    words = load_words(match_id, transcript_name)
    hits = cue_hits(words)
    period_end = {e["period"]: e["t"] for e in events if e["type"] == "period_end"}
    periods = sorted(p for p in period_end if p != 5)  # shootout is excluded downstream

    result = {"match": match_id, "periods": {}, "residuals": []}
    prev_end_audio = None
    for p in periods:
        anchors = anchors_for_period(events, p)
        if p == 1 or prev_end_audio is None:
            center, radius = kickoff_hint(match), SEARCH_P1
        else:
            center, radius = prev_end_audio + BREAK_BEFORE.get(p, 300), SEARCH_LATER
        offset = fit_offset(anchors, hits, center, radius)
        prev_end_audio = offset + period_end[p]
        counts = {k: sum(1 for a in anchors if a["kind"] == k) for k in ANCHOR_KINDS}
        result["periods"][p] = {
            "offset": round(offset, 2), "audio_start": round(offset, 2),
            "audio_end": round(offset + period_end[p], 2), "length_s": round(period_end[p], 1),
            "anchors": counts, "search_center": round(center, 1),
        }
        for i, a in enumerate(anchors):  # leave-one-out residuals
            rest = anchors[:i] + anchors[i + 1 :]
            loo = fit_offset(rest, hits, offset, 120.0) if rest else offset
            result["residuals"].append({"period": p, "kind": a["kind"], "t": a["t"], "minute": a["minute"],
                                        "loo_residual": residual(a, loo, hits),
                                        "fit_residual": residual(a, offset, hits)})
    result["summary"] = summarize(result["residuals"])
    return result


def summarize(residuals: list[dict]) -> dict:
    out = {}
    for kind in ANCHOR_KINDS:
        rs = [r["loo_residual"] for r in residuals if r["kind"] == kind]
        found = [abs(r) for r in rs if r is not None]
        out[kind] = {
            "n": len(rs), "found": len(found),
            "median_abs": round(statistics.median(found), 1) if found else None,
            "max_abs": round(max(found), 1) if found else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", action="append")
    args = ap.parse_args()
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    ids = args.match or [m["id"] for m in load_matches()]
    for mid in ids:
        res = align_match(mid)
        (ALIGN_DIR / f"{mid}.json").write_text(json.dumps(res, indent=1))
        s = res["summary"]
        offs = {p: round(v["offset"] / 60, 2) for p, v in res["periods"].items()}
        print(f"{mid:14s} offsets(min)={offs} goal={s['goal']} corner={s['corner']}")


if __name__ == "__main__":
    main()
