"""StatsBomb open data -> normalized match events.

Each event is a dict with `period` and `t` (seconds since that period's kickoff, StatsBomb `timestamp`),
which is what alignment maps onto audio time. Period 5 (penalty shootout) is kept but tagged so callers
can drop it. StatsBomb data: https://github.com/statsbomb/open-data (attribution required).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from lal.config import load_matches, repo_path

RAW_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{id}.json"
RAW_DIR = repo_path("data/raw/statsbomb")
OUT_DIR = repo_path("data/events")


def parse_ts(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fetch_raw(statsbomb_id: int) -> list[dict]:
    path = RAW_DIR / f"{statsbomb_id}.json"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(RAW_URL.format(id=statsbomb_id), timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return json.loads(path.read_text())


def _base(ev: dict, kind: str) -> dict:
    return {
        "type": kind,
        "period": ev["period"],
        "t": round(parse_ts(ev["timestamp"]), 3),
        "minute": ev["minute"],
        "second": ev["second"],
        "team": ev.get("team", {}).get("name"),
        "player": ev.get("player", {}).get("name"),
        "sb_id": ev["id"],
    }


def normalize(raw: list[dict]) -> list[dict]:
    """Extract the event types the pipeline uses, sorted by (period, t)."""
    out: list[dict] = []
    period_start: dict[int, float] = {}
    period_end: dict[int, float] = {}
    for ev in raw:
        name = ev["type"]["name"]
        p = ev["period"]
        if name == "Half Start":
            period_start[p] = min(period_start.get(p, 1e9), parse_ts(ev["timestamp"]))
        elif name == "Half End":
            period_end[p] = max(period_end.get(p, 0.0), parse_ts(ev["timestamp"]))
        elif name == "Shot":
            shot = ev["shot"]
            e = _base(ev, "shot")
            e.update(
                xg=round(shot.get("statsbomb_xg", 0.0), 4),
                outcome=shot["outcome"]["name"],
                shot_type=shot["type"]["name"],
                shootout=p == 5,
            )
            out.append(e)
            if shot["outcome"]["name"] == "Goal" and p != 5:
                g = _base(ev, "goal")
                g.update(xg=e["xg"], shot_type=e["shot_type"])
                out.append(g)
        elif name == "Own Goal For":
            out.append(_base(ev, "own_goal"))
        elif name == "Foul Committed":
            fc = ev.get("foul_committed", {})
            if fc.get("penalty"):
                out.append(_base(ev, "penalty_foul"))
            if "card" in fc:
                c = _base(ev, "card")
                c.update(card=fc["card"]["name"], via="foul")
                out.append(c)
        elif name == "Bad Behaviour":
            bb = ev.get("bad_behaviour", {})
            if "card" in bb:
                c = _base(ev, "card")
                c.update(card=bb["card"]["name"], via="bad_behaviour")
                out.append(c)
        elif name == "Pass" and ev.get("pass", {}).get("type", {}).get("name") == "Corner":
            out.append(_base(ev, "corner"))
        elif name == "Substitution":
            out.append(_base(ev, "substitution"))
        elif name == "Offside":
            out.append(_base(ev, "offside"))
    for p in sorted(period_start):
        out.append({"type": "period_start", "period": p, "t": period_start[p]})
        out.append({"type": "period_end", "period": p, "t": period_end.get(p)})
    out.sort(key=lambda e: (e["period"], e["t"] if e["t"] is not None else 1e9))
    return out


def summarize(events: list[dict]) -> dict:
    def n(kind, **kw):
        return sum(1 for e in events if e["type"] == kind and all(e.get(k) == v for k, v in kw.items()))

    periods = {e["period"]: e["t"] for e in events if e["type"] == "period_end"}
    return {
        "goals": n("goal") + n("own_goal"),
        "pens_scored": sum(1 for e in events if e["type"] == "goal" and e.get("shot_type") == "Penalty"),
        "yellow": n("card", card="Yellow Card"),
        "second_yellow": n("card", card="Second Yellow"),
        "red": n("card", card="Red Card"),
        "penalty_fouls": n("penalty_foul"),
        "big_chance_shots": sum(
            1
            for e in events
            if e["type"] == "shot"
            and not e["shootout"]
            and e["outcome"] != "Goal"
            and (e["xg"] >= 0.3 or e["outcome"] in ("Post", "Saved to Post"))
        ),
        "shots": sum(1 for e in events if e["type"] == "shot" and not e["shootout"]),
        "corners": n("corner"),
        "subs": n("substitution"),
        "periods": {p: round(t / 60, 1) for p, t in sorted(periods.items()) if t},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-spares", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in load_matches(include_spares=args.include_spares):
        events = normalize(fetch_raw(m["statsbomb_id"]))
        with open(OUT_DIR / f"{m['id']}.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps({"match": m["id"], **e}, ensure_ascii=False) + "\n")
        print(m["id"], m.get("split", "spare"), json.dumps(summarize(events)))


if __name__ == "__main__":
    main()
