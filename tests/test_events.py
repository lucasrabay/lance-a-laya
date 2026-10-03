from lal.events import normalize, parse_ts, summarize


def _ev(name, period, ts, **extra):
    return {"id": f"{name}-{period}-{ts}", "type": {"name": name}, "period": period, "timestamp": ts,
            "minute": 0, "second": 0, "team": {"name": "A"}, "player": {"name": "p"}, **extra}


def test_parse_ts():
    assert parse_ts("00:00:00.000") == 0
    assert parse_ts("00:47:30.500") == 47 * 60 + 30.5
    assert parse_ts("01:02:03.250") == 3723.25


def test_normalize_goals_cards_shootout():
    raw = [
        _ev("Half Start", 1, "00:00:00.000"), _ev("Half Start", 1, "00:00:00.000"),
        _ev("Shot", 1, "00:10:00.000", shot={"statsbomb_xg": 0.4, "outcome": {"name": "Goal"},
                                            "type": {"name": "Open Play"}}),
        _ev("Shot", 1, "00:20:00.000", shot={"statsbomb_xg": 0.35, "outcome": {"name": "Saved"},
                                            "type": {"name": "Open Play"}}),
        _ev("Foul Committed", 1, "00:30:00.000", foul_committed={"card": {"name": "Yellow Card"}}),
        _ev("Foul Committed", 1, "00:31:00.000", foul_committed={"penalty": True}),
        _ev("Bad Behaviour", 1, "00:40:00.000", bad_behaviour={"card": {"name": "Red Card"}}),
        _ev("Half End", 1, "00:46:00.000"),
        _ev("Shot", 5, "00:01:00.000", shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"},
                                            "type": {"name": "Penalty"}}),
    ]
    events = normalize(raw)
    kinds = [e["type"] for e in events]
    assert kinds.count("goal") == 1  # shootout goal excluded
    assert kinds.count("period_start") == 1  # one per period, not per team
    s = summarize(events)
    assert s["goals"] == 1 and s["yellow"] == 1 and s["red"] == 1
    assert s["penalty_fouls"] == 1 and s["big_chance_shots"] == 1
    assert s["shots"] == 2  # shootout kick not counted
    goal = next(e for e in events if e["type"] == "goal")
    assert goal["period"] == 1 and goal["t"] == 600.0
