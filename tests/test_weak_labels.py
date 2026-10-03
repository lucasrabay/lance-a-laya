from lal.weak_labels import event_spans, project

CFG = {"goal_span": [-3, 12], "big_chance_span": [-3, 8], "big_chance_min_xg": 0.3,
       "card_search": [0, 120], "penalty_span": [-3, 30]}
ALIGN = {"periods": {"1": {"offset": 1000.0}, "2": {"offset": 5000.0}}}


def test_event_spans_types_and_shootout():
    events = [
        {"type": "goal", "period": 1, "t": 100.0},
        {"type": "shot", "period": 1, "t": 200.0, "xg": 0.45, "outcome": "Saved"},
        {"type": "shot", "period": 1, "t": 210.0, "xg": 0.05, "outcome": "Post"},
        {"type": "shot", "period": 1, "t": 220.0, "xg": 0.05, "outcome": "Off T"},
        {"type": "card", "period": 2, "t": 50.0, "card": "Yellow Card"},
        {"type": "goal", "period": 5, "t": 10.0},
    ]
    spans = event_spans(events, ALIGN, CFG)
    qs = [s["question"] for s in spans]
    assert qs == ["goal", "big_chance", "big_chance", "card_search"]
    assert spans[0]["start"] == 1097.0 and spans[0]["end"] == 1112.0
    assert spans[3]["start"] == 5050.0


def test_project_positive_ignore_negative():
    spans = [{"start": 100.0, "end": 115.0}]
    wins = [{"start": s, "end": s + 20} for s in (80.0, 95.0, 113.0, 60.0, 40.0)]
    # 80-100 touches at a point (overlap 0, gap 0) -> ignore; 95-115 overlaps 15 -> pos;
    # 113-133 overlaps 2 (< 3) -> ignore; 60-80 gap 20 -> neg; 40-60 -> neg
    assert project(wins, spans, min_overlap=3, ignore_near=10) == [None, 1, None, 0, 0]
