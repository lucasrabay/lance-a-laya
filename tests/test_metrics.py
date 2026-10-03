from lal.metrics import average_precision, ece, event_match, merge_triggers, prf, roc_auc


def test_ranking_metrics():
    assert roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert abs(average_precision([0.9, 0.8, 0.7], [1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-9


def test_prf_and_ece():
    m = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    e, _ = ece([0.9] * 10, [1] * 9 + [0])
    assert abs(e) < 1e-9


def test_merge_and_event_match():
    trig = merge_triggers([(100, 120, 0.6), (105, 125, 0.9), (300, 320, 0.7)], gap=30)
    assert len(trig) == 2 and trig[0]["prob"] == 0.9
    m = event_match(trig, [110.0, 500.0], tol=30)
    assert m["caught"] == 1 and m["true_triggers"] == 1 and m["precision"] == 0.5


def test_merge_respects_max_len():
    wins = [(s, s + 20, 0.9) for s in range(0, 600, 5)]  # fires on every window for 10 minutes
    assert len(merge_triggers(wins, gap=30)) == 1
    assert len(merge_triggers(wins, gap=30, max_len=120)) >= 5
