from lal.build_dataset import gold_probs, laya_rows


def _rows():
    rows = []
    for i in range(100):
        rows.append({"wid": f"w{i}", "labels": {"goal": 1 if i < 2 else 0, "big_chance": None,
                                               "controversy": 0, "card": 0, "intensity": i % 3}})
    return rows


def test_gold_probs_shapes():
    assert gold_probs("goal", 1) == {"probabilities": {"false": 0.0, "true": 1.0}}
    assert gold_probs("intensity", 2)["probabilities"] == {"0": 0.0, "1": 0.0, "2": 1.0}


def test_laya_rows_rebalance_and_ignore():
    texts = {f"w{i}": f"texto {i}" for i in range(100)}
    out = laya_rows(_rows(), texts, rebalance=True, min_pos=10)
    goal_pos = [d for d in out if "goal" in d["gold"] and d["gold"]["goal"]["probabilities"]["true"] == 1.0]
    assert len(goal_pos) == 10  # 2 originals + 8 upsampled copies
    assert all("big_chance" not in d["gold"] for d in out)  # ignored everywhere
    goal_neg = [d for d in out if "goal" in d["gold"] and d["gold"]["goal"]["probabilities"]["true"] == 0.0]
    assert len(goal_neg) <= 40  # capped near 10:1 of 2 positives (random, generous bound)
    assert all(set(d["questions"]) == set(d["gold"]) for d in out)
