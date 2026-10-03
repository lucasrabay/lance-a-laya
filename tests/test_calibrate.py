import random

from lal.calibrate import apply_platt, best_threshold, fit_platt


def test_platt_recovers_overconfident_shift():
    rng = random.Random(0)
    probs, labels = [], []
    for _ in range(4000):
        true_p = rng.random()
        y = int(rng.random() < true_p)
        # model is overconfident: pushes probabilities towards the extremes
        reported = true_p ** 0.4 if true_p > 0.5 else 1 - (1 - true_p) ** 0.4
        probs.append(reported)
        labels.append(y)
    ab = fit_platt(probs, labels)
    cal = [apply_platt(p, ab) for p in probs]
    def ece(ps):
        bins = [[] for _ in range(10)]
        for p, y in zip(ps, labels):
            bins[min(int(p * 10), 9)].append((p, y))
        return sum(len(b) / len(ps) * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b)
    assert ece(cal) < ece(probs)


def test_best_threshold():
    th, f1 = best_threshold([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    assert 0.4 < th <= 0.6 and f1 == 1.0
