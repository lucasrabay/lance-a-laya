"""Metrics without sklearn: threshold-free ranking metrics, P/R/F1, calibration, event matching."""

from __future__ import annotations


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    tp, ap = 0, 0.0
    for k, (_, y) in enumerate(pairs, 1):
        if y:
            tp += 1
            ap += tp / k
    return ap / n_pos


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    # rank-sum with average ranks for ties
    ranks, i = {}, 0
    rank_sum = 0.0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def prf(preds: list[int], labels: list[int], weights: list[float] | None = None) -> dict:
    w = weights or [1.0] * len(preds)
    tp = sum(wi for p, y, wi in zip(preds, labels, w) if p and y)
    fp = sum(wi for p, y, wi in zip(preds, labels, w) if p and not y)
    fn = sum(wi for p, y, wi in zip(preds, labels, w) if not p and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def ece(probs: list[float], labels: list[int], bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error of P(positive) for a binary question, plus the reliability table."""
    table, total, err = [], len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        err += len(idx) / total * abs(conf - acc)
        table.append({"bin": [lo, hi], "n": len(idx), "mean_prob": conf, "frac_pos": acc})
    return err, table


def merge_triggers(times: list[tuple[float, float, float]], gap: float, max_len: float | None = None) -> list[dict]:
    """Merge positive windows (start, end, prob) closer than `gap` seconds into triggers, none longer than
    `max_len` seconds (a method that fires on every window must not collapse into one match-long trigger)."""
    out: list[dict] = []
    for s, e, p in sorted(times):
        if out and s - out[-1]["end"] <= gap and (max_len is None or e - out[-1]["start"] <= max_len):
            out[-1]["end"] = max(out[-1]["end"], e)
            if p > out[-1]["prob"]:
                out[-1]["prob"], out[-1]["peak"] = p, (s + e) / 2
        else:
            out.append({"start": s, "end": e, "prob": p, "peak": (s + e) / 2})
    return out


def event_match(triggers: list[dict], events: list[float], tol: float) -> dict:
    """Event recall/precision: an event is caught if some trigger span lies within +-tol s of it."""
    def near(t, e):
        return t["start"] - tol <= e <= t["end"] + tol

    caught = [e for e in events if any(near(t, e) for t in triggers)]
    good = [t for t in triggers if any(near(t, e) for e in events)]
    return {"events": len(events), "caught": len(caught), "triggers": len(triggers), "true_triggers": len(good),
            "recall": len(caught) / len(events) if events else None,
            "precision": len(good) / len(triggers) if triggers else None}
