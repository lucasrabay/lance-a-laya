"""Calibration and thresholds, fitted on the validation matches only.

1. Laya temperatures: `laya.calibrate.records_from_labeled` + `Agent.fit_temperatures` on val windows
   (runs on Modal via Predictor.fit_calibration), saved to the Volume as calib/{run}.json and applied with
   `laya.load(path, calibration=...)`. Laya fits one temperature per question type, so all four nouls share it.
2. Per-question Platt scaling on top (a, b per noul on logit(p)), fitted locally on val predictions, because
   rebalanced training shifts each question's base rate differently and one shared temperature cannot undo that.
3. Decision thresholds per question (and method) maximising window-level F1 on val.
"""

from __future__ import annotations

import json
import math

import numpy as np

from lal.config import load_matches, repo_path
from lal.metrics import prf
from lal.schema import INTENSITY_LEVELS, NOUL_QUESTIONS, QUESTIONS, state_for


def load_labels(split: str) -> dict[str, dict]:
    out = {}
    for m in load_matches(split=split):
        for line in open(repo_path(f"data/labels/{m['id']}.jsonl")):
            r = json.loads(line)
            out[r["wid"]] = r["labels"]
    return out


def load_preds(tag: str, split: str) -> dict[str, dict]:
    out = {}
    for m in load_matches(split=split):
        path = repo_path(f"data/preds/{tag}/{m['id']}.jsonl")
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                out[r["wid"]] = r["answers"]
    return out


def calibration_pairs(split: str = "val") -> list:
    """(state, questions, targets) for records_from_labeled; targets are probability vectors in option order."""
    from lal.windows import load_windows

    labels = load_labels(split)
    pairs = []
    for m in load_matches(split=split):
        for w in load_windows(m["id"]):
            lab = labels[w["wid"]]
            qs, targets = {}, {}
            for q in NOUL_QUESTIONS:
                if lab[q] is not None:
                    qs[q], targets[q] = QUESTIONS[q], [1.0 - lab[q], float(lab[q])]
            if lab["intensity"] is not None:
                qs["intensity"] = QUESTIONS["intensity"]
                targets["intensity"] = [float(i == lab["intensity"]) for i in range(len(INTENSITY_LEVELS))]
            if qs:
                pairs.append((state_for(w["text"]), qs, targets))
    return pairs


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_platt(probs: list[float], labels: list[int], iters: int = 200) -> tuple[float, float]:
    """Logistic regression y ~ sigmoid(a * logit(p) + b) by Newton's method (tiny L2 for stability)."""
    x = np.array([_logit(p) for p in probs])
    y = np.array(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * x + b, -30, 30)
        s = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((s - y) * x) + 1e-3 * a, np.sum(s - y) + 1e-3 * b])
        w = s * (1 - s)
        h = np.array([[np.sum(w * x * x) + 1e-3, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-3]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(a), float(b)


def apply_platt(p: float, ab: tuple[float, float]) -> float:
    z = ab[0] * _logit(p) + ab[1]
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def best_threshold(probs: list[float], labels: list[int]) -> tuple[float, float]:
    """Threshold maximising F1 on (probs, labels); returns (threshold, f1)."""
    best = (0.5, -1.0)
    for th in sorted(set(round(p, 4) for p in probs)):
        f1 = prf([int(p >= th) for p in probs], labels)["f1"]
        if f1 > best[1]:
            best = (th, f1)
    return best


def fit_val(tag: str, platt: bool) -> dict:
    """Platt params (optional) and F1-optimal thresholds per noul question from val predictions."""
    labels, preds = load_labels("val"), load_preds(tag, "val")
    out = {"tag": tag, "platt": {}, "thresholds": {}, "val_f1": {}}
    for q in NOUL_QUESTIONS:
        ids = [w for w in preds if labels[w][q] is not None]
        p = [preds[w][q]["noul"] for w in ids]
        y = [labels[w][q] for w in ids]
        if platt:
            out["platt"][q] = fit_platt(p, y)
            p = [apply_platt(v, out["platt"][q]) for v in p]
        out["thresholds"][q], out["val_f1"][q] = best_threshold(p, y)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="prediction tag under data/preds/ (val split)")
    ap.add_argument("--platt", action="store_true")
    args = ap.parse_args()
    res = fit_val(args.tag, args.platt)
    out = repo_path(f"outputs/calib_{args.tag}{'_platt' if args.platt else ''}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
