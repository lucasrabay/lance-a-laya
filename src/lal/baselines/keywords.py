"""Keyword baseline: the regex rules a reasonable person would write for PT-BR narration.

Each noul question gets a count of rule hits in the window; a window is positive when the count reaches
a threshold tuned on train+val (never on the test match). `intensity` uses exclamation marks plus the
hits of all rules, with two cut points tuned the same way.
"""

from __future__ import annotations

import re

RULES = {
    "goal": re.compile(r"\bgo+l+\b|golaço|golasso", re.I),
    "big_chance": re.compile(
        r"na trave|no travessão|que chance|quase|perdeu|defendeu|defesa|defesaça|milagre|por cima|"
        r"pra fora|para fora|raspando|tirou em cima|salvou", re.I),
    "controversy": re.compile(
        r"\bvar\b|revis[ãa]o|pênalti|penalti|impedimento|impedido|anulad|polêmic|reclama|"
        r"o árbitro|o juiz|arbitragem", re.I),
    "card": re.compile(r"cartão|amarelo|vermelho|expuls", re.I),
}


def hits(text: str) -> dict[str, int]:
    out = {q: len(rx.findall(text)) for q, rx in RULES.items()}
    out["intensity"] = text.count("!") + sum(out.values())
    return out


def predict(text: str, thresholds: dict) -> dict:
    """Scores in [0, 1] (for ranking/PR curves) and hard decisions from tuned thresholds."""
    h = hits(text)
    out = {}
    for q in RULES:
        out[q] = {"score": min(1.0, h[q] / max(1, thresholds[q])), "pred": int(h[q] >= thresholds[q])}
    lo, hi = thresholds["intensity"]
    out["intensity"] = {"score": float(h["intensity"]), "pred": 0 if h["intensity"] < lo else (1 if h["intensity"] < hi else 2)}
    return out


def tune(texts: list[str], labels: list[dict]) -> dict:
    """Pick per-question thresholds (1..4 hits) maximising F1, and intensity cut points maximising accuracy."""
    hs = [hits(t) for t in texts]
    best = {}
    for q in RULES:
        pairs = [(h[q], lab[q]) for h, lab in zip(hs, labels) if lab.get(q) is not None]
        scores = {}
        for th in (1, 2, 3, 4):
            tp = sum(1 for c, y in pairs if c >= th and y == 1)
            fp = sum(1 for c, y in pairs if c >= th and y == 0)
            fn = sum(1 for c, y in pairs if c < th and y == 1)
            scores[th] = 2 * tp / max(1, 2 * tp + fp + fn)
        best[q] = max(scores, key=scores.get)
    pairs = [(h["intensity"], lab["intensity"]) for h, lab in zip(hs, labels) if lab.get("intensity") is not None]
    best_acc, best_cut = -1.0, (1, 3)
    for lo in range(0, 6):
        for hi in range(lo + 1, 10):
            acc = sum(1 for c, y in pairs if (0 if c < lo else (1 if c < hi else 2)) == y) / max(1, len(pairs))
            if acc > best_acc:
                best_acc, best_cut = acc, (lo, hi)
    best["intensity"] = best_cut
    return best
