"""Render the results tables (markdown) from outputs/results.json -> outputs/results.md."""

from __future__ import annotations

import json

from lal.config import repo_path
from lal.schema import NOUL_QUESTIONS

NAMES = {"keyword": "Keyword rules", "zero-shot": "Zero-shot Laya", "fine-tuned": "**Fine-tuned Laya**",
         "claude-teacher": "*Claude teacher (LLM reference)*"}
QN = {"goal": "goal", "big_chance": "big chance", "controversy": "controversy", "card": "card"}


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def render(r: dict) -> str:
    out = []
    n_scored = r["gold_windows"] - r["gold_unsure"]
    out.append(f"**Window level, human gold set** ({n_scored} windows of the test match; "
               f"{r['gold_unsure']} marked unsure are excluded). F1 at thresholds chosen on validation, "
               "with threshold-free average precision (AP) in parentheses.\n")
    out.append("| method | " + " | ".join(QN[q] for q in NOUL_QUESTIONS) + " | intensity acc. (MAE) |")
    out.append("|---|" + "---|" * (len(NOUL_QUESTIONS) + 1))
    for m, v in r["methods"].items():
        cells = [f"{fmt(v['window'][q]['f1'])} ({fmt(v['window'][q]['ap'])})" for q in NOUL_QUESTIONS]
        it = v["window"]["intensity"]
        out.append(f"| {NAMES[m]} | " + " | ".join(cells) + f" | {fmt(it['accuracy'])} ({fmt(it['mae'])}) |")
    pos = r["methods"]["fine-tuned"]["window"]
    out.append("\nGold positives: " + ", ".join(f"{QN[q]} {pos[q]['positives']}" for q in NOUL_QUESTIONS) + ".\n")

    out.append("**Event level, whole test match.** Positive windows merge into triggers; an event is caught if "
               "a trigger lies within its tolerance (goal ±30 s, card −10 s…+120 s after the foul, big chance and "
               "controversy ±30 s of a human-confirmed moment). A highlight lasts at most 2 min. Cells: recall "
               "(caught / events) · precision (true triggers / triggers) · total flagged time (in-play match ≈ 142 min). "
               "Triggers that only match a moment the reviewer marked unsure are left out.\n")
    qs = [q for q in NOUL_QUESTIONS if q in r["methods"]["fine-tuned"]["event"]]
    out.append("| method | " + " | ".join(QN[q] for q in qs) + " |")
    out.append("|---|" + "---|" * len(qs))
    for m, v in r["methods"].items():
        cells = []
        for q in qs:
            e = v["event"][q]
            cells.append(f"{e['caught']}/{e['events']} · {e['true_triggers']}/{e['triggers']} · {e['highlight_minutes']:.0f} min")
        out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")

    cal = r["calibration"]
    out.append("\n**Calibration of the fine-tuned model on the gold set** (4 yes/no questions pooled, 10 bins): "
               f"ECE {cal['fine-tuned']['ece_pooled']:.3f} with temperatures fitted on validation"
               + (f" vs {cal['fine-tuned-uncalibrated']['ece_pooled']:.3f} at T = 1." if "fine-tuned-uncalibrated" in cal else "."))
    if "latency" in r:
        out.append("\n**Latency** (fine-tuned, all 5 questions in one forward pass per window):\n")
        out.append("| GPU | p50 / p90 per window (batch 1) | batched throughput |")
        out.append("|---|---|---|")
        for row in r["latency"]:
            out.append(f"| {row['gpu']} | {row['p50_ms']} / {row['p90_ms']} ms | {row['batched_windows_per_s']} windows/s |")
    tl = repo_path(f"outputs/timeline_{r['test_match']}.json")
    if tl.exists() and "latency_ms" in json.loads(tl.read_text()):
        e2e = json.loads(tl.read_text())["latency_ms"]
        out.append(f"\nEnd-to-end in the streaming demo (this Mac in Brazil -> deployed Modal T4 -> back, {e2e['n']} "
                   f"sequential windows): p50 {e2e['p50']:.0f} ms, p90 {e2e['p90']:.0f} ms, against a 5 s budget per window.")
    return "\n".join(out) + "\n"


def main() -> None:
    r = json.loads(repo_path("outputs/results.json").read_text())
    md = render(r)
    repo_path("outputs/results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
