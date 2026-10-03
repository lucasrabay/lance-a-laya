"""Figures for the README / post: method comparison, reliability diagram, highlight timeline.

Palette: the dataviz reference categorical slots 1-4 (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9;
slots 3-4 sit below 3:1 on the surface, so values are direct-labelled and the README carries the table view).
Confidence uses one sequential hue (blue, light -> dark).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from lal.config import repo_path  # noqa: E402
from lal.schema import NOUL_QUESTIONS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
SERIES = {"keyword": "#2a78d6", "zero-shot": "#eb6834", "fine-tuned": "#1baf7a", "claude-teacher": "#eda100"}
MLABEL = {"keyword": "keyword rules", "zero-shot": "zero-shot Laya", "fine-tuned": "fine-tuned Laya",
          "claude-teacher": "Claude teacher (LLM reference)"}
BLUES = LinearSegmentedColormap.from_list("conf", ["#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
QLABEL = {"goal": "Goal", "big_chance": "Big chance", "controversy": "Controversy", "card": "Card"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def method_comparison(results: dict, metric: str = "f1", out: str = "outputs/f1_by_method.png"):
    methods = [m for m in SERIES if m in results["methods"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(NOUL_QUESTIONS))]
        vals = [results["methods"][m]["window"][q][metric] for q in NOUL_QUESTIONS]
        bars = ax.bar(xs, vals, width=width - 0.02, color=SERIES[m], label=MLABEL[m], linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(NOUL_QUESTIONS)), [QLABEL[q] for q in NOUL_QUESTIONS], color=INK, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"window-level {metric.upper()} (human gold set)", color=INK2, fontsize=9)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(repo_path(out), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def reliability(results: dict, out: str = "outputs/reliability.png"):
    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (3, 3)), label="perfect calibration")
    series = [("fine-tuned-uncalibrated", "fine-tuned, T = 1", "#eb6834"),
              ("fine-tuned", "fine-tuned, temperature fit on val", "#2a78d6")]
    for key, label, color in series:
        rep = results["calibration"].get(key)
        if not rep:
            continue
        pts = [b for b in rep["reliability"] if b["n"] >= 3]
        ax.plot([b["mean_prob"] for b in pts], [b["frac_pos"] for b in pts], color=color, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{label} (ECE {rep['ece_pooled']:.3f})")
        for b in pts:  # sample count per bin: most windows sit near 0 or 1, the middle is sparse
            ax.annotate(f"n={b['n']}", (b["mean_prob"], b["frac_pos"]), textcoords="offset points",
                        xytext=(9, -14) if color == "#2a78d6" else (9, 9), fontsize=7, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability", color=INK2, fontsize=9)
    ax.set_ylabel("observed frequency (gold)", color=INK2, fontsize=9)
    ax.set_title("Reliability on the gold set, 4 yes/no questions pooled\n(bins with n >= 3)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(repo_path(out), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def timeline(timeline_path: str, truth: dict, align: dict, out: str = "outputs/timeline.png", title: str = ""):
    """One lane per question: predicted highlights (shade = confidence) and true events (dark ticks)."""
    periods = sorted(align["periods"].items(), key=lambda kv: int(kv[0]))
    gap_min = 3.0
    starts, cursor = {}, 0.0
    for p, per in periods:  # concatenate in-play time with a small visual gap between periods
        starts[p] = cursor
        cursor += per["length_s"] / 60 + gap_min

    def x(t_audio: float) -> float | None:
        for p, per in periods:
            if per["audio_start"] - 30 <= t_audio <= per["audio_end"] + 30:
                return starts[p] + (t_audio - per["offset"]) / 60
        return None

    tl = json.loads(repo_path(timeline_path).read_text())
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(False)
    lanes = list(NOUL_QUESTIONS)
    for i, q in enumerate(lanes):
        y = len(lanes) - 1 - i
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)
        for h in (h for h in tl["highlights"] if h["question"] == q):
            x0, x1 = x(h["start"]), x(h["end"])
            if x0 is None:
                continue
            ax.barh(y, max(x1 - x0, 0.6), left=x0, height=0.42, color=BLUES(min(1.0, h["peak_prob"])),
                    edgecolor=SURFACE, linewidth=1, zorder=2)
        xs = [x(t) for t in truth.get(q, []) if x(t) is not None]
        ax.scatter(xs, [y + 0.36] * len(xs), marker="v", s=46, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for p, per in periods[:-1]:
        ax.axvspan(starts[p] + per["length_s"] / 60, starts[p] + per["length_s"] / 60 + gap_min,
                   color="#f0efec", zorder=0)
    for p, per in periods:
        label = {"1": "1st half", "2": "2nd half", "3": "ET 1", "4": "ET 2"}.get(p, p)
        ax.text(starts[p] + 0.5, len(lanes) - 0.35, label, color=INK2, fontsize=8, va="bottom")
    ax.set_yticks(range(len(lanes)), [QLABEL[q] for q in reversed(lanes)], color=INK, fontsize=9)
    ax.set_xlim(-1, cursor - gap_min + 1)
    ax.set_ylim(-0.7, len(lanes) - 0.1)
    ax.set_xlabel("match minutes (in-play, periods concatenated)", color=INK2, fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("predicted highlight: peak probability", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    cb.outline.set_visible(False)
    ax.scatter([], [], marker="v", s=46, color=INK,
               label="true event (goals & cards: StatsBomb; big chance & controversy: human review)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.16), labelcolor=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(repo_path(out), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="")
    args = ap.parse_args()
    from lal.evaluate import test_match, truth_events

    results = json.loads(repo_path("outputs/results.json").read_text())
    method_comparison(results)
    reliability(results)
    mid = args.match or test_match()
    align = json.loads(repo_path(f"data/align/{mid}.json").read_text())
    timeline(f"outputs/timeline_{mid}.json", truth_events(mid), align,
             title="WC22 final ARG–FRA: highlights predicted from PT-BR narration vs. true events")
    print("wrote outputs/f1_by_method.png, outputs/reliability.png, outputs/timeline.png")


if __name__ == "__main__":
    main()
