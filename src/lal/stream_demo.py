"""Streaming simulator: replay a match's narration window by window and emit a highlight timeline.

A new 20 s window completes every 5 s of match audio. Each one is sent to Laya (all five questions in one
forward pass), thresholds picked on the validation matches are applied, and positive windows within the
merge gap extend the current highlight. Output: outputs/timeline_{match}.json.

    uv run python -m lal.stream_demo --tag ft-v1-cal --backend replay            # instant, saved predictions
    uv run python -m lal.stream_demo --tag ft-v1-cal --backend modal --speed 10  # live calls to the deployed predictor
    (deploy once with: uv run modal deploy -m lal.cloud.predict)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from lal.calibrate import apply_platt, fit_val, load_preds
from lal.config import load_matches, repo_path
from lal.schema import INTENSITY_LEVELS, NOUL_QUESTIONS, QUESTIONS, state_for
from lal.teacher import clock
from lal.windows import load_windows, pipeline_cfg

LABEL = {"goal": "GOAL", "big_chance": "BIG CHANCE", "controversy": "CONTROVERSY", "card": "CARD"}


class Highlighter:
    def __init__(self, thresholds: dict, gap: float, max_len: float = 1e9):
        self.th, self.gap, self.max_len = thresholds, gap, max_len
        self.open: dict[str, dict] = {}
        self.done: list[dict] = []

    def push(self, w: dict, probs: dict, latency_ms: float | None) -> list[dict]:
        """Feed one window; return highlights that just started (for live printing)."""
        started = []
        for q in NOUL_QUESTIONS:
            cur = self.open.get(q)
            if cur and (w["start"] - cur["end"] > self.gap or w["end"] - cur["start"] > self.max_len):
                self.done.append(self.open.pop(q))
                cur = None
            if probs[q] >= self.th[q]:
                if cur:
                    cur["end"] = w["end"]
                    if probs[q] > cur["peak_prob"]:
                        cur.update(peak_prob=round(probs[q], 4), peak_wid=w["wid"])
                else:
                    self.open[q] = {"question": q, "start": w["start"], "end": w["end"], "period": w["period"],
                                    "clock": w["clock"], "peak_prob": round(probs[q], 4), "peak_wid": w["wid"],
                                    "detected_after_s": 0.0, "first_latency_ms": latency_ms}
                    started.append(self.open[q])
        return started

    def finish(self) -> list[dict]:
        self.done += self.open.values()
        self.open = {}
        return sorted(self.done, key=lambda h: h["start"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="ft-v1-cal", help="val/test prediction tag (thresholds come from its val preds)")
    ap.add_argument("--model", default="ft-v1")
    ap.add_argument("--calibration", default="ft-v1")
    ap.add_argument("--platt", action="store_true")
    ap.add_argument("--backend", choices=["replay", "modal"], default="replay")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible; 1 = real time")
    ap.add_argument("--match", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    mid = args.match or load_matches(split="test")[0]["id"]
    from lal.windows import load_align

    align = load_align(mid)
    windows = load_windows(mid)
    for w in windows:
        per = align["periods"][str(w["period"])]
        w["clock"] = f"P{w['period']} {clock(w['end'] - per['offset'])}"
    cal = fit_val(args.tag, args.platt)
    ev_cfg = pipeline_cfg()["events"]
    hl = Highlighter(cal["thresholds"], ev_cfg["merge_gap_s"], ev_cfg["max_trigger_s"])

    if args.backend == "modal":
        import modal

        predictor = modal.Cls.from_name("lal-predict", "Predictor")(model=args.model, calibration=args.calibration)
    else:
        saved = load_preds(args.tag, "test") if load_matches(split="test")[0]["id"] == mid else {}
    latencies, intensity_trace = [], []
    t_wall0 = time.perf_counter()
    for w in windows:
        if args.speed > 0:  # wait until this window's end time on the simulated clock
            due = (w["end"] - windows[0]["end"]) / args.speed
            lag = due - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
        t0 = time.perf_counter()
        if args.backend == "modal":
            answers = predictor.predict.remote([state_for(w["text"])], QUESTIONS)[0]
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
        else:
            answers, latency = saved[w["wid"]], None
        probs = {q: answers[q]["noul"] for q in NOUL_QUESTIONS}
        if args.platt:
            probs = {q: apply_platt(p, cal["platt"][q]) for q, p in probs.items()}
        intensity_trace.append({"t": w["end"], "score": answers["intensity"]["score"]})
        for h in hl.push(w, probs, latency):
            if not args.quiet:
                lat = f"  ({latency:.0f} ms)" if latency is not None else ""
                print(f"[{h['clock']}] {LABEL[h['question']]:12s} p={h['peak_prob']:.2f}{lat}", flush=True)
    highlights = hl.finish()
    out = {"match": mid, "model": args.model, "tag": args.tag, "thresholds": cal["thresholds"],
           "merge_gap_s": hl.gap, "highlights": highlights, "intensity": intensity_trace,
           "levels": list(INTENSITY_LEVELS)}
    if latencies:
        out["latency_ms"] = {"n": len(latencies), "p50": round(statistics.median(latencies), 1),
                             "p90": round(sorted(latencies)[int(0.9 * len(latencies)) - 1], 1),
                             "note": "end-to-end from this machine to the deployed Modal predictor"}
    path = repo_path(f"outputs/timeline_{mid}.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(highlights)} highlights -> {path}")


if __name__ == "__main__":
    main()
