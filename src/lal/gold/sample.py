"""Sample the gold test set (~150 windows of the test match), stratified, and cut audio snippets.

Strata (each window in exactly one):
  positive  windows with any teacher/event-positive noul label           (target 60)
  keyword   any keyword-baseline rule hit, not in `positive`              (target 40)
  random    everything else                                               (target 50)
Selected windows are >= MIN_GAP_S apart. Each carries its stratum and inclusion weight N_stratum / n_stratum
so window-level metrics can be reweighted to the full match (Horvitz-Thompson). The labeling page never
shows strata, weights or model/teacher labels: labeling is blind.
Outputs: data/gold/sample.jsonl (has narration text: gitignored), data/gold/audio/{idx}.m4a (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess

from lal.baselines.keywords import hits
from lal.config import load_matches, repo_path
from lal.schema import NOUL_QUESTIONS
from lal.windows import load_windows

GOLD_DIR = repo_path("data/gold")
TARGETS = {"positive": 60, "keyword": 40, "random": 50}
MIN_GAP_S = 10.0


def stratum(labels: dict, text: str) -> str:
    if any(labels.get(q) == 1 for q in NOUL_QUESTIONS):
        return "positive"
    h = hits(text)
    if any(h[q] > 0 for q in NOUL_QUESTIONS):
        return "keyword"
    return "random"


def sample(match_id: str, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    wins = load_windows(match_id)
    labels = {json.loads(l)["wid"]: json.loads(l)["labels"]
              for l in (repo_path("data/labels") / f"{match_id}.jsonl").read_text().splitlines()}
    prev = {w["wid"]: wins[i - 4]["text"] if i >= 4 and wins[i - 4]["period"] == w["period"] else ""
            for i, w in enumerate(wins)}  # the window that ends where this one starts (20 s / 5 s stride)
    groups: dict[str, list[dict]] = {k: [] for k in TARGETS}
    for w in wins:
        if w["n_words"] == 0:
            continue
        groups[stratum(labels[w["wid"]], w["text"])].append(w)
    chosen, starts = [], []
    for name, target in TARGETS.items():
        pool = groups[name][:]
        rng.shuffle(pool)
        picked = []
        for w in pool:
            if len(picked) >= target:
                break
            if all(abs(w["start"] - s) >= MIN_GAP_S for s in starts):
                picked.append(w)
                starts.append(w["start"])
        weight = len(groups[name]) / max(1, len(picked))
        chosen += [{"wid": w["wid"], "start": w["start"], "end": w["end"], "text": w["text"],
                    "prev_text": prev[w["wid"]], "stratum": name, "weight": round(weight, 3),
                    "stratum_size": len(groups[name])} for w in picked]
    rng.shuffle(chosen)  # labeling order mixes strata
    for i, c in enumerate(chosen):
        c["idx"] = i
    return chosen


def cut_audio(match_id: str, items: list[dict]) -> int:
    import imageio_ffmpeg

    src = next(p for p in repo_path("data/audio").glob(f"{match_id}.*"))
    out = GOLD_DIR / "audio"
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n = 0
    for it in items:
        dst = out / f"{it['idx']}.m4a"
        if dst.exists():
            continue
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-ss", str(it["start"]), "-t",
                        str(it["end"] - it["start"]), "-i", str(src), "-vn", "-ac", "1", "-c:a", "aac",
                        "-b:a", "64k", str(dst)], check=True)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    match_id = load_matches(split="test")[0]["id"]
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / "sample.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; the gold sample is frozen once created (delete it deliberately to resample)")
    items = sample(match_id)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {k: sum(1 for i in items if i["stratum"] == k) for k in TARGETS}
    sizes = {k: next((i["stratum_size"] for i in items if i["stratum"] == k), 0) for k in TARGETS}
    print(f"sampled {len(items)} windows from {match_id}: {counts} (stratum sizes {sizes})")
    if not args.no_audio:
        print(f"cut {cut_audio(match_id, items)} audio snippets -> {GOLD_DIR / 'audio'}")


if __name__ == "__main__":
    main()
