"""Cut aligned transcripts into sliding windows (in-play time only; shootout excluded).

Output: data/interim/windows/{match}.jsonl (contains narration text, so gitignored). Each window:
{wid, match, split, period, start, end (stream seconds), t_start (seconds since period kickoff), text, n_words}
"""

from __future__ import annotations

import argparse
import bisect
import json

import yaml

from lal.config import get_match, load_matches, repo_path
from lal.transcripts import load_words

WINDOW_DIR = repo_path("data/interim/windows")


def pipeline_cfg() -> dict:
    return yaml.safe_load(repo_path("configs/pipeline.yaml").read_text())


def make_windows(words: list[dict], align: dict, length: float, stride: float, margin: float) -> list[dict]:
    starts = [w["start"] for w in words]
    out = []
    for p_str, per in sorted(align["periods"].items(), key=lambda kv: int(kv[0])):
        p = int(p_str)
        s = per["audio_start"] - margin
        last = per["audio_end"] + margin - length
        while s <= last + 1e-9:
            e = s + length
            ws = words[bisect.bisect_left(starts, s) : bisect.bisect_left(starts, e)]
            out.append({
                "wid": f"{align['match']}:{p}:{s:.1f}",
                "match": align["match"], "period": p,
                "start": round(s, 2), "end": round(e, 2),
                "t_start": round(s - per["offset"], 2),
                "text": "".join(w["word"] for w in ws).strip(),
                "n_words": len(ws),
            })
            s += stride
    return out


def pseudo_align(match_id: str, words: list[dict]) -> dict:
    """For a match without event data: one 'period' spanning the transcript (clock counts from its start)."""
    a, b = words[0]["start"], words[-1]["end"]
    return {"match": match_id, "periods": {"1": {"offset": a, "audio_start": a, "audio_end": b,
                                                 "length_s": b - a}}, "pseudo": True}


def load_align(match_id: str, words: list[dict] | None = None) -> dict:
    path = repo_path(f"data/align/{match_id}.json")
    if path.exists():
        return json.loads(path.read_text())
    return pseudo_align(match_id, words if words is not None else load_words(match_id))


def load_windows(match_id: str) -> list[dict]:
    with open(WINDOW_DIR / f"{match_id}.jsonl") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", action="append")
    args = ap.parse_args()
    cfg = pipeline_cfg()["windows"]
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    for mid in args.match or [m["id"] for m in load_matches()]:
        words = load_words(mid)
        align = load_align(mid, words)
        wins = make_windows(words, align, cfg["length_s"], cfg["stride_s"], cfg["margin_s"])
        split = get_match(mid)["split"]
        with open(WINDOW_DIR / f"{mid}.jsonl", "w") as f:
            for w in wins:
                f.write(json.dumps({**w, "split": split}, ensure_ascii=False) + "\n")
        empty = sum(1 for w in wins if w["n_words"] == 0)
        print(f"{mid:14s} {split:5s} windows={len(wins)} empty={empty} "
              f"mean_words={sum(w['n_words'] for w in wins) / max(1, len(wins)):.1f}")


if __name__ == "__main__":
    main()
