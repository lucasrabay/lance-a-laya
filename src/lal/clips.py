"""Cut highlight clips for a timeline: download only those segments of the broadcast VOD (yt-dlp
--download-sections with the static ffmpeg from imageio-ffmpeg). Clips stay local (outputs/clips/, gitignored):
broadcast footage is copyrighted and is not republished by this repo.

    uv run python -m lal.clips --pre 10 --post 15 --max-height 480
"""

from __future__ import annotations

import argparse
import json
import subprocess

from lal.config import get_match, load_matches, repo_path


def clip_plan(timeline: dict, questions: set[str], min_prob: float, pre: float, post: float) -> list[dict]:
    """Which highlights get a clip, their file names and stream-time ranges."""
    keep = [h for h in timeline["highlights"] if h["question"] in questions
            and (h["question"] in ("goal", "card") or h["peak_prob"] >= min_prob)]
    plan = []
    for i, h in enumerate(keep):
        name = f"{i:02d}_{h['question']}_{h['clock'].replace(' ', '_').replace(':', 'm')}"
        plan.append({"name": name, "h": h, "start": max(0, h["start"] - pre), "end": h["end"] + post})
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="")
    ap.add_argument("--pre", type=float, default=10.0)
    ap.add_argument("--post", type=float, default=15.0)
    ap.add_argument("--max-height", type=int, default=480)
    ap.add_argument("--questions", default="goal,big_chance,controversy,card")
    ap.add_argument("--min-prob", type=float, default=0.95,
                    help="big_chance/controversy highlights need at least this peak probability (goals/cards: all)")
    args = ap.parse_args()
    import imageio_ffmpeg

    mid = args.match or load_matches(split="test")[0]["id"]
    timeline = json.loads(repo_path(f"outputs/timeline_{mid}.json").read_text())
    video = get_match(mid)["youtube"]
    out_dir = repo_path("outputs/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.questions.split(","))
    for c in clip_plan(timeline, wanted, args.min_prob, args.pre, args.post):
        h, start, end, name = c["h"], c["start"], c["end"], c["name"]
        if (out_dir / f"{name}.mp4").exists():
            continue
        cmd = [
            "yt-dlp", "--js-runtimes", "node", "--no-warnings", "--quiet",
            "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
            "-f", f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]",
            "--download-sections", f"*{start:.1f}-{end:.1f}", "--force-keyframes-at-cuts",
            "--merge-output-format", "mp4", "-o", str(out_dir / f"{name}.%(ext)s"),
            "--retries", "5", "--fragment-retries", "5", f"https://www.youtube.com/watch?v={video}",
        ]
        for attempt in range(3):  # YouTube fragment errors are transient
            if subprocess.run(cmd).returncode == 0:
                print(f"{name}.mp4  [{start:.0f}-{end:.0f} s]  p={h['peak_prob']:.2f}", flush=True)
                break
        else:
            print(f"FAILED {name} after 3 attempts", flush=True)


if __name__ == "__main__":
    main()
