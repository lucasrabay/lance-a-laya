"""Verify narration sources: YouTube metadata + auto-caption speech coverage around the match.

`check` writes data/sources.json (metadata only, no media) with duration, live status, the opus
audio format size, a kickoff hint (StatsBomb kickoff UTC minus live-stream start) and, from YouTube's
auto-captions, how much of the expected match window has narration. Captions are cached under
data/captions/ (gitignored) and are only used for this coverage check, never as transcripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

from lal.config import load_matches, repo_path

YT = "https://www.youtube.com/watch?v={}"
META_DIR = repo_path("data/raw/yt")
CAP_DIR = repo_path("data/captions")


def ytdlp(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["yt-dlp", "--js-runtimes", "node", "--no-warnings", *args],
        capture_output=True, text=True, timeout=600,
    )


def metadata(video_id: str) -> dict:
    path = META_DIR / f"{video_id}.json"
    if not path.exists():
        META_DIR.mkdir(parents=True, exist_ok=True)
        r = ytdlp("-j", "--skip-download", YT.format(video_id))
        if r.returncode != 0:
            return {"error": r.stderr.strip().splitlines()[-1] if r.stderr else "unknown"}
        path.write_text(r.stdout)
    return json.loads(path.read_text())


def kickoff_utc(match: dict) -> datetime | None:
    """Scheduled kickoff from configs/matches.yaml (StatsBomb `kick_off` mixes time zones)."""
    ko = match.get("kickoff_utc")
    return datetime.fromisoformat(ko.replace("Z", "+00:00")) if ko else None


def caption_times(video_id: str) -> list[tuple[float, float]] | None:
    """(start, end) seconds of every auto-caption event, from json3 (pt-orig)."""
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    path = CAP_DIR / f"{video_id}.pt-orig.json3"
    if not path.exists():
        ytdlp("--skip-download", "--write-auto-subs", "--sub-langs", "pt-orig", "--sub-format", "json3",
              "-o", str(CAP_DIR / "%(id)s.%(ext)s"), YT.format(video_id))
    if not path.exists():
        return None
    events = json.loads(path.read_text()).get("events", [])
    return [
        (e["tStartMs"] / 1000, (e["tStartMs"] + e.get("dDurationMs", 0)) / 1000)
        for e in events
        if any(s.get("utf8", "").strip() for s in e.get("segs", []) or [])
    ]


def coverage(times: list[tuple[float, float]], start: float, end: float, step: float = 60.0) -> float:
    """Fraction of `step`-second bins in [start, end) containing at least one caption event."""
    bins = int((end - start) // step)
    if bins <= 0:
        return 0.0
    hit = set()
    for a, _ in times:
        if start <= a < end:
            hit.add(int((a - start) // step))
    return len(hit) / bins


def longest_speech_block(times: list[tuple[float, float]], gap: float = 300.0) -> tuple[float, float]:
    """Longest stretch with no caption gap above `gap` seconds (radio uploads have no kickoff hint)."""
    if not times:
        return (0.0, 0.0)
    best = cur = (times[0][0], times[0][1])
    for a, b in times[1:]:
        cur = (cur[0], b) if a - cur[1] <= gap else (a, b)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def check(match: dict, video_id: str) -> dict:
    meta = metadata(video_id)
    if "error" in meta:
        return {"match": match["id"], "youtube": video_id, "error": meta["error"]}
    audio = {f["format_id"]: f.get("filesize") or f.get("filesize_approx") for f in meta["formats"]
             if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")}
    row = {
        "match": match["id"], "split": match.get("split"), "youtube": video_id,
        "title": meta.get("title"), "channel": meta.get("channel"),
        "duration_s": meta.get("duration"), "live_status": meta.get("live_status"),
        "availability": meta.get("availability"),
        "audio_low_mb": round((audio.get("249") or audio.get("139") or 0) / 1e6, 1),
    }
    ko = kickoff_utc(match)
    start_ts = meta.get("release_timestamp") if meta.get("live_status") == "was_live" else None
    if ko and start_ts:
        row["kickoff_hint_s"] = int(ko.timestamp() - start_ts)
    times = caption_times(video_id)
    if times is None:
        row["captions"] = "missing"
        return row
    if "kickoff_hint_s" in row:
        k = row["kickoff_hint_s"]
        row["coverage_match_window"] = round(coverage(times, k, k + 110 * 60), 3)
    a, b = longest_speech_block(times)
    row["longest_speech_block_min"] = [round(a / 60, 1), round(b / 60, 1)]
    row["coverage_whole"] = round(coverage(times, 0, meta.get("duration") or 0), 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-spares", action="store_true")
    ap.add_argument("--alts", action="store_true", help="also check alternative sources")
    args = ap.parse_args()
    rows = []
    for m in load_matches(include_spares=args.include_spares):
        vids = [m["youtube"]] + (m.get("alt", []) if args.alts else [])
        for v in vids:
            row = check(m, v)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    repo_path("data/sources.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
