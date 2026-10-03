"""Download narration audio (audio-only, low bitrate) and optionally push it to the Modal Volume.

Downloads run locally because YouTube blocks most datacenter IPs and some CazéTV VODs are Brazil-only.
Format 249 (opus ~48 kbps) or 139 (HE-AAC ~49 kbps) is plenty for ASR and needs no ffmpeg.
With --upload the file goes to the `lal-data` Volume under audio/ and the local copy is deleted.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from lal.config import load_matches, repo_path

AUDIO_DIR = repo_path("data/audio")
VOLUME = "lal-data"


def local_audio(match_id: str) -> Path | None:
    hits = [p for p in AUDIO_DIR.glob(f"{match_id}.*") if p.suffix in (".webm", ".m4a", ".opus")]
    return hits[0] if hits else None


def download(match: dict) -> Path:
    existing = local_audio(match["id"])
    if existing:
        return existing
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "yt-dlp", "--js-runtimes", "node", "--no-warnings", "--no-playlist",
            "-f", "249/139/250/251/140", "--concurrent-fragments", "4",
            "-o", str(AUDIO_DIR / f"{match['id']}.%(ext)s"),
            f"https://www.youtube.com/watch?v={match['youtube']}",
        ],
        check=True,
    )
    path = local_audio(match["id"])
    if path is None:
        raise RuntimeError(f"download produced no audio file for {match['id']}")
    return path


def upload(path: Path, keep_local: bool = False) -> None:
    subprocess.run(["modal", "volume", "put", "--force", VOLUME, str(path), f"audio/{path.name}"], check=True)
    if not keep_local:
        path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", action="append", help="match id (repeatable); default: all registered matches")
    ap.add_argument("--upload", action="store_true", help="push to the Modal Volume and delete the local copy")
    ap.add_argument("--keep-local", action="store_true", help="with --upload: keep the local copy")
    ap.add_argument("--include-spares", action="store_true")
    args = ap.parse_args()
    matches = load_matches(include_spares=args.include_spares or bool(args.match))
    if args.match:
        matches = [m for m in matches if m["id"] in args.match]
    for m in matches:
        path = download(m)
        print(f"{m['id']}: {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        if args.upload:
            upload(path, keep_local=args.keep_local)
            print(f"{m['id']}: uploaded to {VOLUME}:audio/{path.name}", flush=True)


if __name__ == "__main__":
    main()
