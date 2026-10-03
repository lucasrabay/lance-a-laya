"""Load raw Whisper transcripts and clean them into a flat, time-sorted word list.

Raw transcripts on the Volume are never modified; cleaning happens here so it can change without
re-running ASR. Cleaning removes stock subtitle hallucinations, likely-silent segments, and runaway
repetition loops (a narrator's "gol, gol, gol" survives: only repeats beyond `MAX_REPEATS` are cut).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lal.config import repo_path

TRANSCRIPT_DIR = repo_path("data/transcripts")
STOCK_HALLUCINATIONS = re.compile(
    r"amara\.org|legendas? (pela|por)|obrigad[oa] por assistir|inscreva-se|ative o sininho|"
    r"legendado por|tradução e legendas|www\.|\.com\b",
    re.I,
)
MAX_REPEATS = 4  # keep at most this many consecutive repeats of the same 1-3 word phrase


def load_segments(match_id: str, name: str | None = None) -> list[dict]:
    path = TRANSCRIPT_DIR / f"{name or match_id}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def _norm(w: str) -> str:
    return re.sub(r"[^\wáéíóúâêôãõçà]+", "", w.lower())


def _cut_loops(words: list[dict]) -> list[dict]:
    """Drop repetitions of an n-gram (n=1..3) beyond MAX_REPEATS consecutive occurrences."""
    out: list[dict] = []
    for w in words:
        out.append(w)
        for n in (1, 2, 3):
            k = n * (MAX_REPEATS + 1)
            if len(out) < k:
                continue
            tail = [_norm(x["word"]) for x in out[-k:]]
            gram = tail[-n:]
            if all(tail[i : i + n] == gram for i in range(0, k, n)) and any(gram):
                del out[-n:]
                break
    return out


def clean_words(segments: list[dict]) -> tuple[list[dict], dict]:
    stats = {"segments": len(segments), "dropped_stock": 0, "dropped_silent": 0, "words_in": 0, "loop_cut": 0}
    words: list[dict] = []
    for seg in segments:
        if STOCK_HALLUCINATIONS.search(seg["text"]):
            stats["dropped_stock"] += 1
            continue
        if seg.get("no_speech_prob", 0) > 0.8 and seg.get("avg_logprob", 0) < -1.0:
            stats["dropped_silent"] += 1
            continue
        seg_words = [w for w in seg.get("words", []) if w["word"].strip()]
        stats["words_in"] += len(seg_words)
        words.extend(seg_words)
    words.sort(key=lambda w: w["start"])
    cleaned = _cut_loops(words)
    stats["loop_cut"] = len(words) - len(cleaned)
    stats["words_out"] = len(cleaned)
    return cleaned, stats


def load_words(match_id: str, name: str | None = None) -> list[dict]:
    return clean_words(load_segments(match_id, name))[0]


def text_between(words: list[dict], start: float, end: float) -> str:
    """Concatenate words whose start time falls in [start, end)."""
    return "".join(w["word"] for w in words if start <= w["start"] < end).strip()
