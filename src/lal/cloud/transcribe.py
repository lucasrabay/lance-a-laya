"""Transcribe narration audio on a Modal GPU with faster-whisper large-v3.

Sequential (not batched) decoding on purpose: the batched pipeline in faster-whisper 1.2.1 ignores
temperature fallback and compression_ratio_threshold, the guards against repetition loops. VAD and the
silence-skipping options are disabled because they drop goal calls (see ASR_OPTIONS).

Each stream is clipped to [kickoff_hint - 25 min, kickoff_hint + match length + margin] so we do not pay
for pre/post-game shows; timestamps in the output are absolute stream seconds.

    modal run src/lal/cloud/transcribe.py --match bra-kor-2022 --start 4400 --end 5400   # smoke test
    modal run src/lal/cloud/transcribe.py                                                 # all matches
"""

from __future__ import annotations

import json
import time

import modal

VOLUME_NAME = "lal-data"
MODEL_REPO = "Systran/faster-whisper-large-v3"
MODEL_DIR = "/models/faster-whisper-large-v3"

ASR_OPTIONS = dict(
    language="pt",
    beam_size=5,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    condition_on_previous_text=False,
    word_timestamps=True,
    # All three "skip" mechanisms are OFF on purpose. A/B on three goal clips (diagnose entrypoint): Silero
    # VAD classifies narration shouted over a roaring crowd as non-speech and dropped 30-50 s around every
    # big goal call; the no-speech skip drops whole 30 s windows; hallucination_silence_threshold drops a
    # 10 s "GOOOOL" as an anomaly. Hallucinations are handled post hoc in lal.transcripts instead.
    vad_filter=False,
    no_speech_threshold=None,
    hallucination_silence_threshold=None,
)


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("faster-whisper==1.2.1", "huggingface_hub>=0.25", "pyyaml")
    .run_function(_download_model)
    .add_local_python_source("lal")
)
app = modal.App("lal-transcribe", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu="L4", volumes={"/data": volume}, timeout=4 * 3600, memory=8192)
def transcribe(match_id: str, start_s: float, end_s: float | None, out_name: str | None = None,
               overrides: dict | None = None) -> dict:
    from pathlib import Path

    from faster_whisper import WhisperModel, decode_audio

    volume.reload()
    hits = sorted(p for p in Path("/data/audio").glob(f"{match_id}.*"))
    if not hits:
        raise FileNotFoundError(f"no audio for {match_id} in volume {VOLUME_NAME}:audio/")
    t0 = time.time()
    sr = 16000
    audio = decode_audio(str(hits[0]), sampling_rate=sr)
    full_s = len(audio) / sr
    end_s = min(end_s or full_s, full_s)
    clip = audio[int(start_s * sr) : int(end_s * sr)]
    decode_s = time.time() - t0

    model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
    t1 = time.time()
    options = {**ASR_OPTIONS, **(overrides or {})}
    segments, info = model.transcribe(clip, **options)

    out_dir = Path("/data/transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_name or match_id
    (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for seg in segments:
            row = {
                "start": round(seg.start + start_s, 3),
                "end": round(seg.end + start_s, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "compression_ratio": round(seg.compression_ratio, 3),
                "temperature": seg.temperature,
                "words": [
                    {"start": round(w.start + start_s, 3), "end": round(w.end + start_s, 3),
                     "word": w.word, "p": round(w.probability, 3)}
                    for w in (seg.words or [])
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    asr_s = time.time() - t1
    meta = {
        "match": match_id, "audio_file": hits[0].name, "model": MODEL_REPO, "options": options,
        "clip_start_s": start_s, "clip_end_s": end_s, "audio_total_s": round(full_s, 1),
        "clip_s": round(end_s - start_s, 1), "segments": n, "decode_s": round(decode_s, 1),
        "asr_s": round(asr_s, 1), "realtime_factor": round((end_s - start_s) / max(asr_s, 1e-6), 1),
        "language_probability": round(info.language_probability, 3),
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    volume.commit()
    return meta


def clip_window(match: dict, events_dir, duration_s: float | None) -> tuple[float, float | None]:
    """Stream seconds to transcribe: 25 min before the kickoff hint to the end of play plus margin."""
    from lal.sources import kickoff_utc, metadata

    meta = metadata(match["youtube"])
    hint = None
    ko = kickoff_utc(match)
    if ko and meta.get("live_status") == "was_live" and meta.get("release_timestamp"):
        hint = ko.timestamp() - meta["release_timestamp"]
    events_file = events_dir / f"{match['id']}.jsonl"
    if hint is None or not events_file.exists():  # new match without event data: whole stream
        return 0.0, None
    periods = set()
    with open(events_file) as f:
        for line in f:
            periods.add(json.loads(line)["period"])
    # 2 halves + stoppage + half-time ~ 125 min; extra time adds ~40 min; shootout ~15 min; then margin.
    length_min = 125 + (40 if 3 in periods else 0) + (15 if 5 in periods else 0) + 30
    return max(0.0, hint - 25 * 60), min(hint + length_min * 60, duration_s or 1e9)


@app.local_entrypoint()
def main(match: str = "", start: float = -1, end: float = -1, include_spares: bool = False):
    from lal.config import load_matches, repo_path
    from lal.sources import metadata

    matches = load_matches(include_spares=include_spares)
    if match:
        matches = [m for m in matches if m["id"] == match]
    jobs = []
    for m in matches:
        if start >= 0:
            jobs.append((m["id"], start, end if end > 0 else None, f"{m['id']}.smoke"))
        else:
            s, e = clip_window(m, repo_path("data/events"), metadata(m["youtube"]).get("duration"))
            jobs.append((m["id"], s, e, None))
    for job in jobs:
        print("queued", job)
    for meta in transcribe.starmap(jobs):
        print(json.dumps({k: meta[k] for k in ("match", "clip_s", "segments", "asr_s", "realtime_factor")}))


DIAG_VARIANTS = {
    "A_current": {},
    "B_noskip": {"no_speech_threshold": None, "hallucination_silence_threshold": None},
    "C_noskip_novad": {"no_speech_threshold": None, "hallucination_silence_threshold": None, "vad_filter": False},
}
DIAG_CLIPS = [("eng-fra-2022", 9960, 10060), ("eng-fra-2022", 11360, 11480), ("bra-sui-2022", 9420, 9520)]


@app.local_entrypoint()
def diagnose():
    """A/B the segment-skipping options on goal clips where the default settings dropped the goal call."""
    jobs = [(m, s, e, f"diag/{m}.{s}.{name}", ov) for m, s, e in DIAG_CLIPS for name, ov in DIAG_VARIANTS.items()]
    for meta in transcribe.starmap(jobs):
        print(json.dumps({k: meta[k] for k in ("match", "clip_start_s", "segments", "options")}, default=str)[:200])
