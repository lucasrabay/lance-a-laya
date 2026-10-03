"""Teacher labels from Claude subagents: export transcript packs, then validate and load their output.

Pack (data/interim/teacher_packs/{match}.txt, gitignored): the in-play transcript grouped into 20 s blocks,
each headed with its id, stream-time range and match clock. Subagents judge from the narration only (they
never see event data) and write data/interim/teacher/{match}.jsonl with two record kinds:
  {"kind": "span", "q": "goal|big_chance|controversy|card", "start": s, "end": e, "uncertain": bool, "note": str}
  {"kind": "intensity", "block": "B0001", "level": "calm|building|peak", "uncertain": bool}
"""

from __future__ import annotations

import argparse
import json

from lal.config import load_matches, repo_path
from lal.schema import INTENSITY_LEVELS, NOUL_QUESTIONS
from lal.transcripts import load_segments, clean_words

PACK_DIR = repo_path("data/interim/teacher_packs")
TEACHER_DIR = repo_path("data/interim/teacher")
BLOCK_S = 20.0
PERIOD_NAMES = {1: "1T", 2: "2T", 3: "1T-prorrogação", 4: "2T-prorrogação"}


def clock(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60):02d}:{int(t % 60):02d}"


def blocks_for(align: dict, margin: float = 10.0) -> list[dict]:
    out = []
    for p_str, per in sorted(align["periods"].items(), key=lambda kv: int(kv[0])):
        s = per["audio_start"] - margin
        while s < per["audio_end"] + margin:
            out.append({"block": f"B{len(out) + 1:04d}", "period": int(p_str), "start": round(s, 1),
                        "end": round(s + BLOCK_S, 1), "clock": clock(s - per["offset"])})
            s += BLOCK_S
    return out


def export_pack(match_id: str) -> tuple[int, int]:
    align = json.loads(repo_path(f"data/align/{match_id}.json").read_text())
    words, _ = clean_words(load_segments(match_id))
    blocks = blocks_for(align)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Narration transcript, match {match_id}. Times are stream seconds. Each [Bxxxx] block is 20 s.",
        "# Lines inside a block: '<stream_second> <words spoken from then on>' (ASR output; may contain errors).",
    ]
    wi, cur_period = 0, None
    starts = [w["start"] for w in words]
    for b in blocks:
        if b["period"] != cur_period:
            cur_period = b["period"]
            per = align["periods"][str(cur_period)]
            lines.append(f"\n=== PERIOD {cur_period} ({PERIOD_NAMES.get(cur_period, cur_period)}): kickoff at "
                         f"stream {per['audio_start']:.1f} s, ends at {per['audio_end']:.1f} s ===")
        lines.append(f"[{b['block']} | {b['start']:.0f}-{b['end']:.0f} | {PERIOD_NAMES.get(b['period'])} "
                     f"{b['clock']}]")
        while wi < len(words) and starts[wi] < b["start"]:
            wi += 1
        # group words into ~5 s lines so the subagent can place spans precisely
        line_start, buf = None, []
        while wi < len(words) and starts[wi] < b["end"]:
            if line_start is None:
                line_start = starts[wi]
            buf.append(words[wi]["word"])
            if starts[wi] - line_start >= 5.0:
                lines.append(f"  {line_start:.1f}{''.join(buf)}")
                line_start, buf = None, []
            wi += 1
        if buf:
            lines.append(f"  {line_start:.1f}{''.join(buf)}")
    (PACK_DIR / f"{match_id}.txt").write_text("\n".join(lines) + "\n")
    # The block time ranges the subagent saw. Labels must be mapped back with these, not with blocks
    # recomputed from a later alignment.
    (PACK_DIR / f"{match_id}.blocks.json").write_text(json.dumps(blocks))
    return len(blocks), len(lines)


def load_blocks(match_id: str) -> list[dict]:
    return json.loads((PACK_DIR / f"{match_id}.blocks.json").read_text())


def load_teacher(match_id: str) -> tuple[list[dict], dict[str, dict], list[str]]:
    """(spans, intensity by block id, validation problems) from a subagent's output file."""
    spans, intensity, problems = [], {}, []
    path = TEACHER_DIR / f"{match_id}.jsonl"
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {i + 1}: not JSON")
            continue
        if r.get("kind") == "span":
            if r.get("q") not in NOUL_QUESTIONS or not isinstance(r.get("start"), (int, float)) \
                    or not isinstance(r.get("end"), (int, float)) or r["end"] < r["start"]:
                problems.append(f"line {i + 1}: bad span {r}")
                continue
            spans.append({"question": r["q"], "start": float(r["start"]), "end": float(r["end"]),
                          "uncertain": bool(r.get("uncertain")), "note": r.get("note", ""), "source": "teacher"})
        elif r.get("kind") == "intensity":
            if r.get("level") not in INTENSITY_LEVELS:
                problems.append(f"line {i + 1}: bad intensity {r}")
                continue
            intensity[r["block"]] = {"level": r["level"], "uncertain": bool(r.get("uncertain"))}
        else:
            problems.append(f"line {i + 1}: unknown kind")
    return spans, intensity, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", action="append")
    args = ap.parse_args()
    for mid in args.match or [m["id"] for m in load_matches()]:
        n_blocks, n_lines = export_pack(mid)
        print(f"{mid:14s} blocks={n_blocks} lines={n_lines}")


if __name__ == "__main__":
    main()
