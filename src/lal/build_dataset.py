"""Merge event labels and teacher labels into per-window labels, log disagreements, build Laya datasets.

Per question (positive spans / ignore spans), projected onto windows with weak_labels.project:
- goal:        positives = StatsBomb goals. Teacher goal spans with no event goal -> ignore + logged.
- card:        positives = teacher card spans inside a StatsBomb card search region. A region where the
               teacher found no card call -> the whole region is ignored (logged); teacher card spans
               outside every region -> ignore + logged.
- big_chance:  positives = StatsBomb high-xG/woodwork shots UNION teacher spans.
- controversy: positives = StatsBomb penalty fouls UNION teacher spans.
- any teacher span marked uncertain -> ignore for that question.
- intensity:   teacher block levels; a window takes the max level over blocks it overlaps by >= 5 s,
               ignored if that block is marked uncertain.
Outputs: data/labels/{match}.jsonl (window ids + labels, no text: committed), data/labels/disagreements.jsonl,
data/labels/balance.json, and data/datasets/{train,val}.jsonl in the Laya fine-tuning format (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from lal.config import load_matches, repo_path
from lal.schema import INTENSITY_LEVELS, NOUL_QUESTIONS, QUESTIONS, state_for
from lal.teacher import load_blocks, load_teacher
from lal.weak_labels import event_spans, load_events, overlap, project
from lal.windows import load_windows, pipeline_cfg

LABEL_DIR = repo_path("data/labels")
DATASET_DIR = repo_path("data/datasets")


def _overlaps_any(span: dict, others: list[dict]) -> bool:
    return any(overlap(span["start"], span["end"], o["start"], o["end"]) > 0 for o in others)


def merge_match(match_id: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    align = json.loads(repo_path(f"data/align/{match_id}.json").read_text())
    windows = load_windows(match_id)
    ev = event_spans(load_events(match_id), align, cfg)
    t_spans, t_intensity, problems = load_teacher(match_id)
    disagreements = [{"match": match_id, "kind": "teacher_file_problem", "detail": p} for p in problems]

    def log(kind, span, q):
        disagreements.append({"match": match_id, "kind": kind, "question": q,
                              "start": span["start"], "end": span["end"], "note": span.get("note", "")})

    teacher = defaultdict(list)
    for s in t_spans:
        teacher[s["question"]].append(s)
    pos, ign = defaultdict(list), defaultdict(list)

    # goal: event is ground truth; the rest of a matching teacher goal span (the long celebration call) is
    # ignored rather than negative
    ev_goal = [s for s in ev if s["question"] == "goal"]
    pos["goal"] += ev_goal
    ign["goal"] += [s for s in teacher["goal"] if _overlaps_any(s, ev_goal)]
    for s in teacher["goal"]:
        if not _overlaps_any(s, ev_goal):
            ign["goal"].append(s)
            log("teacher_goal_without_event", s, "goal")
    for s in ev_goal:
        if not _overlaps_any(s, [t for t in teacher["goal"]]):
            log("event_goal_without_teacher", s, "goal")

    # card: teacher pinpoints inside event search regions
    regions = [s for s in ev if s["question"] == "card_search"]
    for s in teacher["card"]:
        if s["uncertain"]:
            ign["card"].append(s)
        elif _overlaps_any(s, regions):
            pos["card"].append(s)
        else:
            ign["card"].append(s)
            log("teacher_card_outside_event_region", s, "card")
    for r in regions:
        if not _overlaps_any(r, [s for s in teacher["card"] if not s["uncertain"]]):
            ign["card"].append(r)
            log("event_card_not_found_by_teacher", r, "card")

    # big_chance / controversy: union
    for q in ("big_chance", "controversy"):
        ev_q = [s for s in ev if s["question"] == q]
        pos[q] += ev_q
        for s in teacher[q]:
            (ign if s["uncertain"] else pos)[q].append(s)
        for s in ev_q:
            if not _overlaps_any(s, teacher[q]):
                log(f"event_{q}_without_teacher", s, q)
    for s in teacher["goal"]:
        if s["uncertain"]:
            ign["goal"].append(s)

    lab_cfg = cfg
    labels = {w["wid"]: {} for w in windows}
    for q in NOUL_QUESTIONS:
        proj = project(windows, pos[q], lab_cfg["min_overlap_s"], lab_cfg["ignore_near_s"])
        for w, y in zip(windows, proj):
            if y == 0 and any(overlap(w["start"], w["end"], s["start"], s["end"]) > 0 for s in ign[q]):
                y = None
            labels[w["wid"]][q] = y

    blocks = load_blocks(match_id)
    covered = [(b["start"], b["end"]) for b in blocks]
    missing_blocks = [b["block"] for b in blocks if b["block"] not in t_intensity]
    if missing_blocks:
        disagreements.append({"match": match_id, "kind": "teacher_missing_intensity_blocks",
                              "detail": f"{len(missing_blocks)} blocks"})
    for w in windows:
        best, best_unc = None, False
        for b in blocks:
            if overlap(w["start"], w["end"], b["start"], b["end"]) >= 5 and b["block"] in t_intensity:
                lv = INTENSITY_LEVELS.index(t_intensity[b["block"]]["level"])
                if best is None or lv > best:
                    best, best_unc = lv, t_intensity[b["block"]]["uncertain"]
        labels[w["wid"]]["intensity"] = None if best_unc else best
        # windows the teacher pack did not cover (pack exported before an alignment fix) cannot be
        # negatives for teacher-dependent questions
        inside = sum(overlap(w["start"], w["end"], a, b) for a, b in covered)
        if inside < w["end"] - w["start"] - 1e-6:
            for q in ("big_chance", "controversy", "card", "intensity"):
                if labels[w["wid"]][q] != 1:
                    labels[w["wid"]][q] = None

    rows = [{"wid": w["wid"], "match": match_id, "split": w["split"], "period": w["period"],
             "start": w["start"], "end": w["end"], "labels": labels[w["wid"]]} for w in windows]
    return rows, disagreements


def balance(rows: list[dict]) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        out[split] = {"windows": len(rs)}
        for q in NOUL_QUESTIONS:
            c = Counter(r["labels"][q] for r in rs)
            out[split][q] = {"pos": c[1], "neg": c[0], "ignore": c[None]}
        c = Counter(r["labels"]["intensity"] for r in rs)
        out[split]["intensity"] = {lv: c[i] for i, lv in enumerate(INTENSITY_LEVELS)} | {"ignore": c[None]}
    return out


def gold_probs(q: str, y: int) -> dict:
    if QUESTIONS[q]["type"] == "noul":
        return {"probabilities": {"false": 1.0 - y, "true": float(y)}}
    return {"probabilities": {str(i): float(i == y) for i in range(len(INTENSITY_LEVELS))}}


def laya_rows(rows: list[dict], texts: dict[str, str], rebalance: bool, seed: int = 0,
              neg_ratio: float = 10.0, min_pos: int = 300) -> list[dict]:
    """One Laya case per window; per-question negative capping and positive upsampling for rare classes."""
    rng = random.Random(seed)
    keep_prob = {}
    for q in NOUL_QUESTIONS:
        n_pos = sum(1 for r in rows if r["labels"][q] == 1)
        n_neg = sum(1 for r in rows if r["labels"][q] == 0)
        keep_prob[q] = 1.0 if not rebalance or n_neg == 0 else min(1.0, neg_ratio * max(n_pos, 1) / n_neg)
    out = []
    for r in rows:
        gold = {}
        for q in NOUL_QUESTIONS:
            y = r["labels"][q]
            if y is None or (y == 0 and rng.random() > keep_prob[q]):
                continue
            gold[q] = gold_probs(q, y)
        if r["labels"]["intensity"] is not None:
            gold["intensity"] = gold_probs("intensity", r["labels"]["intensity"])
        if gold:
            out.append({"wid": r["wid"], "state": state_for(texts[r["wid"]]),
                        "questions": {q: QUESTIONS[q] for q in gold}, "gold": gold})
    if rebalance:  # upsample rare positives with single-question copies
        for q in NOUL_QUESTIONS:
            pos = [r for r in rows if r["labels"][q] == 1]
            if 0 < len(pos) < min_pos:
                for i in range(min_pos - len(pos)):
                    r = pos[i % len(pos)]
                    out.append({"wid": r["wid"] + f"#up{q}{i}", "state": state_for(texts[r["wid"]]),
                                "questions": {q: QUESTIONS[q]}, "gold": {q: gold_probs(q, 1)}})
        rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    cfg = pipeline_cfg()["labels"]
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, all_dis, texts = [], [], {}
    for m in load_matches():
        if not (repo_path("data/interim/teacher") / f"{m['id']}.jsonl").exists():
            print(f"skip {m['id']}: no teacher labels yet")
            continue
        rows, dis = merge_match(m["id"], cfg)
        with open(LABEL_DIR / f"{m['id']}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        all_rows += rows
        all_dis += dis
        texts |= {w["wid"]: w["text"] for w in load_windows(m["id"])}
    with open(LABEL_DIR / "disagreements.jsonl", "w") as f:
        for d in all_dis:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    bal = balance(all_rows)
    (LABEL_DIR / "balance.json").write_text(json.dumps(bal, indent=1))
    for split, rebalance in (("train", True), ("val", False)):
        rows = [r for r in all_rows if r["split"] == split]
        data = laya_rows(rows, texts, rebalance=rebalance)
        with open(DATASET_DIR / f"{split}.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_items = sum(len(d["gold"]) for d in data)
        print(f"{split}: {len(data)} cases, {n_items} question items")
    print(json.dumps(bal, indent=1))
    print("disagreements:", dict(Counter(d["kind"] for d in all_dis)))


if __name__ == "__main__":
    main()
