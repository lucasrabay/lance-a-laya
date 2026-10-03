"""Paths and config loading shared by every stage."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel: str) -> Path:
    return ROOT / rel


def load_matches(include_spares: bool = False, split: str | None = None) -> list[dict]:
    cfg = yaml.safe_load(repo_path("configs/matches.yaml").read_text())
    matches = list(cfg["matches"])
    if include_spares:
        matches += [{**m, "split": "spare"} for m in cfg.get("spares", [])]
    if split:
        matches = [m for m in matches if m.get("split") == split]
    return matches


def get_match(match_id: str) -> dict:
    for m in load_matches(include_spares=True):
        if m["id"] == match_id:
            return m
    raise KeyError(match_id)
