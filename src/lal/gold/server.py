"""Tiny local labeling server for the gold set (stdlib only).

    uv run python -m lal.gold.server            # http://127.0.0.1:8765  (gold labeling)
    uv run python -m lal.gold.server --review   # pooled trigger review (Phase 5)

Labels are saved on every keystroke-confirmed item to data/gold/gold.jsonl (window ids + labels, no text),
or data/gold/review.jsonl in review mode. The page never sees strata, weights, or model/teacher labels.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lal.config import repo_path
from lal.schema import LABELING_GUIDE

GOLD_DIR = Path(os.environ.get("LAL_GOLD_DIR") or repo_path("data/gold"))  # override for UI demos
HERE = Path(__file__).parent


class State:
    review = False

    @classmethod
    def items_path(cls) -> Path:
        return GOLD_DIR / ("review_items.jsonl" if cls.review else "sample.jsonl")

    @classmethod
    def labels_path(cls) -> Path:
        return GOLD_DIR / ("review.jsonl" if cls.review else "gold.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def save_label(record: dict) -> None:
    path = State.labels_path()
    rows = {r["idx"]: r for r in read_jsonl(path)}
    rows[record["idx"]] = record
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for idx in sorted(rows):
            f.write(json.dumps(rows[idx], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._json({"review": State.review, "guide": LABELING_GUIDE})
        elif self.path == "/api/items":
            blind = ("idx", "wid", "text", "prev_text", "next_text", "question", "clock")
            items = [{k: it[k] for k in blind if k in it} for it in read_jsonl(State.items_path())]
            for it in items:
                it["audio"] = (GOLD_DIR / "audio" / f"{it['idx']}.m4a").exists() and not State.review
            self._json(items)
        elif self.path == "/api/labels":
            self._json({r["idx"]: r for r in read_jsonl(State.labels_path())})
        elif self.path.startswith("/audio/") and self.path.endswith(".m4a"):
            f = GOLD_DIR / "audio" / Path(self.path).name
            if f.exists():
                self._send(200, f.read_bytes(), "audio/mp4")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/label":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        save_label(body)
        self._json({"ok": True})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()
    State.review = args.review
    if not State.items_path().exists():
        raise SystemExit(f"missing {State.items_path()}")
    print(f"labeling {'review' if args.review else 'gold'} items: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
