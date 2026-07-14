"""Shared helpers: project paths, JSONL I/O, text cleanup."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path, records) -> None:
    """Write records as JSON Lines: one JSON object per line.

    JSONL is the standard corpus format for text pipelines because it
    streams (you never need the whole file in memory) and diffs cleanly
    in git.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_whitespace(text) -> str:
    """Collapse all runs of whitespace (incl. newlines) into single spaces."""
    return " ".join((text or "").split())
