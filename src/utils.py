"""Мелочи, общие для train.py и predict.py."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from config import ROOT


def git_commit() -> str:
    """Короткий хеш HEAD — чтобы любую строку журнала можно было воспроизвести."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=ROOT)
        head = r.stdout.strip() or "?"
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, cwd=ROOT).stdout.strip()
        return f"{head}+dirty" if dirty else head
    except Exception:
        return "?"


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"записано в {path}")
