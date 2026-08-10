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
    """Дописать строку, пережив изменение набора колонок.

    Журналы ведут пятеро, и рано или поздно кто-то добавит свою колонку.
    Без миграции заголовок остаётся старым, а строки пишутся новым набором —
    и весь файл разъезжается. Поэтому при расхождении файл перезаписывается
    объединённым набором колонок, старые строки добираются пустыми значениями.
    """
    header: list[str] = []
    old_rows: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            old_rows = list(reader)

    if header and header != fields:
        fields = header + [c for c in fields if c not in header]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for old in old_rows:
                w.writerow({k: old.get(k) or "" for k in fields})
        print(f"схема {path.name} изменилась, файл перезаписан: {len(fields)} колонок")

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"записано в {path}")
