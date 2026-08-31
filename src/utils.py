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


def machine_tag() -> str:
    """На какой машине получена строка журнала.

    Колонка появилась 31.08 после **коллизии имён**: обе стороны кроссмашинной
    сверки назвали свой прогон `gru_w13_mirror`, и в общем журнале оказалась
    одна строка вместо двух. До кроссмашинной работы имя определяло прогон
    однозначно; как только один и тот же состав стали гонять на разных картах,
    перестало — а поля для машины в схеме не было.

    Развести строки суффиксом (`..._4060ti`) можно, но такое соглашение
    держится на памяти участников, а колонка — на схеме. Поэтому значение
    ставится САМО (см. `AUTO_FIELDS`), а не рукой: забыть его нельзя.

    Машина опознаётся по видеокарте. Это единственный признак, который скрипт
    знает сам и который не является персональными данными: имя хоста в общий
    журнал публичного репозитория мы не пишем. Отсюда прямое ограничение,
    которое лучше назвать, чем спрятать: **строки без CUDA между собой
    не различаются** — у бустинга на любой из машин здесь будет `cpu`.

    Префикс `NVIDIA GeForce ` снимается, чтобы значение совпадало с тем, как
    карты названы в документах жюри (`RTX 5070`). Это отрезание literal-строки,
    а не разбор названия: всё остальное идёт как есть.
    """
    try:
        import torch  # noqa: PLC0415  (тяжёлый импорт, бустингу он не нужен)

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    except Exception:
        pass
    return "cpu"


# Колонки, которые журнал заполняет сам, если писатель их не заполнил.
#
# Почему здесь, а не в девяти местах записи. Дефект, ради которого колонка
# заведена, — это соглашение, которое держалось на памяти двоих и потому
# не сработало. Требование «каждый пишущий скрипт не забудь проставить
# машину» — то же самое соглашение уровнем выше, и оно откажет так же.
# Здесь через одну функцию проходят все девять писателей и все будущие.
#
# Заполняется только колонка, которая в наборе уже есть: `submissions/log.csv`
# идёт через ту же функцию, и лишних колонок там не появится.
AUTO_FIELDS = {"machine": machine_tag}


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

    auto = {c: fn() for c, fn in AUTO_FIELDS.items()
            if c in fields and not row.get(c)}
    if auto:
        row = {**row, **auto}  # копия: словарь вызывающего не трогаем

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(f"записано в {path}")
