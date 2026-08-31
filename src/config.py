"""Пути и константы соревнования E-CUP 2026, задача 3 (LTV поиска)."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# Импорт ради побочного действия: в консоли cp1251 (русская Windows по
# умолчанию) печать `−`, `×` или `σ` роняет скрипт после того, как работа
# сделана. Разбор — в самом модуле. Здесь, потому что config импортируют
# 55 скриптов из 61: одно место вместо соглашения «не забудь добавить».
import console  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

# OZON_TRAIN / OZON_SUBMIT позволяют прогнать пайплайн на синтетике (make_fake_data.py)
# или на данных, лежащих вне репозитория, не трогая код.
TRAIN_PARQUET = Path(os.environ.get("OZON_TRAIN", DATA_RAW / "train.parquet"))
SAMPLE_SUBMIT = Path(os.environ.get("OZON_SUBMIT", DATA_RAW / "sample_submit.csv"))

# Горизонт таргета: сумма gmv за 30 дней, начиная с cutoff (включительно).
HORIZON = 30

# Данные: 2025-01-01 .. 2026-02-13. Предсказываем 2026-02-14 .. 2026-03-15.
DATA_START = dt.date(2025, 1, 1)
DATA_END = dt.date(2026, 2, 13)
TEST_CUTOFF = dt.date(2026, 2, 14)

# Обучающие cutoff'ы: таргет должен целиком помещаться в известные данные,
# т.е. cutoff + 29 <= DATA_END  ->  cutoff <= 2026-01-15.
# Шаг 30 дней, чтобы окна таргета не пересекались.
LAST_TRAIN_CUTOFF = dt.date(2026, 1, 15)
N_TRAIN_CUTOFFS = 6

# Шаг между срезами. При шаге меньше HORIZON окна таргета соседних срезов
# пересекаются: примеров больше, но они скоррелированы, а обучающие срезы
# рядом с валидационным приходится отбрасывать (карантин в train.py).
CUTOFF_STRIDE = HORIZON


def train_cutoffs(n: int = N_TRAIN_CUTOFFS, stride: int | None = None) -> list[dt.date]:
    """От свежего к старому: при шаге 30 — 2026-01-15, 2025-12-16, 2025-11-16, ..."""
    step = CUTOFF_STRIDE if stride is None else stride
    return [LAST_TRAIN_CUTOFF - dt.timedelta(days=step * i) for i in range(n)]

# Окна для агрегатов (в днях до cutoff).
WINDOWS = [7, 14, 30, 60, 90, 180, 365]

SEED = 42

for _d in (DATA_RAW, DATA_PROC, MODELS, SUBMISSIONS):
    _d.mkdir(parents=True, exist_ok=True)
