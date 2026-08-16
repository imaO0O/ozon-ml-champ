"""Предсказания нейросетей как признаки бустинга — стекинг вместо бленда.

Бленд соединяет сеть и бустинг взвешенной суммой с одним общим весом на всех
клиентов. Стекинг сильнее: бустинг видит предсказание сети обычным признаком
и может **выучить, где сети верить, а где нет** — например, доверять ей на
активных клиентах и игнорировать на спящих. Проверено: 0.00255 на январском
срезе и 0.00228 на декабрьском.

**Уровень сети подавать нельзя.** Среднее её предсказаний скачет между срезами
от 2.149 до 2.475, а на тесте равно 2.379 — размах 0.32 при стандартном
отклонении остатка около 1.65. Абсолютное значение означает разное на разных
срезах, деревья выучили бы по нему сам срез, а на тесте оказались бы вне
диапазона. Ровно та же болезнь, что лечит блок `ranks`.

Поэтому на каждую сеть подаются две величины, обе не зависящие от уровня:
ранг внутри среза и предсказание минус среднее по срезу. Какую возьмут
деревья — вопрос эксперимента, поэтому даются обе (на первом прогоне взяли
обе, центрированную вдвое охотнее).

## Имена файлов

Одна сеть (текущий рабочий вариант):

    models/netoof_<срез>.npz     поля user_id, pred_log
    models/netoof_test.npz

Несколько сетей — добавляется имя сети, и признаки получают тот же суффикс
(`net_<имя>_rank`, `net_<имя>_centered`):

    models/netoof_<имя>_<срез>.npz
    models/netoof_<имя>_test.npz

Список имён задаётся через `--net-names`, например `--net-names r180,ch180,w90`.
Смысл нескольких сетей: у трека C окна 180 на рангах, 180 на каналах долей и 90
взаимно коррелируют на 0.987–0.994, то есть ошибаются на разных клиентах. Три
признака дают бустингу различать не только «где верить сети», но и «какой сети
верить здесь».

Предсказания out-of-fold: на каждом срезе сеть обучена на пяти срезах строго
старше и таргета этого среза не видела.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from config import MODELS

DEFAULT_NAMES: list[str] = []          # пусто = одна безымянная сеть


def net_path(cutoff: dt.date, test_cutoff: dt.date, name: str = "") -> "object":
    tag = f"{name}_" if name else ""
    tail = "test" if cutoff == test_cutoff else cutoff.isoformat()
    return MODELS / f"netoof_{tag}{tail}.npz"


def load_net(cutoff: dt.date, test_cutoff: dt.date, name: str = "") -> pl.DataFrame | None:
    """Предсказания сети для среза; None, если файла нет."""
    path = net_path(cutoff, test_cutoff, name)
    if not path.exists():
        return None
    with np.load(path) as z:
        if "user_id" not in z or "pred_log" not in z:
            raise SystemExit(f"{path.name}: нужны поля user_id и pred_log, есть {list(z.keys())}")
        return pl.DataFrame({
            "user_id": z["user_id"].astype(np.int64),
            "_net_pred": z["pred_log"].astype(np.float64),
        })


def attach(df: pl.DataFrame, cutoff: dt.date, test_cutoff: dt.date,
           names: list[str] | None = None) -> pl.DataFrame:
    """Добавить к выборке ранг и центрированное предсказание каждой сети."""
    names = names if names else DEFAULT_NAMES or [""]
    out = df
    for name in names:
        net = load_net(cutoff, test_cutoff, name)
        if net is None:
            raise SystemExit(
                f"нет файла предсказаний сети для среза {cutoff}: "
                f"ожидается {net_path(cutoff, test_cutoff, name)}")

        out = out.join(net, on="user_id", how="left")
        missing = out["_net_pred"].null_count()
        if missing:
            print(f"  [{cutoff}] сеть '{name or 'net'}': без предсказания "
                  f"{missing:,} из {out.height:,} — будут пропуски")

        suffix = f"_{name}" if name else ""
        out = out.with_columns(
            (pl.col("_net_pred").rank(method="average") / pl.len())
            .cast(pl.Float32).alias(f"net{suffix}_rank"),
            (pl.col("_net_pred") - pl.col("_net_pred").mean())
            .cast(pl.Float32).alias(f"net{suffix}_centered"),
        ).drop("_net_pred")
    return out
