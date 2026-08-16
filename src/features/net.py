"""Предсказание нейросети как признак бустинга — стекинг вместо бленда.

Сейчас сеть и бустинг соединяются блендом, то есть взвешенной суммой с одним
общим весом на всех клиентов. Стекинг сильнее: бустинг видит предсказание сети
как обычный признак и может **выучить, где сети верить, а где нет** — например,
доверять ей на активных клиентах и игнорировать на спящих.

Файлы готовит трек C: `models/netoof_<срез>.npz` с полями `user_id`, `pred_log`
и `models/netoof_test.npz` для тестового среза. Предсказания out-of-fold, то
есть на каждом срезе сеть не видела его таргет.

**Уровень сети подавать нельзя.** Среднее её предсказаний скачет между срезами
от 2.149 до 2.475, а на тесте равно 2.379 — размах 0.32 при стандартном
отклонении остатка около 1.65. Абсолютное значение означает разное на разных
срезах, деревья выучили бы по нему сам срез, а на тесте оказались бы вне
диапазона. Ровно та же болезнь, что лечит блок `ranks`.

Поэтому подаются две величины, обе не зависящие от уровня:

* `net_rank` — процентильный ранг предсказания внутри среза;
* `net_centered` — предсказание минус среднее по этому же срезу.

Ранг устойчивее к сдвигу формы, центрирование сохраняет масштаб различий.
Какую возьмут деревья — вопрос эксперимента, поэтому даются обе.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from config import MODELS


def net_path(cutoff: dt.date, test_cutoff: dt.date) -> "object":
    name = "netoof_test.npz" if cutoff == test_cutoff else f"netoof_{cutoff.isoformat()}.npz"
    return MODELS / name


def load_net(cutoff: dt.date, test_cutoff: dt.date) -> pl.DataFrame | None:
    """Предсказания сети для среза; None, если файла нет."""
    path = net_path(cutoff, test_cutoff)
    if not path.exists():
        return None
    with np.load(path) as z:
        if "user_id" not in z or "pred_log" not in z:
            raise SystemExit(f"{path.name}: нужны поля user_id и pred_log, есть {list(z.keys())}")
        return pl.DataFrame({
            "user_id": z["user_id"].astype(np.int64),
            "_net_pred": z["pred_log"].astype(np.float64),
        })


def attach(df: pl.DataFrame, cutoff: dt.date, test_cutoff: dt.date) -> pl.DataFrame:
    """Добавить к выборке ранг и центрированное предсказание сети."""
    net = load_net(cutoff, test_cutoff)
    if net is None:
        raise SystemExit(
            f"нет файла предсказаний сети для среза {cutoff}: "
            f"ожидается {net_path(cutoff, test_cutoff)}")

    out = df.join(net, on="user_id", how="left")
    missing = out["_net_pred"].null_count()
    if missing:
        print(f"  [{cutoff}] без предсказания сети: {missing:,} из {out.height:,} — будут пропуски")

    return out.with_columns(
        (pl.col("_net_pred").rank(method="average") / pl.len())
        .cast(pl.Float32).alias("net_rank"),
        (pl.col("_net_pred") - pl.col("_net_pred").mean())
        .cast(pl.Float32).alias("net_centered"),
    ).drop("_net_pred")
