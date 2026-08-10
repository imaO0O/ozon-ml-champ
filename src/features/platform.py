"""Доли пользователя в суточном объёме площадки — признаки, не зависящие от уровня.

Зачем: уровень площадки сильно менялся внутри обучающего периода (GMV на
пользователя за 30 дней — 58 в январе 2025, 111 в декабре, 84 в январе 2026),
а тестовое окно лежит за краем данных. Модель, выученная на абсолютных рублях,
переносит на тест уровень обучающих срезов. Доля «сколько процентов дневного
GMV площадки сделал этот клиент» от уровня свободна, поэтому переносится честно.

Год назад окно, аналогичное тестовому (14.02–15.03), было на 16% выше
предыдущих 30 дней (+0.17 в шкале log1p) — цена ошибки в уровне велика.

Использует суточный контекст из `day_context`: day_gmv, day_ord, day_searches,
day_active_users. Производные ссылаются на агрегаты блоков windows и lifetime.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from .registry import aggs_block, derived_block

EPS = 1e-9
SHARE_WINDOWS = [30, 90, 365]


@aggs_block("platform")
def platform_aggs(cutoff: dt.date) -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for w in SHARE_WINDOWS:
        inw = pl.col("days_ago") <= w
        out += [
            # Сумма суточных долей: устойчива к тому, что сам масштаб площадки плывёт.
            (pl.col("gmv") / (pl.col("day_gmv") + EPS)).filter(inw).sum().alias(f"share_gmv_{w}"),
            (pl.col("to_ord") / (pl.col("day_ord") + EPS)).filter(inw).sum().alias(f"share_ord_{w}"),
            (pl.col("searches") / (pl.col("day_searches") + EPS)).filter(inw).sum().alias(f"share_searches_{w}"),
            # Насколько «людный» день выбирает пользователь: активен ли он в пики площадки.
            pl.col("day_active_users").filter(inw).mean().alias(f"day_crowd_mean_{w}"),
        ]
    out.append(
        (pl.col("gmv") / (pl.col("day_gmv") + EPS)).max().alias("share_gmv_max")
    )
    return out


@derived_block("platform")
def platform_derived() -> list[pl.Expr]:
    return [
        # Пользователь растёт или сдувается относительно площадки, а не в рублях.
        (pl.col("share_gmv_30") * 12 / (pl.col("share_gmv_365") + EPS)).alias("share_momentum_30_365"),
        (pl.col("share_gmv_90") * 4 / (pl.col("share_gmv_365") + EPS)).alias("share_momentum_90_365"),
        (pl.col("share_gmv_30") * 3 / (pl.col("share_gmv_90") + EPS)).alias("share_momentum_30_90"),
        (pl.col("share_ord_90") * 4 / (pl.col("share_ord_365") + EPS)).alias("share_ord_momentum_90_365"),
        # Средняя доля в день активности — «вес» клиента для площадки.
        (pl.col("share_gmv_365") / (pl.col("active_days_365") + EPS)).alias("share_gmv_per_active_365"),
        (pl.col("share_gmv_90") / (pl.col("ord_days_90") + EPS)).alias("share_gmv_per_ord_90"),
    ]
