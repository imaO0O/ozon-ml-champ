"""Оконные агрегаты: что пользователь делал за последние 7 … 365 дней."""
from __future__ import annotations

import datetime as dt

import polars as pl

from config import WINDOWS

from .registry import aggs_block

# Колонки лога (см. описание данных соревнования).
MONEY = ["gmv", "gmv_search", "gmv_cat"]
COUNTS = [
    "to_ord",
    "to_cart",
    "searches",
    "search_to_ord",
    "search_to_cart",
    "cat_to_ord",
    "cat_to_cart",
]
FLAGS = [
    "search",
    "cat",
    "has_search_to_ord",
    "has_search_to_cart",
    "has_cat_to_ord",
    "has_cat_to_cart",
]


def _window_aggs(w: int) -> list[pl.Expr]:
    """Агрегаты по окну [cutoff - w, cutoff)."""
    inw = pl.col("days_ago") <= w
    paid = inw & (pl.col("gmv") > 0)
    out: list[pl.Expr] = [
        inw.sum().alias(f"active_days_{w}"),
        paid.sum().alias(f"ord_days_{w}"),
    ]
    for c in MONEY:
        out.append(pl.col(c).filter(inw).sum().alias(f"{c}_sum_{w}"))
    out.append(pl.col("gmv").filter(inw).max().alias(f"gmv_max_{w}"))
    for c in COUNTS:
        out.append(pl.col(c).filter(inw).sum().alias(f"{c}_sum_{w}"))
    # Флаги как число дней с событием — только для коротких окон, чтобы не раздувать ширину.
    if w <= 90:
        for c in FLAGS:
            out.append(pl.col(c).filter(inw).sum().alias(f"{c}_days_{w}"))
    return out


@aggs_block("windows")
def window_aggs(cutoff: dt.date) -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for w in WINDOWS:
        out.extend(_window_aggs(w))
    return out
