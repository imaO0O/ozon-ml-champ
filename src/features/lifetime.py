"""Вся история, рецентность и BTYD-величины (T, t_x, частота)."""
from __future__ import annotations

import datetime as dt

import polars as pl

from .registry import aggs_block


@aggs_block("lifetime")
def lifetime_aggs(cutoff: dt.date) -> list[pl.Expr]:
    paid = pl.col("gmv") > 0
    return [
        pl.len().alias("lt_active_days"),
        paid.sum().alias("lt_ord_days"),
        pl.col("gmv").sum().alias("lt_gmv"),
        pl.col("gmv").max().alias("lt_gmv_max"),
        pl.col("gmv").filter(paid).mean().alias("lt_avg_check"),
        pl.col("gmv").filter(paid).std().alias("lt_check_std"),
        pl.col("to_ord").sum().alias("lt_to_ord"),
        pl.col("searches").sum().alias("lt_searches"),
        # Рецентность (в днях до cutoff).
        pl.col("days_ago").min().alias("rec_any"),
        pl.col("days_ago").max().alias("tenure"),          # дней с первого события = T в BTYD
        pl.col("days_ago").filter(paid).min().alias("rec_ord"),
        pl.col("days_ago").filter(paid).max().alias("first_ord_ago"),
        pl.col("days_ago").filter(paid).bottom_k(2).max().alias("rec_ord_prev"),
        pl.col("days_ago").filter(pl.col("to_cart") > 0).min().alias("rec_cart"),
        pl.col("days_ago").filter(pl.col("searches") > 0).min().alias("rec_search"),
    ]
