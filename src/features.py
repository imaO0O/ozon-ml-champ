"""Построение признаков на заданный cutoff из прореженного дневного лога.

Данные разрежены: строки есть только за дни с активностью. Полную сетку
user x day не разворачиваем — все оконные агрегаты считаем одним group_by
с filter-выражениями внутри агрегации (ленивый polars, память ~O(users)).
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from config import HORIZON, TRAIN_PARQUET, WINDOWS

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


def scan_log(path=TRAIN_PARQUET) -> pl.LazyFrame:
    """Ленивое чтение лога с приведением event_date к Date."""
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    if schema["event_date"] == pl.Utf8:
        lf = lf.with_columns(pl.col("event_date").str.to_date())
    elif schema["event_date"] != pl.Date:
        lf = lf.with_columns(pl.col("event_date").cast(pl.Date))
    return lf


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
    # Флаги как доля дней с событием — только для коротких окон, чтобы не раздувать ширину.
    if w <= 90:
        for c in FLAGS:
            out.append(pl.col(c).filter(inw).sum().alias(f"{c}_days_{w}"))
    return out


def _lifetime_aggs() -> list[pl.Expr]:
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


def _derived() -> list[pl.Expr]:
    """Отношения и тренды — то, что деревья сами не выведут из сумм."""
    eps = 1e-6
    out = [
        # Средний чек и интенсивность на разных горизонтах.
        (pl.col("gmv_sum_30") / (pl.col("ord_days_30") + eps)).alias("avg_check_30"),
        (pl.col("gmv_sum_90") / (pl.col("ord_days_90") + eps)).alias("avg_check_90"),
        (pl.col("gmv_sum_365") / (pl.col("ord_days_365") + eps)).alias("avg_check_365"),
        (pl.col("gmv_sum_30") / (pl.col("active_days_30") + eps)).alias("gmv_per_active_30"),
        # Тренды: свежая активность против базовой.
        (pl.col("gmv_sum_7") * 4 / (pl.col("gmv_sum_30") + eps)).alias("trend_gmv_7_30"),
        (pl.col("gmv_sum_30") * 3 / (pl.col("gmv_sum_90") + eps)).alias("trend_gmv_30_90"),
        (pl.col("gmv_sum_90") * 4 / (pl.col("gmv_sum_365") + eps)).alias("trend_gmv_90_365"),
        (pl.col("active_days_30") * 3 / (pl.col("active_days_90") + eps)).alias("trend_act_30_90"),
        (pl.col("active_days_90") * 4 / (pl.col("active_days_365") + eps)).alias("trend_act_90_365"),
        (pl.col("to_ord_sum_30") * 3 / (pl.col("to_ord_sum_90") + eps)).alias("trend_ord_30_90"),
        # Конверсия воронки.
        (pl.col("to_ord_sum_90") / (pl.col("to_cart_sum_90") + eps)).alias("cart2ord_90"),
        (pl.col("to_cart_sum_90") / (pl.col("searches_sum_90") + eps)).alias("search2cart_90"),
        (pl.col("gmv_sum_90") / (pl.col("searches_sum_90") + eps)).alias("gmv_per_search_90"),
        # Поиск против каталога.
        (pl.col("gmv_search_sum_365") / (pl.col("gmv_sum_365") + eps)).alias("search_gmv_share_365"),
        (pl.col("gmv_search_sum_90") / (pl.col("gmv_sum_90") + eps)).alias("search_gmv_share_90"),
        # BTYD-подобные: частота, "возраст" и recency в шкале tenure.
        (pl.col("lt_ord_days") - 1).clip(lower_bound=0).alias("btyd_frequency"),
        (pl.col("tenure") - pl.col("rec_ord")).alias("btyd_t_x"),
        (pl.col("rec_ord") / (pl.col("tenure") + eps)).alias("rec_ord_rel"),
        (pl.col("lt_ord_days") / (pl.col("tenure") + eps)).alias("ord_rate_lifetime"),
        (pl.col("lt_gmv") / (pl.col("tenure") + eps)).alias("gmv_per_day_lifetime"),
        # Межпокупочный интервал.
        (pl.col("rec_ord_prev") - pl.col("rec_ord")).alias("last_ipi"),
        (pl.col("tenure") / (pl.col("lt_ord_days") + eps)).alias("mean_ipi"),
        # Доля активных дней с покупкой.
        (pl.col("ord_days_90") / (pl.col("active_days_90") + eps)).alias("ord_day_rate_90"),
        (pl.col("ord_days_365") / (pl.col("active_days_365") + eps)).alias("ord_day_rate_365"),
    ]
    # "Просрочка" относительно обычного интервала: сигнал оттока.
    out.append((pl.col("rec_ord") / (pl.col("tenure") / (pl.col("lt_ord_days") + eps) + eps)).alias("overdue_ratio"))
    return out


def build_features(cutoff: dt.date, lf: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Признаки всех пользователей, у которых есть хоть одно событие до cutoff."""
    lf = scan_log() if lf is None else lf
    hist = (
        lf.filter(pl.col("event_date") < cutoff)
        .with_columns(
            (pl.lit(cutoff) - pl.col("event_date")).dt.total_days().cast(pl.Int32).alias("days_ago")
        )
    )
    aggs: list[pl.Expr] = []
    for w in WINDOWS:
        aggs.extend(_window_aggs(w))
    aggs.extend(_lifetime_aggs())

    feats = hist.group_by("user_id").agg(aggs).with_columns(_derived())
    df = feats.collect(engine="streaming")
    num = [c for c in df.columns if c != "user_id"]
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in num])


def build_target(cutoff: dt.date, lf: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Сумма gmv за [cutoff, cutoff + HORIZON) — то же окно, что в тесте."""
    lf = scan_log() if lf is None else lf
    end = cutoff + dt.timedelta(days=HORIZON)
    return (
        lf.filter((pl.col("event_date") >= cutoff) & (pl.col("event_date") < end))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("target"))
        .collect(engine="streaming")
    )


def build_dataset(cutoff: dt.date, with_target: bool = True) -> pl.DataFrame:
    lf = scan_log()
    df = build_features(cutoff, lf)
    if with_target:
        tgt = build_target(cutoff, lf)
        df = df.join(tgt, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float64)
        )
    return df.with_columns(pl.lit(cutoff).alias("cutoff"))
