"""Отношения и тренды поверх агрегатов — то, что деревья не выведут из сумм."""
from __future__ import annotations

import polars as pl

from .registry import derived_block

EPS = 1e-6


@derived_block("ratios")
def ratio_exprs() -> list[pl.Expr]:
    return [
        # Средний чек и интенсивность на разных горизонтах.
        (pl.col("gmv_sum_30") / (pl.col("ord_days_30") + EPS)).alias("avg_check_30"),
        (pl.col("gmv_sum_90") / (pl.col("ord_days_90") + EPS)).alias("avg_check_90"),
        (pl.col("gmv_sum_365") / (pl.col("ord_days_365") + EPS)).alias("avg_check_365"),
        (pl.col("gmv_sum_30") / (pl.col("active_days_30") + EPS)).alias("gmv_per_active_30"),
        # Тренды: свежая активность против базовой.
        (pl.col("gmv_sum_7") * 4 / (pl.col("gmv_sum_30") + EPS)).alias("trend_gmv_7_30"),
        (pl.col("gmv_sum_30") * 3 / (pl.col("gmv_sum_90") + EPS)).alias("trend_gmv_30_90"),
        (pl.col("gmv_sum_90") * 4 / (pl.col("gmv_sum_365") + EPS)).alias("trend_gmv_90_365"),
        (pl.col("active_days_30") * 3 / (pl.col("active_days_90") + EPS)).alias("trend_act_30_90"),
        (pl.col("active_days_90") * 4 / (pl.col("active_days_365") + EPS)).alias("trend_act_90_365"),
        (pl.col("to_ord_sum_30") * 3 / (pl.col("to_ord_sum_90") + EPS)).alias("trend_ord_30_90"),
        # Конверсия воронки.
        (pl.col("to_ord_sum_90") / (pl.col("to_cart_sum_90") + EPS)).alias("cart2ord_90"),
        (pl.col("to_cart_sum_90") / (pl.col("searches_sum_90") + EPS)).alias("search2cart_90"),
        (pl.col("gmv_sum_90") / (pl.col("searches_sum_90") + EPS)).alias("gmv_per_search_90"),
        # Поиск против каталога.
        (pl.col("gmv_search_sum_365") / (pl.col("gmv_sum_365") + EPS)).alias("search_gmv_share_365"),
        (pl.col("gmv_search_sum_90") / (pl.col("gmv_sum_90") + EPS)).alias("search_gmv_share_90"),
        # BTYD-подобные: частота, «возраст» и recency в шкале tenure.
        (pl.col("lt_ord_days") - 1).clip(lower_bound=0).alias("btyd_frequency"),
        (pl.col("tenure") - pl.col("rec_ord")).alias("btyd_t_x"),
        (pl.col("rec_ord") / (pl.col("tenure") + EPS)).alias("rec_ord_rel"),
        (pl.col("lt_ord_days") / (pl.col("tenure") + EPS)).alias("ord_rate_lifetime"),
        (pl.col("lt_gmv") / (pl.col("tenure") + EPS)).alias("gmv_per_day_lifetime"),
        # Межпокупочный интервал.
        (pl.col("rec_ord_prev") - pl.col("rec_ord")).alias("last_ipi"),
        (pl.col("tenure") / (pl.col("lt_ord_days") + EPS)).alias("mean_ipi"),
        # Доля активных дней с покупкой.
        (pl.col("ord_days_90") / (pl.col("active_days_90") + EPS)).alias("ord_day_rate_90"),
        (pl.col("ord_days_365") / (pl.col("active_days_365") + EPS)).alias("ord_day_rate_365"),
        # «Просрочка» относительно обычного интервала: сигнал оттока.
        (pl.col("rec_ord") / (pl.col("tenure") / (pl.col("lt_ord_days") + EPS) + EPS)).alias("overdue_ratio"),
    ]
