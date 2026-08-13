"""Процентильные ранги ключевых величин внутри среза — уровень площадки уходит.

Главный измеренный дефект решения: модель привязана к абсолютным величинам,
а уровень площадки за год менялся вдвое (GMV на пользователя за 30 дней: 58
в январе 2025, 111 в декабре, 84 в январе 2026). В стресс-тесте это стоило
0.06 RMSLE — на порядок больше любого выигрыша от признаков.

Обрезка истории до общей глубины ту же привязку убирала, но ценой выброшенных
180 дней данных, и проиграла 0.0065. Ранг решает задачу, ничего не выбрасывая:
«верхние 5% по GMV за 30 дней» означают одно и то же на любом срезе, как бы ни
сдвинулся уровень.

Ранги считаются **внутри среза**, то есть по всем пользователям на конкретный
cutoff. Утечки нет: на тесте ранг тоже считается по тестовому срезу, где
известны только признаки. Абсолютные признаки остаются на месте — модель
получает и величину, и позицию, и сама решает, чем пользоваться.

Все выражения опираются только на агрегаты: производные колонки других блоков
здесь недоступны (см. контракт в registry.py), поэтому нужные отношения
вычисляются на месте.
"""
from __future__ import annotations

import polars as pl

from .registry import derived_block

EPS = 1e-6

# (имя ранга, что ранжируем). Отобрано то, что стоит в вершине важности
# и при этом плывёт вместе с уровнем площадки.
RANKED: list[tuple[str, pl.Expr]] = [
    ("gmv_30", pl.col("gmv_sum_30")),
    ("gmv_90", pl.col("gmv_sum_90")),
    ("gmv_365", pl.col("gmv_sum_365")),
    ("ord_30", pl.col("to_ord_sum_30")),
    ("ord_90", pl.col("to_ord_sum_90")),
    ("ord_180", pl.col("to_ord_sum_180")),
    ("ord_365", pl.col("to_ord_sum_365")),
    ("orddays_90", pl.col("ord_days_90")),
    ("orddays_180", pl.col("ord_days_180")),
    ("orddays_365", pl.col("ord_days_365")),
    ("act_30", pl.col("active_days_30")),
    ("act_90", pl.col("active_days_90")),
    ("act_365", pl.col("active_days_365")),
    ("search_30", pl.col("searches_sum_30")),
    ("search_90", pl.col("searches_sum_90")),
    ("lt_gmv", pl.col("lt_gmv")),
    ("lt_ord", pl.col("lt_to_ord")),
    ("lt_orddays", pl.col("lt_ord_days")),
    ("share_gmv_30", pl.col("share_gmv_30")),
    ("share_gmv_365", pl.col("share_gmv_365")),
    ("rec_ord", pl.col("rec_ord")),
    ("rec_any", pl.col("rec_any")),
    # Отношения считаем здесь же — производные чужих блоков недоступны.
    ("avg_check_90", pl.col("gmv_sum_90") / (pl.col("ord_days_90") + EPS)),
    ("unit_price_90", pl.col("gmv_sum_90") / (pl.col("to_ord_sum_90") + EPS)),
    ("gmv_per_day_lt", pl.col("lt_gmv") / (pl.col("tenure") + EPS)),
    ("ord_rate_lt", pl.col("lt_ord_days") / (pl.col("tenure") + EPS)),
    ("mean_ipi", pl.col("tenure") / (pl.col("lt_ord_days") + EPS)),
]


@derived_block("ranks")
def rank_exprs() -> list[pl.Expr]:
    # Ранг в долях от 0 до 1. Пропуски остаются пропусками: LightGBM их умеет.
    return [(e.rank(method="average") / pl.len()).alias(f"rk_{name}") for name, e in RANKED]
