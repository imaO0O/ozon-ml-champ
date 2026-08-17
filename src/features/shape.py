"""Форма активности внутри окна: концентрация, наклон и движение внутри когорты.

Все существующие блоки описывают активность **объёмом**: суммы, счётчики, их
отношения между окнами. Ранги трека A сняли привязку к уровню площадки, но
ранжируют те же объёмы. Здесь добавлено то, что из сумм не выводится.

**Концентрация.** Один заказ на 10 000 и десять по 1 000 дают одинаковый
`gmv_sum_90`, но это разные клиенты: у первого следующий месяц почти наверняка
пустой, у второго — такой же. Индекс Херфиндаля по дневным долям и доля
пикового дня различают эти случаи, а суммы и средние — нет.

**Наклон внутри окна.** Отношения `gmv_30/gmv_90` сравнивают два окна и
чувствительны к границе: клиент, купивший на 31-й день назад, попадает в
знаменатель, но не в числитель. Наклон регрессии по дням внутри одного окна
границы не имеет и меряет ускорение напрямую.

**Разности рангов.** `rk_gmv_30 - rk_gmv_365` показывает, растёт клиент
относительно когорты или падает. Абсолютные тренды этого не различают: они
смешивают движение клиента с движением всей площадки, а разность рангов от
уровня свободна по построению — оба ранга считаются внутри своего среза.

Производные колонки чужих блоков здесь недоступны (контракт в registry.py),
поэтому нужные ранги вычисляются на месте.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from .registry import aggs_block, derived_block

EPS = 1e-6
SHAPE_WINDOWS = [90, 365]


@aggs_block("shape")
def shape_aggs(cutoff: dt.date) -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for w in SHAPE_WINDOWS:
        inw = pl.col("days_ago") <= w
        gmv = pl.col("gmv").filter(inw)
        day = pl.col("days_ago").filter(inw).cast(pl.Float64)
        out += [
            # Для индекса концентрации: сумма квадратов дневных сумм.
            (gmv ** 2).sum().alias(f"gmv_sq_{w}"),
            # Разброс дневного GMV по активным дням — ровный клиент или рваный.
            gmv.std().alias(f"gmv_std_{w}"),
            # Суммы для наклона регрессии gmv по дням внутри окна.
            (gmv * day).sum().alias(f"gmv_x_day_{w}"),
            day.sum().alias(f"day_sum_{w}"),
            (day ** 2).sum().alias(f"day_sq_{w}"),
            # То же для активности: считаем дни, а не деньги.
            pl.col("to_ord").filter(inw).cast(pl.Float64).mul(day).sum().alias(f"ord_x_day_{w}"),
        ]
    return out


def _slope(num: str, w: int) -> pl.Expr:
    """Наклон регрессии величины по days_ago внутри окна, со знаком «к росту».

    days_ago убывает во времени, поэтому знак разворачивается: положительное
    значение означает, что клиент разгоняется к концу окна.

    Две поправки, без которых признак вреден:

    * наклон делится на средний дневной уровень самого клиента. Сырой наклон
      измеряется в рублях в день и потому плывёт вместе с уровнем площадки —
      той самой привязкой, ради ухода от которой сделаны ранги. После деления
      величина безразмерна: «на сколько своих средних в день клиент ускоряется»;
    * меньше трёх активных дней — пропуск. По двум точкам прямая проходит
      всегда, а по одной знаменатель вырождается в ноль, и без этой отсечки
      признак принимал значения до 37 000 при разбросе остальных в единицы.
    """
    n = pl.col(f"active_days_{w}").cast(pl.Float64)
    sx = pl.col(f"day_sum_{w}")
    sxx = pl.col(f"day_sq_{w}")
    sy = pl.col(f"{num}_sum_{w}" if num == "gmv" else f"to_ord_sum_{w}").cast(pl.Float64)
    sxy = pl.col(f"{num}_x_day_{w}")
    raw = -(n * sxy - sx * sy) / (n * sxx - sx ** 2)
    return pl.when(n >= 3).then(raw / (sy / n + EPS)).otherwise(None)


@derived_block("shape")
def shape_derived() -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for w in SHAPE_WINDOWS:
        gmv_sum = pl.col(f"gmv_sum_{w}")
        out += [
            # Индекс Херфиндаля: 1 — весь GMV в одном дне, ~1/n — размазан ровно.
            (pl.col(f"gmv_sq_{w}") / (gmv_sum ** 2 + EPS)).alias(f"gmv_hhi_{w}"),
            # Доля пикового дня — та же мысль, но устойчивее к хвостам.
            (pl.col(f"gmv_max_{w}") / (gmv_sum + EPS)).alias(f"gmv_peak_share_{w}"),
            # Разброс относительно среднего дневного: рваность графика.
            (pl.col(f"gmv_std_{w}")
             / (gmv_sum / (pl.col(f"active_days_{w}") + EPS) + EPS)).alias(f"gmv_cv_{w}"),
            _slope("gmv", w).alias(f"gmv_slope_{w}"),
            _slope("ord", w).alias(f"ord_slope_{w}"),
        ]

    # Разности рангов: движение клиента внутри когорты, а не в рублях.
    def rk(e: pl.Expr) -> pl.Expr:
        return e.rank(method="average") / pl.len()

    r_gmv_30, r_gmv_90 = rk(pl.col("gmv_sum_30")), rk(pl.col("gmv_sum_90"))
    r_gmv_365 = rk(pl.col("gmv_sum_365"))
    r_ord_30, r_ord_365 = rk(pl.col("to_ord_sum_30")), rk(pl.col("to_ord_sum_365"))
    r_act_30, r_act_365 = rk(pl.col("active_days_30")), rk(pl.col("active_days_365"))
    r_srch_30, r_srch_90 = rk(pl.col("searches_sum_30")), rk(pl.col("searches_sum_90"))
    out += [
        (r_gmv_30 - r_gmv_365).alias("rkd_gmv_30_365"),
        (r_gmv_30 - r_gmv_90).alias("rkd_gmv_30_90"),
        (r_gmv_90 - r_gmv_365).alias("rkd_gmv_90_365"),
        (r_ord_30 - r_ord_365).alias("rkd_ord_30_365"),
        (r_act_30 - r_act_365).alias("rkd_act_30_365"),
        (r_srch_30 - r_srch_90).alias("rkd_search_30_90"),
        # Расхождение денег и активности внутри когорты: клиент ходит чаще, но
        # тратит меньше — или наоборот. Ни один объёмный признак этого не ловит.
        (r_gmv_30 - r_act_30).alias("rkd_gmv_vs_act_30"),
        (r_gmv_365 - r_act_365).alias("rkd_gmv_vs_act_365"),
    ]
    return out
