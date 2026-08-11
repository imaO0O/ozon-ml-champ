"""Юнит-экономика и регулярность — сигналы, которых в остальных блоках нет.

Все прочие блоки — это агрегаты сумм GMV, количеств и дней. Отсюда потолок:
они переупаковывают одну и ту же информацию. Здесь добавлены величины,
не выводимые из тех сумм:

* цена единицы товара (GMV на товар, а не на день): десять дешёвых позиций и
  одна дорогая дают одинаковый GMV, но это разные покупатели;
* регулярность в неделях и месяцах, а не в днях: десять дней подряд и десять
  дней по одному в неделю — разное поведение при одинаковом счётчике дней;
* разброс межпокупочных интервалов, а не только средний;
* будни против выходных;
* брошенные корзины — дни с добавлением, но без покупки.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from .registry import aggs_block, derived_block

EPS = 1e-6


@aggs_block("unit")
def unit_aggs(cutoff: dt.date) -> list[pl.Expr]:
    paid = pl.col("gmv") > 0
    weekend = pl.col("event_date").dt.weekday() >= 6
    in90 = pl.col("days_ago") <= 90
    in365 = pl.col("days_ago") <= 365
    # Цена единицы в конкретный день покупки — её разброс отличает «одна дорогая
    # покупка» от «много мелких» даже при одинаковом среднем.
    unit_price = pl.col("gmv") / (pl.col("to_ord") + EPS)
    # Промежутки между днями покупок: days_ago убывает во времени, поэтому
    # сортировка и разности дают длины пауз.
    gaps = pl.col("days_ago").filter(paid).sort().diff()
    return [
        unit_price.filter(paid).std().alias("unit_price_std"),
        unit_price.filter(paid).max().alias("unit_price_max"),
        unit_price.filter(paid & in90).mean().alias("unit_price_mean_90"),
        # Регулярность: сколько разных недель и месяцев человек вообще появлялся.
        (pl.col("days_ago") // 7).filter(in90).n_unique().alias("active_weeks_90"),
        (pl.col("days_ago") // 7).filter(in365).n_unique().alias("active_weeks_365"),
        (pl.col("days_ago") // 7).filter(paid & in365).n_unique().alias("ord_weeks_365"),
        pl.col("event_date").dt.month().n_unique().alias("active_months"),
        # Разброс пауз между покупками, а не только средняя пауза.
        gaps.std().alias("gap_std"),
        gaps.max().alias("gap_max"),
        gaps.mean().alias("gap_mean"),
        # Будни против выходных.
        pl.col("gmv").filter(weekend).sum().alias("weekend_gmv"),
        weekend.filter(in365).sum().alias("weekend_days_365"),
        # Брошенные корзины: добавил и не купил.
        ((pl.col("to_cart") > 0) & (pl.col("to_ord") == 0)).filter(in90).sum().alias("abandon_days_90"),
        ((pl.col("to_cart") > 0) & (pl.col("to_ord") == 0)).sum().alias("abandon_days_lt"),
        # Крупнейшая разовая покупка в товарах и день только с поиском без каталога.
        pl.col("to_ord").max().alias("to_ord_max"),
        ((pl.col("search") == 1) & (pl.col("cat") == 0)).filter(in90).sum().alias("search_only_days_90"),
        ((pl.col("cat") == 1) & (pl.col("search") == 0)).filter(in90).sum().alias("cat_only_days_90"),
    ]


@derived_block("unit")
def unit_derived() -> list[pl.Expr]:
    return [
        # Цена единицы на разных горизонтах — из сумм блока windows.
        (pl.col("gmv_sum_30") / (pl.col("to_ord_sum_30") + EPS)).alias("unit_price_30"),
        (pl.col("gmv_sum_90") / (pl.col("to_ord_sum_90") + EPS)).alias("unit_price_90"),
        (pl.col("gmv_sum_365") / (pl.col("to_ord_sum_365") + EPS)).alias("unit_price_365"),
        # Дорожает или дешевеет корзина пользователя.
        (pl.col("gmv_sum_90") * (pl.col("to_ord_sum_365") + EPS)
         / ((pl.col("to_ord_sum_90") + EPS) * (pl.col("gmv_sum_365") + EPS))).alias("unit_price_trend_90_365"),
        # Товаров за раз: «много мелкого» против «одного крупного».
        (pl.col("to_ord_sum_90") / (pl.col("ord_days_90") + EPS)).alias("items_per_ord_day_90"),
        # Регулярность: сколько недель из возможных человек появлялся.
        (pl.col("active_weeks_90") / 13.0).alias("week_regularity_90"),
        (pl.col("active_weeks_365") / 52.0).alias("week_regularity_365"),
        (pl.col("ord_weeks_365") / (pl.col("active_weeks_365") + EPS)).alias("ord_week_share_365"),
        (pl.col("active_days_90") / (pl.col("active_weeks_90") + EPS)).alias("days_per_active_week_90"),
        # Стабильность ритма покупок: одинаковые паузы или рваный график.
        (pl.col("gap_std") / (pl.col("gap_mean") + EPS)).alias("gap_cv"),
        (pl.col("rec_ord") / (pl.col("gap_max") + EPS)).alias("rec_vs_max_gap"),
        # Выходные.
        (pl.col("weekend_gmv") / (pl.col("lt_gmv") + EPS)).alias("weekend_gmv_share"),
        (pl.col("weekend_days_365") / (pl.col("active_days_365") + EPS)).alias("weekend_day_share_365"),
        # Брошенные корзины.
        (pl.col("abandon_days_90") / (pl.col("active_days_90") + EPS)).alias("abandon_rate_90"),
        ((pl.col("to_cart_sum_90") - pl.col("to_ord_sum_90"))
         / (pl.col("to_cart_sum_90") + EPS)).alias("cart_leftover_90"),
        # Поиск против каталога по дням, а не по деньгам.
        (pl.col("search_only_days_90") / (pl.col("active_days_90") + EPS)).alias("search_only_share_90"),
        (pl.col("cat_only_days_90") / (pl.col("active_days_90") + EPS)).alias("cat_only_share_90"),
        # Разброс цены относительно её среднего.
        (pl.col("unit_price_std") / (pl.col("unit_price_mean_90") + EPS)).alias("unit_price_cv"),
    ]
