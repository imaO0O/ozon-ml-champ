"""Реестр блоков признаков.

Блок — независимый кусок признаков, который добавляет один человек в своём
файле. Так пятеро работают параллельно, не встречаясь в мержах.

Блок состоит из двух необязательных фаз:

* `aggs(cutoff) -> list[pl.Expr]` — выражения внутри `group_by("user_id").agg(...)`.
  Доступны все колонки лога плюс `days_ago` — целое число дней от события
  до cutoff'а (>= 1, то есть строго прошлое, утечки будущего нет).
* `derived() -> list[pl.Expr]` — выражения поверх уже посчитанных агрегатов
  (отношения, тренды). Могут ссылаться на агрегаты любого блока; если нужен
  чужой derived-столбец, считайте его у себя.

Пример своего блока — файл `src/features/my_block.py`:

    from .registry import aggs_block, derived_block

    @aggs_block("weekday")
    def weekday_aggs(cutoff):
        return [(pl.col("days_ago") % 7 == 0).sum().alias("same_weekday_days")]

    @derived_block("weekday")
    def weekday_derived():
        return [(pl.col("same_weekday_days") / 52).alias("same_weekday_rate")]

и одна строка импорта в `src/features/__init__.py`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import polars as pl

AggsFn = Callable[[dt.date], list[pl.Expr]]
DerivedFn = Callable[[], list[pl.Expr]]


@dataclass
class Block:
    name: str
    aggs: AggsFn | None = None
    derived: DerivedFn | None = None


BLOCKS: dict[str, Block] = {}


def _slot(name: str) -> Block:
    return BLOCKS.setdefault(name, Block(name))


def aggs_block(name: str):
    def deco(fn: AggsFn) -> AggsFn:
        _slot(name).aggs = fn
        return fn

    return deco


def derived_block(name: str):
    def deco(fn: DerivedFn) -> DerivedFn:
        _slot(name).derived = fn
        return fn

    return deco


def selected(names: list[str] | None = None) -> list[Block]:
    """Все блоки или подмножество — для замера вклада своего блока (ablation)."""
    if not names:
        return list(BLOCKS.values())
    unknown = set(names) - set(BLOCKS)
    if unknown:
        raise KeyError(f"нет таких блоков: {sorted(unknown)}; есть: {sorted(BLOCKS)}")
    return [BLOCKS[n] for n in names]
