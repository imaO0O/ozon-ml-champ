"""Построение признаков на заданный cutoff из прореженного дневного лога.

Данные разрежены: строки есть только за дни с активностью. Полную сетку
user x day (~100 млн строк) не разворачиваем — все агрегаты считаются одним
ленивым `group_by` в polars.

Признаки собираются из независимых блоков (см. `registry.py`), чтобы команда
могла добавлять свои, не конфликтуя в одном файле. Новый блок = новый модуль
рядом + строка импорта в конце этого файла.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from config import HORIZON, TRAIN_PARQUET

from .registry import BLOCKS, Block, aggs_block, derived_block, selected

__all__ = [
    "BLOCKS", "Block", "aggs_block", "derived_block", "selected",
    "scan_log", "build_features", "build_target", "build_dataset",
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


def day_context(hist: pl.LazyFrame) -> pl.LazyFrame:
    """Суточные итоги площадки — контекст, доступный блокам как обычные колонки.

    Уровень площадки за 2025 год вырос примерно вдвое (GMV на пользователя за
    30 дней: 58 в январе, 111 в декабре, 84 в январе 2026), поэтому абсолютные
    рубли пользователя означают разное в разные месяцы. Доля в дневном объёме
    от уровня не зависит и переносится между режимами.

    Считается только по событиям до cutoff'а (hist уже отфильтрован), так что
    утечки будущего нет.
    """
    return hist.group_by("event_date").agg(
        pl.col("gmv").sum().alias("day_gmv"),
        pl.col("to_ord").sum().alias("day_ord"),
        pl.col("searches").sum().alias("day_searches"),
        pl.len().alias("day_active_users"),
    )


def build_features(cutoff: dt.date, lf: pl.LazyFrame | None = None,
                   blocks: list[str] | None = None) -> pl.DataFrame:
    """Признаки всех пользователей, у которых есть хоть одно событие до cutoff."""
    lf = scan_log() if lf is None else lf
    hist = (
        lf.filter(pl.col("event_date") < cutoff)
        .with_columns(
            (pl.lit(cutoff) - pl.col("event_date")).dt.total_days().cast(pl.Int32).alias("days_ago")
        )
    )
    hist = hist.join(day_context(hist), on="event_date", how="left")
    chosen = selected(blocks)
    aggs = [e for b in chosen if b.aggs for e in b.aggs(cutoff)]
    derived = [e for b in chosen if b.derived for e in b.derived()]

    feats = hist.group_by("user_id").agg(aggs)
    if derived:
        feats = feats.with_columns(derived)
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


def build_dataset(cutoff: dt.date, with_target: bool = True,
                  blocks: list[str] | None = None) -> pl.DataFrame:
    lf = scan_log()
    df = build_features(cutoff, lf, blocks=blocks)
    if with_target:
        tgt = build_target(cutoff, lf)
        df = df.join(tgt, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float64)
        )
    return df.with_columns(pl.lit(cutoff).alias("cutoff"))


# Импорты регистрируют блоки, порядок импорта = порядок колонок в выборке.
# Свой блок — добавьте строку сюда.
from . import windows  # noqa: E402,F401  (после определения реестра)
from . import lifetime  # noqa: E402,F401
from . import ratios  # noqa: E402,F401
from . import platform  # noqa: E402,F401
from . import unit  # noqa: E402,F401
