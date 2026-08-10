"""Быстрый обзор данных: схема, объём, разреженность, распределение таргета."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from config import HORIZON, SAMPLE_SUBMIT, TEST_CUTOFF, train_cutoffs
from features import scan_log

lf = scan_log()
print("схема:", {k: str(v) for k, v in lf.collect_schema().items()})

stats = lf.select(
    pl.len().alias("rows"),
    pl.col("user_id").n_unique().alias("users"),
    pl.col("event_date").min().alias("date_min"),
    pl.col("event_date").max().alias("date_max"),
    (pl.col("gmv") > 0).sum().alias("rows_with_gmv"),
    pl.col("gmv").sum().alias("gmv_total"),
).collect()
print(stats)

sub = pl.read_csv(SAMPLE_SUBMIT)
print(f"\nsample_submit: {sub.height:,} строк, колонки {sub.columns}")
print(sub.head(3))
users_log = lf.select(pl.col("user_id").unique()).collect()
missing = sub.join(users_log, on="user_id", how="anti").height
print(f"пользователей в логе: {users_log.height:,} | в сабмите: {sub['user_id'].n_unique():,} | "
      f"в сабмите, но не в логе: {missing:,}")
print(f"наивный сабмит: нулей {(sub['predict'] == 0).mean():.2%}, "
      f"сумма {sub['predict'].sum():,.0f}, среднее {sub['predict'].mean():.2f}")
users_log = users_log["user_id"]

n_days = (lf.select(pl.col('event_date').max() - pl.col('event_date').min()).collect().item().days) + 1
rows = stats["rows"][0]
print(f"\nразреженность: {rows:,} строк из {len(users_log) * n_days:,} возможных user x day "
      f"= {rows / (len(users_log) * n_days):.1%}")

print("\n--- распределение таргета по cutoff'ам (сумма gmv за 30 дней) ---")
for cut in train_cutoffs(6) + [TEST_CUTOFF]:
    end = cut + dt.timedelta(days=HORIZON)
    hist = lf.filter(pl.col("event_date") < cut).select(pl.col("user_id").unique()).collect()
    if cut == TEST_CUTOFF:
        print(f"{cut}  пользователей с историей: {hist.height:,}  (тестовое окно, таргет неизвестен)")
        continue
    tgt = (
        lf.filter((pl.col("event_date") >= cut) & (pl.col("event_date") < end))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("t")).collect()
    )
    # Только пользователи, известные на момент cutoff'а: как в тесте.
    y = hist.join(tgt, on="user_id", how="left")["t"].fill_null(0.0).to_numpy()
    pos = y[y > 0]
    print(f"{cut}  история: {hist.height:,} | покупателей: {len(pos) / len(y):>6.2%} | "
          f"средний GMV: {y.mean():>9.1f} | медиана>0: {np.median(pos):>8.1f} | "
          f"p99: {np.percentile(y, 99):>10.1f} | max: {y.max():>12.1f} | "
          f"std(log1p): {np.log1p(y).std():.3f}")

print("\n--- «наивный» бенчмарк на последнем cutoff'е: предсказать gmv за предыдущие 30 дней ---")
cut = train_cutoffs(1)[0]
prev = (
    lf.filter((pl.col("event_date") >= cut - dt.timedelta(days=HORIZON)) & (pl.col("event_date") < cut))
    .group_by("user_id").agg(pl.col("gmv").sum().alias("prev")).collect()
)
cur = (
    lf.filter((pl.col("event_date") >= cut) & (pl.col("event_date") < cut + dt.timedelta(days=HORIZON)))
    .group_by("user_id").agg(pl.col("gmv").sum().alias("cur")).collect()
)
base = (
    lf.filter(pl.col("event_date") < cut).select(pl.col("user_id").unique()).collect()
    .join(prev, on="user_id", how="left").join(cur, on="user_id", how="left")
    .fill_null(0.0)
)
y, p = base["cur"].to_numpy(), base["prev"].to_numpy()
print(f"RMSLE(предыдущие 30 дней): {np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)):.5f}")
print(f"RMSLE(всегда 0):           {np.sqrt(np.mean(np.log1p(y) ** 2)):.5f}")
