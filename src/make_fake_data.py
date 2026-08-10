"""Синтетический лог той же схемы — для прогона пайплайна до появления реальных данных.

Пишет data/raw/fake_train.parquet и data/raw/fake_sample_submit.csv.
Использование: python src/make_fake_data.py --users 5000
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import DATA_END, DATA_RAW, DATA_START, SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    n_days = (DATA_END - DATA_START).days + 1
    users = np.arange(2, 2 + args.users)

    # У каждого пользователя своя интенсивность активности и свой средний чек.
    rate = rng.beta(1.2, 12, size=args.users)
    check = rng.lognormal(6.5, 1.0, size=args.users)
    buy_p = rng.beta(1.5, 8, size=args.users)

    rows = []
    for i, u in enumerate(users):
        n_active = rng.poisson(rate[i] * n_days)
        if n_active == 0:
            continue
        days = np.unique(rng.integers(0, n_days, size=n_active))
        k = len(days)
        searches = rng.poisson(3, k) + 1
        to_cart = rng.poisson(0.6, k)
        bought = rng.random(k) < buy_p[i]
        to_ord = np.where(bought, rng.poisson(1.2, k) + 1, 0)
        gmv = np.where(to_ord > 0, rng.lognormal(np.log(check[i]), 0.6, k) * to_ord, 0.0)
        share = rng.random(k)
        rows.append(
            pl.DataFrame({
                "event_date": pl.Series(days, dtype=pl.Int32),
                "user_id": np.full(k, u, dtype=np.int64),
                "search": (searches > 0).astype(np.int8),
                "cat": (rng.random(k) < 0.5).astype(np.int8),
                "searches": searches.astype(np.int32),
                "to_cart": to_cart.astype(np.int32),
                "to_ord": to_ord.astype(np.int32),
                "gmv": gmv,
                "share": share,
            })
        )

    df = pl.concat(rows).with_columns(
        (pl.lit(DATA_START) + pl.duration(days=pl.col("event_date"))).alias("event_date"),
        (pl.col("to_cart") * pl.col("share")).round().cast(pl.Int32).alias("search_to_cart"),
        (pl.col("to_ord") * pl.col("share")).round().cast(pl.Int32).alias("search_to_ord"),
        (pl.col("gmv") * pl.col("share")).alias("gmv_search"),
    ).with_columns(
        (pl.col("to_cart") - pl.col("search_to_cart")).alias("cat_to_cart"),
        (pl.col("to_ord") - pl.col("search_to_ord")).alias("cat_to_ord"),
        (pl.col("gmv") - pl.col("gmv_search")).alias("gmv_cat"),
    ).with_columns(
        (pl.col("search_to_cart") > 0).cast(pl.Int8).alias("has_search_to_cart"),
        (pl.col("search_to_ord") > 0).cast(pl.Int8).alias("has_search_to_ord"),
        (pl.col("cat_to_cart") > 0).cast(pl.Int8).alias("has_cat_to_cart"),
        (pl.col("cat_to_ord") > 0).cast(pl.Int8).alias("has_cat_to_ord"),
    ).drop("share").sort("user_id", "event_date")

    out = DATA_RAW / "fake_train.parquet"
    df.write_parquet(out)
    pl.DataFrame({
        "user_id": df["user_id"].unique().sort(),
        "predict": 0.0,
    }).write_csv(DATA_RAW / "fake_sample_submit.csv")
    print(f"{out}: {df.height:,} строк, {df['user_id'].n_unique():,} пользователей")


if __name__ == "__main__":
    main()
