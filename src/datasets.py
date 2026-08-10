"""Сборка и кэширование обучающих выборок по cutoff'ам."""
from __future__ import annotations

import argparse
import datetime as dt
import time

import polars as pl

from config import DATA_PROC, TEST_CUTOFF, TRAIN_PARQUET, train_cutoffs
from features import build_dataset

ID = "user_id"
NON_FEATURES = {ID, "target", "cutoff"}


def dataset_path(cutoff: dt.date):
    # Имя источника в пути кэша, чтобы синтетика и реальные данные не смешивались.
    return DATA_PROC / f"ds_{TRAIN_PARQUET.stem}_{cutoff.isoformat()}.parquet"


def get_dataset(cutoff: dt.date, with_target: bool = True, rebuild: bool = False) -> pl.DataFrame:
    path = dataset_path(cutoff)
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    t0 = time.time()
    df = build_dataset(cutoff, with_target=with_target)
    df.write_parquet(path)
    print(f"[{cutoff}] {df.height:,} строк x {df.width} колонок за {time.time() - t0:.1f}s -> {path.name}")
    return df


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=None, help="сколько обучающих cutoff'ов собрать")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--test", action="store_true", help="собрать также выборку на TEST_CUTOFF (без таргета)")
    args = ap.parse_args()

    cuts = train_cutoffs() if args.cutoffs is None else train_cutoffs(args.cutoffs)
    for c in cuts:
        get_dataset(c, with_target=True, rebuild=args.rebuild)
    if args.test:
        get_dataset(TEST_CUTOFF, with_target=False, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
