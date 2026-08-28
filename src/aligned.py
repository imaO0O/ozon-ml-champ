"""Сравнение моделей до и после выравнивания уровня.

Трек C заметил: обычное сравнение на валидации смешивает две разные вещи —
насколько модель лучше по существу и насколько ей повезло с уровнем. В сабмит
уровень всё равно правится бесплатным сдвигом (см. --derive-shift), поэтому
«повезло с уровнем» на лидерборде не засчитывается, и валидация завышает
ожидаемый прирост.

Здесь обе версии приводятся к истинному уровню окна (среднее log1p таргета),
и только после этого сравниваются. Ожидание: выровненная разница должна быть
близка к тому, что даёт лидерборд.

    python src/aligned.py --blocks-a windows,lifetime,ratios,platform,unit --blocks-b all
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import SEED
from datasets import parse_blocks
from metrics import rmse_log
from models import GBM
from train import load_split, to_xy

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                   max_bin=255, seed=SEED)


def run(blocks, n_cutoffs, val_cutoff, rounds, early):
    train, val, feats, cuts = load_split(n_cutoffs, blocks=blocks, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    del train, ytr
    m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=early,
            params=BASE_PARAMS, log_period=0)
    m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
    pred = m.predict(Xva)
    raw = rmse_log(yva_log, pred)
    # Выравнивание: сдвигаем предсказания так, чтобы их среднее совпало
    # с истинным уровнем окна. Ровно это делает --derive-shift на сабмите.
    shift = float(yva_log.mean() - pred.mean())
    aligned = rmse_log(yva_log, pred + shift)
    return raw, aligned, shift, len(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-a", default="windows,lifetime,ratios,platform,unit")
    ap.add_argument("--blocks-b", default=None, help="пусто = все блоки")
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=200)
    args = ap.parse_args()

    cases = [("январь", None, 6), ("декабрь", dt.date(2025, 12, 16), 7)]
    for name, cut, n in cases:
        print(f"\n=== {name} ===")
        res = {}
        for tag, b in [("A (без рангов)", args.blocks_a), ("B (с рангами)", args.blocks_b)]:
            raw, aligned, shift, nf = run(parse_blocks(b), n, cut, args.rounds, args.early_stopping)
            res[tag] = (raw, aligned)
            print(f"  {tag:<16} признаков {nf:>3} | RMSLE {raw:.5f} | "
                  f"выровненный {aligned:.5f} | сдвиг {shift:+.4f}")
        (ra, aa), (rb, ab) = res["A (без рангов)"], res["B (с рангами)"]
        print(f"  выигрыш без выравнивания: {ra - rb:+.5f}")
        print(f"  выигрыш с выравниванием:  {aa - ab:+.5f}  <- сопоставимо с лидербордом")


if __name__ == "__main__":
    main()
