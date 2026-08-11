"""Смещение уровня: сколько его настоящего и что даёт глобальный сдвиг.

Разбор ошибки показал среднее смещение -0.22 в log1p-шкале: модель завышает.
Но часть этого может быть артефактом обрезки снизу нулём (отрицательные
предсказания поднимаются до нуля, то есть вверх), а часть — настоящим дрейфом
уровня площадки между обучающими срезами и валидационным.

Здесь они разделяются, и сразу же проверяется, что даёт оптимальный сдвиг —
причём честно, с обрезкой после сдвига, как это происходит на лидерборде.

Дополнительно смещение измеряется на нескольких валидационных срезах: если оно
растёт с удалением обучения от цели, его можно экстраполировать на тестовое окно.

    python src/bias.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import SEED
from metrics import rmse_log
from models import GBM
from train import load_split, to_xy

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                   max_bin=255, seed=SEED)


def analyse(val_cutoff: dt.date, n_cutoffs: int, rounds: int) -> dict:
    train, val, feats, cuts = load_split(n_cutoffs, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    del train
    m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=200,
            params=BASE_PARAMS, log_period=0)
    m.fit(Xtr, np.log1p(ytr), Xva, np.log1p(yva), feature_names=feats)

    raw = m.predict(Xva)                 # без обрезки
    true = np.log1p(yva)
    bias_raw = float((true - raw).mean())
    bias_clipped = float((true - np.clip(raw, 0, None)).mean())
    share_negative = float((raw < 0).mean())

    base = rmse_log(true, raw)           # rmse_log сам обрезает, как лидерборд
    # Оптимальный сдвиг ищем перебором: аналитический оптимум сдвинут обрезкой.
    grid = np.round(np.arange(-0.45, 0.16, 0.01), 3)
    scores = [rmse_log(true, raw + d) for d in grid]
    best_i = int(np.argmin(scores))
    best_d, best_s = float(grid[best_i]), float(scores[best_i])

    print(f"\n=== валидация {cuts[0]} (обучение на {len(cuts) - 1} срезах) ===")
    print(f"  доля отрицательных предсказаний: {share_negative:.2%}")
    print(f"  смещение без обрезки:  {bias_raw:+.4f}")
    print(f"  смещение с обрезкой:   {bias_clipped:+.4f}  "
          f"(разница {bias_clipped - bias_raw:+.4f} — вклад самой обрезки)")
    print(f"  RMSLE без сдвига:      {base:.5f}")
    print(f"  лучший сдвиг {best_d:+.2f}:     {best_s:.5f}  (выигрыш {base - best_s:+.5f})")
    return {"cutoff": cuts[0], "bias_raw": bias_raw, "base": base,
            "best_delta": best_d, "best": best_s, "gain": base - best_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20000)
    args = ap.parse_args()
    d = dt.date.fromisoformat

    rows = [analyse(d("2026-01-15"), 6, args.rounds),
            analyse(d("2025-12-16"), 7, args.rounds),
            analyse(d("2025-11-16"), 8, args.rounds)]

    print("\n=== итог ===")
    for r in rows:
        print(f"{r['cutoff']}  смещение {r['bias_raw']:+.4f} | лучший сдвиг {r['best_delta']:+.2f} "
              f"| выигрыш {r['gain']:+.5f}")
    print("\nЕсли смещение устойчиво по знаку и величине на всех срезах, его можно "
          "переносить на тестовое окно. Если скачет — нельзя, это шум режима.")


if __name__ == "__main__":
    main()
