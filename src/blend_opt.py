"""Оптимальный бленд уже отправленных файлов — считается офлайн, без сабмитов.

Если у двух решений известен public RMSLE, то результат любой их комбинации
вычисляется точно, без единой новой отправки.

В log1p-шкале для остатков r_i = y - p_i:

    MSE(w) = (1-w)^2 * MSE_1 + w^2 * MSE_2 + 2w(1-w) * C,   C = E[r_1 * r_2]

Ключ в том, что C не нужно измерять на лидерборде:

    E[(p_1 - p_2)^2] = E[(r_1 - r_2)^2] = MSE_1 + MSE_2 - 2C

а левая часть считается по файлам на диске. Отсюда

    C  = (MSE_1 + MSE_2 - D) / 2,      D = среднее (p_1 - p_2)^2
    w* = (MSE_1 - C) / (MSE_1 + MSE_2 - 2C)
    MSE* = MSE_1 - (MSE_1 - C)^2 / (MSE_1 + MSE_2 - 2C)

Оптимальный вес лежит вне [0,1], если одна модель доминирует другую — тогда
бленд бесполезен и это видно сразу, тоже без отправки.

    python src/blend_opt.py --a cand4_calib.csv --mse-a 1.6500339212 \
                            --b lgbm_ranks_calib.csv --mse-b 1.6526609501 --out blend.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import SAMPLE_SUBMIT, SUBMISSIONS


def load_log(name: str) -> np.ndarray:
    return np.log1p(pl.read_csv(SUBMISSIONS / name)["predict"].to_numpy().astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--mse-a", type=float, required=True, help="public RMSLE файла A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--mse-b", type=float, required=True, help="public RMSLE файла B")
    ap.add_argument("--out", default=None, help="куда записать бленд с оптимальным весом")
    args = ap.parse_args()

    pa, pb = load_log(args.a), load_log(args.b)
    m1, m2 = args.mse_a ** 2, args.mse_b ** 2
    d = float(np.mean((pa - pb) ** 2))
    c = (m1 + m2 - d) / 2
    denom = m1 + m2 - 2 * c
    w = (m1 - c) / denom
    best = max(m1 - (m1 - c) ** 2 / denom, 0.0) ** 0.5

    print(f"A: {args.a}  RMSLE {args.mse_a:.7f}")
    print(f"B: {args.b}  RMSLE {args.mse_b:.7f}")
    print(f"\nсредний квадрат разности предсказаний: {d:.6f}")
    print(f"корреляция остатков: {c / (m1 * m2) ** 0.5:.6f}")
    print(f"\nоптимальный вес B: {w:.4f}")
    print(f"ожидаемый RMSLE:   {best:.7f}  (выигрыш {args.mse_a - best:+.5f} к лучшему)")

    if not 0 < w < 1:
        print("\nвес вне [0,1]: одна модель доминирует, бленд бесполезен")
        return
    if args.mse_a - best < 0.0002:
        print("\nвыигрыш меньше 0.0002 — сабмита не стоит")

    if args.out:
        blended = np.clip(np.expm1((1 - w) * pa + w * pb), 0, None)
        ref = pl.read_csv(SAMPLE_SUBMIT)
        path = SUBMISSIONS / args.out
        if path.exists():
            raise SystemExit(f"{path.name} уже существует — задайте другое имя")
        pl.DataFrame({"user_id": ref["user_id"],
                      "predict": blended.astype(np.float32)}).write_csv(path)
        print(f"\n{path}")
        print(f"  сумма: {blended.sum():,.0f} | нулевых: {(blended < 1e-6).mean():.2%}")


if __name__ == "__main__":
    main()
