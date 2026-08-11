"""Калиброванная оценка суммарного GMV — для tie-breaker жюри, не для сабмита.

Зачем. RMSLE оптимизируется условным средним логарифмов, поэтому expm1 от
предсказания систематически ниже условного среднего самой величины: наш сабмит
даёт 8.7 млн против фактических ~21 млн. В сабмит поправку вносить нельзя —
RMSLE сразу ухудшится, — но у жюри в критериях отбора финалистов стоит сравнение
суммарного предсказанного GMV по RMSPE, и там нужна честная оценка суммы.

Две поправки:

* скалярная — один множитель на всех;
* подецильная — свой множитель в каждом дециле предсказания, потому что
  занижение неравномерно: у активных покупателей оно одно, у спящих другое.

Главная проверка — **переносимость**: множители подбираются на одном срезе и
проверяются на другом. Подгонка под один срез здесь так же бесполезна, как и
в основной задаче.

    python src/calibrate.py --apply submissions/lgbm_ens_0811_1436.csv
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import MODELS, ROOT, SEED
from metrics import rmspe_total
from models import GBM
from train import load_split, to_xy

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                   max_bin=255, seed=SEED)
N_BUCKETS = 10


def fit_predict(val_cutoff: dt.date, n_cutoffs: int, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """Обучить рабочую конфигурацию и вернуть (истина, предсказание) на валидации."""
    train, val, feats, cuts = load_split(n_cutoffs, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=200,
            params=BASE_PARAMS, log_period=0)
    m.fit(Xtr, np.log1p(ytr), Xva, np.log1p(yva), feature_names=feats)
    return yva, np.clip(np.expm1(m.predict(Xva)), 0, None)


def scalar_factor(y: np.ndarray, p: np.ndarray) -> float:
    return float(y.sum() / (p.sum() + 1e-12))


def bucket_factors(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Границы децилей предсказания и множитель в каждом."""
    edges = np.quantile(p, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
    idx = np.digitize(p, edges)
    factors = np.ones(N_BUCKETS)
    for b in range(N_BUCKETS):
        mask = idx == b
        if mask.sum() and p[mask].sum() > 0:
            factors[b] = y[mask].sum() / p[mask].sum()
    return edges, factors


def apply_buckets(p: np.ndarray, edges: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return p * factors[np.digitize(p, edges)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--apply", default=None, help="csv сабмита, к которому применить поправку")
    args = ap.parse_args()
    d = dt.date.fromisoformat

    print("=== обучение на двух срезах ===")
    y_jan, p_jan = fit_predict(d("2026-01-15"), 6, args.rounds)
    y_dec, p_dec = fit_predict(d("2025-12-16"), 7, args.rounds)

    print("\n=== поправки подбираются на январе, проверяются на декабре ===")
    k = scalar_factor(y_jan, p_jan)
    edges, factors = bucket_factors(y_jan, p_jan)
    print(f"скалярный множитель по январю: x{k:.3f}")
    print("подецильные множители: " + ", ".join(f"{f:.2f}" for f in factors))

    for name, y, p, own in [("январь (подгонка)", y_jan, p_jan, True),
                            ("декабрь (проверка)", y_dec, p_dec, False)]:
        raw = rmspe_total(y, p)
        sc = rmspe_total(y, p * k)
        bu = rmspe_total(y, apply_buckets(p, edges, factors))
        mark = " <- честная оценка переносимости" if not own else ""
        print(f"\n{name}: ошибка суммарного GMV")
        print(f"  без поправки  {raw:>7.2%}")
        print(f"  скалярная     {sc:>7.2%}")
        print(f"  подецильная   {bu:>7.2%}{mark}")

    if args.apply:
        path = ROOT / args.apply if not str(args.apply).startswith(("/", "C:")) else args.apply
        sub = pl.read_csv(path)
        p = sub["predict"].to_numpy()
        print(f"\n=== оценка суммарного GMV на тестовом окне ===")
        print(f"файл: {path}")
        print(f"  сырая сумма предсказаний   {p.sum():>14,.0f}")
        print(f"  со скалярной поправкой     {(p * k).sum():>14,.0f}")
        print(f"  с подецильной поправкой    {apply_buckets(p, edges, factors).sum():>14,.0f}")
        print("\nВ сабмит поправка НЕ вносится: она ухудшит RMSLE. Это отдельная "
              "оценка суммы для критериев жюри.")
        (MODELS / "calibration.txt").write_text(
            f"скалярный множитель: {k:.4f}\n"
            f"границы децилей: {np.round(edges, 3).tolist()}\n"
            f"подецильные множители: {np.round(factors, 3).tolist()}\n"
            f"файл: {path}\n"
            f"сырая сумма: {p.sum():,.0f}\n"
            f"скалярная поправка: {(p * k).sum():,.0f}\n"
            f"подецильная поправка: {apply_buckets(p, edges, factors).sum():,.0f}\n",
            encoding="utf-8")
        print(f"записано в {MODELS / 'calibration.txt'}")


if __name__ == "__main__":
    main()
