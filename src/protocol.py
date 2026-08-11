"""Насколько ранняя остановка по отчётной валидации завышает наши числа.

Проблема. Каждая модель выбирает себе лучшую итерацию на тех же 250 000
пользователей, на которых её потом меряют. Это смещает оценку вниз (в лучшую
сторону), причём **неравномерно**: чем больше итераций перебрала модель, тем
больше у неё шансов поймать удачный минимум. Победитель перебора обучался
207 итераций против 142 у умолчания — правдоподобная причина, почему его
преимущество не перенеслось на другой срез.

Как меряем. Валидация делится пополам **по пользователям** (временная структура
и сдвиг уровня сохраняются в обеих половинах, в отличие от деления по времени).
Обучаются две модели: одна останавливается по половине A, другая по половине B.
Затем обе оцениваются на **одной и той же** половине — тогда единственное
различие между оценками и есть искажение протокола.

Первая версия этого замера сравнивала разные половины и потому меряла не то:
две случайные половины по 125 000 клиентов расходятся на 0.007 RMSLE сами по
себе, и этот разброс полностью перекрывал искомый эффект.

    python src/protocol.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import SEED
from metrics import rmse_log
from models import GBM
from train import load_split, to_xy

CONFIGS = {
    "умолчание (127 листьев, 142 итерации)": dict(
        learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0, max_bin=255, seed=SEED),
    "победитель перебора (511 листьев, 207)": dict(
        learning_rate=0.02, num_leaves=511, min_data_in_leaf=500,
        feature_fraction=0.65, bagging_fraction=0.7, lambda_l2=20.0, max_bin=127, seed=SEED),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--val-cutoff", default=None)
    args = ap.parse_args()

    val_cutoff = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else None
    train, val, feats, cuts = load_split(args.cutoffs, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    del train, ytr

    rng = np.random.default_rng(SEED)
    A = rng.permutation(len(yva)) < len(yva) // 2
    B = ~A
    print(f"валидация {cuts[0]}: половины по {A.sum():,} и {B.sum():,} пользователей\n")

    def fit_on(mask, params):
        m = GBM("lgbm", "reg", "cpu", n_estimators=args.rounds,
                early_stopping=args.early_stopping, params=params, log_period=0)
        m.fit(Xtr, ytr_log, Xva[mask], yva_log[mask], feature_names=feats)
        return m

    for name, params in CONFIGS.items():
        mA, mB = fit_on(A, params), fit_on(B, params)
        pA, pB = mA.predict(Xva), mB.predict(Xva)
        # Оцениваем на половине B: модель mB на ней останавливалась, mA — нет.
        bias_B = rmse_log(yva_log[B], pB[B]) - rmse_log(yva_log[B], pA[B])
        bias_A = rmse_log(yva_log[A], pA[A]) - rmse_log(yva_log[A], pB[A])
        print(f"{name}")
        print(f"  итераций: {mA.best_iter} и {mB.best_iter}")
        print(f"  на половине B: своя остановка {rmse_log(yva_log[B], pB[B]):.5f} против "
              f"чужой {rmse_log(yva_log[B], pA[B]):.5f}  ->  {bias_B:+.5f}")
        print(f"  на половине A: своя остановка {rmse_log(yva_log[A], pA[A]):.5f} против "
              f"чужой {rmse_log(yva_log[A], pB[A]):.5f}  ->  {bias_A:+.5f}")
        print(f"  искажение протокола (среднее): {(bias_A + bias_B) / 2:+.5f}\n")

    print("Отрицательное искажение = отчётная валидация завышает качество.\n"
          "Если у обеих конфигураций оно одинаково, сравнения между ними остаются\n"
          "в силе. Если у более долгой модели заметно больше — часть выводов\n"
          "о гиперпараметрах была артефактом протокола.")


if __name__ == "__main__":
    main()
