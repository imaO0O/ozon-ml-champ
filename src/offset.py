"""Модель остатка поверх собственного уровня пользователя.

Мотивация. Уровень площадки за год менялся вдвое (GMV на пользователя за 30
дней: 58 в январе 2025, 111 в декабре, 84 в январе 2026), а тестовое окно лежит
за краем наблюдаемых данных. Стресс-тест показал, что сдвиг уровня стоит около
0.06 RMSLE — в тридцать раз больше любого нашего выигрыша на признаках.

Обычная модель выучивает абсолютный log1p(GMV) в том диапазоне, который видела.
Здесь вместо этого подаём log1p(GMV пользователя за прошлые 30 дней) как
смещение и учим только отклонение от него. Уровень тогда переносится напрямую,
а деревья занимаются относительным изменением поведения.

Сравнение идёт на трёх постановках сразу: два штатных среза (там эффекты малы
и меряются плохо) и стресс-тест со сдвигом уровня +22% (там эффект крупный,
и именно там разница должна проявиться, если механизм работает).

    python src/offset.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, SEED
from datasets import parse_blocks
from metrics import rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                   max_bin=255, seed=SEED)

# Смещение — недавний уровень самого пользователя. gmv_sum_30 берётся из блока
# windows и есть в выборке как обычный признак.
OFFSET_FEATURE = "gmv_sum_30"


def run(val_cutoff, explicit_train, n_cutoffs, blocks, rounds, early_stopping, tag):
    train, val, feats, cuts = load_split(n_cutoffs, blocks=blocks, val_cutoff=val_cutoff,
                                         explicit_train=explicit_train)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)

    col = feats.index(OFFSET_FEATURE)
    off_tr = np.log1p(np.clip(Xtr[:, col], 0, None)).astype(np.float64)
    off_va = np.log1p(np.clip(Xva[:, col], 0, None)).astype(np.float64)
    del train, ytr

    print(f"  смещение: среднее {off_tr.mean():.3f} против среднего таргета "
          f"{ytr_log.mean():.3f} (в log1p-шкале)")

    plain = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=early_stopping,
                params=BASE_PARAMS, log_period=0)
    plain.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
    s_plain = rmse_log(yva_log, plain.predict(Xva))
    print(f"  обычная модель      RMSLE {s_plain:.5f} | итераций {plain.best_iter}")

    shifted = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=early_stopping,
                  params=BASE_PARAMS, log_period=0)
    shifted.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats,
                init_score=off_tr, init_score_val=off_va)
    # predict возвращает остаток — смещение прибавляем обратно.
    s_shift = rmse_log(yva_log, shifted.predict(Xva) + off_va)
    print(f"  модель остатка      RMSLE {s_shift:.5f} | итераций {shifted.best_iter}")
    print(f"  разница {s_plain - s_shift:+.5f} " +
          ("в пользу остатка" if s_shift < s_plain else "в пользу обычной"))

    append_csv(MODELS / "offset.csv",
               ["created", "commit", "постановка", "val_cutoff", "plain", "shifted", "gain"],
               {"created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
                "постановка": tag, "val_cutoff": str(cuts[0]), "plain": round(s_plain, 5),
                "shifted": round(s_shift, 5), "gain": round(s_plain - s_shift, 5)})
    return s_plain, s_shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--blocks", default=None)
    args = ap.parse_args()
    blocks = parse_blocks(args.blocks)
    d = dt.date.fromisoformat

    cases = [
        ("штатный январь", d("2026-01-15"), None, 6),
        ("штатный декабрь", d("2025-12-16"), None, 7),
        ("сдвиг уровня +22%", d("2025-12-16"),
         [d("2025-06-19"), d("2025-07-19"), d("2025-08-18")], 8),
    ]
    results = []
    for tag, val_cut, expl, n in cases:
        print(f"\n=== {tag} (валидация {val_cut}) ===")
        results.append((tag, *run(val_cut, expl, n, blocks, args.rounds, args.early_stopping, tag)))

    print("\n=== итог ===")
    for tag, plain, shifted in results:
        print(f"{tag:<20} обычная {plain:.5f} | остаток {shifted:.5f} | "
              f"разница {plain - shifted:+.5f}")


if __name__ == "__main__":
    main()
