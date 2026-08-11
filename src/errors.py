"""Из кого складывается наш RMSLE.

Двенадцать часов экспериментов двигали метрику, ни разу не посмотрев на её
структуру. Между тем RMSLE^2 = mean(r^2), где r = log1p(y) - log1p(p), — это
сумма вкладов отдельных пользователей, и она распределена крайне неравномерно.

Смотрим три разреза:

* покупал или нет — граница нуля обычно и есть главный источник ошибки;
* децили предсказания — где модель уверена и где нет;
* давность последней покупки и недавняя активность — сегменты, по которым
  можно действовать признаками или отдельной моделью.

В каждом сегменте важны две величины: доля в общей ошибке (куда смотреть) и
среднее смещение (систематически завышаем или занижаем — это чинится сдвигом).

    python src/errors.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import MODELS, SEED
from models import GBM
from train import load_split, to_xy

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                   max_bin=255, seed=SEED)


def segment_table(name: str, labels: np.ndarray, order: list, r: np.ndarray) -> pl.DataFrame:
    """Доля сегмента в общей квадратичной ошибке и среднее смещение внутри него."""
    total = float((r ** 2).sum())
    rows = []
    for lab in order:
        mask = labels == lab
        n = int(mask.sum())
        if not n:
            continue
        sq = float((r[mask] ** 2).sum())
        rows.append({
            "сегмент": str(lab),
            "клиентов": n,
            "доля клиентов": n / len(r),
            "доля ошибки": sq / total,
            "RMSE внутри": float(np.sqrt(sq / n)),
            "смещение": float(r[mask].mean()),
        })
    df = pl.DataFrame(rows)
    print(f"\n=== {name} ===")
    with pl.Config(tbl_rows=30, tbl_width_chars=140, float_precision=4):
        print(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000)
    args = ap.parse_args()

    train, val, feats, cuts = load_split(args.cutoffs)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    del train
    m = GBM("lgbm", "reg", "cpu", n_estimators=args.rounds, early_stopping=200,
            params=BASE_PARAMS, log_period=0)
    m.fit(Xtr, np.log1p(ytr), Xva, np.log1p(yva), feature_names=feats)

    pred_log = np.clip(m.predict(Xva), 0, None)
    true_log = np.log1p(yva)
    r = true_log - pred_log          # >0 — занизили, <0 — завысили
    print(f"\nвалидация {cuts[0]} | RMSLE {np.sqrt((r ** 2).mean()):.5f} | "
          f"общее смещение {r.mean():+.4f}")

    # 1. Граница нуля: купил или нет.
    bought = np.where(yva > 0, "купил", "не купил")
    segment_table("покупка в целевом окне", bought, ["не купил", "купил"], r)

    # 2. Крупность покупателя.
    bins = [0, 1, 50, 200, 1000, np.inf]
    names = ["0", "1-50", "50-200", "200-1000", ">1000"]
    lab = np.array(names)[np.clip(np.digitize(yva, bins[1:], right=False), 0, len(names) - 1)]
    segment_table("фактический GMV", lab, names, r)

    # 3. Децили предсказания — где модель уверена.
    dec = np.digitize(pred_log, np.quantile(pred_log, np.linspace(0, 1, 11)[1:-1]))
    segment_table("дециль предсказания", dec, list(range(10)), r)

    # 4. Давность последней покупки — сегмент, по которому можно действовать.
    rec = Xva[:, feats.index("rec_ord")]
    rec_bins = [0, 8, 31, 91, 366]
    rec_names = ["1-7 дн", "8-30 дн", "31-90 дн", "91-365 дн", ">365 дн / никогда"]
    rec_lab = np.where(np.isnan(rec), ">365 дн / никогда",
                       np.array(rec_names)[np.clip(np.digitize(np.nan_to_num(rec, nan=999), rec_bins),
                                                   0, len(rec_names) - 1)])
    segment_table("давность последней покупки", rec_lab, rec_names, r)

    # 5. Недавняя активность.
    act = Xva[:, feats.index("active_days_30")]
    act_bins = [0, 1, 4, 11, 21]
    act_names = ["0 дней", "1-3", "4-10", "11-20", "21-30"]
    act_lab = np.array(act_names)[np.clip(np.digitize(np.nan_to_num(act), act_bins), 0, len(act_names) - 1)]
    segment_table("активных дней за 30", act_lab, act_names, r)

    (MODELS / "errors.txt").write_text(
        f"валидация {cuts[0]}, RMSLE {np.sqrt((r ** 2).mean()):.5f}, "
        f"смещение {r.mean():+.4f}, {dt.date.today()}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
