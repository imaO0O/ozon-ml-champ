"""Предсказания бустинга по срезам — основа для сети, обучаемой на остатке.

Зачем. Измерено, что сеть, как её ни улучшай, даёт направление, почти
параллельное бустингу: угол между направлениями 0.9929, и целое новое
направление добавило к оптимуму 0.000003. Улучшение на валидации не означает
новой информации для ансамбля — сеть учит то же самое другими словами.

Лечение по построению: пусть сеть предсказывает не таргет, а **остаток**
бустинга. Тогда её выход это ровно то, чего бустингу не хватает, и параллельным
он быть не может.

Для этого нужны предсказания бустинга на тех же срезах, на которых учится сеть,
причём полученные честно — моделью, не видевшей этот срез. Схема walk-forward:
для среза c бустинг обучается на всех срезах **строго старше** c. Так же
устроен и тест: рабочая модель обучена на срезах до 2026-01-15, а предсказывает
2026-02-14. Обучение на более свежих срезах дало бы сети слишком хорошего
партнёра, которого на тесте не будет.

Дополнительно считается предсказание на TEST_CUTOFF моделью, обученной на всех
срезах — это тот же объект, что и в рабочем сабмите, и именно к нему сеть будет
добавлять свой остаток.

    python -u src/seq_oof.py --cutoffs 8
    python -u src/seq_oof.py --cutoffs 8 --ensemble --name oof_ens
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from config import MODELS, SAMPLE_SUBMIT, TEST_CUTOFF, train_cutoffs
from datasets import feature_names, get_dataset
from metrics import report
from models import GBM
from train import load_split, to_xy


def oof_path(name: str, tag: str):
    return MODELS / f"{name}_{tag}.npz"


def fit_predict(val_cut: dt.date, older: list[dt.date], n_cutoffs: int, rounds: int,
                members=None):
    """Обучить на `older`, предсказать на `val_cut`. Возвращает (user_id, pred_log, y)."""
    train, val, feats, _ = load_split(n_cutoffs, val_cutoff=val_cut, explicit_train=older)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    del train
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)

    if members:
        preds = []
        for mname, kind, params in members:
            m = GBM(kind, "reg", "cpu", n_estimators=rounds, params=params, log_period=0)
            m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
            preds.append(m.predict(Xva))
            print(f"    участник {mname:<20} итераций {m.best_iter}")
        p = np.mean(preds, axis=0)
    else:
        m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=200, log_period=0)
        m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
        p = m.predict(Xva)
        print(f"    итераций {m.best_iter}")
    return val["user_id"].to_numpy(), p, yva


def fit_test(cuts: list[dt.date], rounds: int, members=None):
    """Модель на всех срезах -> предсказание на TEST_CUTOFF (без ранней остановки).

    Число итераций берётся фиксированным: валидации здесь нет по построению,
    как и в `train.py --final`.
    """
    frames = [get_dataset(c) for c in cuts]
    feats = feature_names(frames[0])
    X = np.vstack([f.select(feats).to_numpy().astype(np.float32) for f in frames])
    y = np.concatenate([f["target"].to_numpy().astype(np.float64) for f in frames])
    ylog = np.log1p(y)
    test = get_dataset(TEST_CUTOFF, with_target=False)
    Xte = test.select(feats).to_numpy().astype(np.float32)

    if members:
        preds = []
        for mname, kind, params in members:
            m = GBM(kind, "reg", "cpu", n_estimators=rounds, early_stopping=0, params=params,
                    log_period=0)
            m.fit(X, ylog, feature_names=feats)
            preds.append(m.predict(Xte))
            print(f"    участник {mname:<20} готов")
        p = np.mean(preds, axis=0)
    else:
        m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=0, log_period=0)
        m.fit(X, ylog, feature_names=feats)
        p = m.predict(Xte)
    return test["user_id"].to_numpy(), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=8,
                    help="сколько срезов рассматривать; у самых старых не хватит "
                         "предшественников и они будут пропущены")
    ap.add_argument("--min-train", type=int, default=2,
                    help="минимум обучающих срезов, иначе срез пропускается")
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--final-rounds", type=int, default=170,
                    help="число итераций для модели на всех срезах (валидации там нет)")
    ap.add_argument("--ensemble", action="store_true",
                    help="вместо одиночного LightGBM — рабочий состав из ensemble.py")
    ap.add_argument("--name", default="oof")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    members = None
    if args.ensemble:
        from ensemble import MEMBERS as members  # noqa: PLC0415

    cuts = train_cutoffs(args.cutoffs)          # от свежего к старому
    print(f"срезы: {', '.join(str(c) for c in cuts)}")

    for c in cuts:
        older = [o for o in cuts if o < c]
        if len(older) < args.min_train:
            print(f"\n{c}: предшественников {len(older)} — пропускаю")
            continue
        path = oof_path(args.name, c.isoformat())
        if path.exists():
            print(f"\n{c}: уже посчитан, {path.name}")
            continue
        print(f"\n=== {c}: обучение на {len(older)} более старых срезах ===")
        t0 = time.time()
        uid, p, y = fit_predict(c, older, args.cutoffs, args.rounds, members)
        report(y, np.expm1(np.clip(p, 0, None)), f"oof {c}")
        np.savez(path, user_id=uid, pred_log=p, target=y)
        print(f"  -> {path.name} за {time.time() - t0:.0f}s")

    if args.skip_test:
        return
    path = oof_path(args.name, "test")
    if path.exists():
        print(f"\nтест: уже посчитан, {path.name}")
        return
    print(f"\n=== TEST_CUTOFF {TEST_CUTOFF}: обучение на всех {len(cuts)} срезах ===")
    t0 = time.time()
    uid, p = fit_test(cuts, args.final_rounds, members)
    np.savez(path, user_id=uid, pred_log=p)
    sub = pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy()
    missing = int(np.isin(sub, uid, invert=True).sum())
    print(f"  -> {path.name} за {time.time() - t0:.0f}s | "
          f"пользователей {len(uid):,} | нет в предсказании: {missing}")
    print(f"  сумма expm1(pred): {np.expm1(np.clip(p, 0, None)).sum():,.0f}")


if __name__ == "__main__":
    main()
