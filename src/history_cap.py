"""Одинаковая длина истории у всех срезов — проверка последнего непроверенного эффекта.

Проблема. Данные начинаются 2025-01-01, поэтому у разных cutoff'ов разная
глубина доступного прошлого: у самого старого обучающего среза (2025-08-18)
её 229 дней, у валидационного (2026-01-15) — 379, у тестового (2026-02-14) —
409. При этом в топе важности стоят `to_ord_sum_180`, `to_ord_sum_365`,
`ord_rate_lifetime`, `gmv_per_day_lifetime` — годовые окна и пожизненные
величины. На старом срезе `to_ord_sum_365` это сумма за 229 дней, на тесте —
за 365. Один и тот же признак означает разное.

Отдельно опасен `tenure`: дней с первого события у одного пользователя растёт
от среза к срезу почти линейно, то есть работает как индикатор среза. На тесте
он выходит за пределы всего виденного, а деревья не экстраполируют — они
возвращают значение крайнего листа.

Проверка. Строим признаки по логу, заранее обрезанному до `[cutoff - cap, cutoff)`.
Тогда у **всех** срезов, включая тестовый, ровно `cap` дней истории, и признаки
становятся сопоставимыми. При cap = 229 обрезка ничего не отнимает у самого
старого среза (у него ровно столько и есть) и уравнивает по нему остальные.

Честность сравнения. Обрезка выкидывает из выборки пользователей, чья
активность была только в далёком прошлом, — и тогда два числа считались бы на
разных наборах клиентов. Разброс между двумя случайными половинами валидации
составляет 0.007 RMSLE (PLAN.md, раздел 6), то есть такое сравнение не значило
бы ничего. Поэтому строки и таргеты берутся из полной выборки, а из обрезанной
подставляются только значения признаков: у выпавших пользователей они
становятся пропусками, ровно как в `predict.py` для клиентов без истории.

Ни один существующий файл не меняется: `build_features` принимает готовый
LazyFrame, чем мы и пользуемся. Кэш обрезанных выборок лежит отдельно.

    python -u src/history_cap.py --cap 229
    python -u src/history_cap.py --cap 229 --val-cutoffs 2026-01-15,2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from config import (DATA_PROC, DATA_START, HORIZON, MODELS, TRAIN_PARQUET,
                    train_cutoffs)
from datasets import feature_names, features_version, get_dataset
from features import build_features, build_target, scan_log
from metrics import report, rmse_log
from models import GBM
from utils import append_csv, git_commit

FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "stride", "halflife",
          "val_cutoff", "train_cutoffs", "note"]


def capped_path(cutoff: dt.date, cap: int):
    return DATA_PROC / (f"cap{cap}_{TRAIN_PARQUET.stem}_{cutoff.isoformat()}"
                        f"_{features_version()}.parquet")


def capped_dataset(cutoff: dt.date, cap: int, rebuild: bool = False) -> pl.DataFrame:
    """Выборка на cutoff, где в признаки идут только последние `cap` дней.

    Таргет обрезка не затрагивает: окно `[cutoff, cutoff + 30)` лежит правее
    `cutoff - cap`, поэтому тот же отфильтрованный лог годится и для него.
    """
    path = capped_path(cutoff, cap)
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    t0 = time.time()
    lf = scan_log().filter(pl.col("event_date") >= cutoff - dt.timedelta(days=cap))
    df = build_features(cutoff, lf)
    tgt = build_target(cutoff, lf)
    df = (
        df.join(tgt, on="user_id", how="left")
        .with_columns(pl.col("target").fill_null(0.0).cast(pl.Float64))
        .with_columns(pl.lit(cutoff).alias("cutoff"))
    )
    df.write_parquet(path)
    print(f"  [{cutoff}] cap={cap}: {df.height:,} строк за {time.time() - t0:.0f}s")
    return df


def split_cutoffs(n_cutoffs: int, val_cut: dt.date) -> list[dt.date]:
    """Те же срезы и тот же карантин, что в train.load_split."""
    cuts = train_cutoffs(n_cutoffs)
    latest_ok = val_cut - dt.timedelta(days=HORIZON)
    train_cuts = [c for c in cuts if c < val_cut and c <= latest_ok]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {latest_ok}: увеличьте --cutoffs")
    return train_cuts


def stack(ref_frames: list[pl.DataFrame], src_frames: list[pl.DataFrame] | None,
          feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Строки и таргет из `ref`, значения признаков из `src` (или из самого ref).

    Выпавшие при обрезке пользователи получают пропуски — LightGBM обрабатывает
    их как missing, так же как `predict.py` поступает с клиентами без истории.
    """
    Xs, ys = [], []
    for i, r in enumerate(ref_frames):
        keys = r.select(["user_id", "target"])
        f = keys if src_frames is None else keys.join(
            src_frames[i].select(["user_id", *feats]), on="user_id", how="left")
        if src_frames is None:
            f = keys.join(r.select(["user_id", *feats]), on="user_id", how="left")
        Xs.append(f.select(feats).to_numpy().astype(np.float32))
        ys.append(f["target"].to_numpy().astype(np.float64))
    return np.vstack(Xs), np.concatenate(ys)


def fit_eval(Xtr, ytr, Xva, yva, feats, rounds, tag):
    m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=200, log_period=0)
    m.fit(Xtr, np.log1p(ytr), Xva, np.log1p(yva), feature_names=feats)
    p = m.predict(Xva)
    res = report(yva, np.expm1(np.clip(p, 0, None)), tag)
    return rmse_log(np.log1p(yva), p), m.best_iter, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=229,
                    help="сколько дней истории оставить каждому срезу; 229 — глубина "
                         "самого старого обучающего среза")
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoffs", default="2026-01-15,2025-12-16",
                    help="через запятую; принимаем только при выигрыше на обоих")
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    print("глубина доступной истории по срезам:")
    for c in train_cutoffs(args.cutoffs + 2):
        d = (c - DATA_START).days
        print(f"  {c}: {d} дней" + (f"  -> обрезаем до {args.cap}" if d > args.cap else ""))

    results = []
    for raw in args.val_cutoffs.split(","):
        val_cut = dt.date.fromisoformat(raw.strip())
        n = args.cutoffs if val_cut == train_cutoffs(1)[0] else args.cutoffs + 1
        train_cuts = split_cutoffs(n, val_cut)
        print(f"\n=== валидация {val_cut}, обучение на {len(train_cuts)} срезах ===")

        ref_tr = [get_dataset(c) for c in train_cuts]
        ref_va = [get_dataset(val_cut)]
        feats = feature_names(ref_va[0])

        cap_tr = [capped_dataset(c, args.cap, args.rebuild) for c in train_cuts]
        cap_va = [capped_dataset(val_cut, args.cap, args.rebuild)]
        if feature_names(cap_va[0]) != feats:
            raise SystemExit("набор признаков в обрезанной выборке отличается")

        Xtr, ytr = stack(ref_tr, None, feats)
        Xva, yva = stack(ref_va, None, feats)
        Xtr_c, ytr_c = stack(ref_tr, cap_tr, feats)
        Xva_c, yva_c = stack(ref_va, cap_va, feats)
        assert np.array_equal(yva, yva_c), "таргеты валидации разошлись"
        miss = float(np.isnan(Xva_c[:, feats.index("lt_gmv")]).mean())
        print(f"  строк: train {len(ytr):,}, val {len(yva):,} (одинаковы у обоих вариантов)")
        print(f"  пользователей без активности за последние {args.cap} дней: {miss:.2%}")

        r_base, it_base, _ = fit_eval(Xtr, ytr, Xva, yva, feats, args.rounds, "без обрезки")
        r_cap, it_cap, _ = fit_eval(Xtr_c, ytr_c, Xva_c, yva_c, feats, args.rounds,
                                    f"cap {args.cap}")
        gain = r_base - r_cap
        print(f"  итераций {it_base} против {it_cap}")
        print(f"  ВЫИГРЫШ ОБРЕЗКИ: {gain:+.5f}")
        results.append((val_cut, r_base, r_cap, gain))

        for name, r, it, note in ((f"cap_base_{val_cut:%m%d}", r_base, it_base, "полная история"),
                                  (f"cap{args.cap}_{val_cut:%m%d}", r_cap, it_cap,
                                   f"история обрезана до {args.cap} дней")):
            append_csv(MODELS / "experiments.csv", FIELDS, {
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "feat_ver": features_version(), "blocks": "all",
                "name": name, "model": "lgbm", "cutoffs": len(train_cuts),
                "n_features": len(feats), "rmsle_single": round(r, 5),
                "rmsle_two_stage": "", "rmsle_blend": round(r, 5), "blend_w": "",
                "gini_blend": "", "sum_bias_blend": "", "best_iter_single": it,
                "stride": 30, "halflife": "", "val_cutoff": str(val_cut),
                "train_cutoffs": " ".join(str(c) for c in train_cuts),
                "note": (args.note + "; " if args.note else "")
                        + note + " [одиночный lgbm, умолчания, строки выровнены]"})

    print("\n=== итог ===")
    ok = True
    for val_cut, r_base, r_cap, gain in results:
        ok &= gain > 0
        print(f"{val_cut}: полная {r_base:.5f} | обрезка {r_cap:.5f} | {gain:+.5f}")
    print("\nвердикт: " + (
        "обрезка выигрывает на всех срезах — можно принимать" if ok else
        "выигрыш не на всех срезах — по правилу раздела 2 PLAN.md не принимается"))


if __name__ == "__main__":
    main()
