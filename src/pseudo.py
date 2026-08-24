"""Псевдоразметка: тестовые клиенты в обучении с предсказанием вместо таргета.

Идея. Признаки тестового окна известны, таргеты нет. Добавляем тестовых
клиентов в обучающую выборку, подставив им наше лучшее предсказание как мягкую
метку и малый вес. Модель видит распределение признаков того окна, на котором
её будут спрашивать, — а это единственный способ показать ей тестовый режим,
не зная ответов.

Почему это НЕ то же, что поправка на сдвиг весами. Та требовала пересекающихся
носителей и провалилась именно на этом: дискриминатор отличал срезы идеально,
AUC = 1.0000, и отношение плотностей вырождалось (PLAN, раздел 4а). Здесь
носители не нужны вовсе — тестовые строки добавляются как есть.

Чего приём НЕ может. Новой информации о таргете в нём нет по построению:
метка берётся из той же модели. Работать он может только через регуляризацию —
модель вынуждена быть согласованной на тестовом распределении признаков,
и это иногда убирает часть экстраполяционного произвола. Ожидание низкое,
и мы говорим это заранее.

Проверка честная: на валидации роль «теста» играет валидационный срез,
и его настоящие таргеты в обучение НЕ попадают — только предсказания
контрольной модели, обученной без него.

    python -u src/pseudo.py --cutoffs 6 --weight 0.3
    python -u src/pseudo.py --cutoffs 7 --val-cutoff 2025-12-16 --weight 0.3
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log
from models import LGB_REG
from utils import append_csv, git_commit

LOG = MODELS / "experiments.csv"
FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def fit(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves, weight=None):
    """Обучение с остановкой по ВЫРОВНЕННОМУ RMSLE (см. models.aligned_rmsle)."""
    import lightgbm as lgb

    def feval(preds, data):
        y = data.get_label()
        p = np.asarray(preds, dtype=np.float64)
        p = p - p.mean() + y.mean()
        return "rmsle_aligned", float(np.sqrt(np.mean((p - y) ** 2))), False

    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves, "metric": "None"}
    dtr = lgb.Dataset(Xtr, label=ytr, weight=weight, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv], feval=feval,
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(0)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--weight", type=float, default=0.3,
                    help="вес псевдоразмеченных строк относительно настоящих")
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--early-stopping", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="pseudo")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    blocks = parse_blocks(args.blocks)

    dfv = get_dataset(val_cut, blocks=blocks)
    feats = feature_names(dfv)
    Xv = dfv.select(feats).to_numpy().astype(np.float32)
    yv_log = np.log1p(dfv["target"].to_numpy().astype(np.float64))
    parts = [get_dataset(c, blocks=blocks) for c in train_cuts]
    Xtr = np.vstack([d.select(feats).to_numpy().astype(np.float32) for d in parts])
    ytr_log = np.log1p(np.concatenate(
        [d["target"].to_numpy().astype(np.float64) for d in parts]))
    print(f"валидация {val_cut} | обучение на {len(train_cuts)} срезах, "
          f"{len(ytr_log):,} строк | признаков {len(feats)}")

    print("\n--- контроль: без псевдоразметки ---")
    p_ctl, it_ctl = fit(Xtr, ytr_log, Xv, yv_log, feats,
                        args.rounds, args.early_stopping, args.lr, args.leaves)

    # Мягкие метки берутся у контрольной модели: настоящие таргеты
    # валидационного среза в обучение не попадают ни в каком виде.
    print(f"\n--- рука: + {len(yv_log):,} псевдоразмеченных строк, "
          f"вес {args.weight} ---")
    Xp = np.vstack([Xtr, Xv])
    yp = np.concatenate([ytr_log, p_ctl])
    w = np.concatenate([np.ones(len(ytr_log), dtype=np.float32),
                        np.full(len(p_ctl), args.weight, dtype=np.float32)])
    p_ps, it_ps = fit(Xp, yp, Xv, yv_log, feats, args.rounds,
                      args.early_stopping, args.lr, args.leaves, weight=w)

    print(f"\n{'рука':<24}{'сырой':>10}{'выровн.':>10}{'Gini':>9}{'итераций':>10}")
    rows = []
    for tag, p, it in (("без псевдоразметки", p_ctl, it_ctl),
                       ("с псевдоразметкой", p_ps, it_ps)):
        r, a = rmse_log(yv_log, p), aligned(yv_log, p)
        g = gini_norm(np.expm1(yv_log), np.expm1(p))
        rows.append((tag, r, a, g, it, p))
        print(f"{tag:<24}{r:>10.5f}{a:>10.5f}{g:>9.4f}{it:>10}")
    print(f"\nпсевдоразметка к контролю: {rows[0][2] - rows[1][2]:+.5f}")
    print(f"расстояние между руками D = "
          f"{float(np.mean((rows[0][5] - rows[1][5]) ** 2)):.5f}")

    for tag, r, a, g, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag.split()[0]}",
            "model": "lgbm", "cutoffs": len(train_cuts), "n_features": len(feats),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "best_iter_single": it,
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [псевдоразметка вес {args.weight}, "
                    f"выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")


if __name__ == "__main__":
    main()
