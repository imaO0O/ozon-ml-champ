"""Траектория ранга: где клиент был месяц назад, а не только где он сейчас.

Чем это отличается от уже закрытого. Разности рангов ВНУТРИ среза проверены
и отвергнуты (TASKS, A1: 1.68282 против 1.68151 на январе): пару статических
рангов дерево комбинирует само, отдельный признак ему не нужен. Здесь другое —
ранг клиента на ПРЕДЫДУЩЕМ срезе, то есть его позиция месяц назад. Из текущих
признаков она не выводится ни деревом, ни как угодно: это просто другой момент
времени.

Почему это может сработать, если абсолютная история у модели и так есть.
Абсолютные величины прошлого (`gmv_sum_90`, `gmv_sum_365`) заражены уровнем
площадки, а он за год менялся вдвое — ровно та беда, ради которой заводился
блок `ranks`. Ранг прошлого от уровня не зависит: «был в верхних 10% и остался»
и «был в верхних 10% и упал до 40%» — это разные клиенты, и различить их
абсолютными суммами нельзя, потому что у обоих суммы могли вырасти.

Утечки нет: предыдущий срез старше текущего, его признаки посчитаны по данным
ещё более старым. Для теста предыдущий срез — 2026-01-15, он целиком в прошлом.

    python -u src/rank_traj.py --cutoffs 8
    python -u src/rank_traj.py --cutoffs 8 --val-cutoff 2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

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


def with_prev(cutoff: dt.date, prev: dt.date, blocks) -> tuple[pl.DataFrame, list[str]]:
    """Датасет среза плюс сдвиг ранга относительно предыдущего среза.

    Подаётся именно РАЗНОСТЬ, а не пара рангов: сам текущий ранг у модели уже
    есть, а прошлый в отдельной колонке добавил бы 27 почти дублирующих
    признаков. Ширина входа нам уже дорого обошлась на когортных приорах.

    Клиентов, которых на прошлом срезе не было (новички), помечаем нулём:
    сдвига у них не существует, а не «он равен нулю» — но отличить это модель
    может по стажу, который у неё есть.
    """
    df = get_dataset(cutoff, blocks=blocks)
    pv = get_dataset(prev, blocks=blocks)
    rk = [c for c in df.columns if c.startswith("rk_")]
    pv = pv.select(["user_id", *rk]).rename({c: f"p_{c}" for c in rk})
    df = df.join(pv, on="user_id", how="left")
    delta = [(pl.col(c) - pl.col(f"p_{c}")).fill_null(0.0)
             .cast(pl.Float32).alias(f"d{c}") for c in rk]
    df = df.with_columns(delta).drop([f"p_{c}" for c in rk])
    return df, [f"d{c}" for c in rk]


def fit(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves):
    import lightgbm as lgb
    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(200)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=8)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="traj")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-val-pred", action="store_true")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    # Последнему обучающему срезу нужен ещё более старый — на нём и обрываем.
    if len(train_cuts) < 2:
        raise SystemExit("нужно минимум два обучающих среза: увеличьте --cutoffs")
    train_cuts = train_cuts[:-1]
    prev = {c: cuts[cuts.index(c) + 1] for c in [val_cut, *train_cuts]}
    blocks = parse_blocks(args.blocks)
    print(f"валидация {val_cut} (прошлый {prev[val_cut]}) | обучение на "
          f"{len(train_cuts)}: {', '.join(str(c) for c in train_cuts)}")

    dfv, dcols = with_prev(val_cut, prev[val_cut], blocks)
    base = [f for f in feature_names(dfv) if not f.startswith("drk_")]
    both = base + dcols
    print(f"базовых признаков {len(base)}, добавляем {len(dcols)} сдвигов ранга")

    parts = [with_prev(c, prev[c], blocks)[0] for c in train_cuts]
    Xv_b = dfv.select(base).to_numpy().astype(np.float32)
    Xv_a = dfv.select(both).to_numpy().astype(np.float32)
    yv = dfv["target"].to_numpy().astype(np.float64)
    Xt_b = np.vstack([d.select(base).to_numpy().astype(np.float32) for d in parts])
    Xt_a = np.vstack([d.select(both).to_numpy().astype(np.float32) for d in parts])
    ytr = np.concatenate([d["target"].to_numpy().astype(np.float64) for d in parts])
    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)
    print(f"обучающих строк {len(ytr):,} | валидационных {len(yv):,}")

    sh = dfv.select(dcols).to_numpy()
    print(f"сдвиг ранга: доля ровно нулевых {float((sh == 0).mean()):.1%} "
          f"(новички и неизменившиеся) | разброс {sh.std():.4f}")

    print("\n--- контроль: без траектории ---")
    p_b, it_b, _ = fit(Xt_b, ytr_log, Xv_b, yv_log, base,
                       args.rounds, args.early_stopping, args.lr, args.leaves)
    print("\n--- рука: со сдвигами ранга ---")
    p_a, it_a, m_a = fit(Xt_a, ytr_log, Xv_a, yv_log, both,
                         args.rounds, args.early_stopping, args.lr, args.leaves)

    print(f"\n{'рука':<20}{'сырой':>10}{'выровн.':>10}{'Gini':>9}{'итераций':>10}")
    rows = []
    for tag, p, it in (("без траектории", p_b, it_b), ("со сдвигами", p_a, it_a)):
        r, a = rmse_log(yv_log, p), aligned(yv_log, p)
        g = gini_norm(yv, np.expm1(p))
        rows.append((tag, r, a, g, it, p))
        print(f"{tag:<20}{r:>10.5f}{a:>10.5f}{g:>9.4f}{it:>10}")
    print(f"\nсдвиги к контролю по выровненному: {rows[0][2] - rows[1][2]:+.5f}")

    imp = sorted(zip(both, m_a.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(g for _, g in imp)
    share = sum(g for f, g in imp if f.startswith("drk_")) / tot
    print(f"доля новых признаков во всём выигрыше: {share:.1%}")
    print("лучшие из новых:")
    for f, g in [t for t in imp if t[0].startswith("drk_")][:5]:
        print(f"  {f:<24}{g:>14,.0f}  (место {imp.index((f, g)) + 1})")

    for tag, r, a, g, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag.split()[0]}",
            "model": "lgbm", "cutoffs": len(train_cuts),
            "n_features": len(base) if tag.startswith("без") else len(both),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "best_iter_single": it,
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [траектория ранга, выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    if args.save_val_pred:
        ids = dfv["user_id"].to_numpy()
        for tag, p in (("base", rows[0][5]), ("traj", rows[1][5])):
            np.savez_compressed(MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
                                user_id=ids, pred_log=p, target=yv)


if __name__ == "__main__":
    main()
