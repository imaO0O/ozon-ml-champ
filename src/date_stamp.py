"""Признаки-датчики времени: мешают ли они переносу на тестовое окно.

Откуда взялось. Поправка на сдвиг режима через дискриминатор дала AUC = 1.0000:
классификатор отличает январский срез от старых **идеально**. Разбор важности
показал причину — `tenure` (27% выигрыша), `day_crowd_mean_90`, `first_ord_ago`,
`active_months`. Все они растут монотонно со временем, поэтому для клиента
однозначно кодируют дату среза.

Почему это опасно. Значения этих признаков на тесте лежат ВНЕ обучающего
диапазона: tenure идёт 205.9 -> 337.6 по обучающим срезам, а на тесте 367.6.
Деревья за границу диапазона не экстраполируют — они продолжают крайний лист.
То есть по любому разбиению вида `tenure > порог` весь тест уходит в одну
сторону, и модель применяет к нему поправку, выученную на самом свежем
обучающем срезе.

Как это проверяется честно. Наша валидация УЖЕ находится в том же положении:
январский срез (tenure 337.6) лежит за пределами обучающих (205.9 … 307.6).
Значит эффект, если он есть, виден на валидации напрямую, без гаданий про тест.

    python -u src/date_stamp.py --cutoffs 6
    python -u src/date_stamp.py --cutoffs 7 --val-cutoff 2025-12-16
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

# Признаки, значение которых для одного и того же клиента растёт с датой среза
# и на тесте выходит за обучающий диапазон. Список получен не на глаз, а из
# важностей дискриминатора «свежий срез против старых».
STAMPS = ("tenure", "first_ord_ago", "active_months",
          "day_crowd_mean_30", "day_crowd_mean_90", "day_crowd_mean_365")


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def to_rank(col: np.ndarray) -> np.ndarray:
    """Процентильный ранг ВНУТРИ среза: та же информация без привязки к дате.

    Удаление датчика теряет и порядок, а он полезен: клиент со стажем 300 дней
    всё-таки отличается от новичка. Ранг сохраняет порядок и снимает абсолютную
    шкалу — ровно то, что блок `ranks` делает с денежными величинами. На тесте
    ранг считается по тестовому срезу, поэтому за обучающий диапазон
    не выходит по построению.
    """
    order = np.argsort(col, kind="stable")
    r = np.empty(len(col), dtype=np.float32)
    r[order] = np.arange(1, len(col) + 1, dtype=np.float32) / len(col)
    return r


def fit(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves):
    import lightgbm as lgb
    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(200)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="stamp")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-val-pred", action="store_true")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {val_cut}")
    blocks = parse_blocks(args.blocks)

    dfv = get_dataset(val_cut, blocks=blocks)
    feats = feature_names(dfv)
    keep = [i for i, f in enumerate(feats) if f not in STAMPS]
    dropped = [f for f in feats if f in STAMPS]
    print(f"валидация {val_cut} | обучение на {len(train_cuts)}")
    print(f"признаков {len(feats)}, снимаем {len(dropped)}: {', '.join(dropped)}")

    Xv = dfv.select(feats).to_numpy().astype(np.float32)
    yv = dfv["target"].to_numpy().astype(np.float64)
    parts = [get_dataset(c, blocks=blocks) for c in train_cuts]
    Xtr = np.vstack([d.select(feats).to_numpy().astype(np.float32) for d in parts])
    ytr = np.concatenate([d["target"].to_numpy().astype(np.float64) for d in parts])
    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)

    # Насколько далеко валидация уходит за обучающий диапазон по каждому датчику.
    print("\nвыход валидации за обучающий диапазон:")
    for f in dropped:
        j = feats.index(f)
        hi = Xtr[:, j].max()
        out = float((Xv[:, j] > hi).mean())
        print(f"  {f:<22} максимум обучения {hi:>12,.1f} | за ним "
              f"{out:.1%} валидационных клиентов")

    print("\n--- контроль: все признаки ---")
    p_all, it_all = fit(Xtr, ytr_log, Xv, yv_log, feats,
                        args.rounds, args.early_stopping, args.lr, args.leaves)
    print("\n--- рука: без датчиков времени ---")
    names = [feats[i] for i in keep]
    p_cut, it_cut = fit(Xtr[:, keep], ytr_log, Xv[:, keep], yv_log, names,
                        args.rounds, args.early_stopping, args.lr, args.leaves)

    print("--- рука: датчики заменены рангом внутри среза ---")
    Xtr_r, Xv_r = Xtr.copy(), Xv.copy()
    sizes = [d.height for d in parts]
    for f in dropped:
        j = feats.index(f)
        off = 0
        for n in sizes:
            Xtr_r[off:off + n, j] = to_rank(Xtr_r[off:off + n, j])
            off += n
        Xv_r[:, j] = to_rank(Xv_r[:, j])
    p_rk, it_rk = fit(Xtr_r, ytr_log, Xv_r, yv_log, feats,
                      args.rounds, args.early_stopping, args.lr, args.leaves)

    print(f"\n{'рука':<16}{'сырой':>10}{'выровн.':>10}{'Gini':>9}{'итераций':>10}")
    rows = []
    for tag, p, it in (("все признаки", p_all, it_all),
                       ("без датчиков", p_cut, it_cut),
                       ("датчики рангом", p_rk, it_rk)):
        r, a = rmse_log(yv_log, p), aligned(yv_log, p)
        g = gini_norm(yv, np.expm1(p))
        rows.append((tag, r, a, g, it, p))
        print(f"{tag:<16}{r:>10.5f}{a:>10.5f}{g:>9.4f}{it:>10}")
    print(f"\nбез датчиков к контролю по выровненному: {rows[0][2] - rows[1][2]:+.5f}")
    print(f"датчики рангом к контролю: {rows[0][2] - rows[2][2]:+.5f}")

    for tag, r, a, g, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag.split()[0]}",
            "model": "lgbm", "cutoffs": len(train_cuts),
            "n_features": len(feats) if tag.startswith("все") else len(names),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "best_iter_single": it,
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [датчики времени, выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    if args.save_val_pred:
        ids = dfv["user_id"].to_numpy()
        for tag, p in (("all", rows[0][5]), ("cut", rows[1][5]), ("rk", rows[2][5])):
            np.savez_compressed(MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
                                user_id=ids, pred_log=p, target=yv)
        print("предсказания сохранены")


if __name__ == "__main__":
    main()
