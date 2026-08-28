"""Поправка на сдвиг режима: веса обучающих клиентов из дискриминатора.

Зачем. У нас есть **прямое измерение сдвига**, а не подозрение: оптимальное
растяжение предсказаний на тестовом окне — около 5%, на валидационном — около
1% (PLAN.md, раздел 3). Правда на тесте разбросана иначе, чем на валидации:
площадка выросла, состав активных сменился. Взвешивание срезов по свежести
мы пробовали и получили ноль, но это грубая версия приёма — вес там был один
на весь срез. Здесь вес **поклиентный**.

Механизм стандартный для covariate shift. Обучаем классификатор «этот клиент
из свежего среза или из старого», и если он отличает их с вероятностью p,
то отношение плотностей оценивается как p/(1−p). Обучение с такими весами
двигает модель к тому распределению клиентов, на котором её будут спрашивать.

Почему это проверяемо на валидации, хотя чинится сдвиг к тесту. Дрейф
декабрь→январь того же рода, что январь→тест: те же тридцать дней, тот же
рост площадки. Если механизм реален, он обязан проявиться и внутри нашей
валидации; если не проявится — гипотеза о поклиентном сдвиге не подтверждена,
и тратить на неё сабмиты нельзя.

Веса считаются ВНЕ ОБУЧЕНИЯ: дискриминатор учится на половине строк
и размечает вторую, потом наоборот. Иначе он запоминает конкретные строки,
и веса получаются подгонкой под шум, а не оценкой плотности.

    python -u src/shift_weights.py --cutoffs 6
    python -u src/shift_weights.py --cutoffs 7 --val-cutoff 2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, SEED, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log
from models import LGB_REG, aligned_rmsle
from utils import append_csv, git_commit

LOG = MODELS / "experiments.csv"
FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]


def frame(cutoff, blocks, rebuild: bool):
    df = get_dataset(cutoff, rebuild=rebuild, blocks=blocks)
    feats = feature_names(df)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    return X, y, feats


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def discriminator(Xtr, Xv, feats, rounds, lr):
    """Вероятность «строка из свежего среза», посчитанная вне обучения.

    Две половины: на первой учимся — размечаем вторую, потом наоборот.
    Возвращает p для обучающих строк и AUC как признак того, насколько
    сдвиг вообще различим.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X = np.vstack([Xtr, Xv])
    lab = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xv))])
    rng = np.random.default_rng(SEED)
    half = rng.random(len(X)) < 0.5
    p = np.empty(len(X))
    params = {**LGB_REG, "objective": "binary", "metric": "auc", "learning_rate": lr}
    for fit_on in (half, ~half):
        d = lgb.Dataset(X[fit_on], label=lab[fit_on], feature_name=feats)
        m = lgb.train(params, d, num_boost_round=rounds)
        p[~fit_on] = m.predict(X[~fit_on])
    auc = roc_auc_score(lab, p)
    return p[:len(Xtr)], auc


def fit_reg(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves, weight=None):
    import lightgbm as lgb
    # Остановка по ВЫРОВНЕННОМУ RMSLE: сырая величина включает уровень,
    # который на сабмите правится бесплатно, и обрывает обучение вдвое раньше
    # нужного (PLAN.md, раздел 4а). Все замеры этого файла до 22.08 делались
    # на недообученных руках — сравнение было честным, но точка не та.
    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves,
              "metric": "None"}
    dtr = lgb.Dataset(Xtr, label=ytr, weight=weight, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  feval=aligned_rmsle,
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(200)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--clip", type=float, default=5.0,
                    help="потолок веса. Без него несколько клиентов с p близким "
                         "к единице получают вес в сотни и решают обучение сами")
    ap.add_argument("--disc-rounds", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--name", default="shift")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-val-pred", action="store_true")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {val_cut}")
    blocks = parse_blocks(args.blocks)

    print(f"валидация {val_cut} | обучение на {len(train_cuts)}: "
          f"{', '.join(str(c) for c in train_cuts)}")
    Xv, yv, feats = frame(val_cut, blocks, args.rebuild)
    parts = [frame(c, blocks, args.rebuild) for c in train_cuts]
    Xtr = np.vstack([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)
    print(f"обучающих строк {len(ytr):,} | признаков {len(feats)} | "
          f"валидационных {len(yv):,}")

    print("\n--- дискриминатор: отличим ли свежий срез от старых ---")
    p, auc = discriminator(Xtr, Xv, feats, args.disc_rounds, args.lr)
    w = np.clip(p / np.maximum(1.0 - p, 1e-6), 0.0, args.clip)
    w = w / w.mean()
    # Эффективный размер выборки: сколько строк «осталось» после взвешивания.
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"AUC = {auc:.4f} | вес: медиана {np.median(w):.3f}, "
          f"90-й перцентиль {np.percentile(w, 90):.3f}, доля упёршихся в потолок "
          f"{(w >= args.clip / w.mean() * 0.999).mean():.2%}")
    print(f"эффективный размер выборки {ess:,.0f} из {len(w):,} "
          f"({ess / len(w):.1%})")
    if auc < 0.55:
        print("сдвиг почти неразличим — веса будут близки к единице, "
              "и приём не может дать ничего")

    print("\n--- контроль: без весов ---")
    p_ctl, it_ctl = fit_reg(Xtr, ytr_log, Xv, yv_log, feats,
                            args.rounds, args.early_stopping, args.lr, args.leaves)
    print("\n--- рука: веса из дискриминатора ---")
    p_w, it_w = fit_reg(Xtr, ytr_log, Xv, yv_log, feats, args.rounds,
                        args.early_stopping, args.lr, args.leaves, weight=w)

    print(f"\n{'рука':<12}{'сырой':>10}{'выровн.':>10}{'Gini':>9}{'итераций':>10}")
    rows = []
    for tag, pred, it in (("контроль", p_ctl, it_ctl), ("веса", p_w, it_w)):
        r, a = rmse_log(yv_log, pred), aligned(yv_log, pred)
        g = gini_norm(yv, np.expm1(pred))
        rows.append((tag, r, a, g, it, pred))
        print(f"{tag:<12}{r:>10.5f}{a:>10.5f}{g:>9.4f}{it:>10}")
    print(f"\nвеса к контролю по выровненному: {rows[0][2] - rows[1][2]:+.5f}")
    print(f"расстояние между руками D = "
          f"{float(np.mean((rows[0][5] - rows[1][5]) ** 2)):.5f}")

    for tag, r, a, g, it, pred in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag}",
            "model": "lgbm", "cutoffs": len(train_cuts), "n_features": len(feats),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "best_iter_single": it,
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [поправка на сдвиг, AUC {auc:.4f}, "
                    f"потолок веса {args.clip}, ESS {ess / len(w):.1%}, "
                    f"выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    if args.save_val_pred:
        import polars as pl
        ids = get_dataset(val_cut, blocks=blocks)["user_id"].to_numpy()
        for tag, pred in (("ctl", rows[0][5]), ("w", rows[1][5])):
            np.savez_compressed(MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
                                user_id=ids, pred_log=pred, target=yv)
        print("предсказания сохранены")


if __name__ == "__main__":
    main()
