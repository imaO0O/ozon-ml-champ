"""Распределительная голова у БУСТИНГА: мультикласс по бинам вместо L2.

Зачем. Тот же приём на сети дал +0.0005 и +0.0009 на двух срезах, а через
стекинг +0.00110 и +0.00203 — лучшее, что нашлось за неделю. Но сеть у нас
слабая половина решения, а несёт его бустинг, и он до сих пор учит L2 на
log1p. Здесь проверяется, переносится ли выигрыш на сильную половину.

Механизм тот же. Условное распределение log1p(gmv) бимодально: около 46%
таргетов ровно ноль, остальное — тяжёлый хвост. L2 управляет средним такого
распределения косвенно, сдвигая одну точку под градиентом. Мультикласс учит
форму, а среднее читается из неё точно: E[log1p] = sum p_k * c_k.

Что сравнивается. Одна и та же модель, одни признаки, одни гиперпараметры,
одни срезы — разница только в голове:

    контроль    objective=regression   на log1p
    рука        objective=multiclass   по бинам, предсказание = E[log1p]

Ансамбль здесь намеренно не используется: он усредняет пять конфигураций
и размывает эффект головы, а вопрос стоит про голову.

Цена. Мультикласс строит K деревьев на итерацию, поэтому K=24 идёт примерно
в двадцать раз дольше регрессии. Огрубление при 24 бинах — 0.11 при
стандартном отклонении цели 2.03, и на оценку СРЕДНЕГО оно почти не влияет:
предсказание берётся как непрерывное среднее по распределению, а не как
центр бина.

    python -u src/dist_gbm.py --cutoffs 6 --bins 24
    python -u src/dist_gbm.py --cutoffs 7 --val-cutoff 2025-12-16 --bins 24
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from binning import bin_index, make_bins
from config import MODELS, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log, sum_bias
from models import LGB_REG
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
    return df, X, y, feats


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    """RMSLE после приведения уровня — единственное, что сравнимо между моделями.

    Уровень на сабмите правится бесплатным сдвигом из TEST_LEVEL, поэтому
    выигрыш в уровне на лидерборде не засчитывается (PLAN.md, раздел 2).
    """
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def fit_reg(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves):
    import lightgbm as lgb
    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(200)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


def fit_bins(Xtr, itr, Xv, iv, centers, feats, rounds, stop, lr, leaves):
    import lightgbm as lgb
    params = {**LGB_REG, "objective": "multiclass", "metric": "multi_logloss",
              "num_class": len(centers), "learning_rate": lr, "num_leaves": leaves}
    dtr = lgb.Dataset(Xtr, label=itr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=iv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(50)])
    proba = m.predict(Xv, num_iteration=m.best_iteration)
    return proba @ centers, m.best_iteration, proba


def reliability(proba, yv_log):
    """Совпадает ли обещанная вероятность нуля с наблюдаемой долей нулей.

    Вопрос жюри про вероятностность звучит именно так: не «есть ли у модели
    вероятности», а «можно ли им верить». У BG/NBD мы это мерили и записали,
    что в верхних корзинах она самоуверенна на 5-8 процентных пунктов.
    """
    p0 = proba[:, 0]
    real0 = (yv_log == 0).astype(float)
    print(f"\nвероятность нуля: обещано {p0.mean():.3f}, на деле {real0.mean():.3f}")
    qs = np.quantile(p0, np.linspace(0, 1, 11))
    print("надёжность по децилям P(y=0):")
    worst = 0.0
    for lo, hi in zip(qs[:-1], qs[1:]):
        take = (p0 >= lo) & (p0 < hi)
        if take.sum() > 100:
            said, was = p0[take].mean(), real0[take].mean()
            worst = max(worst, abs(said - was))
            print(f"  обещано {said:.3f} -> на деле {was:.3f}  ({take.sum():,} клиентов)")
    print(f"худшее расхождение по децилю: {worst:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.10,
                    help="одна на обе руки: сравнивается голова, а не темп обучения")
    ap.add_argument("--leaves", type=int, default=63,
                    help="тоже одна на обе руки. Мультикласс строит K деревьев "
                         "на итерацию, поэтому рабочие 127 листьев делают его "
                         "неподъёмным: 24 бина при 127 листьях идут около семи "
                         "часов на срез")
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--name", default="dist_gbm")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-val-pred", action="store_true")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {val_cut}: увеличьте --cutoffs")

    blocks = parse_blocks(args.blocks)
    print(f"валидация {val_cut} | обучение на {len(train_cuts)}: "
          f"{', '.join(str(c) for c in train_cuts)}")
    dfv, Xv, yv, feats = frame(val_cut, blocks, args.rebuild)
    parts = [frame(c, blocks, args.rebuild) for c in train_cuts]
    Xtr = np.vstack([p[1] for p in parts])
    ytr = np.concatenate([p[2] for p in parts])
    print(f"обучающих строк {len(ytr):,} | признаков {len(feats)} | "
          f"валидационных {len(yv):,}")

    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)
    edges, centers = make_bins([p[2] for p in parts], args.bins)
    itr, iv = bin_index(ytr_log, edges), bin_index(yv_log, edges)
    print(f"бинов {len(centers)} | атом в нуле {(itr == 0).mean():.1%} обучающих | "
          f"центры от {centers[1]:.3f} до {centers[-1]:.3f}")

    print("\n--- контроль: regression на log1p ---")
    p_reg, it_reg = fit_reg(Xtr, ytr_log, Xv, yv_log, feats,
                            args.rounds, args.early_stopping, args.lr, args.leaves)
    print(f"\n--- рука: multiclass по {len(centers)} бинам ---")
    p_bin, it_bin, proba = fit_bins(Xtr, itr, Xv, iv, centers, feats,
                                    args.rounds, args.early_stopping, args.lr, args.leaves)

    print(f"\n{'рука':<12}{'сырой':>10}{'выровн.':>10}{'Gini':>9}"
          f"{'смещение':>11}{'итераций':>10}")
    rows = []
    for tag, p, it in (("контроль", p_reg, it_reg),
                       (f"бины", p_bin, it_bin)):
        r, a = rmse_log(yv_log, p), aligned(yv_log, p)
        g, b = gini_norm(yv, np.expm1(p)), sum_bias(yv, np.expm1(p))
        rows.append((tag, r, a, g, b, it, p))
        print(f"{tag:<12}{r:>10.5f}{a:>10.5f}{g:>9.4f}{b:>+11.2%}{it:>10}")

    print(f"\nбины к контролю по выровненному: {rows[0][2] - rows[1][2]:+.5f}")
    print(f"расстояние между руками D = "
          f"{float(np.mean((rows[0][6] - rows[1][6]) ** 2)):.5f}")
    reliability(proba, yv_log)

    for tag, r, a, g, b, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag}",
            "model": "lgbm", "cutoffs": len(train_cuts), "n_features": len(feats),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "sum_bias_blend": round(b, 4),
            "best_iter_single": it, "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [голова бустинга, bins={len(centers)}, "
                    f"lr={args.lr}, листьев {args.leaves}, выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    if args.save_val_pred:
        for tag, p in (("ctl", rows[0][6]), ("bin", rows[1][6])):
            np.savez_compressed(MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
                                user_id=dfv["user_id"].to_numpy(), pred_log=p, target=yv)
        np.savez_compressed(MODELS / f"{args.name}_proba_{val_cut}.npz",
                            user_id=dfv["user_id"].to_numpy(), proba=proba,
                            centers=centers, target=yv)
        print("предсказания и распределение сохранены")


if __name__ == "__main__":
    main()
