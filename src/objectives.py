"""Другие функции потерь — не ради своей точности, а ради непохожести.

Зачем. Бленд у нас исчерпан по измерению: 46 файлов, положительный вес только
у трёх и все прибавки не выше +0.00001. Причина видна прямо в числах — все
наши модели скоррелированы на 0.999, потому что обучены на одних признаках
под одну и ту же квадратичную ошибку в log1p. Складывать почти одинаковые
предсказания бессмысленно.

Критерий партнёрства (PLAN, раздел 3) говорит, чего не хватает. Партнёр
окупается тогда и только тогда, когда расстояние до него больше его
отставания, а выигрыш равен (D − δ)² / (4D). У нынешнего семейства D порядка
0.001–0.009 — отсюда и ноль. Другая функция потерь меняет форму предсказаний
целиком и может дать D на порядок больше. Вопрос единственный: вырастет ли
отставание быстрее расстояния. Он решается офлайн, без единого сабмита.

Что проверяется:

* huber и fair на log1p — та же величина, но выбросы весят иначе;
* regression_l1 (MAE) на log1p — предсказывает медиану, а не среднее;
  при 46% нулей это совсем другая форма, и отставание ожидается большим;
* tweedie и poisson на СЫРОМ gmv — естественные для нуль-раздутой величины,
  предсказывают сырое среднее; в лог-пространство переводятся через log1p.

Про последние две надо понимать заранее: они оценивают E[y], а метрике нужно
E[log1p(y)], и log1p(E[y]) этому не равно. Это не ошибка постановки, а ровно
та причина, по которой они и окажутся непохожими. Часть расхождения снимут
бесплатный сдвиг уровня и растяжение, остальное пойдёт в D.

    python -u src/objectives.py --base models/stk_b64a_jan_valpred_2026-01-15.npz
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

# (имя, параметры поверх LGB_REG, обучать ли на сыром таргете)
ARMS = [
    ("контроль", {}, False),
    ("huber", {"objective": "huber", "alpha": 1.0}, False),
    ("fair", {"objective": "fair", "fair_c": 1.0}, False),
    ("MAE", {"objective": "regression_l1"}, False),
    ("tweedie13", {"objective": "tweedie", "tweedie_variance_power": 1.3}, True),
    ("tweedie17", {"objective": "tweedie", "tweedie_variance_power": 1.7}, True),
    ("poisson", {"objective": "poisson"}, True),
]


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def fit(Xtr, ytr, Xv, yv_log, feats, extra, raw, rounds, stop):
    """Обучение одной руки. Остановка всегда по ВЫРОВНЕННОМУ RMSLE в log1p.

    Для сырых целей (tweedie, poisson) предсказание переводится в log1p прямо
    внутри метрики: иначе рука останавливалась бы по своей внутренней шкале,
    а сравнивать надо по общей. Это то же требование, что и везде —
    останавливаться по той величине, которая идёт в зачёт.
    """
    import lightgbm as lgb

    def feval(preds, data):
        p = np.asarray(preds, dtype=np.float64)
        if raw:
            p = np.log1p(np.maximum(p, 0.0))
        p = p - p.mean() + yv_log.mean()
        return "rmsle_aligned", float(np.sqrt(np.mean((p - yv_log) ** 2))), False

    params = {**LGB_REG, **extra, "metric": "None"}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=np.expm1(yv_log) if raw else yv_log, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv], feval=feval,
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(0)])
    p = m.predict(Xv, num_iteration=m.best_iteration)
    if raw:
        p = np.log1p(np.maximum(p, 0.0))
    return p, m.best_iteration


def partner(base: np.ndarray, arm: np.ndarray, y_log: np.ndarray):
    """Оптимальный вес руки в бленде с базой и получаемый выровненный RMSLE.

    Считается в ВЫРОВНЕННОМ виде: уровень правится бесплатно, поэтому обе руки
    приводятся к уровню цели, и вес ищется только по форме.
    """
    b = base - base.mean() + y_log.mean()
    a = arm - arm.mean() + y_log.mean()
    d = float(np.mean((b - a) ** 2))
    m1 = float(np.sqrt(np.mean((b - y_log) ** 2)))
    if d < 1e-12:
        return 0.0, d, m1
    mse1 = float(np.mean((b - y_log) ** 2))
    mse2 = float(np.mean((a - y_log) ** 2))
    w = float(np.clip((mse1 - mse2 + d) / (2 * d), 0.0, 1.0))
    mix = (1 - w) * b + w * a
    return w, d, float(np.sqrt(np.mean((mix - y_log) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--base", default=None,
                    help="npz с лучшим предсказанием на этом же срезе — партнёр сравнивается с ним")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="obj")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    blocks = parse_blocks(args.blocks)

    dfv = get_dataset(val_cut, blocks=blocks)
    feats = feature_names(dfv)
    Xv = dfv.select(feats).to_numpy().astype(np.float32)
    yv = dfv["target"].to_numpy().astype(np.float64)
    parts = [get_dataset(c, blocks=blocks) for c in train_cuts]
    Xtr = np.vstack([d.select(feats).to_numpy().astype(np.float32) for d in parts])
    ytr = np.concatenate([d["target"].to_numpy().astype(np.float64) for d in parts])
    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)
    del parts
    print(f"валидация {val_cut} | обучение {len(ytr):,} x {len(feats)}\n")

    ids_val = dfv["user_id"].to_numpy()
    order = np.argsort(ids_val)
    base = None
    if args.base:
        d = np.load(args.base)
        o = np.argsort(d["user_id"])
        if not np.array_equal(d["user_id"][o], ids_val[order]):
            raise SystemExit("база посчитана на другом наборе пользователей")
        base = np.empty_like(yv_log)
        base[order] = d["pred_log"][o]
        print(f"база {args.base}: выровненный {aligned(yv_log, base):.5f}\n")

    print(f"{'рука':<12}{'выровн.':>10}{'отставание':>12}{'D до базы':>11}"
          f"{'вес':>7}{'бленд':>10}{'выигрыш':>10}{'итер.':>7}")
    rows = []
    for tag, extra, raw in ARMS:
        p, it = fit(Xtr, ytr if raw else ytr_log, Xv, yv_log, feats,
                    extra, raw, args.rounds, args.early_stopping)
        a = aligned(yv_log, p)
        line = f"{tag:<12}{a:>10.5f}"
        if base is not None:
            m1 = aligned(yv_log, base)
            w, D, mix = partner(base, p, yv_log)
            line += f"{a - m1:>+12.5f}{D:>11.5f}{w:>7.2f}{mix:>10.5f}{m1 - mix:>+10.5f}"
        line += f"{it:>7}"
        print(line)
        rows.append((tag, a, it, p))

    for tag, a, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag}",
            "model": "lgbm", "cutoffs": len(train_cuts), "n_features": len(feats),
            "rmsle_single": round(float(rmse_log(yv_log, p)), 5),
            "rmsle_blend": round(float(rmse_log(yv_log, p)), 5),
            "gini_blend": round(float(gini_norm(yv, np.expm1(p))), 4),
            "best_iter_single": it, "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [функция потерь {tag}, выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    for tag, a, it, p in rows:
        np.savez_compressed(
            MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
            user_id=ids_val[order], pred_log=p[order], target=yv[order])
    print("предсказания сохранены")


if __name__ == "__main__":
    main()
