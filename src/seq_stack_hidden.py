"""Стекинг на скрытом состоянии сети: даёт ли оно больше, чем её предсказание.

Организаторы намекали отдать бустингу «скрытое состояние» — команда отдала
предсказание, то есть **одно число**. Но предсказание есть одномерная проекция
скрытого состояния: сеть сжимает 128 чисел в одно, и всё, что не легло на эту
ось, теряется безвозвратно. Бустинг же умеет строить пороги по каждой оси
отдельно.

Замер того стоит: стекинг на одном числе дал на лидерборде 1.6515847 против
1.6526610 у чистого бустинга — единственное моделирование за всю работу,
перешагнувшее порог. Здесь тот же приём с полным состоянием.

Признаки берутся честно, схемой walk-forward: скрытое состояние на срезе `c`
получено сетью, обученной только на срезах старше `c` (см. seq_oof_net.py).
Состояния с обучающих срезов дали бы бустингу утечку.

Сравнение всегда парное и в одном прогоне: одна и та же конфигурация LightGBM
с добавленными колонками и без них, на одном и том же наборе строк. Принимаем
только при положительном знаке на обоих срезах.

    python -u src/seq_stack_hidden.py --name nethid
    python -u src/seq_stack_hidden.py --name nethid --pca 32
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import MODELS, train_cutoffs
from datasets import feature_names, get_dataset
from metrics import report, rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "stride", "halflife",
          "val_cutoff", "train_cutoffs", "note"]


def load_hidden(name: str, cut: dt.date, users: np.ndarray) -> np.ndarray:
    """Скрытое состояние, выровненное по порядку `users` выборки признаков."""
    path = MODELS / f"{name}_{cut}.npz"
    if not path.exists():
        raise SystemExit(f"нет {path.name} — сначала: python -u src/seq_oof_net.py "
                         f"--hidden --seeds 42 --name {name}")
    d = np.load(path)
    order = np.argsort(d["user_id"])
    su, sz = d["user_id"][order], d["hidden"][order]
    pos = np.clip(np.searchsorted(su, users), 0, len(su) - 1)
    ok = su[pos] == users
    out = np.full((len(users), sz.shape[1]), np.nan, dtype=np.float32)
    out[ok] = sz[pos[ok]].astype(np.float32)
    if (~ok).sum():
        print(f"  {path.name}: нет состояния для {(~ok).sum():,} пользователей — пропуски")
    return out


def fit_pca(mats: list[np.ndarray], k: int) -> np.ndarray:
    """Оси главных компонент, общие для всех срезов.

    Считать PCA отдельно на каждом срезе нельзя: оси получились бы разные, и
    колонка `pc_3` означала бы на разных срезах разное — ровно та ошибка, из-за
    которой у нас уже проваливались признаки, зависящие от среза.
    """
    x = np.vstack([m[~np.isnan(m).any(axis=1)] for m in mats])
    x = x - x.mean(axis=0)
    # SVD по подвыборке: полная матрица тут не нужна, оси устойчивы и так.
    idx = np.random.default_rng(0).choice(len(x), min(100_000, len(x)), replace=False)
    _, _, vt = np.linalg.svd(x[idx], full_matrices=False)
    return vt[:k].T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nethid", help="префикс .npz из seq_oof_net --hidden")
    ap.add_argument("--val-cutoffs", default="2026-01-15,2025-12-16")
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--pca", type=int, default=0,
                    help="сжать состояние до k компонент (0 — подавать все оси)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    results = []
    for raw in args.val_cutoffs.split(","):
        val_cut = dt.date.fromisoformat(raw.strip())
        n = args.cutoffs if val_cut == train_cutoffs(1)[0] else args.cutoffs + 1
        print(f"\n=== валидация {val_cut} ===")
        train, val, feats, cuts = load_split(n, val_cutoff=val_cut)
        Xtr, ytr = to_xy(train, feats)
        Xva, yva = to_xy(val, feats)

        # Скрытые состояния подаются в том же порядке строк, что и признаки.
        ztr = np.vstack([load_hidden(args.name, c, get_dataset(c)["user_id"].to_numpy())
                         for c in cuts[1:]])
        zva = load_hidden(args.name, val_cut, val["user_id"].to_numpy())
        if args.pca:
            axes = fit_pca([ztr], args.pca)
            ztr, zva = ztr @ axes, zva @ axes
            print(f"  сжато до {args.pca} компонент")
        assert len(ztr) == len(Xtr) and len(zva) == len(Xva), "строки разошлись"

        base, _ = fit_eval(Xtr, ytr, Xva, yva, feats, args.rounds, "без состояния")
        znames = [f"nz_{i}" for i in range(ztr.shape[1])]
        with_z, it = fit_eval(np.hstack([Xtr, ztr]), ytr, np.hstack([Xva, zva]), yva,
                              feats + znames, args.rounds, "со состоянием")
        gain = base - with_z
        print(f"  ВЫИГРЫШ СОСТОЯНИЯ: {gain:+.5f}")
        results.append((val_cut, base, with_z, gain))

        append_csv(MODELS / "experiments.csv", FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": f"hidden{ztr.shape[1]}", "blocks": "all+hidden",
            "name": f"stackhid_{val_cut:%m%d}", "model": "lgbm", "cutoffs": len(cuts) - 1,
            "n_features": len(feats) + ztr.shape[1], "rmsle_single": round(with_z, 5),
            "rmsle_two_stage": "", "rmsle_blend": round(with_z, 5), "blend_w": "",
            "gini_blend": "", "sum_bias_blend": "", "best_iter_single": it,
            "stride": 30, "halflife": "", "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in cuts[1:]),
            "note": (args.note + "; " if args.note else "")
                    + f"стекинг на скрытом состоянии ({ztr.shape[1]} осей), "
                      f"без него {base:.5f}"})

    print("\n=== итог ===")
    ok = True
    for val_cut, base, with_z, gain in results:
        ok &= gain > 0
        print(f"{val_cut}: без {base:.5f} | со состоянием {with_z:.5f} | {gain:+.5f}")
    print("\nвердикт: " + ("выигрыш на всех срезах — можно принимать" if ok else
                           "знак не совпал — по правилу раздела 2 PLAN.md не принимается"))


def fit_eval(Xtr, ytr, Xva, yva, feats, rounds, tag):
    m = GBM("lgbm", "reg", "cpu", n_estimators=rounds, early_stopping=200, log_period=0)
    m.fit(Xtr, np.log1p(ytr), Xva, np.log1p(yva), feature_names=feats)
    p = m.predict(Xva)
    report(yva, np.expm1(np.clip(p, 0, None)), tag)
    return rmse_log(np.log1p(yva), p), m.best_iter


if __name__ == "__main__":
    main()
