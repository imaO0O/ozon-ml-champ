"""Ансамбль разнородных LightGBM с проверкой сразу на двух валидационных срезах.

Зачем ансамбль. Перебор гиперпараметров дал три конфигурации, различающиеся
на 0.00007 — выбрать из них одну нельзя (проверено: выбор не переносится на
другой срез), но усреднить их предсказания можно, и разнородность как раз
и делает усреднение полезным.

Зачем два среза. Расхождение оценок между 2026-01-15 и 2025-12-16 — порядка
0.001, поэтому эффект меньше 0.002, измеренный на одном срезе, ничего не значит.
Ансамбль принимаем, только если он выигрывает на обоих.

Усреднение идёт в log1p-шкале: там живёт метрика, и там же складываются
предсказания моделей.

    python src/ensemble.py --val-cutoffs 2026-01-15,2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, SEED
from datasets import parse_blocks
from metrics import report, rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

# Участники: умолчание с двумя сидами плюс три верхние конфигурации перебора.
# Они статистически неразличимы между собой, но устроены по-разному — 511 и 1023
# листа, lr от 0.02 до 0.05, разная доля признаков. Именно это и нужно ансамблю.
MEMBERS = [
    ("default_s42", dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                         feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                         max_bin=255, seed=SEED)),
    ("default_s7", dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                        feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0,
                        max_bin=255, seed=7)),
    ("deep511_lr02", dict(learning_rate=0.02, num_leaves=511, min_data_in_leaf=500,
                          feature_fraction=0.65, bagging_fraction=0.7, lambda_l2=20.0,
                          max_bin=127, seed=SEED)),
    ("deep1023_ff04", dict(learning_rate=0.02, num_leaves=1023, min_data_in_leaf=200,
                           feature_fraction=0.4, bagging_fraction=0.7, lambda_l2=20.0,
                           max_bin=255, seed=SEED)),
    ("deep511_lr05", dict(learning_rate=0.05, num_leaves=511, min_data_in_leaf=1000,
                          feature_fraction=0.65, bagging_fraction=0.85, lambda_l2=20.0,
                          max_bin=255, seed=SEED)),
]


def run_cutoff(val_cutoff: dt.date | None, n_cutoffs: int, blocks, rounds: int,
               early_stopping: int) -> dict:
    train, val, feats, cuts = load_split(n_cutoffs, blocks=blocks, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    del train, ytr

    preds, scores = [], {}
    for name, params in MEMBERS:
        m = GBM("lgbm", "reg", "cpu", n_estimators=rounds,
                early_stopping=early_stopping, params=params, log_period=0)
        m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
        p = m.predict(Xva)
        preds.append(p)
        scores[name] = rmse_log(yva_log, p)
        print(f"  {name:<16} RMSLE {scores[name]:.5f} | итераций {m.best_iter}")

    ens = np.mean(preds, axis=0)
    out = {
        "val_cutoff": str(cuts[0]),
        "members": scores,
        "best_member": min(scores.values()),
        "ensemble": rmse_log(yva_log, ens),
    }
    print(f"  {'среднее':<16} RMSLE {np.mean(list(scores.values())):.5f}")
    print(f"  {'АНСАМБЛЬ':<16} RMSLE {out['ensemble']:.5f} "
          f"(лучший участник {out['best_member']:.5f})")
    report(yva, np.expm1(ens), "ансамбль")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cutoffs", default="2026-01-15,2025-12-16",
                    help="через запятую; ансамбль принимаем только при выигрыше на всех")
    ap.add_argument("--cutoffs", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    results = []
    for raw in args.val_cutoffs.split(","):
        cut = dt.date.fromisoformat(raw.strip())
        print(f"\n=== валидация {cut} ===")
        results.append(run_cutoff(cut, args.cutoffs, parse_blocks(args.blocks),
                                  args.rounds, args.early_stopping))

    print("\n=== итог ===")
    verdict = True
    for r in results:
        gain = r["best_member"] - r["ensemble"]
        verdict &= gain > 0
        print(f"{r['val_cutoff']}: ансамбль {r['ensemble']:.5f} против лучшего "
              f"участника {r['best_member']:.5f} — выигрыш {gain:+.5f}")
        append_csv(MODELS / "ensemble.csv",
                   ["created", "commit", "val_cutoff", "n_members", "ensemble",
                    "best_member", "mean_member", "gain", "note"],
                   {"created": dt.datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit(), "val_cutoff": r["val_cutoff"],
                    "n_members": len(MEMBERS), "ensemble": round(r["ensemble"], 5),
                    "best_member": round(r["best_member"], 5),
                    "mean_member": round(float(np.mean(list(r["members"].values()))), 5),
                    "gain": round(gain, 5), "note": args.note})
    print("\nвердикт: " + ("ансамбль выигрывает на всех срезах — можно принимать"
                          if verdict else
                          "выигрыш не на всех срезах — принимать нельзя"))


if __name__ == "__main__":
    main()
