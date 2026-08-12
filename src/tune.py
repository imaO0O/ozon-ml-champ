"""Случайный перебор гиперпараметров LightGBM на штатном протоколе валидации.

Тюнится одноголовая регрессия log1p(target): её метрика — это и есть RMSLE
соревнования, поэтому вариант стоит втрое дешевле полного прогона train.py
(нет классификатора и регрессора на покупателях). Выборка грузится один раз.

Каждая попытка дописывается в models/tuning.csv, так что серию можно прервать
и продолжить, а результаты сравнимы между машинами команды.

    python src/tune.py --trials 24
    python src/tune.py --trials 10 --val-cutoff 2025-12-16   # проверка победителя
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np

from config import MODELS, SEED
from datasets import parse_blocks
from metrics import rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

# broad — первая серия. Строилась от гипотезы «ранняя остановка = деревья
# слишком сложные», поэтому включала простые варианты. Гипотеза не подтвердилась:
# все варианты с 255 листьями заняли первые места, все с 31 — последние,
# 63 и 127 ровно посередине, без единого исключения.
#
# deep — вторая серия вокруг найденного угла: число листьев вынесено за прежнюю
# границу (оптимум лежал на краю сетки), а сдерживать переобучение при таких
# деревьях должен размер листа и доля признаков.
SPACES = {
    "broad": {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [31, 63, 127, 255],
        "min_data_in_leaf": [100, 200, 500, 1000, 2000],
        "feature_fraction": [0.5, 0.65, 0.8, 1.0],
        "bagging_fraction": [0.7, 0.85, 1.0],
        "lambda_l2": [1.0, 5.0, 20.0, 100.0],
        "max_bin": [127, 255],
    },
    "deep": {
        "learning_rate": [0.02, 0.03, 0.05],
        "num_leaves": [255, 511, 1023],
        "min_data_in_leaf": [200, 500, 1000, 2000],
        "feature_fraction": [0.4, 0.5, 0.65],
        "bagging_fraction": [0.7, 0.85, 1.0],
        "lambda_l2": [1.0, 5.0, 20.0],
        "max_bin": [127, 255],
    },
}

FIELDS = ["created", "commit", "space", "val_cutoff", "rmsle", "best_iter", "seconds",
          *SPACES["broad"].keys(), "note"]


def sample_params(rng: np.random.Generator, space: dict) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def parse_params(text: str) -> dict:
    """JSON или пары ключ=значение через запятую/пробел.

    Второй формат нужен потому, что PowerShell срезает двойные кавычки при
    передаче аргумента в python, и JSON до скрипта не доезжает. Заодно строку
    в этом формате можно копировать прямо из вывода перебора.
    """
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    out: dict = {}
    for part in text.replace(",", " ").split():
        k, _, v = part.partition("=")
        if not v:
            raise SystemExit(f"не разобрать параметр {part!r}: нужно вида num_leaves=511")
        out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000, help="потолок итераций; решает ранняя остановка")
    ap.add_argument("--early-stopping", type=int, default=300)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--history", type=int, default=None,
                    help="обрезать историю до K дней на всех срезах")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--space", default="broad", choices=sorted(SPACES),
                    help="broad — первая, широкая сетка; deep — вокруг найденного угла")
    ap.add_argument("--params", default=None,
                    help="конкретная конфигурация вместо перебора: пары вида "
                         "num_leaves=511,learning_rate=0.02 или JSON")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    space = SPACES[args.space]
    # Победителя перебора нужно уметь прогнать на срезе, которого перебор не видел:
    # иначе не отличить настоящий выигрыш от подгонки под валидацию.
    fixed = parse_params(args.params) if args.params else None
    if fixed:
        args.trials = 1

    val_cutoff = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else None
    train, val, feats, cuts = load_split(args.cutoffs, blocks=parse_blocks(args.blocks),
                                         val_cutoff=val_cutoff, history=args.history)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    del train, ytr
    print(f"перебор: {args.trials} попыток | сетка {args.space} | "
          f"{Xtr.shape[0]:,} x {len(feats)} | валидация {cuts[0]}\n")

    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(1, args.trials + 1):
        params = dict(fixed) if fixed else sample_params(rng, space)
        t0 = time.time()
        m = GBM("lgbm", "reg", "cpu", n_estimators=args.rounds,
                early_stopping=args.early_stopping, params=params, log_period=0)
        m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
        score = rmse_log(yva_log, m.predict(Xva))
        secs = time.time() - t0
        results.append((score, m.best_iter, params))
        print(f"[{i:>2}/{args.trials}] RMSLE {score:.5f} | итераций {m.best_iter:>5} | {secs:>5.0f}s | "
              + " ".join(f"{k}={v}" for k, v in params.items()))

        append_csv(MODELS / "tuning.csv", FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
            "space": args.space, "val_cutoff": str(cuts[0]),
            "rmsle": round(score, 5), "best_iter": m.best_iter,
            "seconds": round(secs), **params, "note": args.note,
        })

    print("\n--- лучшие пять ---")
    for score, best_iter, params in sorted(results, key=lambda r: r[0])[:5]:
        print(f"RMSLE {score:.5f} | итераций {best_iter:>5} | "
              + " ".join(f"{k}={v}" for k, v in params.items()))


if __name__ == "__main__":
    main()
