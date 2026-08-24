"""Сетка гиперпараметров бустинга по ВЫРОВНЕННОЙ метрике — перебор заново.

Зачем понадобился заново. Прежний перебор 71 конфигурации судился по сырой
RMSLE, в которую входит промах по уровню, а на сабмите уровень правится
бесплатно. Из 71 прогона 65 были испорчены, и вывод «гиперпараметры не
переносятся» опирался на них.

Что это меняет на числах. Разница между скоростью 0.03 и 0.015 составляет
0.00015 по сырой метрике и 0.00039 по выровненной — втрое. Величина такого
порядка тонет в сыром сравнении, и именно поэтому перебор её не нашёл.

Почему одним модулем, а не девятью вызовами train.py. Выборка на пять срезов
это 1.2 млн строк на 242 признака, её подъём стоит дороже иной подгонки.
Здесь она поднимается один раз, а конфигурации перебираются в цикле.

Уровень выравнивается перед метрикой — тем же приёмом, что везде: предсказание
сдвигается так, чтобы его среднее совпало со средним истины. Это ровно то, что
делает бесплатный сдвиг на сабмите.

    python -u src/lgb_grid.py
    python -u src/lgb_grid.py --val-cutoff 2025-12-16 --cutoffs 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import time

import numpy as np

from config import MODELS
from metrics import gini_norm, rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

# Скорость обучения и пара «листья / минимум в листе». Умолчание рабочей
# модели — 0.03 и (127, 200), оно входит в сетку как точка отсчёта.
RATES = [0.008, 0.015, 0.03]
SHAPES = [(127, 200), (255, 500), (511, 1000)]

# Пространство случайного поиска. Первые три оси мы уже прошли сеткой и знаем,
# что там лежит 0.0008. Остальные четыре не трогали НИ РАЗУ — ни мы, ни
# недействительный перебор трека A, — а именно регуляризация и подвыборки
# обычно и дают на такой размерности основную часть выигрыша.
SPACE = {
    "learning_rate": ("logu", 0.004, 0.04),
    "num_leaves": ("int", 63, 1023),
    "min_data_in_leaf": ("int", 50, 2000),
    "feature_fraction": ("u", 0.4, 1.0),
    "bagging_fraction": ("u", 0.5, 1.0),
    "lambda_l2": ("logu", 0.1, 100.0),
    "min_gain_to_split": ("u", 0.0, 0.5),
}

FIELDS = ["created", "commit", "val_cutoff", "lr", "leaves", "min_data",
          "feature_fraction", "bagging_fraction", "lambda_l2", "min_gain",
          "iters", "rmsle_raw", "rmsle_aligned", "gini", "seconds"]


def sample_config(rng) -> dict:
    """Одна точка пространства. Логравномерно там, где важен порядок величины."""
    out = {}
    for name, (kind, lo, hi) in SPACE.items():
        if kind == "logu":
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "int":
            out[name] = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[name] = float(rng.uniform(lo, hi))
    # bagging_freq обязателен, иначе LightGBM молча игнорирует bagging_fraction.
    out["bagging_freq"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=400,
                    help="при малой скорости каждая итерация двигает модель меньше, "
                         "и прежний запас 200 обрывал бы прогон рано")
    ap.add_argument("--random", type=int, default=0,
                    help="случайный поиск: сколько точек взять вместо сетки 3x3")
    ap.add_argument("--seed", type=int, default=0, help="сид случайного поиска")
    args = ap.parse_args()

    val_cutoff = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else None
    train, val, feats, cuts = load_split(args.cutoffs, val_cutoff=val_cutoff)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    print(f"обучение {len(ytr):,} строк | валидация {len(yva):,} | признаков {len(feats)}")


    if args.random:
        rng = np.random.default_rng(args.seed)
        configs = [sample_config(rng) for _ in range(args.random)]
    else:
        configs = [{"learning_rate": lr, "num_leaves": lv, "min_data_in_leaf": md}
                   for lr, (lv, md) in itertools.product(RATES, SHAPES)]
    kind = f"случайный поиск, сид {args.seed}" if args.random else "сетка 3x3"
    print(f"срез валидации {cuts[0]} | {kind} | конфигураций {len(configs)}")
    print()

    rows = []
    best_so_far = float("inf")
    for i, cfg in enumerate(configs, 1):
        lr = cfg["learning_rate"]
        leaves, min_data = cfg["num_leaves"], cfg["min_data_in_leaf"]
        t0 = time.time()
        m = GBM(kind="lgbm", task="reg", device="cpu", n_estimators=args.rounds,
                early_stopping=args.early_stopping, log_period=0, params=cfg)
        m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
        p = m.predict(Xva)
        raw = rmse_log(yva_log, p)
        pa = p - p.mean() + yva_log.mean()
        ali = rmse_log(yva_log, pa)
        gini = gini_norm(yva, np.expm1(np.clip(pa, 0, None)))
        secs = time.time() - t0
        rows.append((lr, leaves, min_data, m.best_iter, raw, ali, gini, secs))
        mark = ""
        if ali < best_so_far:
            best_so_far, mark = ali, "  <- лучшая"
        print(f"[{i}/{len(configs)}] lr {lr:.4f} листья {leaves:<4} лист>={min_data:<5} "
              f"ff {cfg.get('feature_fraction', 1):.2f} bag {cfg.get('bagging_fraction', 1):.2f} "
              f"l2 {cfg.get('lambda_l2', 0):.2f} | итер {m.best_iter:<5} "
              f"выровн. {ali:.5f} Gini {gini:.4f} | {secs:.0f}s{mark}", flush=True)
        append_csv(MODELS / "lgb_grid.csv", FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "val_cutoff": str(cuts[0]),
            "lr": round(lr, 5), "leaves": leaves, "min_data": min_data,
            "feature_fraction": round(cfg.get("feature_fraction", 1.0), 3),
            "bagging_fraction": round(cfg.get("bagging_fraction", 1.0), 3),
            "lambda_l2": round(cfg.get("lambda_l2", 0.0), 3),
            "min_gain": round(cfg.get("min_gain_to_split", 0.0), 3),
            "iters": m.best_iter, "rmsle_raw": round(raw, 5),
            "rmsle_aligned": round(ali, 5), "gini": round(gini, 4),
            "seconds": round(secs)})

    print(f"\n{'по выровненной метрике':<24}{'lr':>7}{'листья':>8}{'лист>=':>8}"
          f"{'итер':>7}{'выровн.':>10}{'сырой':>10}{'Gini':>8}")
    for r in sorted(rows, key=lambda x: x[5])[:5]:
        lr, lv, md, it, raw, ali, g, _ = r
        print(f"{'':<24}{lr:>7}{lv:>8}{md:>8}{it:>7}{ali:>10.5f}{raw:>10.5f}{g:>8.4f}")

    best_a = min(rows, key=lambda x: x[5])
    best_r = min(rows, key=lambda x: x[4])
    print(f"\nлучшая по выровненной: lr {best_a[0]}, листья {best_a[1]} -> {best_a[5]:.5f}")
    print(f"лучшая по сырой:       lr {best_r[0]}, листья {best_r[1]} -> {best_r[5]:.5f}")
    if best_a[:3] != best_r[:3]:
        print("Конфигурации РАЗНЫЕ — сырая метрика выбрала бы не ту. Это и есть "
              "причина, по которой прежний перебор ничего не нашёл.")


if __name__ == "__main__":
    main()
