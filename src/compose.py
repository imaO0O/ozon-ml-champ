"""Метрики совместного состава по предсказаниям двух треков — без обмена кодом.

Зачем. Половины решения живут на разных машинах: бустинг на ноутбуке, сети
на видеокарте. Чтобы узнать Gini и ошибку суммы у их взвешенной смеси, не нужно
переносить ни веса, ни код — достаточно предсказаний на одном и том же срезе.

Оба файла — npz с полями user_id, pred_log; таргет берётся из того, где он есть.
Уровень каждой половины выравнивается по истине перед смешиванием: иначе
разница уровней подмешается в состав постоянным сдвигом, и сравнивать формы
будет нельзя.

    python -u src/compose.py --a models/ens_for_cand4_valpred_2026-01-15.npz \
                             --b models/cand4_netpart_2026-01-15.npz --wb 0.542
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import console  # noqa: F401  (печать в консоли cp1251 — разбор в модуле)
from metrics import gini_norm, rmse_log, sum_bias


def load(path: Path) -> dict:
    with np.load(path) as z:
        out = {"user_id": z["user_id"].astype(np.int64),
               "pred_log": z["pred_log"].astype(np.float64)}
        if "target" in z:
            out["target"] = z["target"].astype(np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="npz первой половины")
    ap.add_argument("--b", required=True, help="npz второй половины")
    ap.add_argument("--wb", type=float, required=True, help="вес второй половины в составе")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = load(Path(args.a)), load(Path(args.b))
    tgt = a.get("target", b.get("target"))
    if tgt is None:
        raise SystemExit("ни в одном файле нет поля target — метрику посчитать не из чего")

    src = a if "target" in a else b
    frame = (pl.DataFrame({"user_id": src["user_id"], "target": tgt})
             .join(pl.DataFrame({"user_id": a["user_id"], "pa": a["pred_log"]}), on="user_id")
             .join(pl.DataFrame({"user_id": b["user_id"], "pb": b["pred_log"]}), on="user_id"))
    if frame.height != len(src["user_id"]):
        print(f"внимание: совпало {frame.height:,} из {len(src['user_id']):,} клиентов")

    y = np.log1p(frame["target"].to_numpy())
    y_raw = frame["target"].to_numpy()
    pa, pb = frame["pa"].to_numpy(), frame["pb"].to_numpy()

    # Выравнивание уровня: сравниваем формы предсказаний, а не сдвиги.
    pa = pa - pa.mean() + y.mean()
    pb = pb - pb.mean() + y.mean()
    mix = (1 - args.wb) * pa + args.wb * pb

    print(f"клиентов: {len(y):,} | вес {args.label_b} = {args.wb:.3f}, "
          f"{args.label_a} = {1 - args.wb:.3f}")
    print(f"корреляция половин: {np.corrcoef(pa, pb)[0, 1]:.4f}\n")
    print(f"{'модель':<24} {'RMSLE':>9} {'Gini':>8} {'ошибка суммы':>14}")
    for label, p in ((args.label_a, pa), (args.label_b, pb), ("состав", mix)):
        pred = np.expm1(np.clip(p, 0, None))
        print(f"{label:<24} {rmse_log(y, p):>9.5f} {gini_norm(y_raw, pred):>8.4f} "
              f"{sum_bias(y_raw, pred):>13.1%}")

    grid = [(w, rmse_log(y, (1 - w) * pa + w * pb)) for w in np.linspace(0, 1, 51)]
    best_w, best_r = min(grid, key=lambda t: t[1])
    print(f"\nлучший вес {args.label_b} по RMSLE: {best_w:.2f} -> {best_r:.5f} "
          f"(в составе {args.wb:.3f})")


if __name__ == "__main__":
    main()
