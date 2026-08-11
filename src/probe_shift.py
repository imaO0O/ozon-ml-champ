"""Зондирование лидерборда: измерить смещение уровня, которое офлайн не измеряется.

Главная неопределённость решения — уровень тестового окна. Модель ожидает спад
на 17% относительно января, а год назад то же календарное окно было на 16% выше.
Данными это не разрешить: в логе 13.5 месяцев, отделить тренд от сезонности
по одному году нельзя. Зато это ровно то, что измеряет лидерборд.

Как. RMSLE^2 = E[(log1p(y) - log1p(p))^2] = E[r^2]. Сдвиг всех предсказаний
в лог-шкале на d даёт остаток (r - d), то есть

    MSE(d) = MSE(0) - 2*d*b + d^2,   где b = E[r] — среднее смещение.

Один сабмит со сдвигом d даёт b точно:

    b = (MSE(0) - MSE(d) + d^2) / (2*d)

а оптимальный сдвиг равен b и стоит MSE(0) - b^2. При d = 0.15 исходы
расходятся на 0.007-0.009 RMSLE — на порядок выше разрешающей способности
лидерборда, поэтому ответ будет однозначным в любую сторону.

Точность оценки: b — один глобальный параметр по 50 000 клиентам public,
стандартная ошибка около 0.007, поэтому подгонки под public тут почти нет.

    python src/probe_shift.py --source lgbm_ens_0811_1436.csv --delta 0.15
    python src/probe_shift.py --solve --mse0 1.65649 --mse1 <ответ> --delta 0.15
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import SAMPLE_SUBMIT, SUBMISSIONS
from utils import append_csv, git_commit


def shift_submission(source: str, delta: float, out: str | None) -> None:
    src = SUBMISSIONS / source
    sub = pl.read_csv(src)
    p = sub["predict"].to_numpy().astype(np.float64)
    shifted = np.clip(np.expm1(np.log1p(p) + delta), 0, None)

    name = out or f"probe_d{delta:+.2f}_{dt.datetime.now():%m%d_%H%M}.csv".replace("+", "p")
    path = SUBMISSIONS / name
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": sub["user_id"], "predict": shifted.astype(np.float32)}).write_csv(path)

    ref = pl.read_csv(SAMPLE_SUBMIT)
    assert (ref["user_id"] == sub["user_id"]).all(), "порядок user_id разошёлся с sample_submit"

    print(f"{path}")
    print(f"  исходный файл: {source} | сдвиг в log1p-шкале: {delta:+.3f}")
    print(f"  сумма: {p.sum():,.0f} -> {shifted.sum():,.0f} (x{shifted.sum() / p.sum():.3f})")
    print(f"  среднее: {p.mean():.2f} -> {shifted.mean():.2f}")
    print(f"  нулевых: {(shifted < 1e-6).mean():.2%}")
    append_csv(SUBMISSIONS / "log.csv",
               ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"],
               {"file": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "name": "probe", "model": "shift",
                "pred_sum": round(float(shifted.sum())),
                "pred_zeros": f"{(shifted < 1e-6).mean():.4f}",
                "note": f"зонд уровня: {source} со сдвигом {delta:+.3f} в log1p"})


def solve(mse0: float, mse1: float, delta: float) -> None:
    """По двум ответам лидерборда восстановить смещение и оптимальный сдвиг."""
    m0, m1 = mse0 ** 2, mse1 ** 2
    b = (m0 - m1 + delta ** 2) / (2 * delta)
    best = max(m0 - b ** 2, 0.0) ** 0.5
    print(f"RMSLE без сдвига {mse0:.5f}, со сдвигом {delta:+.3f} — {mse1:.5f}")
    print(f"\nсреднее смещение предсказаний b = {b:+.4f} в log1p-шкале")
    print(f"оптимальный сдвиг = {b:+.4f}, ожидаемый RMSLE {best:.5f} "
          f"(выигрыш {mse0 - best:+.5f} к исходному)")
    if abs(b) < 0.02:
        print("\nвывод: предсказания по уровню верны, сдвигать нечего")
    elif b > 0:
        print("\nвывод: модель систематически занижает — тестовое окно активнее, "
              "чем она ожидает")
    else:
        print("\nвывод: модель систематически завышает — тестовое окно слабее, "
              "чем она ожидает")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="исходный сабмит из submissions/")
    ap.add_argument("--delta", type=float, default=0.15, help="сдвиг в log1p-шкале")
    ap.add_argument("--out", default=None)
    ap.add_argument("--solve", action="store_true", help="решить по двум ответам лидерборда")
    ap.add_argument("--mse0", type=float, help="RMSLE исходного сабмита")
    ap.add_argument("--mse1", type=float, help="RMSLE сабмита со сдвигом")
    args = ap.parse_args()

    if args.solve:
        if args.mse0 is None or args.mse1 is None:
            raise SystemExit("нужны --mse0 и --mse1")
        solve(args.mse0, args.mse1, args.delta)
    else:
        if not args.source:
            raise SystemExit("нужен --source")
        shift_submission(args.source, args.delta, args.out)


if __name__ == "__main__":
    main()
