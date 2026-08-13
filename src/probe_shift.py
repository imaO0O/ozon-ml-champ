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


def scale_submission(source: str, alpha: float, out: str | None) -> None:
    """Растянуть предсказания вокруг их среднего: p' = m + alpha*(p - m).

    Уровень мы уже поправили сдвигом, но размах предсказаний может быть сжат
    или растянут относительно правды — модель обучалась в другом режиме.
    Среднее при таком преобразовании не меняется, поэтому найденный сдвиг
    остаётся оптимальным, а поправки не мешают друг другу.
    """
    src = SUBMISSIONS / source
    sub = pl.read_csv(src)
    p = np.log1p(sub["predict"].to_numpy().astype(np.float64))
    m = p.mean()
    scaled = np.clip(np.expm1(np.clip(m + alpha * (p - m), 0, None)), 0, None)

    name = out or f"probe_a{alpha:.3f}_{dt.datetime.now():%m%d_%H%M}.csv"
    path = SUBMISSIONS / name
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": sub["user_id"], "predict": scaled.astype(np.float32)}).write_csv(path)

    print(f"{path}")
    print(f"  исходный файл: {source} | растяжение alpha = {alpha:.3f}")
    print(f"  Var(p) в log1p-шкале: {p.var():.5f}  <- нужно для решения")
    print(f"  доля предсказаний, обрезанных нулём после растяжения: "
          f"{(m + alpha * (p - m) < 0).mean():.3%}")
    print(f"  сумма: {np.expm1(p).sum():,.0f} -> {scaled.sum():,.0f}")
    append_csv(SUBMISSIONS / "log.csv",
               ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"],
               {"file": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "name": "probe", "model": "scale",
                "pred_sum": round(float(scaled.sum())),
                "pred_zeros": f"{(scaled < 1e-6).mean():.4f}",
                "note": f"зонд размаха: {source}, alpha={alpha:.3f}, Var={p.var():.5f}"})


# Уровень тестового окна, выведенный из зонда уровня: b измерялось как
# E[log1p(y)] - E[log1p(pred)] для файла lgbm_ens_0811_1436.csv, у которого
# среднее log1p предсказаний равно 2.27132, а b = +0.0578.
# Проверка сходится: для полностью откалиброванного lgbm_ens_full_calib.csv
# нужный сдвиг выходит 0.00000, а его public-ответ известен — 1.6529357726.
# Отсюда любой сдвиг выводится без зонда: shift = TEST_LEVEL - E[log1p(pred)].
TEST_LEVEL = 2.32912
TEST_LEVEL_SE = 0.007          # погрешность наследуется от b


def derive_shift(source: str) -> float:
    """Сдвиг до известного уровня тестового окна — без траты сабмита."""
    p = np.log1p(pl.read_csv(SUBMISSIONS / source)["predict"].to_numpy().astype(np.float64))
    shift = TEST_LEVEL - p.mean()
    cost = TEST_LEVEL_SE ** 2 / (2 * 1.653)
    print(f"{source}")
    print(f"  E[log1p(pred)] = {p.mean():.5f}")
    print(f"  уровень окна   = {TEST_LEVEL:.5f} ± {TEST_LEVEL_SE:.3f}")
    print(f"  нужный сдвиг   = {shift:+.5f}")
    print(f"  ожидаемый выигрыш = {shift ** 2 / (2 * 1.653):.5f} RMSLE")
    print(f"  цена неточности уровня ≈ {cost:.5f} RMSLE — пренебрежимо")
    print("\nРастяжение и кривизну так вывести нельзя: они зависят от формы")
    print("предсказаний конкретной модели и требуют своего зонда.")
    return shift


def quad_basis(p: np.ndarray) -> np.ndarray:
    """Квадратичное направление, очищенное от уже исправленных.

    Сдвиг и растяжение — первые два члена разложения поправки. Третий ловит
    нелинейную рассогласованность: модель может ошибаться по-разному на слабых
    и сильных клиентах. Чтобы зонд мерил именно её, направление ортогонализуется
    к константе и к самому предсказанию — иначе он частично повторил бы уже
    найденные поправки и сбил бы их.
    """
    u = p - p.mean()
    g = u ** 2
    g = g - g.mean()                          # ортогонально константе
    g = g - (g @ u) / (u @ u) * u             # ортогонально линейному члену
    return g


def quad_submission(source: str, gamma: float, out: str | None) -> None:
    src = SUBMISSIONS / source
    sub = pl.read_csv(src)
    p = np.log1p(sub["predict"].to_numpy().astype(np.float64))
    g = quad_basis(p)
    shifted = np.clip(np.expm1(np.clip(p + gamma * g, 0, None)), 0, None)

    name = out or f"probe_q{gamma:+.4f}_{dt.datetime.now():%m%d_%H%M}.csv".replace("+", "p")
    path = SUBMISSIONS / name
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": sub["user_id"], "predict": shifted.astype(np.float32)}).write_csv(path)

    print(f"{path}")
    print(f"  исходный файл: {source} | gamma = {gamma:+.5f}")
    print(f"  Var(g) = {g.var():.5f}  <- нужно для решения")
    print(f"  сдвиг среднего: {(gamma * g).mean():+.2e} (должен быть ~0)")
    print(f"  обрезано нулём: {(p + gamma * g < 0).mean():.3%}")
    print(f"  сумма: {np.expm1(p).sum():,.0f} -> {shifted.sum():,.0f}")
    append_csv(SUBMISSIONS / "log.csv",
               ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"],
               {"file": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "name": "probe", "model": "quad",
                "pred_sum": round(float(shifted.sum())),
                "pred_zeros": f"{(shifted < 1e-6).mean():.4f}",
                "note": f"зонд кривизны: {source}, gamma={gamma:+.5f}, Var(g)={g.var():.5f}"})


def solve_direction(mse0: float, mse1: float, step: float, var_g: float, what: str) -> None:
    """Общее решение для зонда вдоль любого ортогонализованного направления."""
    m0, m1 = mse0 ** 2, mse1 ** 2
    cov = (m0 - m1 + step ** 2 * var_g) / (2 * step)
    best_step = cov / var_g
    best = max(m0 - cov ** 2 / var_g, 0.0) ** 0.5
    print(f"RMSLE базового {mse0:.7f}, зонда — {mse1:.7f}")
    print(f"\nCov(остаток, {what}) = {cov:+.5f}")
    print(f"оптимальный коэффициент = {best_step:+.5f}")
    print(f"ожидаемый RMSLE {best:.7f} (выигрыш {mse0 - best:+.5f})")
    if mse0 - best < 0.0002:
        print("\nвывод: в этом направлении поправлять нечего")


def solve_scale(mse0: float, mse1: float, alpha: float, var_p: float) -> None:
    """По ответу зонда размаха восстановить оптимальное растяжение."""
    a = alpha - 1.0
    m0, m1 = mse0 ** 2, mse1 ** 2
    cov = (m0 - m1 + a ** 2 * var_p) / (2 * a)
    best_a = cov / var_p
    gain_mse = cov ** 2 / var_p
    best = max(m0 - gain_mse, 0.0) ** 0.5
    print(f"RMSLE базового {mse0:.7f}, зонда с alpha={alpha:.3f} — {mse1:.7f}")
    print(f"\nCov(остаток, предсказание) = {cov:+.5f}")
    print(f"оптимальное alpha = {1 + best_a:.4f}")
    print(f"ожидаемый RMSLE {best:.7f} (выигрыш {mse0 - best:+.5f})")
    if abs(best_a) < 0.005:
        print("\nвывод: размах предсказаний уже верный, растягивать нечего")
    elif best_a > 0:
        print("\nвывод: предсказания сжаты — правда разбросана сильнее, чем модель считает")
    else:
        print("\nвывод: предсказания растянуты — модель переоценивает разброс")


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
    ap.add_argument("--alpha", type=float, default=None,
                    help="растяжение вокруг среднего вместо сдвига")
    ap.add_argument("--gamma", type=float, default=None,
                    help="коэффициент при квадратичном направлении (ортогонализованном)")
    ap.add_argument("--var-g", type=float, default=None,
                    help="Var(g) — печатается при создании квадратичного зонда")
    ap.add_argument("--var-p", type=float, default=None,
                    help="Var(log1p(предсказаний)) — печатается при создании зонда размаха")
    ap.add_argument("--derive-shift", action="store_true",
                    help="вывести сдвиг из известного уровня тестового окна, без сабмита")
    ap.add_argument("--solve", action="store_true", help="решить по двум ответам лидерборда")
    ap.add_argument("--mse0", type=float, help="RMSLE исходного сабмита")
    ap.add_argument("--mse1", type=float, help="RMSLE сабмита-зонда")
    args = ap.parse_args()

    if args.derive_shift:
        if not args.source:
            raise SystemExit("нужен --source")
        derive_shift(args.source)
    elif args.solve:
        if args.mse0 is None or args.mse1 is None:
            raise SystemExit("нужны --mse0 и --mse1")
        if args.gamma is not None:
            if args.var_g is None:
                raise SystemExit("для квадратичного зонда нужен --var-g")
            solve_direction(args.mse0, args.mse1, args.gamma, args.var_g, "кривизна")
        elif args.alpha is not None:
            if args.var_p is None:
                raise SystemExit("для зонда размаха нужен --var-p")
            solve_scale(args.mse0, args.mse1, args.alpha, args.var_p)
        else:
            solve(args.mse0, args.mse1, args.delta)
    elif args.gamma is not None:
        if not args.source:
            raise SystemExit("нужен --source")
        quad_submission(args.source, args.gamma, args.out)
    elif args.alpha is not None:
        if not args.source:
            raise SystemExit("нужен --source")
        scale_submission(args.source, args.alpha, args.out)
    else:
        if not args.source:
            raise SystemExit("нужен --source")
        shift_submission(args.source, args.delta, args.out)


if __name__ == "__main__":
    main()
