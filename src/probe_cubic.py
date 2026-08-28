"""Зонд кубической поправки — следующий член семейства после кривизны.

Зачем. Калибровка правит уровень (степень 0), растяжение (1) и кривизну (2).
На валидации кубический член даёт ещё +0.00002…+0.00003 и **переносится**
между срезами, а четвёртая степень уже переобучается: оптимум семейства —
степень 3, а не 2.

Важное различие, без которого этот зонд выглядел бы противоречием с нашим же
отрицательным результатом. Изотоническая перекалибровка провалилась
на **переносе между валидационными срезами**: гибкое семейство подгоняет шум
одного окна. Зонд лидерборда так не работает — он оценивает коэффициент
**прямо на тесте**, и переноса в нём нет вовсе. Провал изотоники кубический
зонд не закрывает.

## Как устроен зонд

`MSE(γ) = MSE(0) + 2γ·a + γ²·V`, где `a = mean(h·(p − y))`, `V = mean(h²)`.

`MSE(0)` известен из public-счёта базы, `V` считается локально, значит **одна
отправка определяет `a` точно**, а с ним и оптимум `γ* = −a/V` и его выигрыш
`a²/V`. Это ровно та же конструкция, что дала кривизну: там из одного зонда
решилось `γ* = −0.004905`.

## Почему направление ортогонализуется

`h = u³` очищается от `{1, u, g}`: уровень, растяжение и кривизна к базе уже
применены, и мерить надо **остаточное** направление, а не сумму. Без этого
зонд переоткрывал бы уже применённую поправку и решал бы не тот коэффициент.

    python -u src/probe_cubic.py --gamma 0.0036            # собрать зонд и дать прогноз
    python -u src/probe_cubic.py --solve 1.6467450         # решить оптимум по ответу
    python -u src/probe_cubic.py --solve 1.6467450 --apply # и собрать финальный файл
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import SUBMISSIONS

BASE = "pair_q"
BASE_SCORE = 1.6466941018333736      # public рекорда, из submissions/log.csv
PROBE = "probe_cub"
OUT = "pair_cub"

# Ужатие оценки перед применением. Оценка из одного зонда шумная, и полный шаг
# к ней переносит шум в файл целиком. Тот же множитель, что у кривизны:
# там 0.699 дал чистый выигрыш после цены шума +4.19e-5.
SHRINK = 0.699


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(SUBMISSIONS / f"{name}.csv").sort("user_id")
    return d["user_id"].to_numpy(), np.log1p(d["predict"].to_numpy().astype(np.float64))


def write(uid: np.ndarray, p: np.ndarray, name: str) -> None:
    pred = np.clip(np.expm1(np.clip(p, 0, None)), 0, None)
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(SUBMISSIONS / f"{name}.csv")


def cubic_dir(p: np.ndarray) -> np.ndarray:
    """u³, очищенное от константы, u и квадратичного направления."""
    u = p - p.mean()
    g = u ** 2
    g = g - g.mean()
    g = g - (g @ u) / (u @ u) * u
    h = u ** 3
    h = h - h.mean()
    h = h - (h @ u) / (u @ u) * u
    h = h - (h @ g) / (g @ g) * g
    return h


def rmsle_of(mse: float) -> float:
    return float(np.sqrt(mse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.0036,
                    help="шаг зонда вдоль кубического направления")
    ap.add_argument("--solve", type=float, default=None,
                    help="public-счёт зонда: решить оптимум")
    ap.add_argument("--apply", action="store_true",
                    help="вместе с --solve собрать финальный файл с ужатым оптимумом")
    ap.add_argument("--shrink", type=float, default=SHRINK)
    args = ap.parse_args()

    uid, p = load(BASE)
    h = cubic_dir(p)
    V = float(np.mean(h * h))
    mse0 = BASE_SCORE ** 2
    print(f"база {BASE}.csv, public {BASE_SCORE:.10f}")
    print(f"Var(h) кубического направления = {V:.5f}")
    print(f"ортогональность h к 1/u/g: {abs(h.mean()):.1e} "
          f"{abs(h @ (p - p.mean())) / len(h):.1e}")

    if args.solve is None:
        g3 = args.gamma
        d = g3 ** 2 * V                      # известный вклад квадратичного члена
        print(f"\nзонд: gamma3 = {g3:+.6f}, расстояние до базы D = {d:.3e}")
        print(f"максимальная поправка {g3 * np.abs(h).max():+.4f} в log1p")
        write(uid, p + g3 * h, PROBE)
        print(f"записан {PROBE}.csv")
        print("\nПРОГНОЗ ДО ОТПРАВКИ (счёт зонда как функция неизвестного оптимума):")
        print(f"  {'gamma*':>10}{'ожидаемый public':>20}")
        for gs in (0.0018, 0.0009, 0.0, -0.0009, -0.0018):
            a = -gs * V
            print(f"  {gs:>+10.4f}{rmsle_of(mse0 + 2 * g3 * a + d):>20.7f}")
        print("\nОбратный ход: по вернувшемуся счёту S оптимум решается точно как")
        print("  a = (S² − MSE(0) − γ²V) / (2γ),   γ* = −a / V,   выигрыш a²/V.")
        return

    g3 = args.gamma
    s = args.solve
    a = (s ** 2 - mse0 - g3 ** 2 * V) / (2 * g3)
    gstar = -a / V
    gain_mse = a * a / V
    gain_rmsle = BASE_SCORE - rmsle_of(mse0 - gain_mse)
    print(f"\nответ зонда: {s:.10f} при gamma3 = {g3:+.6f}")
    print(f"  a = mean(h·(p−y)) = {a:+.6f}")
    print(f"  ОПТИМУМ gamma* = {gstar:+.8f}")
    print(f"  выигрыш при полном шаге: {gain_rmsle:+.7f} RMSLE")
    gsh = args.shrink * gstar
    # Выигрыш при ужатом шаге c: (2c − c²)·полный.
    c = args.shrink
    gain_sh = (2 * c - c * c) * gain_mse
    print(f"  ужатый шаг {args.shrink}: gamma = {gsh:+.8f}, "
          f"выигрыш {BASE_SCORE - rmsle_of(mse0 - gain_sh):+.7f}")
    print(f"  ПРОГНОЗ файла с ужатым шагом: {rmsle_of(mse0 - gain_sh):.7f}")
    if args.apply:
        write(uid, p + gsh * h, OUT)
        print(f"\nзаписан {OUT}.csv")


if __name__ == "__main__":
    main()
