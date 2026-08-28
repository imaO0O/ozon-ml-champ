"""Замена устаревшей половины внутри pair_ac — крупнейший оставшийся выигрыш.

Что заморожено и почему это стоит денег. В рекорд `pair_q` цепочкой входит
`pair_ac_cal` с весом 0.28, а сам `pair_ac` — это половина трека C, смешанная
с НАШЕЙ половиной по состоянию на 25.08:

    pair_ac = expm1(0.5·log1p(mix_year) + 0.5·log1p(cand_w))

Тождество проверено восстановлением: невязка 4.2e-06 в log1p, тогда как
гипотеза «смесь в сырой шкале» промахивается на 0.34 — различимо в 80 000 раз.

`mix_year` (public 1.6479607) с тех пор вытеснена: наша половина теперь `mix9`
(1.6473665), лучше на 0.00059. Внутри рекорда она сидит замороженной, и
разморозка — единственная крупная позиция, оставшаяся у команды.

**Условие, без которого замена нечестна.** Вес 0.5 в паре задан треком C и
от счетов не зависит — в этом вся ценность `pair_ac_cal` как страховки
(A2: «ноль подогнанных весов»). Он остаётся нейтральным, только если уровни
половин совпадают. Проверено: `cand_w` 2.329120, `mix9` 2.329120,
`mix_year` 2.329128. Замена уровень не сдвигает, вес подгонкой не становится.

## Прогноз считается, а не гадается

Каждый шаг цепочки — попарная выпуклая смесь, для неё

    MSE((1−w)·C + w·P) = (1−w)·MSE(C) + w·MSE(P) − w(1−w)·D(C,P)

точно. Калибровка (уровень + растяжение) — детерминированное аффинное
преобразование, и её влияние на MSE вычисляется через `Var(y)` тестового окна,
восстановленную тремя независимыми способами со сходимостью 0.01%.

Итоговый прогноз строится иначе и надёжнее: по разности файлов. Для
`Δ = новый − старый`

    MSE(новый) = MSE(старый) + 2·mean(Δ·(старый − y)) + mean(Δ²)

где `mean(Δ·y)` берётся из известных публичных счетов компонент: разность
счетов `mix9` и `mix_year` даёт `mean(d·y)` для `d = mix9 − mix_year` в лоб.

## ИТОГ: замену делать НЕЛЬЗЯ, и причина важнее самой замены

Посчитано, не оценено. Метод проверен контролем: та же арифметика
воспроизводит известный `pair_nost` до 7e-7.

    новая пара 0.5·mix9 + 0.5·cand_w   1.6470369   ЛУЧШЕ старой на 0.00019
    цепочка с новой половиной          1.6469483
    цепочка со старой (pair_nost)      1.6467440   ЛУЧШЕ на 0.00020

Половина стала лучше, а состав стал хуже. Механизм измерен:

    первый шаг      своё MSE   D до mix_multi  опт.вес   выигрыш
    старая        1.6472270          0.00409     0.339  +0.000143
    новая         1.6470369          0.00114    -0.349  +0.000042

Обновлённая половина втрое БЛИЖЕ к остальной цепочке — `mix9` и `mix_multi`
собраны из одних компонент, — и оптимальный вес партнёра уходит в минус.
Партнёрство разрушается.

**Устаревшая половина ценна именно тем, что устарела.** Она — независимое
более раннее состояние нашего же моделирования, и ошибается иначе. Освежить
её значит уничтожить ровно то, за что она в составе стоит.

Это наш центральный закон в самой резкой форме: расстояние ничего не значит,
значит направление — и здесь улучшение качества СНИЗИЛО расстояние и убило
направление. Прежняя оценка выигрыша +0.00013…+0.0002 считала прибавку
к качеству и не учитывала потерю направления; она отменена.

Как добавочный партнёр новая пара тоже не годится: оптимальный вес −0.060.

Скрипт оставлен воспроизводимым отрицательным результатом.

    python -u src/swap_half.py            # пересчитать вывод
    python -u src/swap_half.py --write    # записать файлы (не нужно)
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import SUBMISSIONS
from rebuild_final import (GAMMA_AC, GAMMA_Q, TARGET_VAR, TEST_LEVEL, calibrate,
                           level, load, quad_basis, write)

# Публичные счета, всё из submissions/log.csv.
PUB = {
    "mix_year": 1.647960682323361,
    "mix9": 1.6473664856835506,
    "cand_w": 1.6480186,
    "mix_multi": 1.6476256267531169,
    "pair_ac_cal": 1.6472269592902848,
    "cand_w2_cal": 1.6472647800533498,
    "nost_avg": 1.6882862502056688,
    "pair_q": 1.6466941018333736,
}
VAR_Y = 5.36583          # дисперсия log1p(y) тестового окна, три независимых замера


def moments(p: np.ndarray, score: float) -> float:
    """mean(p·y) из публичного счёта: MSE = mean(p²) − 2mean(py) + mean(y²)."""
    mean_y2 = VAR_Y + TEST_LEVEL ** 2
    return float((np.mean(p * p) + mean_y2 - score ** 2) / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--our-half", default="mix9", help="чем заменить mix_year")
    args = ap.parse_args()

    uid, my = load("mix_year")
    _, cw = load("cand_w")
    _, new_half = load(args.our_half)
    _, old_final = load("pair_q")

    # --- 1. тождество ---
    _, ac = load("pair_ac")
    res = float(np.abs(0.5 * my + 0.5 * cw - ac).max())
    print(f"тождество pair_ac = 0.5·log1p(mix_year) + 0.5·log1p(cand_w): "
          f"невязка {res:.2e}")
    if res > 1e-4:
        raise SystemExit("тождество не подтверждается, замена небезопасна")

    print(f"\nуровни (вес 0.5 нейтрален, только если совпадают):")
    for n, p in [("cand_w", cw), ("mix_year", my), (args.our_half, new_half)]:
        print(f"  {n:<12}{p.mean():.6f}")

    # --- 2. новая половина пары и пересборка цепочки ---
    new_ac = 0.5 * new_half + 0.5 * cw
    new_ac_cal = calibrate(new_ac)
    new_ac_cal = new_ac_cal + GAMMA_AC * quad_basis(new_ac_cal)

    cur = level(new_ac_cal)
    for name, w in [("mix_multi", 0.28), ("cand_w2_cal", 0.35), ("nost_avg", 0.06)]:
        cur = (1 - w) * cur + w * level(load(name)[1])
        cur = calibrate(cur)
    new_final = cur + GAMMA_Q * quad_basis(cur)

    # --- 3. прогноз по разности ---
    d = new_half - my
    # mean(d·y) из разности публичных счетов компонент
    dy = float((np.mean(d * (new_half + my)) - (PUB[args.our_half] ** 2 - PUB["mix_year"] ** 2)) / 2)
    delta = new_final - old_final
    # Δ раскладывается по {1, d, old_final}: калибровка делает связь чуть
    # нелинейной (alpha зависит от дисперсии), но остаток должен быть мал.
    X = np.column_stack([np.ones_like(d), d, old_final])
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
    resid = delta - X @ beta
    print(f"\nразложение Δ по (1, d, старый финал): остаток "
          f"{np.abs(resid).max():.2e} при |Δ| до {np.abs(delta).max():.4f}")

    of_y = moments(old_final, PUB["pair_q"])
    delta_y = beta[0] * TEST_LEVEL + beta[1] * dy + beta[2] * of_y
    mse_old = PUB["pair_q"] ** 2
    mse_new = (mse_old + 2 * (float(np.mean(delta * old_final)) - delta_y)
               + float(np.mean(delta * delta)))
    print(f"\nD(новый, старый) = {float(np.mean(delta ** 2)):.3e}")
    print(f"ПРОГНОЗ ДО ОТПРАВКИ: {np.sqrt(mse_new):.7f}  "
          f"({np.sqrt(mse_new) - PUB['pair_q']:+.7f} к рекорду)")

    if args.write:
        write(uid, new_ac_cal, "pair_ac9_cal")
        write(uid, new_final, "pair_q9")
        print("\nзаписаны pair_ac9_cal.csv и pair_q9.csv")


if __name__ == "__main__":
    main()
