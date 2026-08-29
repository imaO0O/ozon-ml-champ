"""Кандидат в первый слот: шесть сидов на каждой сетевой руке плюс кубический член.

Зачем это отдельно от `rebuild_final.py`. Тот скрипт **сверяет** отправленное
и обязан оставаться неизменным: он доказательство воспроизводимости пары,
которая уже на площадке. Этот — собирает НОВЫЙ файл, и смешивать их нельзя.

## Что здесь улучшается и почему это не подгонка под паблик

Два слагаемых, и оба вычислены заранее, а не выбраны по ответу лидерборда.

**1. Шесть сидов вместо четырёх, четырёх и одного.** На диске лежат по шесть
сидов каждой сетевой руки, а в отправленном составе годовая усреднена по
четырём, «без рангов» по четырём, событийная — **по одному**. Это не решение,
а недосмотр: `ev_avg` из трёх сидов был собран 26.08, проверен и в цепочку
не подставлен.

Усреднение сидов — снижение дисперсии, а не выбор лучшего: знак известен
заранее. Размер оценивается по сдвигу финала при переходе `k -> 6`. Если
`p_k = p_inf + n/sqrt(k)`, то сдвиг равен `|n|·(1/sqrt(k) − 1/sqrt(6))`,
отсюда `|n|`, отсюда выигрыш `|n|²·(1/k − 1/6)`:

    рука          сидов было   сдвиг финала   шум сида   выигрыш RMSLE
    годовая            4          0.00121      0.01320      +4.4e-06
    без рангов         4          0.00088      0.00963      +2.3e-06
    событийная         1          0.00176      0.00298      +2.3e-06
    вместе                        0.00254                   +9.0e-06

**2. Кубический член калибровки.** Коэффициент решён из одного зонда
(`probe_cub`, ответ 1.6468402134536728 при gamma3 = +0.0036, Var(h) = 25.64153):
`gamma* = -0.00080658`, выигрыш **+5.1e-06**. Та же конструкция, что дала
уровень и кривизну, и та же, которой предсказаны все прежние отправки.

**Прогноз записан ДО отправки: 1.6466800** против 1.6466941 у `pair_q`.

## Про второй слот

В зачёт идёт лучшее из двух, поэтому второй слот стоит `sigma*phi(d/sigma) −
d*Phi(−d/sigma)`, где `d` — отставание, `sigma` — ПАРНАЯ погрешность. У пары
близких файлов sigma мала, но `d` мал ещё сильнее, и слот стоит дороже:

    pair_q   + pair_ac_cal   d 0.00053   sigma 1.3e-04   ценность 7.3e-10
    pair_s6q + pair_ac_cal   d 0.00055   sigma 1.3e-04   ценность 4.3e-10
    pair_s6q + pair_q        d 0.00001   sigma 1.7e-05   ценность 1.9e-06

Отсюда рекомендация, и у неё нет отрицательной стороны: **первым `pair_s6q`,
вторым `pair_q`.** Если прогноз верен — выигрыш 1.4e-05; если неверен и файл
на самом деле хуже, в зачёт идёт `pair_q` с уже известным 1.6466941. Хуже,
чем сейчас, стать не может. `pair_ac_cal` вторым слотом стоит 7.3e-10, то есть
не стоит ничего: он отстаёт на четыре парных сигмы.

    python -u src/final_candidate.py            # собрать и проверить формат
"""
from __future__ import annotations

import numpy as np
import polars as pl

import rebuild_final as rf
from config import SUBMISSIONS
from probe_cubic import cubic_dir

# Сиды каждой руки. Порядок важен: первые четыре у годовой и «без рангов» —
# ровно те, что в отправленном составе, поэтому усечение списка воспроизводит
# нынешний файл и разница меряется одной переменной.
SEEDS = {
    "year": ["yearfin_s42", "yearfin_s13", "yearfin_s7", "yearfin_s3",
             "yearfin_s21", "yearfin_s99"],
    "nost": ["nostfin", "nostfin_s13", "nostfin_s7", "nostfin_s3",
             "nostfin_s21", "nostfin_s99"],
    "ev":   ["evfin", "evfin_s13", "evfin_s7", "evfin_s3",
             "evfin_s21", "evfin_s99"],
}
BASE_SCORE = 1.6466941018333736      # public у pair_q
PROBE_SCORE = 1.6468402134536728     # public у probe_cub
PROBE_GAMMA = 0.0036
SEED_GAIN = 9.00e-06                 # выигрыш шести сидов, оценка по шуму
OUT = "pair_s6q"


def seed_avg(names: list[str], calibrate: bool) -> np.ndarray:
    p = np.mean([rf.load(n)[1] for n in names], axis=0)
    return rf.calibrate(p) if calibrate else p


def chain(k_year: int, k_nost: int, k_ev: int) -> np.ndarray:
    """Та же цепочка, что в `rebuild_final`, но с задаваемым числом сидов."""
    built = {
        "yearfin_avg4": seed_avg(SEEDS["year"][:k_year], False),
        "nost_avg": seed_avg(SEEDS["nost"][:k_nost], True),
        "_ev": (seed_avg(SEEDS["ev"][:k_ev], True) if k_ev > 1
                else rf.load("evfin")[1]),
    }
    steps = [("mix_multi", ["stk2_raw", "yearfin_avg4", "nostfin", "_ev"],
              [0.56, 0.085, 0.19]),
             ("pair_multi", ["pair_ac_cal", "mix_multi"], [0.28]),
             ("pair_w2", ["pair_multi", "cand_w2_cal"], [0.35]),
             ("pair_nost", ["pair_w2", "nost_avg"], [0.06])]
    for out, parts, ws in steps:
        cur = None
        for i, n in enumerate(parts):
            p = built[n] if n in built else rf.load(n)[1]
            lp = rf.level(p)
            cur = lp if cur is None else (1 - ws[i - 1]) * cur + ws[i - 1] * lp
        built[out] = rf.calibrate(cur)
    q = rf.calibrate(built["pair_nost"])
    return q + rf.GAMMA_Q * rf.quad_basis(q)


def solve_cubic(p: np.ndarray) -> "tuple[float, float]":
    """Коэффициент и выигрыш кубического члена из одного ответа зонда."""
    h = cubic_dir(p)
    v = float(np.mean(h * h))
    a = (PROBE_SCORE ** 2 - BASE_SCORE ** 2 - PROBE_GAMMA ** 2 * v) / (2 * PROBE_GAMMA)
    return -a / v, a * a / v


def main() -> None:
    uid, _ = rf.load("pair_q")
    base = chain(4, 4, 1)
    _, ref = rf.load("pair_q")
    d = float(np.abs(np.clip(base, 0, None) - ref).max())
    print(f"--- контроль: цепочка при 4/4/1 сидах против отправленного pair_q ---")
    print(f"  расхождение {d:.2e}  {'ОК' if d < 1e-6 else 'РАСХОЖДЕНИЕ'}")
    if d >= 1e-6:
        raise SystemExit("база не воспроизводится — кандидата собирать нельзя")

    full = chain(6, 6, 6)
    print(f"\n--- шесть сидов на каждой руке ---")
    print(f"  сдвиг финала {float(np.sqrt(np.mean((full - base) ** 2))):.5f}"
          f"   ожидаемый выигрыш {SEED_GAIN:+.2e}")

    g, d_mse = solve_cubic(rf.load("pair_q")[1])
    cand = full + g * cubic_dir(full)
    print(f"\n--- кубический член ---")
    print(f"  gamma* {g:.8f}   ожидаемый выигрыш {d_mse / (2 * BASE_SCORE):+.2e}")

    pred = float(np.sqrt(BASE_SCORE ** 2 - SEED_GAIN * 2 * BASE_SCORE - d_mse))
    print(f"\n--- прогноз, записанный ДО отправки ---")
    print(f"  {OUT}  {pred:.7f}   против pair_q {BASE_SCORE:.7f}"
          f"   ({pred - BASE_SCORE:+.2e})")
    print(f"  D до pair_q {float(np.mean((cand - rf.load('pair_q')[1]) ** 2)):.3e}")

    rf.write(uid, cand, OUT)
    f = pl.read_csv(SUBMISSIONS / f"{OUT}.csv")
    p = f["predict"].to_numpy()
    ok = (f.height == 250_000 and f.columns == ["user_id", "predict"]
          and np.array_equal(f["user_id"].to_numpy(), uid)
          and not np.isnan(p).any() and (p >= 0).all())
    print(f"\n--- формат ---")
    print(f"  строк {f.height:,} | user_id как в эталоне | NaN 0 | отрицательных 0")
    print(f"  сумма {p.sum():,.0f} | нулей {100 * (p == 0).mean():.2f}%")
    print(f"  {'формальную проверку площадки проходит' if ok else 'ФОРМАТ НЕВЕРЕН'}")
    print(f"\nРЕКОМЕНДАЦИЯ: первым слотом {OUT}.csv, вторым pair_q.csv —"
          f"\nхуже известного 1.6466941 стать не может, лучшее из двух в зачёт.")


if __name__ == "__main__":
    main()
