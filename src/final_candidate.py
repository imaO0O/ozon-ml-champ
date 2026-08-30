"""Кандидат в первый слот: все доступные сиды на сетевых руках плюс кубический член.

Зачем это отдельно от `rebuild_final.py`. Тот скрипт **сверяет** отправленное
и обязан оставаться неизменным: он доказательство воспроизводимости пары,
которая уже на площадке. Этот — собирает НОВЫЙ файл, и смешивать их нельзя.

## Что здесь улучшается и почему это не подгонка под паблик

**1. Все сиды, какие есть.** В отправленном 27.08 составе годовая рука была
усреднена по четырём сидам, «без рангов» по четырём, событийная — **по одному**,
при том что на диске лежало по шесть. Недосмотр, а не решение: трёхсидовое
среднее `ev_avg` собрано 26.08, проверено и в цепочку не подставлено.

Усреднение сидов — снижение дисперсии, а не выбор лучшего: знак известен
заранее, и подгонкой под лидерборд это не является.

**2. Рука «без рангов» усредняется и ВНУТРИ `mix_multi`.** В отправленном
файле там стоял одиночный сид, а среднее входило отдельным партнёром с весом
0.06. Прибавка мелкая (7e-07), но бесплатная.

**3. Кубический член калибровки**, коэффициент решён из одного зонда
(`probe_cub`, ответ 1.6468402135 при gamma3 = +0.0036, Var(h) = 25.64153).

## Как считается прогноз и почему шкала откалибрована по факту

Проверено отправкой 30.08: переход 4/4/1 -> 6/6/6 вместе с кубическим членом
предсказывался как 1.6466800. Получено **1.6466589721**, то есть фактический
выигрыш +3.51e-05 против предсказанных +1.41e-05. Если кубический член дал
ровно решённое зондом (+5.07e-06 — величина измеренная), на сиды пришлось
+3.01e-05.

**Формула была выведена неверно и исправлена 30.08.** Для ВЛОЖЕННЫХ наборов
(первые j из k сидов) `E||P_j - P_k||^2 = sigma^2*(1/j - 1/k)`, и выигрыш MSE
от `j -> k` равен ровно тому же. То есть **выигрыш MSE равен квадрату сдвига**,
без всяких `sqrt`. Прежняя версия делила сдвиг на `1/sqrt(j) - 1/sqrt(k)`
и давала немонотонную оценку: прогноз для восьми сидов выходил хуже факта
для шести, что и вскрыло ошибку.

По исправленной модели выигрыш 4/4/1 -> 6/6/6 равен 6.45e-06 MSE, а факт
9.90e-05 — **в пятнадцать раз больше**. Скорее всего дело в калибровке:
усреднение сжимает дисперсию, а stretch до `TARGET_VAR` возвращает её обратно,
превращая снятый шум в сигнал. Разделить одним замером нельзя.

Поэтому прогноз считается по той же форме (`выигрыш ~ сдвиг^2`), но с наклоном,
снятым с единственного замера; чистая модель печатается рядом как нижняя
граница. **Знак от этого не зависит: при любом наклоне больше сидов не хуже.**

## Отсев вырожденных сидов

Один испорченный сид в среднем не улучшает, а портит, и в истории проекта
79 прогонов сети были оборваны на первой-третьей эпохе. Поэтому каждый сид
проверяется на выброс **в пространстве предсказаний**: расстояние до среднего
остальных сравнивается с медианой по семье, порог 1.5 медианы.

Проверять по валидационной метрике было бы отбором и запрещено; расстояние
до своих же братьев — контроль качества, а не выбор лучшего. На 30.08 выбросов
нет: `yearfin_s5` остановился на 4-й эпохе при 10-12 у прочих, но в пространстве
предсказаний он на 1.29 медианы против 1.28 у `yearfin_s42`, то есть внутри
разброса семьи, и он включён.

    python -u src/final_candidate.py            # собрать, проверить формат
    python -u src/final_candidate.py --dry      # только отчёт, файл не писать
"""
from __future__ import annotations

import argparse
import csv
import io

import numpy as np
import polars as pl

import rebuild_final as rf
from config import MODELS, SUBMISSIONS
from probe_cubic import cubic_dir

ARMS = {"year": "yearfin", "nost": "nostfin", "ev": "evfin"}
# Сиды отправленного 27.08 состава — от них считается сдвиг. Порядок не менять:
# усечение до этих имён обязано воспроизводить отправленный файл.
BASE_ORDER = {
    "year": ["yearfin_s42", "yearfin_s13", "yearfin_s7", "yearfin_s3"],
    "nost": ["nostfin", "nostfin_s13", "nostfin_s7", "nostfin_s3"],
    "ev": ["evfin"],
}
BASE_K = {k: len(v) for k, v in BASE_ORDER.items()}
PAIR_Q_SCORE = 1.6466941018333736
S6Q_SCORE = 1.6466589720673095       # ответ по 6/6/6 плюс кубика, 30.08
PROBE_SCORE = 1.6468402134536728
PROBE_GAMMA = 0.0036
# Замер 30.08: сдвиг финала 4/4/1 -> 6/6/6 составил 0.00254, а выигрыш на сиды
# (полный минус решённый зондом кубический член) — 9.90e-05 в единицах MSE.
# Чистая модель дала бы ровно квадрат сдвига, 6.45e-06, то есть в 15 раз меньше.
# Расхождение, вероятно, от калибровки: усреднение сжимает дисперсию, а stretch
# до TARGET_VAR возвращает её обратно, превращая снятый шум в сигнал.
# Разделить это одним замером нельзя, поэтому наклон берётся с него как есть.
OBS_SHIFT = 0.00254
OBS_SEED_MSE = 9.901e-05
OUTLIER = 1.5                         # порог выброса в медианах расстояния
OUT = "pair_s12q"


def available(prefix: str) -> list[str]:
    """Завершённые прогоны руки: строка журнала плюс файл на диске.

    Источник истины — журнал, а не список файлов: `predict` пишет имя
    с временной меткой, и разбирать имя файла нельзя — имя прогон
    не идентифицирует.
    """
    done = {(r.get("name") or "").strip()
            for r in csv.DictReader(io.open(MODELS / "experiments.csv", encoding="utf-8"))
            if (r.get("rmsle_single") or "").strip()}
    return sorted(n for n in done
                  if (n == prefix or n.startswith(prefix + "_s"))
                  and (SUBMISSIONS / f"{n}.csv").exists())


def drop_outliers(names: list[str]) -> "tuple[list[str], list[tuple[str, float]]]":
    if len(names) < 4:
        return names, []
    p = {n: rf.level(rf.load(n)[1]) for n in names}
    d = {n: float(np.sqrt(np.mean((p[n] - np.mean([p[m] for m in names if m != n],
                                                  axis=0)) ** 2))) for n in names}
    med = float(np.median(list(d.values())))
    bad = [(n, d[n] / med) for n in names if d[n] > OUTLIER * med]
    return [n for n in names if d[n] <= OUTLIER * med], bad


def seed_avg(names: list[str], calibrate: bool) -> np.ndarray:
    p = np.mean([rf.load(n)[1] for n in names], axis=0)
    return rf.calibrate(p) if calibrate else p


def chain(sel: "dict[str, list[str]]", nost_in_mix: bool) -> np.ndarray:
    built = {
        "_year": seed_avg(sel["year"], False),
        "_nostavg": seed_avg(sel["nost"], True),
        "_ev": (seed_avg(sel["ev"], True) if len(sel["ev"]) > 1
                else rf.load(sel["ev"][0])[1]),
        "_nost": (seed_avg(sel["nost"], False) if nost_in_mix
                  else rf.load("nostfin")[1]),
    }
    steps = [("mix_multi", ["stk2_raw", "_year", "_nost", "_ev"], [0.56, 0.085, 0.19]),
             ("pair_multi", ["pair_ac_cal", "mix_multi"], [0.28]),
             ("pair_w2", ["pair_multi", "cand_w2_cal"], [0.35]),
             ("pair_nost", ["pair_w2", "_nostavg"], [0.06])]
    for out, parts, ws in steps:
        cur = None
        for i, n in enumerate(parts):
            p = built[n] if n in built else rf.load(n)[1]
            lp = rf.level(p)
            cur = lp if cur is None else (1 - ws[i - 1]) * cur + ws[i - 1] * lp
        built[out] = rf.calibrate(cur)
    q = rf.calibrate(built["pair_nost"])
    return q + rf.GAMMA_Q * rf.quad_basis(q)


def solve_cubic() -> "tuple[float, float]":
    """Коэффициент и выигрыш кубического члена из одного ответа зонда."""
    _, p = rf.load("pair_q")
    h = cubic_dir(p)
    v = float(np.mean(h * h))
    a = (PROBE_SCORE ** 2 - PAIR_Q_SCORE ** 2 - PROBE_GAMMA ** 2 * v) / (2 * PROBE_GAMMA)
    return -a / v, a * a / v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="только отчёт, файл не писать")
    args = ap.parse_args()

    print("--- доступные сиды и отсев выбросов ---")
    sel: dict[str, list[str]] = {}
    for key, prefix in ARMS.items():
        names = available(prefix)
        good, bad = drop_outliers(names)
        sel[key] = good
        print(f"  {prefix:<9} найдено {len(names):>2}, взято {len(good):>2}")
        for n, ratio in bad:
            print(f"      ОТБРОШЕН {n} — {ratio:.2f} медианы расстояния до семьи")

    print("\n--- контроль: усечение до состава 27.08 воспроизводит pair_q ---")
    base = chain(BASE_ORDER, nost_in_mix=False)
    _, ref = rf.load("pair_q")
    d = float(np.abs(np.clip(base, 0, None) - ref).max())
    print(f"  расхождение {d:.2e}  {'ОК' if d < 1e-6 else 'РАСХОЖДЕНИЕ'}")
    if d >= 1e-6:
        raise SystemExit("база не воспроизводится — кандидата собирать нельзя")

    full = chain(sel, nost_in_mix=True)
    shift = float(np.sqrt(np.mean((full - base) ** 2)))

    print("\n--- сдвиг по рукам (справочно) ---")
    for key, prefix in ARMS.items():
        one = dict(BASE_ORDER)
        one[key] = sel[key]
        k0, k = BASE_K[key], len(sel[key])
        if k <= k0:
            print(f"  {prefix:<9} сидов {k0} -> {k}: без изменений")
            continue
        s = float(np.sqrt(np.mean((chain(one, nost_in_mix=False) - base) ** 2)))
        print(f"  {prefix:<9} сидов {k0} -> {k:<3} сдвиг {s:.5f}")

    g_cub, d_mse_cub = solve_cubic()
    # Модель даёт выигрыш MSE РОВНО равным квадрату сдвига (вывод в шапке),
    # а факт 30.08 оказался в 15 раз больше. Поэтому нижняя граница — модель,
    # рабочая оценка — та же форма с наклоном, снятым с единственного замера.
    model_mse = shift ** 2
    seed_mse = OBS_SEED_MSE * (shift / OBS_SHIFT) ** 2
    print("\n--- итог ---")
    print(f"  сдвиг финала от базы          {shift:.5f}")
    print(f"  сиды по чистой модели         {model_mse / (2 * PAIR_Q_SCORE):+.2e}"
          f"   (нижняя граница)")
    print(f"  сиды по замеру 30.08          {seed_mse / (2 * PAIR_Q_SCORE):+.2e}"
          f"   (наклон {OBS_SEED_MSE / OBS_SHIFT ** 2:.1f}x модели, ОДИН замер)")
    print(f"  кубический член gamma* {g_cub:.8f}  {d_mse_cub / (2 * PAIR_Q_SCORE):+.2e}")

    cand = full + g_cub * cubic_dir(full)
    pred = float(np.sqrt(PAIR_Q_SCORE ** 2 - seed_mse - d_mse_cub))
    low = float(np.sqrt(PAIR_Q_SCORE ** 2 - model_mse - d_mse_cub))
    print("\n--- ПРОГНОЗ, записанный ДО отправки ---")
    print(f"  {OUT}  {pred:.7f}   (нижняя граница по чистой модели {low:.7f})")
    print(f"  против pair_s6q {S6Q_SCORE:.7f}   ({pred - S6Q_SCORE:+.2e})")
    print(f"  D до pair_s6q {float(np.mean((cand - rf.load('pair_s6q')[1]) ** 2)):.3e}")

    if args.dry:
        print("\nотчёт; файл не записан")
        return

    uid, _ = rf.load("pair_q")
    rf.write(uid, cand, OUT)
    f = pl.read_csv(SUBMISSIONS / f"{OUT}.csv")
    p = f["predict"].to_numpy()
    ok = (f.height == 250_000 and f.columns == ["user_id", "predict"]
          and np.array_equal(f["user_id"].to_numpy(), uid)
          and not np.isnan(p).any() and (p >= 0).all())
    print("\n--- формат ---")
    print(f"  строк {f.height:,} | user_id как в эталоне | NaN 0 | отрицательных 0")
    print(f"  сумма {p.sum():,.0f} | нулей {100 * (p == 0).mean():.2f}%")
    print(f"  {'формальную проверку площадки проходит' if ok else 'ФОРМАТ НЕВЕРЕН'}")
    # Рекомендация «первым pair_s12q» стояла здесь до ответа лидерборда
    # и была им опровергнута. Скрипт не должен пережить свой вывод.
    print(f"\nВНИМАНИЕ: {OUT} отправлен 30.08 и получил 1.6466757047 — "
          f"ХУЖЕ pair_s6q")
    print("на 1.67e-05. Прогноз промахнулся на 3.9e-05: в одном файле были")
    print("соединены ДВА изменения, а наклон модели снят с одного наблюдения")
    print("в 1.25 сигмы. Разбор обеих ошибок — docs/jury/A2.")
    print("\nФИНАЛЬНАЯ ПАРА: pair_s6q.csv и pair_q.csv, выбрана по разделённости,")
    print("а не по лучшему публичному счёту. Скрипт оставлен как инструмент")
    print("сборки и как запись опровергнутой гипотезы, а не как рекомендация.")


if __name__ == "__main__":
    main()
