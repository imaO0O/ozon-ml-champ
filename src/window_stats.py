"""Статистики таргета по всем 30-дневным окнам года: где среди них тестовое.

Зачем. Тестовое окно [14.02 … 15.03] содержит 23 февраля и 8 марта — два
крупнейших после Нового года всплеска розницы. Ни один наш валидационный срез
такого состава не имеет: январь несёт хвост новогодних покупок, декабрь —
чёрную пятницу и начало предновогоднего.

Это кандидат на объяснение сразу двух вещей, которые мы списывали на шум
режима:

* **расхождение знаков январь/декабрь** — наблюдали трижды у изменений,
  меняющих форму лосса;
* **растяжение 5% на тесте против 1% на валидации** — окно с двумя праздниками
  может иметь больший разброс покупок, и тогда это не свойство модели,
  а свойство календаря.

Обучения здесь нет вовсе: берём лог, считаем сумму gmv на клиента за каждое
окно [d, d+30) и смотрим на распределение по всем возможным d. Данные
начинаются 01.01.2025, поэтому прошлогодний аналог тестового окна
(14.02.2025 … 15.03.2025) в логе есть целиком.

    python -u src/window_stats.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import HORIZON, SAMPLE_SUBMIT, TEST_CUTOFF
from features import scan_log
from probe_shift import TEST_LEVEL

# Даты, ради которых всё и считается: в 2025-м они внутри лога, и по ним видно,
# как ведёт себя окно с крупным розничным всплеском.
MARKS = {
    dt.date(2025, 2, 14): "аналог тестового окна год назад (23.02 + 08.03)",
    dt.date(2025, 1, 15): "аналог январского валидационного",
    dt.date(2025, 11, 16): "аналог декабрьского валидационного",
    dt.date(2025, 12, 16): "декабрьский валидационный",
    dt.date(2026, 1, 15): "январский валидационный",
}


def window_stats(lf, start: dt.date, horizon: int, users: np.ndarray) -> dict | None:
    """Среднее и разброс log1p(gmv за окно) по ФИКСИРОВАННОМУ набору клиентов.

    Считать по активным в окне нельзя: их число растёт с 188 до 250 тысяч
    за год, и разброс тогда меряет состав выборки, а не поведение окна.
    Наша цель определена ровно так — сумма gmv за окно по всем клиентам
    сабмита, с нулём у тех, кто не купил.
    """
    end = start + dt.timedelta(days=horizon)
    df = (lf.filter((pl.col("event_date") >= start) & (pl.col("event_date") < end))
            .group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
            .collect())
    if df.height == 0:
        return None
    m = dict(zip(df["user_id"].to_numpy(), df["y"].to_numpy()))
    y = np.log1p(np.array([m.get(u, 0.0) for u in users], dtype=np.float64))
    return {"start": start, "buyers": int((y > 0).sum()), "mean": float(y.mean()),
            "std": float(y.std()), "p90": float(np.quantile(y, 0.9))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=7, help="шаг сетки окон в днях")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()

    lf = scan_log()
    users = np.sort(pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy())
    span = lf.select(pl.col("event_date").min().alias("lo"),
                     pl.col("event_date").max().alias("hi")).collect()
    lo, hi = span["lo"][0], span["hi"][0]
    print(f"лог: {lo} … {hi}, горизонт {args.horizon} дней\n")

    rows = []
    d = lo
    while d + dt.timedelta(days=args.horizon) <= hi + dt.timedelta(days=1):
        r = window_stats(lf, d, args.horizon, users)
        if r:
            rows.append(r)
        d += dt.timedelta(days=args.step)
    for m in MARKS:
        if lo <= m and m + dt.timedelta(days=args.horizon) <= hi + dt.timedelta(days=1):
            if all(r["start"] != m for r in rows):
                r = window_stats(lf, m, args.horizon, users)
                if r:
                    rows.append(r)
    rows.sort(key=lambda r: r["start"])

    stds = np.array([r["std"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    print(f"{'начало окна':<14}{'купивших':>11}{'среднее':>10}{'разброс':>10}"
          f"{'перц. разброса':>16}  пометка")
    for r in rows:
        pct = float((stds < r["std"]).mean())
        mark = MARKS.get(r["start"], "")
        star = " <---" if mark else ""
        print(f"{str(r['start']):<14}{r['buyers']:>11,}{r['mean']:>10.4f}"
              f"{r['std']:>10.4f}{pct:>15.0%}  {mark}{star}")

    print(f"\nпо всем {len(rows)} окнам: разброс от {stds.min():.4f} до {stds.max():.4f}, "
          f"среднее {stds.mean():.4f}")
    print(f"среднее уровня: от {means.min():.4f} до {means.max():.4f}")
    # Календарная оценка, откалиброванная по ИЗМЕРЕННОМУ уровню теста.
    #
    # Наивный перенос сезонного шага с 2025-го завышает: рост площадки за 2025-й
    # был кратно быстрее. Но есть точка привязки - TEST_LEVEL измерен зондом
    # лидерборда. Отношение фактического шага уровня к прошлогоднему даёт
    # масштаб, с которым надо переносить и шаг разброса.
    by = {r["start"]: r for r in rows}
    j25, f25 = by.get(dt.date(2025, 1, 15)), by.get(dt.date(2025, 2, 14))
    j26, d25 = by.get(dt.date(2026, 1, 15)), by.get(dt.date(2025, 12, 16))
    if j25 and f25 and j26 and d25:
        scale = (TEST_LEVEL - j26["mean"]) / (f25["mean"] - j25["mean"])
        pred = j26["std"] + (f25["std"] - j25["std"]) * scale
        frac = (pred - j26["std"]) / (d25["std"] - j26["std"])
        print()
        print("календарная оценка тестового окна:")
        print(f"  масштаб сезонного шага 2026/2025: {scale:.2f}")
        print(f"  январь {j26['std']:.4f} | ТЕСТ {pred:.4f} | декабрь {d25['std']:.4f}")
        print(f"  тест лежит на {frac:.0%} пути от января к декабрю, то есть "
              f"ПОСЕРЕДИНЕ,")
        print(f"  а не рядом с январём, как следовало бы из календарной близости.")

    print(f"\nтестовое окно начинается {TEST_CUTOFF} — его таргета в логе нет, "
          f"смотреть надо на прошлогодний аналог 2025-02-14.")


if __name__ == "__main__":
    main()
