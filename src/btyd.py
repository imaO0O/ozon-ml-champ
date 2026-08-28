"""Вероятностная иерархическая модель покупательского поведения: BG/NBD + Gamma-Gamma.

Зачем она в решении, где уже есть бустинг и нейросеть. Бустинг выдаёт **точку** —
условное среднее log1p(GMV), и больше ничего. На вопрос «с какой вероятностью
этот клиент вообще купит в следующие 30 дней», «жив ли он ещё», «каков разброс
его чека» бустинг не отвечает: у него этих величин просто нет.

BG/NBD отвечает на них из коробки, потому что это не регрессия, а **порождающая
модель поведения**:

* пока клиент «жив», он покупает по пуассоновскому процессу с личной
  интенсивностью λ;
* после каждой покупки он с вероятностью p уходит навсегда;
* λ по популяции распределена как Gamma(r, alpha), p — как Beta(a, b).

Последние две строки и делают модель **иерархической**: у каждого клиента свои
λ и p, но они не свободны — они порождены общим для всех популяционным
распределением. Клиент с двумя покупками не получает оценку «λ = 2 / T»,
а стягивается к популяции тем сильнее, чем меньше о нём известно. Это partial
pooling в чистом виде, и он же виден в формуле Gamma-Gamma ниже: ожидаемый чек
там — **взвешенное среднее личного и популяционного**, где вес личного растёт
с числом покупок.

Что модель отдаёт на каждого клиента:

    p_alive      вероятность, что клиент ещё не ушёл
    e_trans      ожидаемое число покупок в следующие 30 дней
    e_value      ожидаемый чек одной покупки (с усадкой к популяции)
    e_gmv        произведение двух последних — ожидаемый GMV окна
    p_zero       вероятность нулевого GMV в окне

Точность по RMSLE у неё заведомо ниже бустинга: три-четыре параметра против
242 признаков. Ценность в другом — в ответах на вопросы, которых бустинг
не даёт, и в непохожести: модель ошибается там, где ошибается предположение
о пуассоновости, а не там, где деревьям не хватило данных.

    python -u src/btyd.py --cutoff 2026-01-15
    python -u src/btyd.py --cutoff 2026-01-15 --save models/btyd_2026-01-15.npz
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

from config import HORIZON, TRAIN_PARQUET
from metrics import gini_norm, rmse_log


def summary(cutoff: dt.date) -> pl.DataFrame:
    """RFM-сводка на срез: x покупок сверх первой, t_x возраст последней, T возраст клиента.

    Транзакцией считается день с покупкой: в логе нет отдельных заказов, есть
    суточные итоги. Для BG/NBD это корректно — процесс всё равно считается
    в непрерывном времени, а день здесь единица.
    """
    lf = pl.scan_parquet(TRAIN_PARQUET)
    if lf.collect_schema()["event_date"] != pl.Date:
        lf = lf.with_columns(pl.col("event_date").cast(pl.Date))
    buys = (lf.filter((pl.col("event_date") < cutoff) & (pl.col("gmv") > 0))
              .group_by("user_id")
              .agg(pl.len().alias("n_buy"),
                   pl.col("event_date").min().alias("first"),
                   pl.col("event_date").max().alias("last"),
                   pl.col("gmv").sum().alias("gmv_total"))
              .collect())
    return buys.with_columns(
        (pl.col("n_buy") - 1).cast(pl.Float64).alias("x"),
        (pl.col("last") - pl.col("first")).dt.total_days().cast(pl.Float64).alias("t_x"),
        (pl.lit(cutoff) - pl.col("first")).dt.total_days().cast(pl.Float64).alias("T"),
        (pl.col("gmv_total") / pl.col("n_buy")).alias("m_bar"),
    )


def bgnbd_nll(par, x, t_x, T):
    """Минус лог-правдоподобие BG/NBD; параметры оптимизируются в логарифмах."""
    r, alpha, a, b = np.exp(par)
    ln_A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_A2 = -(r + x) * np.log(alpha + T)
    ln_A3 = betaln(a, b + x) - betaln(a, b)
    # Второе слагаемое существует только у клиентов с повторной покупкой:
    # «ушёл сразу после последней» — событие, требующее хотя бы одной повторной.
    ln_A4 = np.where(x > 0,
                     betaln(a + 1, b + x - 1) - betaln(a, b) - (r + x) * np.log(alpha + t_x),
                     -np.inf)
    m = np.maximum(ln_A3 + ln_A2, ln_A4)
    ll = ln_A1 + m + np.log(np.exp(ln_A3 + ln_A2 - m) + np.where(x > 0, np.exp(ln_A4 - m), 0.0))
    return -np.sum(ll)


def fit_bgnbd(x, t_x, T):
    best = None
    for start in ([0.0, 1.0, 0.0, 1.0], [-0.5, 2.0, -0.5, 0.5], [0.5, 3.0, 0.5, 1.5]):
        res = minimize(bgnbd_nll, start, args=(x, t_x, T), method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x), best.fun


def expected_transactions(par, x, t_x, T, horizon):
    """E[число покупок за horizon | история] — стандартная формула BG/NBD."""
    r, alpha, a, b = par
    z = horizon / (alpha + T + horizon)
    hyp = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
    first = (a + b + x - 1.0) / (a - 1.0)
    second = 1.0 - ((alpha + T) / (alpha + T + horizon)) ** (r + x) * hyp
    denom = 1.0 + np.where(x > 0, (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return first * second / denom


def p_alive(par, x, t_x, T):
    r, alpha, a, b = par
    ratio = np.where(x > 0, (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return 1.0 / (1.0 + ratio)


def gg_nll(par, x, m_bar):
    """Gamma-Gamma на клиентах с повторными покупками."""
    p, q, g = np.exp(par)
    return -np.sum(gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
                   + q * np.log(g) + (p * x - 1.0) * np.log(m_bar)
                   + (p * x) * np.log(x) - (p * x + q) * np.log(g + x * m_bar))


def fit_gg(x, m_bar):
    mask = (x > 0) & (m_bar > 0)
    best = None
    for start in ([0.0, 0.0, 2.0], [0.5, 1.0, 3.0], [-0.5, 0.5, 1.0]):
        res = minimize(gg_nll, start, args=(x[mask], m_bar[mask]), method="Nelder-Mead",
                       options={"maxiter": 8000})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x)


def expected_value(par, x, m_bar):
    """Ожидаемый чек: взвешенное среднее личного и популяционного.

    Вес личного среднего растёт с числом покупок — это и есть иерархическая
    усадка. У клиента с одной покупкой оценка почти популяционная, у клиента
    с двадцатью — почти его собственная.
    """
    p, q, g = par
    pop = g * p / (q - 1.0)
    w = (p * x) / (p * x + q - 1.0)
    return np.where(x > 0, w * m_bar + (1.0 - w) * pop, pop)


def predict(cutoff: dt.date, horizon: int = HORIZON) -> pl.DataFrame:
    """Полный набор вероятностных величин на каждого клиента с историей покупок."""
    s = summary(cutoff)
    x = s["x"].to_numpy(); t_x = s["t_x"].to_numpy(); T = s["T"].to_numpy()
    m_bar = s["m_bar"].to_numpy()

    bg, nll = fit_bgnbd(x, t_x, T)
    gg = fit_gg(x, m_bar)
    print(f"BG/NBD  r={bg[0]:.4f} alpha={bg[1]:.4f} a={bg[2]:.4f} b={bg[3]:.4f}  "
          f"(-logL {nll:,.0f})")
    print(f"Gamma-Gamma  p={gg[0]:.4f} q={gg[1]:.4f} gamma={gg[2]:.2f}  "
          f"популяционный чек {gg[2] * gg[0] / (gg[1] - 1):.2f}")

    et = expected_transactions(bg, x, t_x, T, horizon)
    pa = p_alive(bg, x, t_x, T)
    ev = expected_value(gg, x, m_bar)
    # Число покупок за окно — пуассон со средним et; отсюда вероятность нуля.
    return s.select("user_id").with_columns(
        pl.Series("p_alive", pa),
        pl.Series("e_trans", et),
        pl.Series("e_value", ev),
        pl.Series("e_gmv", et * ev),
        pl.Series("p_zero", np.exp(-et)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-01-15")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--save", default=None, help="npz с вероятностными величинами")
    args = ap.parse_args()
    cutoff = dt.date.fromisoformat(args.cutoff)

    out = predict(cutoff, args.horizon)
    print(f"\nклиентов с историей покупок: {out.height:,}")

    # Оценка качества, если окно таргета доступно
    lf = pl.scan_parquet(TRAIN_PARQUET)
    if lf.collect_schema()["event_date"] != pl.Date:
        lf = lf.with_columns(pl.col("event_date").cast(pl.Date))
    end = cutoff + dt.timedelta(days=args.horizon)
    tgt = (lf.filter((pl.col("event_date") >= cutoff) & (pl.col("event_date") < end))
             .group_by("user_id").agg(pl.col("gmv").sum().alias("target")).collect())
    j = out.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    if j["target"].sum() == 0:
        print("окно таргета вне данных — оценка не считается")
    else:
        y = np.log1p(j["target"].to_numpy()); p = np.log1p(np.clip(j["e_gmv"].to_numpy(), 0, None))
        al = p - p.mean() + y.mean()
        print(f"\nRMSLE сырой      {rmse_log(y, p):.5f}")
        print(f"RMSLE выровненный {rmse_log(y, al):.5f}")
        print(f"Gini             {gini_norm(j['target'].to_numpy(), np.expm1(p)):.4f}")
        buy = (j["target"].to_numpy() > 0).astype(float)
        pz = j["p_zero"].to_numpy()
        print(f"\nкалибровка вероятности покупки (предсказано против факта):")
        for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
            m = (1 - pz >= lo) & (1 - pz < hi)
            if m.sum() > 100:
                print(f"  P(покупка) в [{lo:.1f},{hi:.1f}): предсказано "
                      f"{(1 - pz)[m].mean():.3f}, фактически {buy[m].mean():.3f}, "
                      f"клиентов {m.sum():,}")
    if args.save:
        np.savez_compressed(args.save, **{c: out[c].to_numpy() for c in out.columns})
        print(f"\nсохранено в {args.save}")


if __name__ == "__main__":
    main()
