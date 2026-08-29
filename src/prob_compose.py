"""Предсказательное распределение для ИТОГОВОГО состава, а не для одной сети.

Пробел, который это закрывает. Распределение есть у сетей с распределительной
головой, но в сабмит идёт **состав** из пяти компонент, и у него вероятностного
выхода нет вовсе — только точечное предсказание. На вопрос «какое у вас
распределение для финального решения» ответить было нечем.

Конструкция. Форма берётся у СМЕСИ распределений всех сетей состава,
положение — у состава (его точечное предсказание лучшее).

**Смесь, а не одна сеть — правка 29.08, и она существенна.** До неё в этом
месте стояло «форма берётся у сети, она единственная её оценивает». Это было
неверно: распределительная голова есть у ВСЕХ четырёх сетей, просто
вероятности сохранялись у одной. Распределение состава заимствовало форму
у одной компоненты из четырёх.

Смесь берётся с весами состава (нормированными по сетям, у которых есть
распределение) и даёт три независимых улучшения:

    CRPS                       январь +0.00159, декабрь +0.00209
    оптимальный масштаб        1.08 на ОБОИХ срезах (было 1.12 и 1.16)
    разрыв критериев масштаба  0.44 -> 0.40 на обоих

Второе важнее первого: у одной сети масштаб приходилось подбирать под срез,
у смеси он общий. Параметр перестал быть подгоняемым.

Для каждого клиента:

    P(y = 0)      = p₀ сети, НЕ ТРОГАЕТСЯ
    центры k > 0  = m' + s·(cₖ + δ − m')

где δ подобрано так, чтобы среднее всего распределения равнялось предсказанию
состава, а s — масштаб разброса.

**Нулевой атом остаётся на месте.** `log1p(0) = 0` ровно, и там сидит 46%
клиентов; сдвинуть или растянуть эту точку значило бы утверждать, что клиент
без покупок потратил не ноль. Сдвиг применяется только к положительной части,
и δ пересчитывается через условное среднее `m' = comp / (1 − p₀)`.

Иерархическая часть. Масштаб s оценивается **по когортам** с частичным
объединением: когорта с малым числом клиентов тянется к общему s, с большим —
к своему. Это уместно именно здесь: средние у нас насыщены и улучшению
не поддаются, а **разброс по когорте оценивается надёжнее, чем по клиенту**.

Прежние когортные приоры отвергались как способ улучшить среднее. К
неопределённости они не применялись ни разу.

    python -u src/prob_compose.py --net yearprob --cut 2026-01-15
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS

EPS = 1e-12


def load_net(name: str, cut: str):
    z = np.load(MODELS / f"{name}_proba_{cut}.npz")
    return (z["proba"].astype(np.float64), z["centers"].astype(np.float64),
            np.log1p(z["target"].astype(np.float64)), z["user_id"])


def load_point(name: str, cut: str):
    d = np.load(MODELS / f"{name}_valpred_{cut}.npz")
    o = np.argsort(d["user_id"])
    return d["user_id"][o], d["pred_log"][o], np.log1p(d["target"][o])


def shift_centers(proba: np.ndarray, centers: np.ndarray, comp: np.ndarray,
                  s: np.ndarray | float):
    """Центры, сдвинутые под предсказание состава и растянутые на s.

    Нулевой бин не трогается: он стоит ровно в нуле по построению.
    Возвращает матрицу центров (клиенты × бины).
    """
    p0 = proba[:, 0]
    w = np.maximum(1.0 - p0, EPS)                      # вес положительной части
    m_pos_new = comp / w                               # нужное условное среднее
    pos = proba[:, 1:]
    c_pos = centers[1:]
    m_pos_old = (pos @ c_pos) / np.maximum(pos.sum(1), EPS)
    delta = m_pos_new - m_pos_old
    s = np.asarray(s, dtype=np.float64).reshape(-1, 1) if np.ndim(s) else s
    C = m_pos_new[:, None] + s * (c_pos[None, :] + delta[:, None] - m_pos_new[:, None])
    return np.concatenate([np.zeros((len(comp), 1)), C], axis=1)


def crps_shifted(proba: np.ndarray, C: np.ndarray, y: np.ndarray) -> float:
    """CRPS для дискретного распределения с ИНДИВИДУАЛЬНЫМИ центрами.

    Считается по определению через ступенчатую CDF: центры у каждого клиента
    свои, поэтому общей сетки нет и векторизовать через один набор точек
    нельзя. Сортируем центры внутри клиента и интегрируем (F − 1{x ≥ y})².
    """
    idx = np.argsort(C, axis=1)
    Cs = np.take_along_axis(C, idx, axis=1)
    Ps = np.take_along_axis(proba, idx, axis=1)
    F = np.cumsum(Ps, axis=1)
    dx = np.diff(Cs, axis=1)
    ind = (Cs[:, :-1] >= y[:, None]).astype(np.float64)
    inner = np.sum((F[:, :-1] - ind) ** 2 * dx, axis=1)
    # ХВОСТЫ ОБЯЗАТЕЛЬНЫ. Интеграл идёт по всей прямой, а не между крайними
    # центрами. Слева от c_min функция F равна нулю, справа от c_max — единице,
    # и там подынтегральное выражение равно единице ровно на отрезке до y.
    #
    # Без этих слагаемых CRPS падает при сжатии распределения к точке: сжимается
    # сам интервал интегрирования. В первом прогоне это дало «оптимум» на краю
    # сетки при любом её расширении — 0.70, потом 0.30, и все когорты просили
    # минимум. Признак был виден сразу: правильная функция оценки не может
    # монотонно улучшаться от вырождения распределения в точку.
    left = np.maximum(0.0, Cs[:, 0] - y)
    right = np.maximum(0.0, y - Cs[:, -1])
    return float((inner + left + right).mean())


def coverage(proba: np.ndarray, C: np.ndarray, y: np.ndarray, q: float) -> float:
    """Доля клиентов, попавших в центральный интервал уровня q."""
    idx = np.argsort(C, axis=1)
    Cs = np.take_along_axis(C, idx, axis=1)
    F = np.cumsum(np.take_along_axis(proba, idx, axis=1), axis=1)
    lo_i = (F < (1 - q) / 2).sum(1).clip(0, C.shape[1] - 1)
    hi_i = (F < 1 - (1 - q) / 2).sum(1).clip(0, C.shape[1] - 1)
    lo = np.take_along_axis(Cs, lo_i[:, None], 1)[:, 0]
    hi = np.take_along_axis(Cs, hi_i[:, None], 1)[:, 0]
    return float(((y >= lo) & (y <= hi)).mean())


def fit_scale(proba, centers, comp, y, grid) -> tuple[float, float]:
    """Масштаб разброса по минимуму CRPS."""
    best = (None, np.inf)
    for s in grid:
        v = crps_shifted(proba, shift_centers(proba, centers, comp, s), y)
        if v < best[1]:
            best = (s, v)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="имя прогона с сохранённым распределением")
    ap.add_argument("--point", required=True, help="имя точечного предсказания состава")
    ap.add_argument("--cut", required=True)
    ap.add_argument("--cohorts", type=int, default=10,
                    help="сколько когорт по величине предсказания")
    ap.add_argument("--k", type=float, default=2000.0,
                    help="сила ужатия к общему масштабу: s_когорты = "
                         "(n·s_своё + k·s_общее)/(n + k)")
    args = ap.parse_args()

    proba, centers, y, uid_n = load_net(args.net, args.cut)
    o = np.argsort(uid_n)
    proba, y = proba[o], y[o]
    uid_p, comp, y2 = load_point(args.point, args.cut)
    if not np.array_equal(uid_n[o], uid_p):
        raise SystemExit("сеть и состав посчитаны на разных наборах клиентов")
    # состав приводится к уровню цели: сравниваем формы, а не сдвиги
    comp = comp - comp.mean() + y.mean()

    print(f"=== распределение состава, срез {args.cut} ===")
    print(f"клиентов {len(y):,} | бинов {proba.shape[1]} | "
          f"нулей на деле {float((y == 0).mean()):.3f}, обещано {float(proba[:, 0].mean()):.3f}")

    base_C = shift_centers(proba, centers, comp, 1.0)
    print(f"\nбез калибровки разброса (s = 1):")
    print(f"  CRPS {crps_shifted(proba, base_C, y):.5f}")
    for q in (0.5, 0.8, 0.9):
        print(f"  покрытие {q:.0%}: {coverage(proba, base_C, y, q):.3f}")

    # Сетка начинается с 0.30, а не с 0.70: в первом прогоне оптимум упёрся
    # в нижний край, то есть настоящий мог лежать ниже. Оптимум на краю сетки —
    # это не оптимум, а сообщение о том, что сетка узка (тот же урок, что
    # с числом листьев у бустинга, PLAN, раздел про перевёрнутый перебор).
    grid = np.round(np.arange(0.30, 1.41, 0.02), 2)
    s_glob, crps_glob = fit_scale(proba, centers, comp, y, grid)
    # CRPS и покрытие оптимизируются РАЗНЫМИ масштабами: первый оценивает
    # распределение целиком, второе — одно его свойство. Показываем оба,
    # чтобы выбор был осознанным, а не побочным следствием критерия.
    s_cov = min(grid, key=lambda t: abs(
        coverage(proba, shift_centers(proba, centers, comp, t), y, 0.8) - 0.8))
    print(f"масштаб по попаданию в 80%% покрытия: s = {s_cov:.2f} | "
          f"CRPS {crps_shifted(proba, shift_centers(proba, centers, comp, s_cov), y):.5f}")
    print(f"\nобщий масштаб по минимуму CRPS: s = {s_glob:.2f} | CRPS {crps_glob:.5f}")
    C = shift_centers(proba, centers, comp, s_glob)
    for q in (0.5, 0.8, 0.9):
        print(f"  покрытие {q:.0%}: {coverage(proba, C, y, q):.3f}")

    # --- иерархия: масштаб по когортам с частичным объединением ---
    edges = np.quantile(comp, np.linspace(0, 1, args.cohorts + 1))
    coh = np.clip(np.searchsorted(edges, comp, side="right") - 1, 0, args.cohorts - 1)
    s_user = np.full(len(y), s_glob)
    print(f"\n{'когорта':<10}{'клиентов':>10}{'своё s':>9}{'ужатое':>9}"
          f"{'покрытие 80%':>15}")
    for c in range(args.cohorts):
        m = coh == c
        if m.sum() < 50:
            continue
        s_own, _ = fit_scale(proba[m], centers, comp[m], y[m], grid)
        n = m.sum()
        s_sh = (n * s_own + args.k * s_glob) / (n + args.k)
        s_user[m] = s_sh
        cov = coverage(proba[m], shift_centers(proba[m], centers, comp[m], s_own), y[m], 0.8)
        print(f"{c:<10}{n:>10,}{s_own:>9.2f}{s_sh:>9.3f}{cov:>15.3f}")

    C_h = shift_centers(proba, centers, comp, s_user)
    print(f"\nиерархический масштаб: CRPS {crps_shifted(proba, C_h, y):.5f} "
          f"против {crps_glob:.5f} у общего")
    for q in (0.5, 0.8, 0.9):
        print(f"  покрытие {q:.0%}: {coverage(proba, C_h, y, q):.3f}")
    np.savez_compressed(MODELS / f"{args.point}_prob_{args.cut}.npz",
                        user_id=uid_p, proba=proba.astype(np.float32),
                        centers_user=C_h.astype(np.float32), s_user=s_user,
                        target=np.expm1(y))
    print(f"\nсохранено {args.point}_prob_{args.cut}.npz "
          f"(распределение состава с иерархическим масштабом)")


if __name__ == "__main__":
    main()
