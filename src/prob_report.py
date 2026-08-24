"""Вероятностная часть решения: можно ли верить числам, которые модель выдаёт.

Жюри назвало вероятностность отдельным плюсом, и после замены головы сети
на распределительную она у нас появилась по построению: сеть предсказывает
распределение по 64 бинам, а в сабмит идёт его среднее. Но «модель выдаёт
вероятности» и «этим вероятностям можно верить» — разные утверждения,
и второе надо мерить.

Что считается:

* **надёжность `P(y = 0)`** по децилям: обещали 30% — не купили ли ровно 30%.
  Это главный вопрос: у 46% клиентов таргет ровно ноль, и вероятность нуля —
  самая содержательная величина, которую модель может отдать бизнесу;
* **изотоническая калибровка** этой вероятности, с честной перекрёстной
  проверкой: подгонка на одной половине, замер на другой. Без этого
  калибровка всегда выглядит идеальной, потому что подогнана по тем же точкам;
* **покрытие интервалов**: 80-процентный интервал обязан накрывать 80%
  клиентов, иначе он не интервал, а украшение;
* **CRPS** — правильная функция оценки для распределения целиком. RMSLE
  оценивает только среднее и про форму ничего не говорит.

Ни одна из этих величин не влияет на RMSLE: изотоническая калибровка
вероятностей монотонна и среднее не меняет. Это материал для жюри, а не
для лидерборда, и написано так намеренно.

    python -u src/prob_report.py --name prob_jan --cut 2026-01-15
"""
from __future__ import annotations

import argparse

import numpy as np

from config import MODELS


def reliability(p0: np.ndarray, zero: np.ndarray, bins: int = 10) -> float:
    """Диаграмма надёжности по децилям обещанной вероятности. Возвращает ECE."""
    qs = np.quantile(p0, np.linspace(0, 1, bins + 1))
    qs[-1] += 1e-9
    print(f"  {'обещано':>10}{'на деле':>10}{'клиентов':>11}{'разрыв':>9}")
    ece = 0.0
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 100:
            continue
        said, was = p0[m].mean(), zero[m].mean()
        ece += m.mean() * abs(said - was)
        print(f"  {said:>10.3f}{was:>10.3f}{m.sum():>11,}{said - was:>+9.3f}")
    return ece


def isotonic_cv(p0: np.ndarray, zero: np.ndarray, seed: int = 0) -> np.ndarray:
    """Изотоническая калибровка с перекрёстной проверкой на двух половинах.

    Подгонять и мерить на одних точках нельзя: изотоническая регрессия
    воспроизводит наблюдаемые доли почти точно, и любая калибровка выглядела
    бы идеальной. Здесь половина учит, вторая проверяется, потом наоборот.
    """
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(seed)
    half = rng.random(len(p0)) < 0.5
    out = np.empty_like(p0)
    for fit_on in (half, ~half):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p0[fit_on], zero[fit_on])
        out[~fit_on] = iso.predict(p0[~fit_on])
    return out


def quantile_from_cdf(cdf: np.ndarray, centers: np.ndarray, q: float) -> np.ndarray:
    """Квантиль дискретного распределения: первый бин, где накопилось q."""
    idx = (cdf < q).sum(axis=1)
    return centers[np.clip(idx, 0, len(centers) - 1)]


def crps(proba: np.ndarray, centers: np.ndarray, y: np.ndarray) -> float:
    """CRPS для дискретного предсказания на опорных точках centers.

    Ступенчатая функция распределения, интеграл берётся по отрезкам между
    опорными точками плюс два хвоста. Форма по CDF, а не энергетическая:
    та требует двойной суммы по бинам и на 250 000 клиентов не считается.
    """
    cdf = np.cumsum(proba, axis=1)
    width = np.diff(centers)
    ind = (centers[None, :] >= y[:, None]).astype(np.float64)
    inner = (((cdf[:, :-1] - ind[:, :-1]) ** 2) * width[None, :]).sum(axis=1)
    left = np.clip(centers[0] - y, 0, None)
    right = np.clip(y - centers[-1], 0, None)
    return float((inner + left + right).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cut", required=True)
    args = ap.parse_args()

    z = np.load(MODELS / f"{args.name}_proba_{args.cut}.npz")
    proba = z["proba"].astype(np.float64)
    centers = z["centers"].astype(np.float64)
    y = np.log1p(z["target"].astype(np.float64))
    p0, zero = proba[:, 0], (y == 0).astype(np.float64)
    mean = proba @ centers

    print(f"=== {args.name}, срез {args.cut} ===")
    print(f"клиентов {len(y):,} | бинов {len(centers)} | "
          f"доля нулей на деле {zero.mean():.3f}, обещано {p0.mean():.3f}")

    print("\nнадёжность P(y = 0), как есть:")
    ece = reliability(p0, zero)
    print(f"  средний взвешенный разрыв (ECE): {ece:.4f}")

    cal = isotonic_cv(p0, zero)
    print("\nона же после изотонической калибровки (перекрёстной):")
    ece2 = reliability(cal, zero)
    print(f"  средний взвешенный разрыв (ECE): {ece2:.4f}  "
          f"(было {ece:.4f}, стало лучше в {ece / max(ece2, 1e-9):.1f} раза)")

    print("\nпокрытие центральных интервалов:")
    cdf = np.cumsum(proba, axis=1)
    for level in (0.5, 0.8, 0.9):
        lo = quantile_from_cdf(cdf, centers, (1 - level) / 2)
        hi = quantile_from_cdf(cdf, centers, 1 - (1 - level) / 2)
        cov = float(((y >= lo) & (y <= hi)).mean())
        print(f"  обещано {level:.0%}  ->  накрывает {cov:.1%}  "
              f"(средняя ширина {np.mean(hi - lo):.2f} в log1p)")

    print(f"\nCRPS: {crps(proba, centers, y):.5f}")
    print(f"для сравнения, RMSLE того же предсказания: "
          f"{np.sqrt(np.mean((mean - y) ** 2)):.5f}")
    print("\nCRPS оценивает распределение целиком, RMSLE — только его среднее.")
    print("Изотоническая калибровка на RMSLE не влияет: она монотонна "
          "и среднего не меняет.")


if __name__ == "__main__":
    main()
