"""Сравнение двух рук сети по сохранённым валидационным предсказаниям.

Зачем отдельный скрипт. Сеть печатает СЫРОЙ RMSLE, а сравнивать модели можно
только по выровненному (PLAN.md, раздел 2): уровень предсказаний на сабмите
правится бесплатным сдвигом из TEST_LEVEL, поэтому «повезло с уровнем»
в зачёт не идёт. Ловушка уже четырежды завышала выводы команды, а на
распределительной голове она перевернула знак: сырое число говорило -0.00168,
выровненное +0.00063.

Второе; сравнивать надо ТО, ЧТО ПОЙДЁТ В СОСТАВ. В состав идёт среднее трёх
сидов, а не одиночный прогон, и это не одно и то же: усреднение выигрывает
тем больше, чем сильнее сиды расходятся между собой. У распределительной
головы они расходятся сильнее, поэтому по одиночным сидам январь давал ноль,
а по усреднённой руке — плюс.

    python -u src/seq_cmp.py --cut 2026-01-15 --a ctl_s42 --b b64_s42
    python -u src/seq_cmp.py --cut 2026-01-15 --a ctl_s42,ctl_s13,ctl_s7 \
                                              --b b64_s42,b64_s13,b64_s7
"""
from __future__ import annotations

import argparse

import numpy as np

from config import MODELS
from metrics import gini_norm, rmse_log


def load(names: list[str], cut: str):
    """Предсказания рук в log1p и общий таргет; несколько имён — среднее сидов."""
    preds, y = [], None
    for n in names:
        path = MODELS / f"{n}_valpred_{cut}.npz"
        if not path.exists():
            raise SystemExit(f"нет {path.name} — прогон делался без --save-val-pred?")
        z = np.load(path)
        preds.append(z["pred_log"].astype(np.float64))
        t = np.log1p(z["target"].astype(np.float64))
        if y is not None and not np.allclose(t, y):
            raise SystemExit(f"{n}: таргет не совпадает с предыдущими — разные срезы?")
        y = t
    return np.mean(preds, axis=0), y, preds


def describe(tag: str, names: list[str], cut: str) -> tuple[float, np.ndarray]:
    p, y, preds = load(names, cut)
    # Выравнивание — ровно то, что делает бесплатный сдвиг на сабмите.
    aligned = p - p.mean() + y.mean()
    score = rmse_log(y, aligned)
    singles = [rmse_log(y, q - q.mean() + y.mean()) for q in preds]
    print(f"  {tag:<10} выровненный {score:.5f} | Gini {gini_norm(np.expm1(y), np.expm1(p)):.4f}"
          f" | уровень {p.mean():.4f} против истинного {y.mean():.4f}")
    if len(preds) > 1:
        print(f"{'':<13}по сидам {' '.join(f'{v:.5f}' for v in singles)} | "
              f"усреднение даёт {np.mean(singles) - score:+.5f}")
    return score, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, help="валидационный срез, ГГГГ-ММ-ДД")
    ap.add_argument("--a", required=True, help="имена прогонов первой руки через запятую")
    ap.add_argument("--b", required=True, help="то же для второй руки")
    args = ap.parse_args()

    print(f"=== срез {args.cut} ===")
    sa, pa = describe("рука A", args.a.split(","), args.cut)
    sb, pb = describe("рука B", args.b.split(","), args.cut)
    d = float(np.mean((pa - pb) ** 2))
    print(f"  B к A: {sa - sb:+.5f} | расстояние D = {d:.5f} | "
          f"корреляция {np.corrcoef(pa, pb)[0, 1]:.5f}")


if __name__ == "__main__":
    main()
