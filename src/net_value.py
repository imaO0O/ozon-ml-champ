"""Стоит ли новая сеть прогона в стекинге — проверка за секунды вместо часа.

Зачем. Добавить сеть признаком бустинга стоит шести прогонов (две руки на двух
валидационных срезах плюс пересборка кэша) — около получаса. При этом потолок
направления считается заранее и точно: если регрессия таргета на предсказания
**обеих** сетей выигрывает у регрессии на одну старую меньше порога, то
бустингу тем более взять нечего — он видит те же два столбца плюс 242 признака,
которые с ними коррелируют.

Так был закрыт стекинг на двух сетях: офлайн-оценка дала 0.00073 в среднем по
шести срезам, и последующие шесть прогонов подтвердили ноль (PLAN, раздел 4).

**Границы применимости — важно.** Оценка линейная, поэтому она честный потолок
только тогда, когда проверяемый столбец сам по себе уже хороший предсказатель:
предсказание сети монотонно связано с таргетом, и деревьям сверх линейной
комбинации взять почти нечего (это и подтвердилось на шести прогонах).

Для **сырых внутренних величин** — координат скрытого состояния, эмбеддингов,
любых осей, по отдельности ничего не предсказывающих, — вывод обратный:
регрессия меряет лучшую линейную комбинацию, а бустинг возьмёт ещё и
нелинейную структуру. Там результат этого скрипта — **нижняя граница**, и
низкое значение направление НЕ закрывает. Проверять только прогоном.

Как читать вывод:
  выигрыш   — насколько оптимальная комбинация двух сетей лучше одной старой;
              это **потолок**, бустинг обычно берёт меньше;
  t(новая)  — значимость вклада новой сети; большое t при мелком выигрыше
              означает «эффект есть, но он ничтожен», а не «стоит брать»;
  корр.     — корреляция предсказаний сетей. Ниже 0.97 — повод присмотреться
              даже при скромном выигрыше: непохожесть важна для бленда.

Порог решения — 0.002, тот же, что и везде: разницу мельче валидация не
различает (PLAN, раздел 2).

    python -u src/net_value.py --base w90 --new sh
    python -u src/net_value.py --base w90 --new r180 --cutoffs 3
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import MODELS, TEST_CUTOFF, train_cutoffs
from datasets import get_dataset
from features.net import net_path

THRESHOLD = 0.002


def load(name: str, cutoff: dt.date) -> pl.DataFrame:
    path = net_path(cutoff, TEST_CUTOFF, name)
    if not path.exists():
        raise SystemExit(f"нет файла сети '{name}' для среза {cutoff}: ожидается {path}")
    with np.load(path) as z:
        return pl.DataFrame({
            "user_id": z["user_id"].astype(np.int64),
            name: z["pred_log"].astype(np.float64),
        })


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Остаток МНК — считаем через lstsq, матрицы здесь крошечные."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="имя работающей сети, например w90")
    ap.add_argument("--new", required=True, help="имя проверяемой сети")
    ap.add_argument("--cutoffs", type=int, default=6)
    args = ap.parse_args()

    print(f"{'срез':<12} {'вес новой':>10} {'одна':>9} {'обе':>9} {'выигрыш':>9} "
          f"{'t(новая)':>9} {'корр.':>7}")
    gains = []
    for cutoff in sorted(train_cutoffs(args.cutoffs)):
        base, new = load(args.base, cutoff), load(args.new, cutoff)
        df = (get_dataset(cutoff).select(["user_id", "target"])
              .join(base, on="user_id", how="inner")
              .join(new, on="user_id", how="inner"))
        y = np.log1p(df["target"].to_numpy())
        b, n = df[args.base].to_numpy(), df[args.new].to_numpy()

        one = np.column_stack([np.ones_like(y), b])
        both = np.column_stack([np.ones_like(y), b, n])
        res_one, res_both = fit(one, y), fit(both, y)
        r_one = float(np.sqrt(res_one @ res_one / len(y)))
        r_both = float(np.sqrt(res_both @ res_both / len(y)))

        # значимость коэффициента при новой сети — обычная t-статистика МНК
        beta, *_ = np.linalg.lstsq(both, y, rcond=None)
        sigma2 = res_both @ res_both / (len(y) - both.shape[1])
        se = np.sqrt(sigma2 * np.diag(np.linalg.inv(both.T @ both)))
        gains.append(r_one - r_both)
        print(f"{str(cutoff):<12} {beta[2]:>10.3f} {r_one:>9.5f} {r_both:>9.5f} "
              f"{gains[-1]:>9.5f} {beta[2] / se[2]:>9.1f} "
              f"{float(np.corrcoef(b, n)[0, 1]):>7.4f}")

    mean = float(np.mean(gains))
    print(f"\nлинейный потолок: {mean:.5f} (среднее по {len(gains)} срезам)")
    if mean < THRESHOLD:
        print(f"вывод: ниже порога {THRESHOLD} — прогон в стекинге не окупится,")
        print("если проверяемая величина сама по себе предсказывает таргет")
        print("(предсказание другой сети, готовый скор). Для сырых внутренних")
        print("осей — скрытого состояния, эмбеддингов — это нижняя граница:")
        print("бустинг возьмёт ещё и нелинейную структуру, закрывать нельзя.")
    else:
        print(f"вывод: выше порога {THRESHOLD} — сеть стоит прогнать в стекинге:")
        print("  python -u src/tune.py --cutoffs 6 --net-names "
              f"{args.base},{args.new} --params <рабочие>")
        print("  то же с --val-cutoff 2025-12-16; принимать только при плюсе на обоих")


if __name__ == "__main__":
    main()
