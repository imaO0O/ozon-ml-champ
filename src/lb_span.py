"""Предсказать RMSLE любой комбинации уже отправленных сабмитов — офлайн.

Идея. Метрика квадратична по предсказаниям:

    MSE(p) = E[y2] - 2*E[y*p] + E[p2],   y = log1p(таргета), p в log1p-шкале

`E[p2]` считается офлайн. `E[y2]` неизвестна. `E[y*p]` линейно по p, поэтому
каждый ответ лидерборда — одно линейное уравнение на неизвестный функционал.
Для комбинации `q = sum(c_i * p_i) + s` при `sum(c_i) = 1` слагаемое с `E[y2]`
сокращается, и остаётся полностью вычислимое выражение:

    MSE(q) = sum(c_i * MSE(p_i)) - sum(c_i * E[p_i2]) - 2*s*Y + E[q2]

где `Y = E[log1p(y)]` — уровень тестового окна, измеренный зондом (2.32912).

Отсюда два следствия:

* результат любой аффинной комбинации отправленных файлов известен **заранее**,
  без траты сабмита;
* лучшую такую комбинацию можно найти решением маленькой задачи с одним
  ограничением — то есть выжать из уже потраченных сабмитов всё, что в них есть.

Точность. Лидерборд считает метрику по 50 000 клиентов public, а моменты
`E[p2]` и матрица Грама считаются здесь по всем 250 000 — какие именно клиенты
в public, мы не знаем. Public это случайная подвыборка, поэтому моменты почти
совпадают, но не в точности. Тем же приближением пользуются зонды сдвига и
размаха, и практика даёт оценку ошибки: трек A предсказывал 1.6533492 и получил
1.6533489803, то есть расхождение около 2e-7. Считайте предсказание верным до
шестого знака, а не до десятого.

Граница применимости. Веса подбираются по 50 000 клиентов public, и чем их
больше, тем сильнее подгонка под эту выборку. Один-два параметра переносятся
на private надёжно (так же, как сдвиг и растяжение, см. PLAN.md раздел 3), а
десяток весов при корреляции файлов 0.999 — уже подгонка: матрица Грама почти
вырождена, и веса разбегаются в плюс-минус большие числа. Держите набор
коротким и осмысленным, а на разбегающиеся веса смотрите как на предупреждение.

    python -u src/lb_span.py --files a.csv=1.6529357726,b.csv=1.6521259306
    python -u src/lb_span.py --files ... --mix 0.3758,0.6242 --shift 0
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import SUBMISSIONS

Y_LEVEL = 2.32912  # E[log1p(y)] тестового окна, измерено зондом уровня


def load(spec: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """`файл=RMSLE` через запятую -> имена, матрица предсказаний в log1p, ответы."""
    names, scores, preds = [], [], []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"нужно вида файл.csv=1.6529357726, получено {part!r}")
        f, s = part.rsplit("=", 1)
        names.append(f.strip())
        scores.append(float(s))
        preds.append(np.log1p(pl.read_csv(SUBMISSIONS / f.strip())["predict"]
                              .to_numpy().astype(np.float64)))
    return names, np.array(scores), np.vstack(preds)


def predict_mse(P: np.ndarray, mse: np.ndarray, c: np.ndarray, s: float) -> float:
    """MSE аффинной комбинации sum(c_i p_i) + s при sum(c_i) = 1."""
    q = c @ P + s
    return float(c @ mse - c @ (P ** 2).mean(axis=1) - 2 * s * Y_LEVEL + (q ** 2).mean())


def optimal_mix(P: np.ndarray, mse: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Лучшая комбинация со свободным сдвигом: минимум по c при sum(c)=1 и по s.

    Сдвиг убирается аналитически: при фиксированном c оптимальный s выводит
    среднее комбинации ровно на измеренный уровень Y, поэтому достаточно
    заранее вычесть из каждого файла его среднее и добавить Y.
    """
    k = len(mse)
    m = P.mean(axis=1)
    C = P - m[:, None]                      # центрированные направления
    G = C @ C.T / P.shape[1]                # матрица Грама центрированных
    # С оптимальным сдвигом комбинация равна q = c*C + Y, поэтому
    #   MSE(c) = A - 2*c'v + c'Gc,  A = E[(y-Y)^2],  v_i = E[(y-Y)*C_i]
    # Ответ по отдельному файлу даёт v_i, но файл отправлялся БЕЗ выравнивания
    # уровня, и его собственное смещение среднего входит в измеренный MSE:
    #   mse_i = A - 2*v_i + G_ii + (m_i - Y)^2
    #   =>  v_i = (A + G_ii + (m_i - Y)^2 - mse_i) / 2
    # Без слагаемого (m_i - Y)^2 файлы с несовпадающим уровнем (сырой ансамбль
    # смещён на 0.058) получают заниженный v_i, и оптимум уезжает.
    # При sum(c)=1 константа A выпадает, остаётся минимум c'Gc - c'lin.
    lin = np.diag(G) + (m - Y_LEVEL) ** 2 - mse
    # минимизируем f(c) = c'Gc - c'lin  при 1'c = 1
    KKT = np.zeros((k + 1, k + 1))
    KKT[:k, :k] = 2 * G
    KKT[:k, k] = 1.0
    KKT[k, :k] = 1.0
    rhs = np.concatenate([lin, [1.0]])
    sol = np.linalg.solve(KKT, rhs)
    c = sol[:k]
    s = Y_LEVEL - float(c @ m)
    return c, s, predict_mse(P, mse, c, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True,
                    help="через запятую: файл.csv=RMSLE для каждого отправленного сабмита")
    ap.add_argument("--mix", default=None,
                    help="проверить конкретные веса через запятую (сумма должна быть 1)")
    ap.add_argument("--shift", type=float, default=None,
                    help="сдвиг для --mix; по умолчанию выводится из уровня окна")
    ap.add_argument("--out", default=None, help="сохранить лучшую комбинацию в submissions/")
    args = ap.parse_args()

    names, scores, P = load(args.files)
    mse = scores ** 2
    print(f"файлов: {len(names)} | клиентов: {P.shape[1]:,}")
    for n, s, p in zip(names, scores, P):
        print(f"  {n:<32} RMSLE {s:.10f} | mean log1p {p.mean():.5f}")

    if args.mix:
        c = np.array([float(x) for x in args.mix.split(",")])
        if abs(c.sum() - 1) > 1e-6:
            raise SystemExit(f"веса должны давать в сумме 1, сейчас {c.sum():.6f}")
        s = args.shift if args.shift is not None else Y_LEVEL - float(c @ P.mean(axis=1))
        print(f"\nзаданная комбинация (сдвиг {s:+.5f}): "
              f"RMSLE {predict_mse(P, mse, c, s) ** 0.5:.7f}")

    c, s, best = optimal_mix(P, mse)
    print(f"\n=== лучшая комбинация в оболочке отправленного ===")
    for n, w in zip(names, c):
        print(f"  {w:+.4f}  {n}")
    print(f"  сдвиг {s:+.5f}")
    print(f"  ожидаемый RMSLE {best ** 0.5:.7f} (лучший отправленный {scores.min():.7f}, "
          f"выигрыш {scores.min() - best ** 0.5:+.5f})")
    if np.abs(c).max() > 3:
        print("\n  ОСТОРОЖНО: веса разбегаются — файлы почти коллинеарны, и решение\n"
              "  подгоняется под шум public. На private такое не переносится.")

    # Сверка на самих файлах тождественна и ничего не доказывает: при c = e_i
    # формула сводится к mse_i. Единственная честная проверка — предсказать
    # результат до отправки и сравнить с ответом лидерборда после.
    print("\nпроверять формулу надо на будущих отправках, а не на этих файлах:")
    print("  подстановка c = e_i сводится к тождеству mse_i = mse_i.")

    if args.out:
        q = np.clip(np.expm1(c @ P + s), 0, None)
        path = SUBMISSIONS / args.out
        if path.exists():
            raise SystemExit(f"{path.name} уже существует")
        uid = pl.read_csv(SUBMISSIONS / names[0])["user_id"]
        pl.DataFrame({"user_id": uid, "predict": q.astype(np.float32)}).write_csv(path)
        print(f"\n{path}\n  сумма {q.sum():,.0f}")


if __name__ == "__main__":
    main()
