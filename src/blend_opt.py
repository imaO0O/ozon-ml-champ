"""Оптимальный бленд уже отправленных файлов — считается офлайн, без сабмитов.

Если у двух решений известен public RMSLE, то результат любой их комбинации
вычисляется точно, без единой новой отправки.

В log1p-шкале для остатков r_i = y - p_i:

    MSE(w) = (1-w)^2 * MSE_1 + w^2 * MSE_2 + 2w(1-w) * C,   C = E[r_1 * r_2]

Ключ в том, что C не нужно измерять на лидерборде:

    E[(p_1 - p_2)^2] = E[(r_1 - r_2)^2] = MSE_1 + MSE_2 - 2C

а левая часть считается по файлам на диске. Отсюда

    C  = (MSE_1 + MSE_2 - D) / 2,      D = среднее (p_1 - p_2)^2
    w* = (MSE_1 - C) / (MSE_1 + MSE_2 - 2C)
    MSE* = MSE_1 - (MSE_1 - C)^2 / (MSE_1 + MSE_2 - 2C)

Оптимальный вес лежит вне [0,1], если одна модель доминирует другую — тогда
бленд бесполезен и это видно сразу, тоже без отправки.

    python src/blend_opt.py --a cand4_calib.csv --mse-a 1.6500339212 \
                            --b lgbm_ranks_calib.csv --mse-b 1.6526609501 --out blend.csv
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import SAMPLE_SUBMIT, SUBMISSIONS
from utils import append_csv, git_commit

LOG_FIELDS = ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
              "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"]


def load_log(name: str, ref_ids: np.ndarray) -> np.ndarray:
    """Предсказания в log1p-шкале, с проверкой порядка строк.

    Блендинг идёт поэлементно, а user_id в итоговый файл подставляется из
    sample_submit. Если у присланного файла другой порядок строк — а файлы
    приходят с чужих машин и из чужих скриптов, — склеятся предсказания разных
    клиентов, и это не проявится ничем, кроме плохого ответа лидерборда.
    """
    df = pl.read_csv(SUBMISSIONS / name)
    ids = df["user_id"].to_numpy()
    if len(ids) != len(ref_ids) or not np.array_equal(ids, ref_ids):
        raise SystemExit(
            f"{name}: порядок или состав user_id не совпадает с sample_submit "
            f"({len(ids):,} строк против {len(ref_ids):,}). Блендить нельзя — "
            "предсказания склеятся по разным клиентам.")
    return np.log1p(df["predict"].to_numpy().astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--mse-a", type=float, required=True, help="public RMSLE файла A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--mse-b", type=float, default=None,
                    help="public RMSLE файла B; можно опустить, если B ещё не отправлялся — "
                         "тогда обязателен --weight, состав соберётся на заданном весе, "
                         "а MSE файла B восстановится из ответа лидерборда (--solve-b)")
    ap.add_argument("--out", default=None, help="куда записать бленд")
    ap.add_argument("--weight", type=float, default=None,
                    help="вес B вместо оптимального: оптимум оценён по 50 000 клиентов "
                         "public, а применяется к 200 000 private, поэтому его имеет "
                         "смысл ужать к нулю — кривая обычно плоская, цена мала")
    args = ap.parse_args()

    ref = pl.read_csv(SAMPLE_SUBMIT)
    ref_ids = ref["user_id"].to_numpy()
    pa, pb = load_log(args.a, ref_ids), load_log(args.b, ref_ids)

    # Неотправленный партнёр: оптимум посчитать не из чего, но состав на заданном
    # весе собрать можно — а один ответ лидерборда потом восстановит MSE партнёра
    # точно, потому что среднее квадрата разности файлов известно офлайн.
    if args.mse_b is None:
        if args.weight is None:
            raise SystemExit("без --mse-b нужен --weight: оптимальный вес вычислить не из чего")
        wt = args.weight
        d_ab = float(np.mean((pa - pb) ** 2))
        print(f"A: {args.a}  RMSLE {args.mse_a:.7f}")
        print(f"B: {args.b}  на лидерборде ещё не был")
        print()
        print(f"среднее квадрата разности предсказаний: {d_ab:.6f}")
        print(f"корреляция в log1p-шкале: {np.corrcoef(pa, pb)[0, 1]:.4f}")
        print()
        print(f"состав на весе {wt:.4f}. После ответа лидерборда MSE партнёра")
        print("восстанавливается точно:")
        print(f"  MSE_C = (1-w)^2*MSE_A + w^2*MSE_B + w(1-w)*(MSE_A + MSE_B - D),  D = {d_ab:.6f}")
        print("  разрешить относительно MSE_B — одно уравнение, одна неизвестная")
        blended = np.clip(np.expm1((1 - wt) * pa + wt * pb), 0, None)
        path = SUBMISSIONS / args.out if args.out else None
        if path is None:
            raise SystemExit("нужен --out: без оптимума считать нечего, смысл только в сборке файла")
        if path.exists():
            raise SystemExit(f"{path.name} уже существует — задайте другое имя")
        pl.DataFrame({"user_id": ref["user_id"],
                      "predict": blended.astype(np.float32)}).write_csv(path)
        append_csv(SUBMISSIONS / "log.csv", LOG_FIELDS, {
            "file": args.out, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "name": "blend", "model": "blend",
            "blend_w": round(wt, 4),
            "pred_sum": round(float(blended.sum())),
            "pred_zeros": f"{(blended < 1e-6).mean():.4f}",
            "note": f"состав {args.a} ({args.mse_a:.7f}) и {args.b} (без оценки), "
                    f"вес {wt:.4f}, D={d_ab:.6f} — ожидание не вычисляется"})
        lvl = float(np.log1p(blended).mean())
        print()
        print(f"{path}")
        print(f"  сумма: {blended.sum():,.0f} | уровень log1p {lvl:.5f} "
              f"| остаточный сдвиг {2.32912 - lvl:+.5f}")
        print("  строка записана в submissions/log.csv")
        return
    m1, m2 = args.mse_a ** 2, args.mse_b ** 2
    d = float(np.mean((pa - pb) ** 2))
    c = (m1 + m2 - d) / 2
    denom = m1 + m2 - 2 * c
    w = (m1 - c) / denom
    best = max(m1 - (m1 - c) ** 2 / denom, 0.0) ** 0.5

    print(f"A: {args.a}  RMSLE {args.mse_a:.7f}")
    print(f"B: {args.b}  RMSLE {args.mse_b:.7f}")
    print(f"\nсредний квадрат разности предсказаний: {d:.6f}")
    print(f"корреляция остатков: {c / (m1 * m2) ** 0.5:.6f}")
    print(f"\nоптимальный вес B: {w:.4f}")
    print(f"ожидаемый RMSLE:   {best:.7f}  (выигрыш {args.mse_a - best:+.5f} к лучшему)")

    # Кривая по весу: если она плоская, вес можно смело ужимать к нулю —
    # это снижает зависимость от оценки по 50 000 клиентов почти даром.
    print("\nкривая по весу (точная арифметика, без отправок):")
    for wt in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        mse = (1 - wt) ** 2 * m1 + wt ** 2 * m2 + 2 * wt * (1 - wt) * c
        mark = "  <- оптимум рядом" if abs(wt - w) < 0.025 else ""
        print(f"  {wt:.2f}  {mse ** 0.5:.7f}  выигрыш {args.mse_a - mse ** 0.5:+.5f}{mark}")

    if not 0 < w < 1:
        print("\nвес вне [0,1]: одна модель доминирует, бленд бесполезен")
        return
    if args.mse_a - best < 0.0002:
        print("\nвыигрыш меньше 0.0002 — по общему правилу сабмита не стоит.")
        print("Исключение: если сабмиты всё равно не израсходовать до конца этапа,")
        print("то цена отправки нулевая, а выигрыш здесь не оценка, а арифметика.")

    if args.out:
        wt = args.weight if args.weight is not None else w
        if args.weight is not None:
            mse = (1 - wt) ** 2 * m1 + wt ** 2 * m2 + 2 * wt * (1 - wt) * c
            print(f"\nвес задан вручную: {wt:.4f} вместо оптимального {w:.4f}, "
                  f"ожидаемый RMSLE {mse ** 0.5:.7f}")
        blended = np.clip(np.expm1((1 - wt) * pa + wt * pb), 0, None)
        path = SUBMISSIONS / args.out
        if path.exists():
            raise SystemExit(f"{path.name} уже существует — задайте другое имя")
        pl.DataFrame({"user_id": ref["user_id"],
                      "predict": blended.astype(np.float32)}).write_csv(path)
        exp = ((1 - wt) ** 2 * m1 + wt ** 2 * m2 + 2 * wt * (1 - wt) * c) ** 0.5
        # Ожидание пишется в журнал ДО отправки: иначе проверка арифметики
        # превращается в подгонку объяснения под уже полученный ответ.
        append_csv(SUBMISSIONS / "log.csv", LOG_FIELDS, {
            "file": args.out, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "name": "blend", "model": "blend",
            "blend_w": round(wt, 4),
            "pred_sum": round(float(blended.sum())),
            "pred_zeros": f"{(blended < 1e-6).mean():.4f}",
            "note": f"бленд {args.a} ({args.mse_a:.7f}) и {args.b} ({args.mse_b:.7f}), "
                    f"вес {wt:.4f} (оптимум {w:.4f}), ожидается {exp:.7f}"})
        print(f"\n{path}")
        print(f"  сумма: {blended.sum():,.0f} | нулевых: {(blended < 1e-6).mean():.2%}")
        print(f"  строка записана в submissions/log.csv, ожидание {exp:.7f}")


if __name__ == "__main__":
    main()
