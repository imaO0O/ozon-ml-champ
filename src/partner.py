"""Вклад кандидата в состав выпуклой смесью, против пола дублирования.

Отбор партнёра — не «лучше ли он состава», а «добавляет ли он к составу».
Величина меряется так: состав и кандидат выравниваются по истинному уровню
окна, затем ищется вес w в отрезке [0, 1], минимизирующий MSE смеси
(1-w)*состав + w*кандидат. Выигрыш — разница RMSLE до и после.

Почему вес выпуклый, а не безусловный. Безусловный оптимум w = C^-1 c даёт
двойнику, отличающемуся только точностью арифметики, вес 0.387 и вклад
+0.00022 — он эксплуатирует разницу в последнем бите. Выпуклые веса
(неотрицательные, сумма единица) снижают это вдвое на январе и обнуляют на
декабре. Половина прежнего порога была свойством решателя, а не данных.

Пол дублирования — вклад заведомо пустой руки, прогона-двойника той же
конфигурации с другой точностью арифметики. Вклад ниже пола результатом
не является.

    python -u src/partner.py --cand models/gru_w52_h256l2_valpred_2026-01-15.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from config import MODELS, ROOT
from metrics import rmse_log
from utils import append_csv, git_commit

# Пол дублирования, измеренный двойником fp32/bf16 (см. C1_measurement.md).
FLOOR = {dt.date(2026, 1, 15): 0.00011, dt.date(2025, 12, 16): 0.00001}

HANDOFF = {
    dt.date(2026, 1, 15): ROOT / "handoff_trackC_jan.npz",
    dt.date(2025, 12, 16): ROOT / "handoff_trackC_dec.npz",
}


def aligned(pred: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Уровень правится на сабмите бесплатным сдвигом, значит сравнивать надо
    выровненные версии — иначе «непохожесть» окажется разницей уровней."""
    return pred - pred.mean() + y_log.mean()


def load_composition(cutoff: dt.date, key: str = "composition_C"):
    d = np.load(HANDOFF[cutoff])
    order = np.argsort(d["user_id"])
    return (d["user_id"][order].astype(np.int64),
            d[key][order].astype(np.float64),
            np.log1p(d["target"][order].astype(np.float64)))


def load_candidate(path: Path, users: np.ndarray):
    """Кандидат выравнивается по строкам состава: у npz свой порядок."""
    d = np.load(path)
    order = np.argsort(d["user_id"])
    cu, cp = d["user_id"][order].astype(np.int64), d["pred_log"][order].astype(np.float64)
    pos = np.searchsorted(cu, users)
    pos = np.clip(pos, 0, len(cu) - 1)
    ok = cu[pos] == users
    if not ok.all():
        raise SystemExit(f"{path.name}: нет предсказания для {(~ok).sum():,} пользователей")
    return cp[pos]


def convex_gain(comp: np.ndarray, cand: np.ndarray, y_log: np.ndarray, steps: int = 2001):
    """Точный минимум по одному параметру в отрезке.

    Перебор, а не формула: формула дала бы и отрицательный вес, то есть
    экстраполяцию прочь от кандидата, а это ровно то, что мы забраковали.
    """
    base = rmse_log(y_log, comp)
    ws = np.linspace(0.0, 1.0, steps)
    best_w, best = 0.0, base
    for w in ws:
        r = rmse_log(y_log, (1.0 - w) * comp + w * cand)
        if r < best:
            best_w, best = float(w), r
    return base, best, best_w


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """D между откалиброванными версиями — средний квадрат расхождения."""
    return float(np.mean((a - b) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True,
                    help="npz кандидата на январе; декабрьский ищется по имени "
                         "(суффикс _dec перед _valpred), либо задайте --cand-dec")
    ap.add_argument("--cand-dec", default=None)
    ap.add_argument("--key", default="composition_C")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    jan, dec = dt.date(2026, 1, 15), dt.date(2025, 12, 16)
    jan_path = Path(args.cand)
    if args.cand_dec:
        dec_path = Path(args.cand_dec)
    else:
        dec_path = Path(str(jan_path).replace("_valpred_2026-01-15", "_dec_valpred_2025-12-16"))
    name = args.name or jan_path.stem.replace("_valpred_2026-01-15", "")

    print(f"=== вклад кандидата {name} в состав ({args.key}) ===")
    rows = []
    for cutoff, path in ((jan, jan_path), (dec, dec_path)):
        if not path.exists():
            print(f"  {cutoff}: нет файла {path.name} — срез пропущен")
            continue
        users, comp_raw, y_log = load_composition(cutoff, args.key)
        cand_raw = load_candidate(path, users)
        comp, cand = aligned(comp_raw, y_log), aligned(cand_raw, y_log)

        base, best, w = convex_gain(comp, cand, y_log)
        solo = rmse_log(y_log, cand)
        d = distance(comp, cand)
        floor = FLOOR[cutoff]
        gain = base - best
        rows.append((cutoff, base, solo, d, w, gain, floor))
        verdict = "выше пола" if gain > floor else "НИЖЕ ПОЛА"
        print(f"  {cutoff}: состав {base:.5f} | кандидат сам {solo:.5f} | "
              f"D {d:.5f} | вес {w:.3f} | вклад {gain:+.5f} | пол {floor:.5f} — {verdict}")

    if len(rows) == 2:
        g1, g2 = rows[0][5], rows[1][5]
        f1, f2 = rows[0][6], rows[1][6]
        both = g1 > f1 and g2 > f2
        print(f"\n  Вклад выше пола на обоих срезах: {'ДА' if both else 'НЕТ'} "
              f"({g1:+.5f} и {g2:+.5f})")
        if not both:
            print("  Направление как партнёр не принимается: правило требует "
                  "положительного знака на обоих срезах.")

    for cutoff, base, solo, d, w, gain, floor in rows:
        append_csv(
            MODELS / "experiments.csv",
            ["created", "commit", "name", "model", "val_cutoff", "rmsle_single",
             "rmsle_blend", "blend_w", "note"],
            {"created": dt.datetime.now().isoformat(timespec="seconds"),
             "commit": git_commit(), "name": f"partner_{name}", "model": "blend",
             "val_cutoff": str(cutoff), "rmsle_single": round(solo, 5),
             "rmsle_blend": round(base - gain, 5), "blend_w": round(w, 3),
             "note": (f"вклад в состав {gain:+.5f} при поле {floor:.5f}, D {d:.5f}, "
                      f"состав без него {base:.5f}. {args.note}")},
        )


if __name__ == "__main__":
    main()
