"""Кто из наших файлов окупается как партнёр к текущему рекорду. Ноль сабмитов."""
import sys, csv, io
sys.path.insert(0, "src")
import numpy as np, polars as pl
from config import SAMPLE_SUBMIT, SUBMISSIONS

REC = "mix_c2.csv"
solved = {"gru_dr90_avg3.csv": 1.6515714, "stack_dr_sh_sh.csv": 1.6527429}

scores = {}
with io.open("submissions/log.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["lb_score"]:
            scores[row["file"]] = float(row["lb_score"])
scores.update(solved)

ref = pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy()
def load(f):
    df = pl.read_csv(SUBMISSIONS / f)
    if not np.array_equal(df["user_id"].to_numpy(), ref):
        return None
    return np.log1p(df["predict"].to_numpy().astype(np.float64))

pr = load(REC); m1 = scores[REC] ** 2
rows = []
for f, r in sorted(scores.items(), key=lambda kv: kv[1]):
    if f == REC or not (SUBMISSIONS / f).exists():
        continue
    p = load(f)
    if p is None:
        print(f"  {f}: другой порядок user_id, пропущен"); continue
    m2 = r * r
    D = float(np.mean((pr - p) ** 2))
    delta = m2 - m1
    C = (m1 + m2 - D) / 2
    w = (m1 - C) / (m1 + m2 - 2 * C)
    mse = m1 - (m1 - C) ** 2 / (m1 + m2 - 2 * C)
    rows.append((f, r, delta, D, D - delta, w, np.sqrt(mse), scores[REC] - np.sqrt(mse)))

print(f"рекорд {REC} = {scores[REC]:.7f}, партнёров проверено {len(rows)}\n")
print(f"{'файл':<26}{'public':>11}{'δ':>10}{'D':>10}{'D-δ':>10}{'вес':>8}{'пара':>12}{'выигрыш':>10}")
for f, r, d, D, diff, w, mse, g in sorted(rows, key=lambda t: -t[7]):
    mark = "  <-- окупается" if (diff > 0 and 0 < w < 1 and g > 0) else ""
    print(f"{f.replace('.csv',''):<26}{r:>11.7f}{d:>10.5f}{D:>10.5f}{diff:>10.5f}{w:>8.4f}{mse:>12.7f}{g:>+10.5f}{mark}")
