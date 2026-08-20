"""Гипотеза плотности на шести точках: растёт ли выигрыш дневных рангов с плотностью.

Откуда взялась. Дневные ранги дали на январе больше, чем на декабре, и мы
объяснили это ростом плотности сетки: ранг нормирует на состав активных за
день, значит его польза должна расти по мере ухода окна от обучающих срезов.
Объяснение было подогнано к двум точкам, и мы это оговорили.

Почему проверка была важна. У гипотезы есть решающее следствие: если выигрыш
растёт с плотностью, то январская оценка ЗАНИЖАЕТ тестовую, и вес сети в
составе следует брать ВЫШЕ публичного оптимума. Это было бы единственное за
всю работу основание сознательно отклониться от оптимума вверх — во всех
остальных случаях мы вес наоборот ужимали.

ОПРОВЕРГНУТО. Шесть срезов, обе руки одним протоколом (walk-forward, три сида,
уровень выровнен):

    срез        плотность  контроль   ранги    выигрыш
    2025-08-18    25.0%    1.74757   1.74500   +0.00258
    2025-09-17    25.6%    1.72626   1.72386   +0.00240
    2025-10-17    26.8%    1.71360   1.71197   +0.00163
    2025-11-16    28.4%    1.73516   1.73438   +0.00078
    2025-12-16    30.2%    1.73883   1.73754   +0.00129
    2026-01-15    30.8%    1.67255   1.67113   +0.00141

Корреляция -0.766: выигрыш ПАДАЕТ с плотностью. Экстраполяция на тестовые
31.2% даёт +0.00094 против среднего +0.00168 по шести срезам. Основания
поднимать вес выше оптимума нет.

Осторожно с обратным выводом. Плотность спутана с двумя вещами сразу: у
старых срезов меньше обучающих данных и они дальше от теста. Модель, которая
лучше обобщает, помогает сильнее там, где данных меньше, — это объясняет тот
же наклон, ничего не говоря о плотности. Шести точек с таким смешением
недостаточно, чтобы пользоваться наклоном для поправки в любую сторону.
Честный итог: довод за повышение веса снят, довод за понижение не появился.

Побочно и важнее самой гипотезы: на трёх сидах эффект дневных рангов
положителен на ВСЕХ шести срезах, от +0.00078 до +0.00258, среднее +0.00168,
и Gini тоже растёт везде. Прежние одиночные прогоны давали +0.00421 на январе
и +0.00036 на декабре — разброс вдесятеро, и он оказался почти целиком шумом
сида. Усреднение трёх сидов его сняло.

    python -u src/density_gain.py
"""
import sys
sys.path.insert(0, "src")
import numpy as np
from config import MODELS, TEST_CUTOFF, train_cutoffs
from metrics import gini_norm, rmse_log
from seq_data import RAW_CHANNELS, build, gather, history_mask

def leveled(y, p):
    return p + (y.mean() - p.mean())

def aligned(path):
    d = np.load(MODELS / path)
    y = np.log1p(d["target"])
    p = leveled(y, d["pred_log"])
    return rmse_log(y, p), gini_norm(d["target"], np.expm1(np.clip(p, 0, None)))

seq, users, first_day, _ = build()
rng = np.random.default_rng(0)
n_raw = len(RAW_CHANNELS)

rows_out = []
for cut in train_cutoffs(6):
    r_dr, g_dr = aligned(f"netoof_dr_{cut}.npz")
    r_ct, g_ct = aligned(f"netoof_ctl_{cut}.npz")
    idx = np.nonzero(history_mask(first_day, cut))[0]
    sel = np.sort(rng.choice(idx, size=min(20000, len(idx)), replace=False))
    act = np.concatenate([
        (gather(seq, sel[i:i+4000], cut, 90)[:, :, :n_raw] != 0).any(axis=2).sum(axis=1)
        for i in range(0, len(sel), 4000)])
    rows_out.append((cut, act.mean() / 90, r_ct, r_dr, r_ct - r_dr, g_dr - g_ct))

# Плотность тестового окна — для экстраполяции.
idx = np.nonzero(history_mask(first_day, TEST_CUTOFF))[0]
sel = np.sort(rng.choice(idx, size=20000, replace=False))
act = np.concatenate([
    (gather(seq, sel[i:i+4000], TEST_CUTOFF, 90)[:, :, :n_raw] != 0).any(axis=2).sum(axis=1)
    for i in range(0, len(sel), 4000)])
dens_test = act.mean() / 90

print(f"{'срез':<13}{'плотн.':>8}{'контроль':>11}{'ранги':>10}{'выигрыш':>11}{'dGini':>9}")
for cut, dens, rc, rd, gain, dg in sorted(rows_out, key=lambda r: r[1]):
    print(f"{str(cut):<13}{dens:>7.1%}{rc:>11.5f}{rd:>10.5f}{gain:>+11.5f}{dg:>+9.4f}")

d = np.array([r[1] for r in rows_out])
g = np.array([r[4] for r in rows_out])
slope, icept = np.polyfit(d, g, 1)
corr = np.corrcoef(d, g)[0, 1]
print(f"\nтест: плотность {dens_test:.1%}")
print(f"корреляция выигрыша с плотностью: {corr:+.3f}")
print(f"наклон: {slope:+.5f} на единицу плотности ({slope/100:+.6f} на процентный пункт)")
print(f"экстраполяция на тест: {icept + slope*dens_test:+.5f}")
print(f"среднее по шести срезам:  {g.mean():+.5f}")
