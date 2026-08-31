"""Зависит ли оптимальное растяжение от разногласия участников состава.

Механизм: MSE-оптимальная модель стягивает предсказание к середине тем сильнее,
чем меньше знает о клиенте. Растяжение alpha=1.0586 правит это стягивание одним
числом на всех, хотя неопределённость у клиентов различается на порядки.

Разногласие между участниками ансамбля — прямая оценка этой неопределённости, и
она у нас лежит готовая: четыре модели, обученные по-разному, на предсказуемом
клиенте сходятся, на спорном расходятся.

Проверка: разбить клиентов на три группы по разбросу между участниками и найти
оптимальное alpha в каждой отдельно. Если группы расходятся и картина
повторяется на обоих срезах — направление живое.

РЕЗУЛЬТАТ: отвергнуто.

    группа       alpha январь   alpha декабрь
    согласные        1.002          1.004
    средние          1.016          1.006
    спорные          1.012          1.016

Выигрыш от групповых alpha: +0.00002 на обоих срезах. Монотонности нет: на
декабре порядок правильный, на январе средняя группа выше спорной. Будь связь
систематической, порядок повторился бы и при малой величине.

Оговорка, которую надо знать, если кто-то захочет вернуться: общее оптимальное
alpha на валидации 1.010, а на тесте измерено 1.0586. Проверка идёт в режиме,
где эффект в шесть раз меньше, и потому недомощна. Отсутствие монотонности —
довод против связи, не зависящий от величины, и он перевесил. Но если появится
причина проверить на тесте, направление задаётся одним зондом: это
`(p - m) * нормированное разногласие`, ортогонализованное к `(p - m)` и
константе, и решается точно, как сдвиг, растяжение и кривизна.
"""
import pathlib

import numpy as np

def _need(path):
    """Внятный отказ вместо traceback.

    Скрипт — разовый разбор, и он опирается на промежуточные файлы
    конкретных прогонов. Если их нет, надо сказать какой прогон их
    делает, а не падать на open() из недр numpy: репозиторий, где
    вход валится сырым traceback, читается как сломанный.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(
            f"нет промежуточного файла {p}.\n"
            "Это разовый разбор поверх сохранённых предсказаний; сначала\n"
            "нужен прогон, который их создаёт (см. заголовок файла).")
    return path


W = np.array([0.458, 0.097, 0.097, 0.348])
FILES = {
    "2026-01-15": ("models/oofrk_2026-01-15.npz",
                   ["models/gru_rk_valpred_2026-01-15.npz",
                    "models/gru_sh_valpred_2026-01-15.npz",
                    "models/gru_w90_valpred_2026-01-15.npz"]),
    "2025-12-16": ("models/oofrk_2025-12-16.npz",
                   ["models/gru_rk_dec_valpred_2025-12-16.npz",
                    "models/gru_sh_dec_valpred_2025-12-16.npz",
                    "models/gru_w90_dec_valpred_2025-12-16.npz"]),
}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - np.clip(p, 0, None)) ** 2)))


def leveled(y, p):
    grid = np.arange(-0.35, 0.26, 0.0025)
    return p + grid[int(np.argmin([rmse(y, p + d) for d in grid]))]


def best_alpha(y, p, m, mask=None):
    """Оптимальное растяжение вокруг общего среднего m, считая только по mask."""
    sel = slice(None) if mask is None else mask
    grid = np.arange(0.90, 1.30, 0.002)
    sc = [rmse(y[sel], (m + a * (p - m))[sel]) for a in grid]
    i = int(np.argmin(sc))
    return float(grid[i]), sc[i]


for cut, (oof, nets) in FILES.items():
    d = np.load(_need(oof))
    o = np.argsort(d["user_id"])
    uid, pb, t = d["user_id"][o], d["pred_log"][o], d["target"][o]
    y = np.log1p(t)

    P = [leveled(y, pb)]
    for f in nets:
        q = np.load(_need(f))
        z = np.empty(len(uid))
        z[np.searchsorted(uid, q["user_id"])] = q["pred_log"]
        P.append(leveled(y, z))
    P = np.vstack(P)

    p = leveled(y, W @ P)
    m = p.mean()
    # Разногласие: разброс участников вокруг их взвешенного среднего.
    dis = np.sqrt((W[:, None] * (P - (W @ P)) ** 2).sum(axis=0))

    a_glob, s_glob = best_alpha(y, p, m)
    print(f"\n=== {cut} ===")
    print(f"  разногласие: медиана {np.median(dis):.4f}, "
          f"1-й и 99-й процентили {np.percentile(dis, 1):.4f} / {np.percentile(dis, 99):.4f}")
    print(f"  общее оптимальное alpha {a_glob:.3f}, RMSLE {s_glob:.5f}")

    edges = np.quantile(dis, [1 / 3, 2 / 3])
    grp = np.digitize(dis, edges)
    tot = np.zeros_like(p)
    print(f"  {'группа':<10}{'клиентов':>10}{'медиана разн.':>15}{'alpha':>9}{'RMSLE в группе':>16}")
    alphas = []
    for g in range(3):
        mask = grp == g
        a, s = best_alpha(y, p, m, mask)
        alphas.append(a)
        tot[mask] = m + a * (p[mask] - m)
        print(f"  {['согласные', 'средние', 'спорные'][g]:<10}{mask.sum():>10,}"
              f"{np.median(dis[mask]):>15.4f}{a:>9.3f}{s:>16.5f}")
    s_grp = rmse(y, tot)
    print(f"  по группам {s_grp:.5f} против общего {s_glob:.5f} -> "
          f"выигрыш {s_glob - s_grp:+.5f}")
    print(f"  разброс alpha между группами: {max(alphas) - min(alphas):.3f}")
