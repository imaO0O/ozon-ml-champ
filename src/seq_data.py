"""Плотная матрица `пользователь x день x канал` для последовательностных моделей.

Зачем отдельный формат. Бустингам нужны агрегаты, и `features/` считает их
ленивым `group_by`, не разворачивая сетку. Сети нужна ровно эта сетка: форма
поведения по дням, а не свёрнутые в суммы объёмы. Разворачивать её на каждый
cutoff заново — это ноутбук организаторов и его «40GB+ RAM»; вместо этого
матрица строится **один раз на весь период**, а срез любого cutoff'а получается
обычным слайсом по оси дней. Шесть обучающих срезов достаются бесплатно.

Размер: 250 000 x 409 x 12 в float16 — около 2.5 ГБ на диске. Читается как
`np.memmap`, поэтому в оперативную память целиком не поднимается: батч берёт
только свои строки, а горячая часть оседает в файловом кэше ОС.

Каналы. Из лога взяты 12 колонок: три денежных, семь счётчиков и два флага.
Четыре флага `has_*` не хранятся — они выводятся из счётчиков (`has_x = x > 0`)
и заняли бы треть объёма впустую. Деньги и счётчики кладутся в log1p: сеть
получает величины одного порядка, а `expm1` для метрики нигде не нужен, так как
таргет тоже живёт в log1p-шкале.

Утечки будущего здесь нет по построению: матрица содержит только то, что было
в логе, а окно всегда обрывается на дне, предшествующем cutoff'у (см.
`window_bounds`). Ни одна функция этого модуля не смотрит на таргет.

    python -u src/seq_data.py            # собрать (или показать готовую)
    python -u src/seq_data.py --rebuild  # пересобрать с нуля
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from config import DATA_END, DATA_PROC, DATA_START, TRAIN_PARQUET
from features import scan_log

# Порядок каналов фиксирован: он попадает в веса обученной сети, менять его
# задним числом нельзя — иначе старые чекпойнты молча начнут читать не те данные.
CHANNELS = [
    "gmv", "gmv_search", "gmv_cat",                     # деньги
    "to_ord", "to_cart", "searches",                    # счётчики общие
    "search_to_ord", "search_to_cart",                  # счётчики поиска
    "cat_to_ord", "cat_to_cart",                        # счётчики каталога
    "search", "cat",                                    # флаги «был в поиске / каталоге»
]
LOG_CHANNELS = CHANNELS[:10]  # флаги в log1p не переводим — они и так 0/1

N_DAYS = (DATA_END - DATA_START).days + 1
N_CHANNELS = len(CHANNELS)


def seq_path() -> "tuple":
    """Матрица и её спутник с индексом пользователей.

    Имя включает источник данных (реальные/синтетика) и форму, поэтому прогон
    на `fake_train.parquet` не затрёт настоящую матрицу и наоборот.
    """
    stem = f"seq_{TRAIN_PARQUET.stem}_{N_DAYS}x{N_CHANNELS}"
    return DATA_PROC / f"{stem}.npy", DATA_PROC / f"{stem}.meta.npz"


def day_index(d: dt.date) -> int:
    """Номер дня в матрице. Для cutoff'а это индекс *первого неизвестного* дня."""
    return (d - DATA_START).days


def window_bounds(cutoff: dt.date, lookback: int) -> "tuple[int, int]":
    """Границы окна `[d0, d1)` строго до cutoff'а. d0 может быть отрицательным.

    d1 = day_index(cutoff) — сам день cutoff'а в окно не входит: таргет считается
    начиная с него, и попадание этого дня в признаки было бы утечкой.
    Отрицательный d0 означает, что окно длиннее доступной истории; недостающие
    дни добиваются нулями, а канал `observed` отличает их от настоящего затишья.
    """
    d1 = day_index(cutoff)
    return d1 - lookback, d1


def build(rebuild: bool = False, chunks: int = 8) -> "tuple":
    """Собрать матрицу. Возвращает (memmap, user_ids, first_day, last_day).

    Пишется по диапазонам пользователей, а не по датам: раскладка массива
    C-порядковая, поэтому вся история одного пользователя лежит в файле подряд,
    и запись диапазоном идёт последовательно, а не случайными точками по 2.5 ГБ.
    """
    path, meta_path = seq_path()
    if path.exists() and meta_path.exists() and not rebuild:
        return open_seq()

    lf = scan_log()
    t0 = time.time()
    users = (
        lf.select(pl.col("user_id").unique())
        .collect(engine="streaming")["user_id"]
        .sort()
        .to_numpy()
    )
    n_users = len(users)
    print(f"пользователей в логе: {n_users:,} | дней: {N_DAYS} | каналов: {N_CHANNELS}")
    print(f"размер матрицы: {n_users * N_DAYS * N_CHANNELS * 2 / 1024 ** 3:.2f} ГБ (float16)")

    # Границы активности каждого пользователя — одним проходом, чтобы потом не
    # пересчитывать «есть ли история до cutoff'а» по самой матрице.
    bounds = (
        lf.with_columns(
            (pl.col("event_date") - pl.lit(DATA_START)).dt.total_days().cast(pl.Int32).alias("d")
        )
        .group_by("user_id")
        .agg(pl.col("d").min().alias("first"), pl.col("d").max().alias("last"))
        .collect(engine="streaming")
        .sort("user_id")
    )
    first_day = bounds["first"].to_numpy().astype(np.int32)
    last_day = bounds["last"].to_numpy().astype(np.int32)

    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(n_users, N_DAYS, N_CHANNELS)
    )

    edges = np.linspace(0, n_users, chunks + 1).astype(np.int64)
    rows_total = 0
    for i in range(chunks):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if lo == hi:
            continue
        u_lo, u_hi = int(users[lo]), int(users[hi - 1])
        df = (
            lf.filter((pl.col("user_id") >= u_lo) & (pl.col("user_id") <= u_hi))
            .with_columns(
                (pl.col("event_date") - pl.lit(DATA_START)).dt.total_days().cast(pl.Int32).alias("_d"),
                *[pl.col(c).cast(pl.Float32).log1p() for c in LOG_CHANNELS],
            )
            .select(["user_id", "_d", *CHANNELS])
            .collect(engine="streaming")
        )
        if df.height == 0:
            continue
        # searchsorted, а не join: users отсортирован, и это дешевле любого join'а.
        ui = np.searchsorted(users, df["user_id"].to_numpy())
        di = df["_d"].to_numpy()
        vals = df.select(CHANNELS).to_numpy().astype(np.float16)
        arr[ui, di] = vals
        rows_total += df.height
        print(f"  [{i + 1}/{chunks}] пользователи {lo:,}..{hi:,} | строк {df.height:,}")
        del df, ui, di, vals

    arr.flush()
    np.savez(meta_path, users=users, first_day=first_day, last_day=last_day,
             channels=np.array(CHANNELS), n_days=N_DAYS)
    dens = rows_total / (n_users * N_DAYS)
    print(f"\n{path}")
    print(f"строк лога: {rows_total:,} | заполненность сетки: {dens:.1%} | "
          f"{time.time() - t0:.0f}s")
    return open_seq()


def open_seq() -> "tuple":
    """Открыть готовую матрицу только на чтение."""
    path, meta_path = seq_path()
    if not (path.exists() and meta_path.exists()):
        raise SystemExit(f"нет матрицы {path.name} — соберите: python -u src/seq_data.py")
    meta = np.load(meta_path, allow_pickle=False)
    arr = np.load(path, mmap_mode="r")
    return arr, meta["users"], meta["first_day"], meta["last_day"]


def gather(seq, rows: np.ndarray, cutoff: dt.date, lookback: int) -> np.ndarray:
    """Окна нескольких пользователей: (len(rows), lookback, N_CHANNELS + 1).

    Последний канал — `observed`: 1 для дней, попавших в наблюдаемый период,
    0 для добитых нулями. Без него сеть не отличает «пользователь молчал» от
    «данных за этот день вообще нет», а это разные вещи: у старых срезов история
    короче, чем у тестового (229 дней против 409), и без явного признака модель
    выучила бы эту разницу как поведение клиента.
    """
    d0, d1 = window_bounds(cutoff, lookback)
    n = len(rows)
    out = np.zeros((n, lookback, N_CHANNELS + 1), dtype=np.float32)
    pad = max(0, -d0)
    src_lo = max(0, d0)
    if d1 > src_lo:
        out[:, pad:, :N_CHANNELS] = seq[rows, src_lo:d1]
        out[:, pad:, N_CHANNELS] = 1.0
    return out


def history_mask(first_day: np.ndarray, cutoff: dt.date) -> np.ndarray:
    """Пользователи, у которых есть хотя бы одно событие до cutoff'а.

    Это ровно тот набор, на котором обучается и меряется бустинг: `build_features`
    оставляет пользователей с непустой историей, а `predict.py` добивает
    остальных нулями уже на этапе сабмита.
    """
    return first_day < day_index(cutoff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--chunks", type=int, default=8,
                    help="на сколько диапазонов пользователей резать проход (меньше = больше RAM)")
    args = ap.parse_args()

    seq, users, first_day, last_day = build(rebuild=args.rebuild, chunks=args.chunks)
    print(f"\nматрица: {seq.shape} {seq.dtype}")
    print(f"пользователей: {len(users):,} | первый день активности: медиана {np.median(first_day):.0f}")
    from config import TEST_CUTOFF, train_cutoffs

    for c in train_cutoffs(6) + [TEST_CUTOFF]:
        d0, d1 = window_bounds(c, 180)
        print(f"  {c}: окно 180 дней -> дни [{d0}, {d1}) | "
              f"с историей {history_mask(first_day, c).sum():,}")


if __name__ == "__main__":
    main()
