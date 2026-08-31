"""Свежий срез с укороченным горизонтом таргета.

Замысел. Полный 30-дневный таргет существует только у cutoff'ов не позже
2026-01-15: данные кончаются 2026-02-13. Поэтому последние 15 дней лога
работают исключительно входом, и самый свежий обучающий пример отстоит от
тестового окна на 30 дней.

Но 15-дневный таргет у среза 2026-01-30 существует: его окно [01-30, 02-14)
целиком внутри известных данных и кончается ровно на TEST_CUTOFF. Такой срез
на 15 дней свежее любого полного и карантин не нарушает по построению —
его окно заканчивается там, где начинается предсказываемое.

Чем это НЕ является. Плотные срезы шагом 15 проверены и отвергнуты: при
30-дневном таргете окна соседних срезов делят одни покупки. Здесь окно само
укорочено до 15 дней, поэтому таргет свежего среза не пересекается с тем,
что предсказывается. Пересечение остаётся с окном последнего полного среза,
и оно есть и в реальном случае — поэтому в валидационной реплике оно
воспроизводится, а не устраняется.

Как соединяются два горизонта:

  feature  — колонка `horizon` (30 или 15) подаётся признаком;
  scale    — 15-дневный таргет переносится на шкалу 30-дневного квантильным
             отображением положительной части (ноль остаётся нулём),
             отображение считается на тех же обучающих срезах.

Три руки, и третья обязательна: свежий срез отличается от базовой руки ДВУМЯ
вещами — он свежее и он лишние строки. Контрольная рука добавляет такой же
частичный срез на 30 дней раньше, то есть не свежее имеющихся. Если B и C
дают одно и то же, дело в количестве строк, а не в свежести.

    python -u src/fresh.py --mode feature
    python -u src/fresh.py --mode scale --seeds 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from tqdm.auto import tqdm

from config import HORIZON, MODELS, SEED
from datasets import dataset_path, features_version, parse_blocks
from ens_size import iter_bar
from features import build_features, scan_log
from metrics import rmse_log
from models import LGB_REG
from train import load_split, to_xy
from utils import append_csv, git_commit


def fresh_cutoff(anchor: dt.date, h: int) -> dt.date:
    """Срез, чьё окно таргета кончается ровно на anchor — карантин соблюдён точно."""
    return anchor - dt.timedelta(days=h)


def short_target(cutoff: dt.date, h: int, lf: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Сумма gmv за [cutoff, cutoff + h) — то же, что features.build_target, но короче.

    Считается здесь, а не параметром в features/, намеренно: любая правка
    внутри пакета признаков меняет их хеш и обесценивает кэш команды целиком.
    """
    lf = scan_log() if lf is None else lf
    end = cutoff + dt.timedelta(days=h)
    return (
        lf.filter((pl.col("event_date") >= cutoff) & (pl.col("event_date") < end))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("target"))
        # in-memory, а не streaming: потоковый движок суммирует внутри группы в
        # недетерминированном порядке, и две сборки подряд расходятся. Замерено
        # здесь же, срез 2025-12-31: streaming даёт max |разницу| 3.64e-12,
        # in-memory ровно ноль. Двенадцатый знак не пыль -- трек A измерил, что
        # через хаотичность бустинга он стоит 0.00086 RMSLE (та же конфигурация
        # после пересборки кэша встала на 144-й итерации вместо 152-й).
        # Признаки намеренно остаются на streaming: там движок нужен по памяти,
        # а детерминизм и так есть. Окно цели короткое, потоковый режим тут
        # ничего не экономит.
        .collect(engine="in-memory")
    )


def get_partial(cutoff: dt.date, h: int, blocks=None, rebuild: bool = False) -> pl.DataFrame:
    """Выборка с h-дневным таргетом. Кэш отдельный: ключ содержит _hz{h}.

    Собирается здесь, а не в datasets.py, чтобы не трогать features/ — любая
    правка там меняет хеш версии признаков и обесценивает весь кэш команды.
    """
    path = dataset_path(cutoff, blocks, None, False, "rank_centered", h)
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    t0 = time.time()
    lf = scan_log()
    df = build_features(cutoff, lf, blocks=blocks)
    tgt = short_target(cutoff, h, lf)
    df = (df.join(tgt, on="user_id", how="left")
            .with_columns(pl.col("target").fill_null(0.0).cast(pl.Float64),
                          pl.lit(cutoff).alias("cutoff")))
    df.write_parquet(path)
    print(f"[{cutoff} h={h}] {df.height:,} x {df.width} за {time.time() - t0:.1f}s -> {path.name}")
    return df


def quantile_map(y_short: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Положительную часть короткого таргета — на шкалу опорного, ноль в ноль.

    Монотонное отображение не может размазать атом в нуле, поэтому атомы
    сопоставляются атомам: покупок за 15 дней нет -> считаем, что нет и за 30.
    Это вносит ошибку у клиентов, купивших только во второй половине окна, —
    цена за то, что маргинальное распределение таргета остаётся правильным.
    """
    out = np.zeros_like(y_short, dtype=np.float64)
    pos = y_short > 0
    if not pos.any():
        return out
    ref_pos = np.sort(y_ref[y_ref > 0])
    if ref_pos.size == 0:
        return out
    vals = y_short[pos]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = (np.arange(order.size) + 0.5) / order.size
    ref_q = (np.arange(ref_pos.size) + 0.5) / ref_pos.size
    out[pos] = np.interp(ranks, ref_q, ref_pos)
    return out


def build_arm(Xb: np.ndarray, yb: np.ndarray, feats: list[str],
              Xe: np.ndarray | None, ye: np.ndarray | None, mode: str, h: int):
    """Матрица и таргет одной руки поверх уже готовой базовой матрицы.

    Собирается в предвыделенный буфер, а не через vstack/hstack: на 1.45 млн
    строк x 243 признака каждая лишняя копия стоит 1.4 ГБ, а свободной памяти
    на машине около семи. Прогон на 6000 итераций падал именно на этом.
    """
    if mode not in ("feature", "scale"):
        raise SystemExit(f"неизвестный режим {mode!r}")
    add_col = mode == "feature"
    names = list(feats) + (["horizon"] if add_col else [])
    nb, nf = len(Xb), Xb.shape[1]
    ne = 0 if Xe is None else len(Xe)

    X = np.empty((nb + ne, nf + int(add_col)), dtype=np.float32)
    X[:nb, :nf] = Xb
    y = np.empty(nb + ne, dtype=np.float64)
    y[:nb] = yb
    if add_col:
        X[:nb, nf] = HORIZON
    if ne:
        X[nb:, :nf] = Xe
        if add_col:
            X[nb:, nf] = h
        y[nb:] = ye if add_col else quantile_map(ye, yb)
    return X, np.log1p(y), names


def fit_arm(X, ylog, Xv, yvlog, names, seed, rounds, early, desc):
    """Одна посадка. Зовётся не через `models.GBM`, а напрямую — по той же
    причине, что и `ens_size.fit_member`: GBM не принимает свой callback, а без
    него посадка на четыре минуты идёт без признаков жизни. Параметры берутся
    из того же `LGB_REG`, поэтому разойтись с рабочей моделью они не могут.
    """
    base = {**LGB_REG, "seed": seed}
    dtrain = lgb.Dataset(X, label=ylog, feature_name=names)
    dvalid = lgb.Dataset(Xv, label=yvlog, reference=dtrain)

    prog = iter_bar(desc)

    def on_iter(env):
        prog.update(1)
        if env.evaluation_result_list:
            prog.set_postfix_str(f"rmse {env.evaluation_result_list[0][2]:.5f}")

    on_iter.before_iteration = False
    on_iter.order = 30
    cbs = [on_iter]
    if early:
        cbs.insert(0, lgb.early_stopping(early, verbose=False))
    model = lgb.train(base, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                      callbacks=cbs)
    prog.close()
    best = model.best_iteration or rounds
    pred = model.predict(Xv, num_iteration=best)
    raw = rmse_log(yvlog, pred)
    shift = float(yvlog.mean() - pred.mean())
    return raw, rmse_log(yvlog, pred + shift), best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="feature", choices=["feature", "scale"])
    ap.add_argument("--horizon", type=int, default=15, help="длина укороченного окна")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--only", default=None, help="только один срез: январь|декабрь")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    blocks = parse_blocks(args.blocks)
    h = args.horizon

    cases = [("январь", dt.date(2026, 1, 15), 6), ("декабрь", dt.date(2025, 12, 16), 7)]
    if args.only:
        cases = [c for c in cases if c[0] == args.only]

    for label, val_cut, n_cut in cases:
        fresh_cut = fresh_cutoff(val_cut, h)
        ctl_cut = fresh_cut - dt.timedelta(days=HORIZON)
        print(f"\n{'=' * 62}")
        print(f"=== {label}: валидация {val_cut} | свежий срез {fresh_cut} | "
              f"контроль {ctl_cut} (h={h}, режим {args.mode})")

        train, val, feats, cuts = load_split(n_cut, blocks=blocks, val_cutoff=val_cut)
        Xva, yva = to_xy(val, feats)
        yva_log = np.log1p(yva)
        Xb = train.select(feats).to_numpy().astype(np.float32)
        yb = train["target"].to_numpy().astype(np.float64)
        del train, val
        gc.collect()

        def extra_of(cut: dt.date):
            df = get_partial(cut, h, blocks)
            Xe = df.select(feats).to_numpy().astype(np.float32)
            ye = df["target"].to_numpy().astype(np.float64)
            share = float((ye > 0).mean())
            del df
            gc.collect()
            return Xe, ye, share

        slices = tqdm(total=2, desc="  частичные срезы", unit="срез", disable=None,
                      leave=False, dynamic_ncols=True,
                      bar_format="  {desc}: {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
        Xf, yf, share_f = extra_of(fresh_cut)
        slices.update(1)
        slices.set_postfix_str(f"свежий {fresh_cut}")
        Xc, yc, _ = extra_of(ctl_cut)
        slices.update(1)
        slices.close()
        tqdm.write(f"  свежий +{len(Xf):,} строк, покупателей {share_f:.3%} "
                   f"(в валидации за 30 дней {(yva > 0).mean():.3%})")

        Xv = Xva
        if args.mode == "feature":
            Xv = np.empty((len(Xva), Xva.shape[1] + 1), dtype=np.float32)
            Xv[:, :Xva.shape[1]] = Xva
            Xv[:, -1] = HORIZON
            del Xva
            gc.collect()

        arms = [("A база", None, None), ("B свежий", Xf, yf), ("C контроль", Xc, yc)]
        total = len(arms) * args.seeds
        outer = tqdm(total=total, desc="  посадки", unit="модель", disable=None,
                     leave=False, dynamic_ncols=True,
                     bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}{postfix}]")
        table: dict[str, list[float]] = {}
        done = 0
        for tag, Xe, ye in arms:
            X, ylog, names = build_arm(Xb, yb, feats, Xe, ye, args.mode, h)
            got = []
            for s in range(args.seeds):
                t0 = time.time()
                seed = SEED + 101 * s
                done += 1
                raw, al, it = fit_arm(X, ylog, Xv, yva_log, names, seed,
                                      args.rounds, args.early_stopping,
                                      f"[{done}/{total}] {tag} сид {seed}")
                got.append(al)
                outer.update(1)
                outer.set_postfix_str(f"{tag} {al:.5f}")
                tqdm.write(f"  {tag:<12} сид {seed:<4} строк {len(X):>9,} | "
                           f"RMSLE {raw:.5f} | выровненный {al:.5f} | "
                           f"итераций {it} | {time.time() - t0:.0f}s")
            table[tag] = got
            del X, ylog
            gc.collect()
        outer.close()

        base = float(np.mean(table["A база"]))
        print(f"\n  --- {label}, выровненный RMSLE (среднее {args.seeds} сид.) ---")
        for tag in ("A база", "B свежий", "C контроль"):
            v = float(np.mean(table[tag]))
            spread = ""
            if args.seeds > 1:
                spread = f" | разброс {max(table[tag]) - min(table[tag]):.5f}"
            print(f"  {tag:<12} {v:.5f}   выигрыш {base - v:+.5f}{spread}")
        gain_b = base - float(np.mean(table["B свежий"]))
        gain_c = base - float(np.mean(table["C контроль"]))
        print(f"  свежесть сверх количества строк: {gain_b - gain_c:+.5f}")

        append_csv(
            MODELS / "experiments.csv",
            # `train_cutoffs` пишется не для порядка, а чтобы прогон был
            # ПРОВЕРЯЕМ: `src/leak_audit.py` сверяет по этой колонке, что окно
            # цели обучения не достаёт до валидационного среза. Без неё восемь
            # прогонов A/B/C выпадали из аудита утечки — не как подозрительные,
            # а как непроверяемые, что хуже.
            ["created", "commit", "feat_ver", "name", "model", "note",
             "val_cutoff", "train_cutoffs", "rmsle_single", "rmsle_two_stage",
             "rmsle_blend"],
            {"created": dt.datetime.now().isoformat(timespec="seconds"),
             "commit": git_commit(), "feat_ver": features_version(blocks),
             "name": f"fresh_{args.mode}_h{h}", "model": "lgbm",
             "val_cutoff": str(val_cut),
             # Частичные срезы записываются ТОЖЕ, с суффиксом горизонта: без
             # них аудит не видел именно те два среза, ради которых прогон и
             # ставился. Срез val-15 при горизонте 30 выглядел бы нарушением,
             # при своём 15 кончается ровно на валидации — суффикс отличает
             # одно от другого, не выводя прогон в исключения.
             "train_cutoffs": " ".join(
                 [str(c) for c in cuts if c < val_cut]
                 + [f"{fresh_cut}@{h}", f"{ctl_cut}@{h}"]),
             "rmsle_single": round(base, 5),
             "rmsle_two_stage": round(float(np.mean(table["B свежий"])), 5),
             "rmsle_blend": round(float(np.mean(table["C контроль"])), 5),
             "note": (f"A/B/C выровненные: база/свежий/контроль, {args.seeds} сид. "
                      f"{args.note}")},
        )


if __name__ == "__main__":
    main()
