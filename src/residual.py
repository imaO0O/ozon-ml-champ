"""C-1: предсказуем ли остаток нашего состава вне своего среза.

Вопрос не «можно ли улучшить состав», а «осталась ли в наших 242 признаках
информация, которой состав не воспользовался». Проверяется так: бустинг учится
на остатке состава на одном срезе и проверяется на остатке на другом. Если
обучение останавливается на первой итерации, значит переносимой структуры в
остатке нет — и направления закрыты измерением, а не перебором идей.

Остаток берётся ПОСЛЕ выравнивания уровня: сдвиг выводится из TEST_LEVEL
бесплатно, поэтому уровневая часть остатка не является ошибкой модели и её
предсказание ничего не стоило бы. Без выравнивания бустинг выучил бы разницу
уровней двух срезов и показал бы мнимый перенос.

Прогон в обе стороны намеренно. Одна сторона говорит про один остаток; две
разделяют «в остатке нет структуры» и «структура есть, но своя на каждом срезе».

    python -u src/residual.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import lightgbm as lgb
import numpy as np
import polars as pl
from tqdm.auto import tqdm

from config import MODELS, ROOT, SEED
from datasets import feature_names, get_dataset
from ens_size import iter_bar
from models import LGB_REG
from utils import append_csv, git_commit

HANDOFF = {
    dt.date(2026, 1, 15): ROOT / "handoff_trackC_jan.npz",
    dt.date(2025, 12, 16): ROOT / "handoff_trackC_dec.npz",
}


def load_residual(cutoff: dt.date, key: str = "composition_C"):
    """Остаток состава на срезе, выровненный по уровню, плюс матрица признаков.

    Строки выравниваются по user_id: в npz свой порядок, в выборке свой, и
    молчаливое несовпадение дало бы шум вместо остатка.
    """
    path = HANDOFF[cutoff]
    if not path.exists():
        raise SystemExit(f"нет файла состава {path.name} — он не отслеживается git")
    d = np.load(path)
    if key not in d.files:
        raise SystemExit(f"в {path.name} нет ключа {key!r}; есть: {', '.join(d.files)}")

    ids = pl.DataFrame({"user_id": d["user_id"].astype(np.int64),
                        "pred_log": d[key].astype(np.float64),
                        "y": np.log1p(d["target"].astype(np.float64))})
    df = get_dataset(cutoff)
    feats = feature_names(df)
    joined = df.join(ids, on="user_id", how="inner")
    if joined.height != df.height:
        print(f"  [{cutoff}] по user_id совпало {joined.height:,} из {df.height:,}")

    y = joined["y"].to_numpy()
    p = joined["pred_log"].to_numpy()
    shift = float(y.mean() - p.mean())
    r = y - (p + shift)
    X = joined.select(feats).to_numpy().astype(np.float32)
    print(f"  [{cutoff}] строк {len(r):,} | сдвиг уровня {shift:+.5f} | "
          f"std остатка {r.std():.5f} | std цели {y.std():.5f}")
    return X, r, feats


def run(train_cut: dt.date, val_cut: dt.date, rounds: int, early: int, feats_cache: dict):
    Xtr, rtr, feats = feats_cache[train_cut]
    Xva, rva, _ = feats_cache[val_cut]

    base = {**LGB_REG, "seed": SEED}
    dtrain = lgb.Dataset(Xtr, label=rtr, feature_name=feats)
    dvalid = lgb.Dataset(Xva, label=rva, reference=dtrain)
    prog = iter_bar(f"остаток {train_cut} -> {val_cut}")

    def on_iter(env):
        prog.update(1)
        if env.evaluation_result_list:
            prog.set_postfix_str(f"rmse {env.evaluation_result_list[0][2]:.5f}")

    on_iter.before_iteration = False
    on_iter.order = 30
    model = lgb.train(base, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(early, verbose=False), on_iter])
    prog.close()

    best = model.best_iteration or rounds
    pred = model.predict(Xva, num_iteration=best)
    after = rva - pred
    corr = float(np.corrcoef(pred, rva)[0, 1]) if pred.std() > 0 else 0.0
    ev = 1.0 - float(after.var() / rva.var())
    return {"train": str(train_cut), "val": str(val_cut), "iters": int(best),
            "std_before": float(rva.std()), "std_after": float(after.std()),
            "corr": corr, "explained": ev}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="composition_C", help="какой столбец npz считать составом")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    jan, dec = dt.date(2026, 1, 15), dt.date(2025, 12, 16)
    print(f"=== C-1: предсказуемость остатка состава ({args.key}) ===")
    cache = {}
    load = tqdm(total=2, desc="  срезы", unit="срез", disable=None, leave=False,
                dynamic_ncols=True,
                bar_format="  {desc}: {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
    for c in (dec, jan):
        cache[c] = load_residual(c, args.key)
        load.update(1)
    load.close()

    rows = []
    outer = tqdm(total=2, desc="  направления", unit="прогон", disable=None, leave=False,
                 dynamic_ncols=True,
                 bar_format="  {desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
    for a, b in ((dec, jan), (jan, dec)):
        r = run(a, b, args.rounds, args.early_stopping, cache)
        rows.append(r)
        outer.update(1)
        outer.set_postfix_str(f"итераций {r['iters']}")
        tqdm.write(f"  обучение {r['train']} -> проверка {r['val']}: "
                   f"итераций до остановки {r['iters']}")
    outer.close()

    print("\n  обучение   проверка   итер.  std до    std после  корр.    объясн. дисп.")
    for r in rows:
        print(f"  {r['train']}  {r['val']}  {r['iters']:>5}  "
              f"{r['std_before']:.5f}  {r['std_after']:.5f}  "
              f"{r['corr']:+.4f}  {r['explained']:+.6f}")

    stuck = [r for r in rows if r["iters"] <= 1]
    if len(stuck) == 2:
        print("\n  Обе стороны останавливаются на первой итерации: переносимой структуры "
              "в остатке состава нет. Направления на этих признаках закрыты измерением.")
    elif stuck:
        print(f"\n  Останавливается одна сторона из двух ({stuck[0]['train']} -> "
              f"{stuck[0]['val']}). Утверждение односторонее: на другом срезе "
              f"структура нашлась, значит она своя на каждом срезе, а не общая.")
    else:
        print("\n  Ни одна сторона не встала на первой итерации — остаток чем-то "
              "предсказуем. Смотреть на объяснённую дисперсию: если она "
              "отрицательна, перенос всё равно во вред.")

    for r in rows:
        append_csv(
            MODELS / "experiments.csv",
            ["created", "commit", "name", "model", "val_cutoff", "train_cutoffs",
             "best_iter_single", "rmsle_single", "note"],
            {"created": dt.datetime.now().isoformat(timespec="seconds"),
             "commit": git_commit(), "name": f"residual_{args.key}", "model": "lgbm",
             "val_cutoff": r["val"], "train_cutoffs": r["train"],
             "best_iter_single": r["iters"], "rmsle_single": round(r["std_after"], 5),
             "note": (f"C-1 остаток состава: std {r['std_before']:.5f} -> "
                      f"{r['std_after']:.5f}, корр {r['corr']:+.4f}, "
                      f"объясн.дисп {r['explained']:+.6f}. {args.note}")},
        )


if __name__ == "__main__":
    main()
