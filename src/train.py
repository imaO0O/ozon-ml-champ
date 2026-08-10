"""Обучение и временная валидация.

Схема валидации: обучаемся на старых cutoff'ах, валидируемся на самом свежем
(2026-01-15) — он ближе всего к тестовому окну по сезонности и по «возрасту»
пользовательской базы. Ни один таргет из валидации не попадает в обучение.

Две модели:
  single   — регрессия log1p(gmv_30d) напрямую;
  two-stage — P(покупка) * E[log1p(gmv) | покупка]. Разложение точное, т.к.
              log1p(0) = 0, поэтому E[log1p(y)] = p * E[log1p(y) | y > 0].
              Обычно лучше ранжирует (Gini), что важно как tie-breaker жюри.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import polars as pl

from config import MODELS, SEED, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import report, rmse_log
from models import GBM
from utils import append_csv, git_commit


def load_split(n_cutoffs: int, rebuild: bool = False, blocks: list[str] | None = None,
               val_cutoff: dt.date | None = None, explicit_train: list[dt.date] | None = None):
    """По умолчанию — валидация на самом свежем срезе, обучение на всех предыдущих.

    `val_cutoff` и `explicit_train` позволяют собрать нестандартную пару: например,
    обучение на низком уровне площадки и валидация на высоком — так проверяется
    устойчивость к сдвигу уровня, которого штатная валидация не видит.
    """
    cuts = train_cutoffs(n_cutoffs)
    val_cut = val_cutoff or cuts[0]
    train_cuts = explicit_train if explicit_train else [c for c in cuts if c < val_cut]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {val_cut}: увеличьте --cutoffs")
    cuts = [val_cut, *train_cuts]
    val = get_dataset(val_cut, rebuild=rebuild, blocks=blocks)
    trains = [get_dataset(c, rebuild=rebuild, blocks=blocks) for c in train_cuts]
    feats = feature_names(val)
    trains = [t.select(["user_id", *feats, "target"]) for t in trains]
    train = pl.concat(trains, how="vertical")
    print(f"train: {train.height:,} строк ({len(train_cuts)} cutoff) | val: {val.height:,} ({val_cut})")
    return train, val, feats, cuts


def to_xy(df: pl.DataFrame, feats: list[str]):
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    return X, y


def fit_single(X, ylog, Xv, yvlog, feats, kind, device, rounds=6000):
    m = GBM(kind=kind, task="reg", device=device, n_estimators=rounds)
    m.fit(X, ylog, Xv, yvlog, feature_names=feats)
    return m


def fit_two_stage(X, y, Xv, yv, feats, kind, device, rounds=6000):
    clf = GBM(kind=kind, task="bin", device=device, n_estimators=rounds)
    clf.fit(X, (y > 0).astype(np.int8), Xv, (yv > 0).astype(np.int8), feature_names=feats)

    pos, posv = y > 0, yv > 0
    reg = GBM(kind=kind, task="reg", device=device, n_estimators=rounds)
    reg.fit(X[pos], np.log1p(y[pos]), Xv[posv], np.log1p(yv[posv]), feature_names=feats)
    return clf, reg


def two_stage_predict(clf: GBM, reg: GBM, X) -> np.ndarray:
    return clf.predict(X) * reg.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "cat", "xgb"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--rebuild", action="store_true", help="пересобрать признаки, игнорируя кэш")
    ap.add_argument("--name", default=None, help="имя артефактов (по умолчанию = --model)")
    ap.add_argument("--final", action="store_true",
                    help="дообучить на train+val фиксированным числом итераций и сохранить для сабмита")
    ap.add_argument("--note", default="", help="что проверяем — попадёт в журнал экспериментов")
    ap.add_argument("--val-cutoff", default=None, help="валидационный срез, ГГГГ-ММ-ДД")
    ap.add_argument("--train-cutoffs", default=None,
                    help="явный список обучающих срезов через запятую, ГГГГ-ММ-ДД")
    ap.add_argument("--blocks", default=None,
                    help="подмножество блоков признаков через запятую (ablation)")
    args = ap.parse_args()
    name = args.name or args.model
    blocks = parse_blocks(args.blocks)
    parse_date = dt.date.fromisoformat
    val_cutoff = parse_date(args.val_cutoff) if args.val_cutoff else None
    explicit_train = [parse_date(d.strip()) for d in args.train_cutoffs.split(",")] if args.train_cutoffs else None

    train, val, feats, cuts = load_split(args.cutoffs, rebuild=args.rebuild, blocks=blocks,
                                         val_cutoff=val_cutoff, explicit_train=explicit_train)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    print(f"признаков: {len(feats)} | доля покупателей в val: {(yva > 0).mean():.3%}")

    single = fit_single(Xtr, ytr_log, Xva, yva_log, feats, args.model, args.device, args.rounds)
    p_single = single.predict(Xva)
    clf, reg = fit_two_stage(Xtr, ytr, Xva, yva, feats, args.model, args.device, args.rounds)
    p_two = two_stage_predict(clf, reg, Xva)

    print("\n--- валидация (cutoff %s) ---" % cuts[0])
    res = {}
    res["single"] = report(yva, np.expm1(p_single), "single")
    res["two_stage"] = report(yva, np.expm1(p_two), "two-stage")

    best_w, best_rmse = 0.0, float("inf")
    for w in np.linspace(0, 1, 21):
        r = rmse_log(yva_log, w * p_two + (1 - w) * p_single)
        if r < best_rmse:
            best_w, best_rmse = float(w), r
    p_blend = best_w * p_two + (1 - best_w) * p_single
    res["blend"] = report(yva, np.expm1(p_blend), f"blend w={best_w:.2f}")

    print("\ntop-20 признаков (gain, single):")
    for f, g in single.feature_importance(feats)[:20]:
        print(f"  {f:<28} {g:,.0f}")

    meta = {
        "name": name, "model": args.model, "device": args.device,
        "features": feats, "blend_w": best_w, "val_cutoff": str(cuts[0]),
        "metrics": res, "seed": SEED, "features_version": features_version(blocks),
        "blocks": blocks or "all",
        "best_iter": {"single": single.best_iter, "clf": clf.best_iter, "reg": reg.best_iter},
    }

    # Журнал экспериментов: одна строка на прогон, чтобы результаты команды
    # можно было сравнивать между собой, а не по скриншотам в чате.
    append_csv(
        MODELS / "experiments.csv",
        ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs", "n_features",
         "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
         "gini_blend", "sum_bias_blend", "best_iter_single", "val_cutoff", "train_cutoffs", "note"],
        {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(blocks),
            "blocks": ",".join(blocks) if blocks else "all", "name": name,
            "model": args.model, "cutoffs": len(cuts) - 1, "n_features": len(feats),
            # Без этих двух колонок строка стресс-теста неотличима от штатной,
            # а сравнивать их между собой нельзя.
            "val_cutoff": str(cuts[0]),
            "train_cutoffs": " ".join(str(c) for c in cuts[1:]),
            "rmsle_single": round(res["single"]["rmsle"], 5),
            "rmsle_two_stage": round(res["two_stage"]["rmsle"], 5),
            "rmsle_blend": round(res["blend"]["rmsle"], 5), "blend_w": round(best_w, 2),
            "gini_blend": round(res["blend"]["gini"], 4),
            "sum_bias_blend": round(res["blend"]["sum_bias"], 4),
            "best_iter_single": single.best_iter, "note": args.note,
        },
    )

    if args.final:
        # Данных стало больше на 1/(n-1) — примерно во столько же раз растим число итераций.
        scale = 1.0 + 1.0 / max(len(cuts) - 1, 1)
        Xall = np.vstack([Xtr, Xva])
        yall = np.concatenate([ytr, yva])
        yall_log = np.log1p(yall)
        print(f"\n--- финальное обучение на всех {len(cuts)} cutoff'ах (итераций x{scale:.2f}) ---")

        f_single = GBM(args.model, "reg", args.device, int(single.best_iter * scale), early_stopping=0)
        f_single.fit(Xall, yall_log, feature_names=feats)
        f_clf = GBM(args.model, "bin", args.device, int(clf.best_iter * scale), early_stopping=0)
        f_clf.fit(Xall, (yall > 0).astype(np.int8), feature_names=feats)
        pos = yall > 0
        f_reg = GBM(args.model, "reg", args.device, int(reg.best_iter * scale), early_stopping=0)
        f_reg.fit(Xall[pos], yall_log[pos], feature_names=feats)

        f_single.save(MODELS / f"{name}_single.pkl")
        f_clf.save(MODELS / f"{name}_clf.pkl")
        f_reg.save(MODELS / f"{name}_reg.pkl")
        meta["final"] = True
        # Мета пишется только вместе с весами: иначе прогон без --final оставлял бы
        # рядом со старыми .pkl мету от другого набора признаков.
        (MODELS / f"{name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"модели и мета сохранены в {MODELS}")
    else:
        print("прогон без --final: веса не сохранялись, результат — строкой в experiments.csv")


if __name__ == "__main__":
    main()
