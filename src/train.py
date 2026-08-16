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

from config import CUTOFF_STRIDE, HORIZON, MODELS, SEED, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import report, rmse_log
from models import GBM, Ensemble
from utils import append_csv, git_commit


def load_split(n_cutoffs: int, rebuild: bool = False, blocks: list[str] | None = None,
               val_cutoff: dt.date | None = None, explicit_train: list[dt.date] | None = None,
               stride: int | None = None, history: int | None = None, net: bool = False):
    """По умолчанию — валидация на самом свежем срезе, обучение на всех предыдущих.

    `val_cutoff` и `explicit_train` позволяют собрать нестандартную пару: например,
    обучение на низком уровне площадки и валидация на высоком — так проверяется
    устойчивость к сдвигу уровня, которого штатная валидация не видит.
    """
    cuts = train_cutoffs(n_cutoffs, stride)
    val_cut = val_cutoff or cuts[0]
    train_cuts = explicit_train if explicit_train else [c for c in cuts if c < val_cut]

    # Карантин: окно таргета обучающего среза не должно заходить в окно валидации,
    # иначе модель учится на тех же днях, по которым её проверяют. При шаге 30
    # это условие выполняется само, при шаге 14 отбрасывает ближайшие срезы.
    latest_ok = val_cut - dt.timedelta(days=HORIZON)
    dropped = [c for c in train_cuts if c > latest_ok]
    train_cuts = [c for c in train_cuts if c <= latest_ok]
    if dropped:
        print(f"карантин: отброшено срезов {len(dropped)} "
              f"({', '.join(str(c) for c in dropped)}) — их таргет пересекается с валидацией")
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {latest_ok}: увеличьте --cutoffs")
    cuts = [val_cut, *train_cuts]
    val = get_dataset(val_cut, rebuild=rebuild, blocks=blocks, history=history, net=net)
    trains = [get_dataset(c, rebuild=rebuild, blocks=blocks, history=history, net=net)
              for c in train_cuts]
    feats = feature_names(val)
    # _gap — на сколько дней срез примера отстоит от валидации. Не признак:
    # feats берётся из колонок val, поэтому в X эта колонка не попадёт.
    trains = [
        t.select(["user_id", *feats, "target"]).with_columns(
            pl.lit((val_cut - c).days).cast(pl.Int32).alias("_gap")
        )
        for c, t in zip(train_cuts, trains)
    ]
    train = pl.concat(trains, how="vertical")
    print(f"train: {train.height:,} строк ({len(train_cuts)} cutoff) | val: {val.height:,} ({val_cut})")
    return train, val, feats, cuts


def to_xy(df: pl.DataFrame, feats: list[str]):
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    return X, y


def recency_weights(train: pl.DataFrame, halflife: float) -> np.ndarray | None:
    """Вес примера падает вдвое на каждые `halflife` дней удаления от валидации.

    Два независимых опыта показали, что свежесть среза здесь дороже количества
    примеров: возврат среза, который ближе на 12 дней, дал больше, чем миллион
    дополнительных строк. Веса — прямая проверка этого наблюдения.
    """
    if not halflife:
        return None
    gap = train["_gap"].to_numpy().astype(np.float64)
    return (0.5 ** (gap / halflife)).astype(np.float32)


def fit_single(X, ylog, Xv, yvlog, feats, kind, device, rounds=6000, w=None, members=None):
    """Одна модель или ансамбль разнородных конфигураций (см. ensemble.py).

    Ансамбль выигрывает у рабочего умолчания на обоих валидационных срезах,
    тогда как отдельные конфигурации-чемпионы между срезами не переносятся.
    """
    if not members:
        m = GBM(kind=kind, task="reg", device=device, n_estimators=rounds)
        m.fit(X, ylog, Xv, yvlog, feature_names=feats, sample_weight=w)
        return m

    models = []
    for mname, mkind, params in members:
        m = GBM(kind=mkind, task="reg", device=device, n_estimators=rounds, params=params)
        m.fit(X, ylog, Xv, yvlog, feature_names=feats, sample_weight=w)
        print(f"  участник {mname:<20} итераций {m.best_iter}")
        models.append(m)
    return Ensemble(models, [n for n, _, _ in members])


def fit_two_stage(X, y, Xv, yv, feats, kind, device, rounds=6000, w=None):
    clf = GBM(kind=kind, task="bin", device=device, n_estimators=rounds)
    clf.fit(X, (y > 0).astype(np.int8), Xv, (yv > 0).astype(np.int8),
            feature_names=feats, sample_weight=w)

    pos, posv = y > 0, yv > 0
    reg = GBM(kind=kind, task="reg", device=device, n_estimators=rounds)
    reg.fit(X[pos], np.log1p(y[pos]), Xv[posv], np.log1p(yv[posv]),
            feature_names=feats, sample_weight=w[pos] if w is not None else None)
    return clf, reg


def two_stage_predict(clf: GBM, reg: GBM, X) -> np.ndarray:
    return clf.predict(X) * reg.predict(X)


def parse_net(flag: bool, names: str | None):
    """True = одна безымянная сеть, список имён = стекинг на нескольких."""
    if names:
        return [n.strip() for n in names.split(",") if n.strip()]
    return bool(flag)


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
    ap.add_argument("--stride", type=int, default=None,
                    help="шаг между срезами в днях (по умолчанию 30 = длина окна таргета)")
    ap.add_argument("--val-cutoff", default=None, help="валидационный срез, ГГГГ-ММ-ДД")
    ap.add_argument("--train-cutoffs", default=None,
                    help="явный список обучающих срезов через запятую, ГГГГ-ММ-ДД")
    ap.add_argument("--blocks", default=None,
                    help="подмножество блоков признаков через запятую (ablation)")
    ap.add_argument("--halflife", type=float, default=0,
                    help="период полураспада веса примера в днях; 0 = все срезы равнозначны")
    ap.add_argument("--ensemble", action="store_true",
                    help="одноголовую модель заменить ансамблем конфигураций из ensemble.py")
    ap.add_argument("--members", default="lgb",
                    help="состав ансамбля: lgb (рабочий), cat, mixed — см. ensemble.py")
    ap.add_argument("--net", action="store_true",
                    help="добавить предсказание сети признаком (стекинг, см. features/net.py)")
    ap.add_argument("--net-names", default=None,
                    help="имена сетей через запятую для стекинга на нескольких, например r180,ch180,w90")
    args = ap.parse_args()
    name = args.name or args.model
    blocks = parse_blocks(args.blocks)
    parse_date = dt.date.fromisoformat
    val_cutoff = parse_date(args.val_cutoff) if args.val_cutoff else None
    explicit_train = [parse_date(d.strip()) for d in args.train_cutoffs.split(",")] if args.train_cutoffs else None

    train, val, feats, cuts = load_split(args.cutoffs, rebuild=args.rebuild, blocks=blocks,
                                         val_cutoff=val_cutoff, explicit_train=explicit_train,
                                         stride=args.stride, net=parse_net(args.net, args.net_names))
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    print(f"признаков: {len(feats)} | доля покупателей в val: {(yva > 0).mean():.3%}")

    # sw, а не w: ниже `w` занято весом бленда, и одноимённая переменная
    # молча затёрла бы веса примеров перед финальным обучением.
    sw = recency_weights(train, args.halflife)
    if sw is not None:
        gaps = sorted(set(train["_gap"].to_list()))
        print("веса срезов (полураспад %g дн.): %s" % (
            args.halflife,
            ", ".join(f"{g}дн={0.5 ** (g / args.halflife):.2f}" for g in gaps)))

    members = None
    if args.ensemble:
        from ensemble import MEMBER_SETS  # noqa: PLC0415  (импорт по требованию)

        if args.members not in MEMBER_SETS:
            raise SystemExit(f"нет состава '{args.members}'; есть: {', '.join(sorted(MEMBER_SETS))}")
        members = MEMBER_SETS[args.members]
        print(f"одноголовая модель — ансамбль '{args.members}' из {len(members)} конфигураций: "
              + ", ".join(n for n, _, _ in members))

    single = fit_single(Xtr, ytr_log, Xva, yva_log, feats, args.model, args.device, args.rounds,
                        w=sw, members=members)
    p_single = single.predict(Xva)
    clf, reg = fit_two_stage(Xtr, ytr, Xva, yva, feats, args.model, args.device, args.rounds, w=sw)
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
        "blocks": blocks or "all", "net": parse_net(args.net, args.net_names),
        "best_iter": {"single": single.best_iter, "clf": clf.best_iter, "reg": reg.best_iter},
    }

    # Журнал экспериментов: одна строка на прогон, чтобы результаты команды
    # можно было сравнивать между собой, а не по скриншотам в чате.
    append_csv(
        MODELS / "experiments.csv",
        ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs", "n_features",
         "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
         "gini_blend", "sum_bias_blend", "best_iter_single", "stride", "halflife", "val_cutoff",
         "train_cutoffs", "note"],
        {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(blocks),
            "blocks": ",".join(blocks) if blocks else "all", "name": name,
            "model": args.model, "cutoffs": len(cuts) - 1, "n_features": len(feats),
            # Без этих двух колонок строка стресс-теста неотличима от штатной,
            # а сравнивать их между собой нельзя.
            "stride": args.stride or CUTOFF_STRIDE,
            "halflife": args.halflife or "",
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
        # Валидационный срез в финальном обучении — самый свежий, вес 1.
        sw_all = None if sw is None else np.concatenate([sw, np.ones(len(yva), dtype=np.float32)])
        print(f"\n--- финальное обучение на всех {len(cuts)} cutoff'ах (итераций x{scale:.2f}) ---")

        if members:
            # Каждый участник дообучается со своим числом итераций: они сильно
            # разные (77 у быстрого lr=0.05 против 207 у медленного lr=0.02).
            fitted = []
            # mname, а не name: в name лежит имя артефактов, и цикл его затирал —
            # модели сохранялись под именем последнего участника.
            for (mname, mkind, params), m in zip(members, single.models):
                fm = GBM(mkind, "reg", args.device, int(m.best_iter * scale),
                         early_stopping=0, params=params)
                fm.fit(Xall, yall_log, feature_names=feats, sample_weight=sw_all)
                print(f"  участник {mname:<20} итераций {fm.n_estimators}")
                fitted.append(fm)
            f_single = Ensemble(fitted, [n for n, _, _ in members])
        else:
            f_single = GBM(args.model, "reg", args.device, int(single.best_iter * scale),
                           early_stopping=0)
            f_single.fit(Xall, yall_log, feature_names=feats, sample_weight=sw_all)
        f_clf = GBM(args.model, "bin", args.device, int(clf.best_iter * scale), early_stopping=0)
        f_clf.fit(Xall, (yall > 0).astype(np.int8), feature_names=feats, sample_weight=sw_all)
        pos = yall > 0
        f_reg = GBM(args.model, "reg", args.device, int(reg.best_iter * scale), early_stopping=0)
        f_reg.fit(Xall[pos], yall_log[pos], feature_names=feats,
                  sample_weight=None if sw_all is None else sw_all[pos])

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
