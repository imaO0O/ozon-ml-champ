"""Намеренно урезанные бустинги — партнёры, а не чемпионы.

Откуда идея. Закон партнёрства (PLAN, «расстояние ничего не значит, значит
направление») проверен дважды: семь альтернативных функций потерь дали
расстояние на два порядка больше при нулевом направлении и не заплатили
ничего, а годовая сеть при скромном расстоянии 0.026 и направлении −0.044
принесла рекорд. Полезен не непохожий, а тот, чьё отличие направлено ПРОТИВ
нашей ошибки.

Третье подтверждение нашлось случайно. Сети, обученные по ошибке БЕЗ
статических рангов, хуже базы на 0.037 — по обычным меркам брак, я их и
забраковал как испорченный замер. Но они дают +0.00074 сверх годовой сети,
потому что, лишённые рангов, вынуждены опираться на одну последовательность
и потому ошибаются иначе.

Отсюда приём: **калечить намеренно**. Если отнять у бустинга целое семейство
признаков, он станет хуже, но его ошибки сместятся в другую сторону. Вопрос
единственный и тот же: вырастет ли отставание быстрее направления.

Важно, чего здесь НЕ проверяется. Это не отбор признаков и не вклад блоков —
такие замеры у нас есть, и они отвечают на вопрос «что нужно чемпиону».
Здесь вопрос обратный: «что можно отнять, чтобы получился полезный партнёр».
Рука, проигравшая как замена, может выиграть как партнёр — ровно это и
случилось с годовой историей (TASKS, П5).

    python -u src/handicap.py --base models/stk2_jan_valpred_2026-01-15.npz
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log
from models import LGB_REG, aligned_rmsle
from utils import append_csv, git_commit

LOG = MODELS / "experiments.csv"
FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]

SHORT = (7, 14, 30)
LONG = (90, 180, 365)


def groups(feats: list[str]) -> dict[str, list[str]]:
    """Разбиение признаков на семейства, каждое из которых можно отнять."""
    rk = [f for f in feats if f.startswith("rk_")]
    lt = [f for f in feats if f.startswith("lt_")]
    short = [f for f in feats if any(f.endswith(f"_{w}") for w in SHORT)]
    long_ = [f for f in feats if any(f.endswith(f"_{w}") for w in LONG)]
    return {"ранги": rk, "пожизненные": lt, "короткие окна": short,
            "длинные окна": long_}


def arms(feats: list[str]) -> list[tuple[str, list[str]]]:
    g = groups(feats)
    out = [("контроль", feats)]
    for name, cols in g.items():
        keep = [f for f in feats if f not in set(cols)]
        if keep and len(keep) < len(feats):
            out.append((f"без «{name}»", keep))
    # Крайний случай: только ранги. Уровень площадки из них вычтен по
    # построению, поэтому такая рука видит ТОЛЬКО положение клиента среди
    # прочих и ничего об абсолютных величинах.
    if g["ранги"]:
        out.append(("только ранги", g["ранги"]))
    return out


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def fit(Xtr, ytr, Xv, yv, feats, rounds, stop):
    import lightgbm as lgb
    params = {**LGB_REG, "metric": "None"}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  feval=aligned_rmsle,
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(0)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


def partner(base: np.ndarray, arm: np.ndarray, y: np.ndarray):
    """Вес и выигрыш партнёра. ОБЕ величины держим в MSE.

    Именно здесь мы уже ошиблись однажды: сравнили D в единицах MSE
    с отставанием в единицах RMSLE, и направление, давшее потом рекорд,
    было отброшено (PLAN, «седьмая ошибка»). Поэтому delta считается
    явно как m2 - m1, а не как разность корней.
    """
    b = base - base.mean() + y.mean()
    a = arm - arm.mean() + y.mean()
    m1 = float(np.mean((b - y) ** 2))
    m2 = float(np.mean((a - y) ** 2))
    D = float(np.mean((b - a) ** 2))
    if D < 1e-12:
        return 0.0, 0.0, 0.0, float(np.sqrt(m1))
    delta = m2 - m1                      # ОТСТАВАНИЕ В MSE, не в RMSLE
    r = float(np.corrcoef(b - y, a - b)[0, 1])
    w = float(np.clip((m1 - m2 + D) / (2 * D), 0.0, 1.0))
    mix = (1 - w) * b + w * a
    return w, D, r, float(np.sqrt(np.mean((mix - y) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=7)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--base", default=None, help="npz с нынешним составом на этом срезе")
    ap.add_argument("--extra", default=None,
                    help="ещё один npz, подмешать к базе перед сравнением "
                         "(например годовая сеть), формат имя:вес")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="hcap")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    blocks = parse_blocks(args.blocks)

    dfv = get_dataset(val_cut, blocks=blocks)
    feats = feature_names(dfv)
    yv = dfv["target"].to_numpy().astype(np.float64)
    y = np.log1p(yv)
    ids = dfv["user_id"].to_numpy()
    order = np.argsort(ids)
    parts = [get_dataset(c, blocks=blocks) for c in train_cuts]
    ytr = np.log1p(np.concatenate(
        [d["target"].to_numpy().astype(np.float64) for d in parts]))
    print(f"валидация {val_cut} | обучение {len(ytr):,} строк | признаков {len(feats)}")

    base = None
    if args.base:
        d = np.load(args.base); o = np.argsort(d["user_id"])
        if not np.array_equal(d["user_id"][o], ids[order]):
            raise SystemExit("база на другом наборе пользователей")
        base = np.empty_like(y); base[order] = d["pred_log"][o]
        base = base - base.mean() + y.mean()
        if args.extra:
            nm, _, wt = args.extra.partition(":")
            e = np.load(nm); oe = np.argsort(e["user_id"])
            ex = np.empty_like(y); ex[order] = e["pred_log"][oe]
            ex = ex - ex.mean() + y.mean()
            w = float(wt) if wt else 0.48
            base = (1 - w) * base + w * ex
            print(f"база = состав + {nm} с весом {w}")
        print(f"база: выровненный {aligned(y, base):.5f}")

    print(f"\n{'рука':<20}{'призн.':>8}{'выровн.':>10}{'D':>9}"
          f"{'направл.':>10}{'вес':>7}{'выигрыш':>10}{'итер.':>7}")
    rows = []
    for tag, cols in arms(feats):
        Xv = dfv.select(cols).to_numpy().astype(np.float32)
        Xtr = np.vstack([d.select(cols).to_numpy().astype(np.float32) for d in parts])
        p, it = fit(Xtr, ytr, Xv, y, cols, args.rounds, args.early_stopping)
        a = aligned(y, p)
        line = f"{tag:<20}{len(cols):>8}{a:>10.5f}"
        if base is not None:
            w, D, r, mix = partner(base, p, y)
            line += f"{D:>9.4f}{r:>10.4f}{w:>7.2f}{aligned(y, base) - mix:>+10.5f}"
        print(line + f"{it:>7}")
        rows.append((tag, a, it, p, cols))

    def safe_tag(tag: str) -> str:
        """Имя строки журнала и имя файла обязаны строиться ОДНИМ правилом.

        Строилось разными: строка журнала брала последнее слово тега, файл —
        весь тег без кавычек. Теги «без короткие окна» и «без длинные окна»
        кончаются одинаково, и два разных опыта легли в журнал под одним
        именем `hcap_окна»`, различаясь только числом (1.68360 и 1.68469).
        Файлы при этом остались разными — то есть по журналу прогон было
        уже не восстановить, а по диску ещё можно.
        """
        return tag.replace("«", "").replace("»", "").replace(" ", "_")

    for tag, a, it, p, cols in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{safe_tag(tag)}",
            "model": "lgbm", "cutoffs": len(train_cuts), "n_features": len(cols),
            "rmsle_single": round(float(rmse_log(y, p)), 5),
            "rmsle_blend": round(float(rmse_log(y, p)), 5),
            "gini_blend": round(float(gini_norm(yv, np.expm1(p))), 4),
            "best_iter_single": it, "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [урезанный бустинг: {tag}, выровненный {a:.5f}]",
        })
    for tag, a, it, p, cols in rows:
        np.savez_compressed(MODELS / f"{args.name}_{safe_tag(tag)}_valpred_{val_cut}.npz",
                            user_id=ids[order], pred_log=p[order], target=yv[order])
    print(f"\nзаписано в {LOG}, предсказания сохранены")


if __name__ == "__main__":
    main()
