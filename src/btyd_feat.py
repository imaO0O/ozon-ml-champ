"""Величины BG/NBD признаками бустинга — направление, закрытое не тем измерением.

Почему пересматривается. Порождающая модель (`src/btyd.py`) была отвергнута
как источник признаков на основании `net_value.py`, который дал **линейный
потолок +0.00054** при t = 9.8. Но в нашем же README записано прямо: линейный
потолок — честная оценка сверху для **готового предсказания** и **нижняя
граница** для сырых величин. Сигнал признан статистически различимым (t = 9.8)
и закрыт по величине, посчитанной не тем инструментом: деревья извлекают
нелинейное, а мерили только линейное.

Что подаётся, пять величин на клиента:

    p_alive   вероятность, что клиент ещё не ушёл
    e_trans   ожидаемое число покупок в следующие 30 дней
    e_value   ожидаемый чек одной покупки, стянутый к популяции
    e_gmv     произведение двух предыдущих
    p_zero    вероятность нулевого GMV в окне

Почему это может быть не пересказом уже имеющегося. Все наши признаки —
описательные: сколько, когда, как часто. BG/NBD — **порождающая** модель
с явным partial pooling: клиент с двумя покупками не получает наивную оценку
2/T, он стягивается к популяционной тем сильнее, чем меньше о нём известно.
Такого преобразования у бустинга нет, и вывести его из счётчиков он не может:
это не комбинация признаков, а результат оценки максимума правдоподобия.

Утечки нет: параметры и величины считаются по данным строго до своего cutoff'а
(`src/btyd.py`), отдельно для каждого среза.

    python -u src/btyd_feat.py --cutoffs 6
    python -u src/btyd_feat.py --cutoffs 7 --val-cutoff 2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from config import MODELS, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log
from models import LGB_REG
from utils import append_csv, git_commit

LOG = MODELS / "experiments.csv"
FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]
COLS = ("p_alive", "e_trans", "e_value", "e_gmv", "p_zero")


def aligned(y_log: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y_log, p - p.mean() + y_log.mean())


def with_btyd(cutoff, blocks) -> pl.DataFrame:
    """Датасет среза плюс пять величин BG/NBD, выровненных по user_id.

    Клиентов, которых нет в файле BG/NBD (у них не было ни одной покупки
    до среза, и модель на них не определена), помечаем −1 вместо нуля:
    ноль здесь означал бы «оценка равна нулю», а верно «оценки не существует»,
    и это разные вещи. Дерево различит их по любому признаку активности.
    """
    df = get_dataset(cutoff, blocks=blocks)
    path = MODELS / f"btyd_{cutoff}.npz"
    if not path.exists():
        raise SystemExit(f"нет {path.name} — сначала: "
                         f"python -u src/btyd.py --cutoff {cutoff} --save {path}")
    z = np.load(path)
    add = pl.DataFrame({"user_id": z["user_id"],
                        **{f"bt_{c}": z[c].astype(np.float32) for c in COLS}})
    df = df.join(add, on="user_id", how="left")
    return df.with_columns([pl.col(f"bt_{c}").fill_null(-1.0) for c in COLS])


def fit(Xtr, ytr, Xv, yv, feats, rounds, stop, lr, leaves):
    import lightgbm as lgb
    params = {**LGB_REG, "learning_rate": lr, "num_leaves": leaves}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(stop, verbose=False),
                             lgb.log_evaluation(200)])
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="btyd")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-val-pred", action="store_true")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {val_cut}")
    blocks = parse_blocks(args.blocks)

    dfv = with_btyd(val_cut, blocks)
    base = [f for f in feature_names(dfv) if not f.startswith("bt_")]
    both = base + [f"bt_{c}" for c in COLS]
    print(f"валидация {val_cut} | обучение на {len(train_cuts)}")
    print(f"базовых признаков {len(base)}, добавляем {len(COLS)} величин BG/NBD")

    parts = [with_btyd(c, blocks) for c in train_cuts]
    Xv_b = dfv.select(base).to_numpy().astype(np.float32)
    Xv_a = dfv.select(both).to_numpy().astype(np.float32)
    yv = dfv["target"].to_numpy().astype(np.float64)
    Xt_b = np.vstack([d.select(base).to_numpy().astype(np.float32) for d in parts])
    Xt_a = np.vstack([d.select(both).to_numpy().astype(np.float32) for d in parts])
    ytr = np.concatenate([d["target"].to_numpy().astype(np.float64) for d in parts])
    ytr_log, yv_log = np.log1p(ytr), np.log1p(yv)
    miss = float((dfv["bt_p_alive"].to_numpy() < 0).mean())
    print(f"обучающих строк {len(ytr):,} | без оценки BG/NBD {miss:.1%} клиентов")

    print("\n--- контроль: без BG/NBD ---")
    p_b, it_b, _ = fit(Xt_b, ytr_log, Xv_b, yv_log, base,
                       args.rounds, args.early_stopping, args.lr, args.leaves)
    print("\n--- рука: с величинами BG/NBD ---")
    p_a, it_a, m_a = fit(Xt_a, ytr_log, Xv_a, yv_log, both,
                         args.rounds, args.early_stopping, args.lr, args.leaves)

    print(f"\n{'рука':<18}{'сырой':>10}{'выровн.':>10}{'Gini':>9}{'итераций':>10}")
    rows = []
    for tag, p, it in (("без BG/NBD", p_b, it_b), ("с BG/NBD", p_a, it_a)):
        r, a = rmse_log(yv_log, p), aligned(yv_log, p)
        g = gini_norm(yv, np.expm1(p))
        rows.append((tag, r, a, g, it, p))
        print(f"{tag:<18}{r:>10.5f}{a:>10.5f}{g:>9.4f}{it:>10}")
    print(f"\nBG/NBD к контролю по выровненному: {rows[0][2] - rows[1][2]:+.5f}")

    imp = sorted(zip(both, m_a.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(g for _, g in imp)
    share = sum(g for f, g in imp if f.startswith("bt_")) / tot
    print(f"доля новых признаков во всём выигрыше: {share:.1%}")
    for f, g in [t for t in imp if t[0].startswith("bt_")]:
        print(f"  {f:<16}{g:>14,.0f}  (место {imp.index((f, g)) + 1} из {len(imp)})")

    for tag, r, a, g, it, p in rows:
        append_csv(LOG, FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": features_version(),
            "blocks": args.blocks or "all", "name": f"{args.name}_{tag.split()[0]}",
            "model": "lgbm", "cutoffs": len(train_cuts),
            "n_features": len(base) if tag.startswith("без") else len(both),
            "rmsle_single": round(r, 5), "rmsle_blend": round(r, 5),
            "gini_blend": round(g, 4), "best_iter_single": it,
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
            "note": f"{args.note} [BG/NBD признаками, выровненный {a:.5f}]",
        })
    print(f"\nзаписано в {LOG}")

    if args.save_val_pred:
        ids = dfv["user_id"].to_numpy()
        for tag, p in (("base", rows[0][5]), ("btyd", rows[1][5])):
            np.savez_compressed(MODELS / f"{args.name}_{tag}_valpred_{val_cut}.npz",
                                user_id=ids, pred_log=p, target=yv)


if __name__ == "__main__":
    main()
