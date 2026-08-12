"""Добавляет ли сеть что-то ансамблю — проверка без единого сабмита.

Зачем. Сеть проигрывает бустингу как самостоятельная модель, но это ещё не
приговор: ансамбль зарабатывает на **непохожести ошибок**, а не на силе
участника. Именно поэтому CatBoost почти ничего не добавил (PLAN.md, раздел 4)
— он был равен LightGBM и ошибался в тех же местах. Сеть устроена принципиально
иначе, и её ошибки могут лежать в других местах даже при худшем среднем.

Что считается:

* RMSLE бустинга и сети по отдельности на одной и той же валидации;
* корреляция их предсказаний в log1p-шкале — чем ниже, тем больше надежды;
* оптимальный вес бленда и выигрыш относительно **лучшего** участника, а не
  среднего: смесь, которая хуже своего лучшего участника, бесполезна.

Бустинг обучается здесь же, на вашей машине: числа из журнала измерены на
чужом железе, а сравнивать предсказания можно только полученные в одном прогоне.
Нужен кэш признаков — `python -u src/datasets.py --test`.

Несколько `.npz` усредняются в log1p-шкале до бленда: это ансамбль сетей по
сидам, и он сам по себе обычно сильнее одиночной сети.

    python -u src/seq_blend.py --seq models/gru_fp32_valpred_2026-01-15.npz
    python -u src/seq_blend.py --seq models/gru_fp32_valpred_2026-01-15.npz,models/gru_v1_valpred_2026-01-15.npz
    python -u src/seq_blend.py --seq models/gru_fp32_dec_valpred_2025-12-16.npz --val-cutoff 2025-12-16 --cutoffs 7
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, ROOT
from metrics import report, rmse_log
from models import GBM
from seq_train import EXPERIMENT_FIELDS
from train import load_split, to_xy
from utils import append_csv, git_commit


def load_seq(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Предсказания сети из одного или нескольких .npz, усреднённые в log1p."""
    users = target = None
    preds = []
    for p in paths:
        path = ROOT / p if not str(p).startswith(("/", "E:", "C:")) else p
        d = np.load(path)
        if users is None:
            users, target = d["user_id"], d["target"]
        elif not np.array_equal(users, d["user_id"]):
            raise SystemExit(f"{p}: другой набор пользователей — файлы с разных срезов?")
        preds.append(d["pred_log"])
        print(f"  {path.name}: RMSLE {rmse_log(np.log1p(d['target']), d['pred_log']):.5f}")
    return users, np.mean(preds, axis=0), target


def average_submissions(sources: list[str], out: str | None) -> None:
    """Усреднить несколько сабмитов одной модели в log1p-шкале (сеть по сидам).

    Зачем отдельно от blend_submissions. На валидации сеть проверялась средним
    трёх сидов (1.68446), а один сид даёт 1.68266…1.68804 — разброс 0.0054,
    больше, чем весь выигрыш бленда. Сабмит из одного сида поэтому слабее того,
    что измерено, и с вероятностью около трети попадёт на худший сид.
    """
    import polars as pl

    from config import SUBMISSIONS

    if len(sources) < 2:
        raise SystemExit("нужно хотя бы два файла через запятую")
    subs = [pl.read_csv(SUBMISSIONS / s) for s in sources]
    for s in subs[1:]:
        if not (s["user_id"] == subs[0]["user_id"]).all():
            raise SystemExit("порядок user_id в файлах различается")
    logs = [np.log1p(s["predict"].to_numpy().astype(np.float64)) for s in subs]
    mixed = np.clip(np.expm1(np.mean(logs, axis=0)), 0, None)

    name = out or f"avg{len(sources)}_{dt.datetime.now():%m%d_%H%M}.csv"
    path = SUBMISSIONS / name
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": subs[0]["user_id"],
                  "predict": mixed.astype(np.float32)}).write_csv(path)
    print(f"{path}")
    print(f"  усреднено файлов: {len(sources)} ({', '.join(sources)})")
    print(f"  суммы: {', '.join(f'{np.expm1(l).sum():,.0f}' for l in logs)} -> {mixed.sum():,.0f}")
    append_csv(SUBMISSIONS / "log.csv",
               ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"],
               {"file": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "name": "avg_seeds", "model": "avg",
                "pred_sum": round(float(mixed.sum())),
                "pred_zeros": f"{(mixed < 1e-6).mean():.4f}",
                "note": f"среднее {len(sources)} сидов сети в log1p: {', '.join(sources)}"})


def blend_submissions(sources: list[str], weight: float, out: str | None) -> None:
    """Смешать два готовых сабмита в log1p-шкале: p = (1-w)*первый + w*второй.

    Веса подбираются на валидации (`--full`), а применяются здесь. Смешивание
    идёт в log1p — там же, где живёт метрика и где складываются предсказания
    участников ансамбля в `models.Ensemble`.

    Поправки калибровки к результату не относятся: они измерены зондами для
    другой модели. Для бленда нужны свои зонды (PLAN.md, раздел 3).
    """
    import polars as pl

    from config import SAMPLE_SUBMIT, SUBMISSIONS

    if len(sources) != 2:
        raise SystemExit("нужно ровно два файла через запятую: бустинг,сеть")
    subs = [pl.read_csv(SUBMISSIONS / s) for s in sources]
    if not (subs[0]["user_id"] == subs[1]["user_id"]).all():
        raise SystemExit("порядок user_id в файлах различается")
    ref = pl.read_csv(SAMPLE_SUBMIT)
    if not (ref["user_id"] == subs[0]["user_id"]).all():
        raise SystemExit("порядок user_id разошёлся с sample_submit")

    logs = [np.log1p(s["predict"].to_numpy().astype(np.float64)) for s in subs]
    mixed = np.clip(np.expm1((1 - weight) * logs[0] + weight * logs[1]), 0, None)

    name = out or f"blend_w{weight:.2f}_{dt.datetime.now():%m%d_%H%M}.csv"
    path = SUBMISSIONS / name
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": subs[0]["user_id"],
                  "predict": mixed.astype(np.float32)}).write_csv(path)
    print(f"{path}")
    print(f"  {sources[0]} (вес {1 - weight:.2f}) + {sources[1]} (вес {weight:.2f})")
    print(f"  суммы: {np.expm1(logs[0]).sum():,.0f} и {np.expm1(logs[1]).sum():,.0f} "
          f"-> {mixed.sum():,.0f}")
    print("\nПоправки калибровки старой модели к этому файлу НЕ относятся —\n"
          "нужны свои зонды уровня и размаха (PLAN.md, раздел 3; TASKS.md, A2).")
    append_csv(SUBMISSIONS / "log.csv",
               ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"],
               {"file": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit(), "name": "blend_seq", "model": "blend",
                "blend_w": round(weight, 2), "pred_sum": round(float(mixed.sum())),
                "pred_zeros": f"{(mixed < 1e-6).mean():.4f}",
                "note": f"бленд в log1p: {sources[0]} и {sources[1]}, вес сети {weight:.2f}; "
                        f"поправки калибровки не применялись"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-submissions", default=None,
                    help="смешать два готовых сабмита через запятую: бустинг,сеть "
                         "(вместо проверки на валидации)")
    ap.add_argument("--average-submissions", default=None,
                    help="усреднить несколько сабмитов одной модели в log1p (сеть по сидам)")
    ap.add_argument("--weight", type=float, default=0.4,
                    help="вес сети при смешивании сабмитов")
    ap.add_argument("--out", default=None, help="имя файла результата")
    ap.add_argument("--seq", default=None,
                    help="через запятую: .npz с предсказаниями сети (--save-val-pred)")
    ap.add_argument("--val-cutoff", default="2026-01-15")
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--full", action="store_true",
                    help="сравнивать не с одиночным LightGBM, а с рабочей моделью целиком: "
                         "ансамбль пяти конфигураций + двухголовая + их бленд. Это то, из "
                         "чего собирается сабмит, но считается в разы дольше")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.average_submissions:
        average_submissions([s.strip() for s in args.average_submissions.split(",") if s.strip()],
                            args.out)
        return
    if args.blend_submissions:
        blend_submissions([s.strip() for s in args.blend_submissions.split(",") if s.strip()],
                          args.weight, args.out)
        return
    if not args.seq:
        raise SystemExit("нужен --seq (проверка на валидации), --blend-submissions "
                         "или --average-submissions")
    val_cut = dt.date.fromisoformat(args.val_cutoff)

    print("=== предсказания сети ===")
    seq_users, p_seq, seq_target = load_seq([s.strip() for s in args.seq.split(",") if s.strip()])
    if len(p_seq.shape) != 1:
        raise SystemExit("неожиданная форма предсказаний сети")

    print(f"\n=== бустинг на вашей машине (валидация {val_cut}) ===")
    train, val, feats, cuts = load_split(args.cutoffs, val_cutoff=val_cut)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    del train
    ylog_tr = np.log1p(ytr)
    if args.full:
        # Рабочая модель целиком, теми же функциями, что и train.py — иначе
        # сравнение шло бы с одиночным LightGBM, а сабмит собирается не из него.
        from ensemble import MEMBERS
        from train import fit_single, fit_two_stage, two_stage_predict

        single = fit_single(Xtr, ylog_tr, Xva, np.log1p(yva), feats, "lgbm", "cpu",
                            args.rounds, members=MEMBERS)
        p_s = single.predict(Xva)
        clf, reg = fit_two_stage(Xtr, ytr, Xva, yva, feats, "lgbm", "cpu", args.rounds)
        p_t = two_stage_predict(clf, reg, Xva)
        gbm_w, gbm_r = 0.0, float("inf")
        for w in np.linspace(0, 1, 21):
            r = rmse_log(np.log1p(yva), w * p_t + (1 - w) * p_s)
            if r < gbm_r:
                gbm_w, gbm_r = float(w), r
        p_gbm = gbm_w * p_t + (1 - gbm_w) * p_s
        best_iter = single.best_iter
        print(f"  рабочая модель: ансамбль {len(MEMBERS)} конфигураций + двухголовая, "
              f"вес двухголовой {gbm_w:.2f}")
    else:
        m = GBM("lgbm", "reg", "cpu", n_estimators=args.rounds, early_stopping=200)
        m.fit(Xtr, ylog_tr, Xva, np.log1p(yva), feature_names=feats)
        p_gbm = m.predict(Xva)
        best_iter = m.best_iter

    # Выравнивание по user_id: порядок строк выборки признаков не совпадает
    # с порядком в .npz, а складывать предсказания разных пользователей —
    # ошибка, которая тихо испортит все числа ниже.
    val_users = val["user_id"].to_numpy()
    pos = np.searchsorted(seq_users, val_users)
    if pos.max() >= len(seq_users) or not np.array_equal(seq_users[pos], val_users):
        raise SystemExit("наборы пользователей сети и бустинга не совпадают: "
                         "проверьте, что .npz с того же среза")
    p_seq = p_seq[pos]
    if not np.allclose(seq_target[pos], yva):
        raise SystemExit("таргеты сети и бустинга разошлись — файлы с разных срезов")

    ylog = np.log1p(yva)
    r_gbm, r_seq = rmse_log(ylog, p_gbm), rmse_log(ylog, p_seq)
    corr = float(np.corrcoef(p_gbm, p_seq)[0, 1])

    best_w, best_r = 0.0, float("inf")
    for w in np.linspace(0, 1, 101):
        r = rmse_log(ylog, w * p_seq + (1 - w) * p_gbm)
        if r < best_r:
            best_w, best_r = float(w), r

    print(f"\n=== итог на {val_cut}, {len(yva):,} пользователей ===")
    report(yva, np.expm1(np.clip(p_gbm, 0, None)),
           "бустинг" if args.full else "одиночный lgbm")
    report(yva, np.expm1(np.clip(p_seq, 0, None)), "сеть")
    print(f"  корреляция предсказаний в log1p-шкале: {corr:.4f}")
    print(f"  лучший бленд       RMSLE {best_r:.5f} при весе сети {best_w:.2f}")
    gain = min(r_gbm, r_seq) - best_r
    print(f"  выигрыш к лучшему участнику: {gain:+.5f}")
    report(yva, np.expm1(np.clip(best_w * p_seq + (1 - best_w) * p_gbm, 0, None)), "бленд")

    # Оптимальный вес подобран на той же валидации, по которой отчитываемся, и
    # между срезами он разный (0.50 на январе, 0.30 на декабре). Поэтому важнее
    # оптимума то, насколько он плоский: если фиксированный вес, выбранный
    # заранее, даёт почти столько же, результат переносится, а не подгоняется.
    print("\n  вес сети:  " + "  ".join(f"{w:>7.2f}" for w in (0.2, 0.3, 0.4, 0.5, 0.6)))
    print("  RMSLE:     " + "  ".join(
        f"{rmse_log(ylog, w * p_seq + (1 - w) * p_gbm):.5f}" for w in (0.2, 0.3, 0.4, 0.5, 0.6)))

    if gain < 0.002:
        print("\nвывод: выигрыш ниже порога различимости 0.002 — на одном срезе\n"
              "это ничего не значит. Проверяйте на втором срезе и принимайте\n"
              "только при положительном знаке на обоих (PLAN.md, раздел 2).")
    if corr > 0.99:
        print(f"\nосторожно: корреляция {corr:.4f} — сеть предсказывает почти то же самое,\n"
              "что и бустинг. Пара с корреляцией 0.9997 уже проверялась как вторая\n"
              "финальная кандидатура и не дала ничего (PLAN.md, раздел 8).")

    append_csv(MODELS / "experiments.csv", EXPERIMENT_FIELDS, {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
        "feat_ver": "seq+lgbm", "blocks": "all+seq", "name": "blend_seq_lgbm",
        "model": "blend", "cutoffs": len(cuts) - 1, "n_features": len(feats),
        "rmsle_single": round(r_gbm, 5), "rmsle_two_stage": round(r_seq, 5),
        "rmsle_blend": round(best_r, 5), "blend_w": round(best_w, 2),
        "gini_blend": "", "sum_bias_blend": "", "best_iter_single": best_iter,
        "stride": 30, "halflife": "", "val_cutoff": str(val_cut),
        "train_cutoffs": " ".join(str(c) for c in cuts[1:]),
        "note": (args.note or f"бленд сети и бустинга "
                              f"({'рабочая модель' if args.full else 'одиночный LightGBM'}): "
                              f"corr={corr:.4f}, выигрыш к лучшему {gain:+.5f}")
                + f" [колонки: single=бустинг, two_stage=сеть]",
    })


if __name__ == "__main__":
    main()
