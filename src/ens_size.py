"""Сколько участников нужно ансамблю бустинга: кривая качества по размеру.

Рабочий состав — пять конфигураций LightGBM. Переход от одной модели к пяти дал
0.0019 (1.68612 -> 1.68423), это принято и записано в PLAN, раздел 5. Но дальше
пяти никто не пробовал, и вот почему это пропустили: вывод «гиперпараметры не
переносятся» закрыл тему целиком, хотя относился он к **выбору одной** лучшей
конфигурации.

У усреднения логика обратная. Дисперсия среднего n участников с попарной
корреляцией ошибок p равна sigma^2 * (p + (1 - p) / n): непереносимость
отдельных конфигураций означает низкую p, то есть работает НА усреднение, а не
против. При p около 0.9 переход с пяти на двадцать снимает ещё примерно четверть
того, что снял переход с одного на пять.

Здесь меряется не «лучше или хуже», а **кривая**: где насыщение и стоит ли
городить двадцать моделей ради результата. Участники берутся как четыре
различные конфигурации из ensemble.py, размноженные по сидам: у LightGBM
bagging_fraction 0.85 и feature_fraction 0.75, поэтому разные сиды дают
по-настоящему разные модели. Ни одного нового гиперпараметра — только снижение
дисперсии, единственный приём в проекте, который ни разу не подвёл.

Порядок участников чередуется по конфигурациям, а не по сидам: тогда любой
префикс сбалансирован, и точка «8 участников» означает «четыре конфигурации по
два сида», а не «две конфигурации по четыре».

Сравнение всегда по выровненному уровню: уровень правится в сабмите бесплатно,
и засчитывать модели его исправление нельзя (эта ловушка срабатывала в проекте
трижды).

    python -u src/ens_size.py --seeds 42,7,13,21,99
    python -u src/ens_size.py --seeds 42,7,13 --val-cutoffs 2026-01-15
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

import lightgbm as lgb
from tqdm.auto import tqdm

from config import MODELS, train_cutoffs
from ensemble import MEMBERS
from metrics import report, rmse_log
from models import LGB_REG
from train import load_split, to_xy
from utils import append_csv, git_commit

FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]


def distinct_configs() -> list[tuple[str, str, dict]]:
    """Конфигурации из рабочего состава без учёта сида.

    В MEMBERS два участника — одна и та же конфигурация с сидами 42 и 7.
    Размножать её по сидам ещё раз значило бы посчитать одну модель дважды.
    """
    seen, out = set(), []
    for name, kind, params in MEMBERS:
        key = tuple(sorted((k, v) for k, v in params.items() if k != "seed"))
        if key in seen:
            continue
        seen.add(key)
        out.append((name.rsplit("_s", 1)[0], kind, {k: v for k, v in params.items()
                                                    if k != "seed"}))
    return out


def leveled(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Оптимальный сдвиг перебором: аналитический оптимум смещён обрезкой нуля."""
    grid = np.arange(-0.40, 0.26, 0.0025)
    return p + grid[int(np.argmin([rmse_log(y, p + d) for d in grid]))]


def iter_bar(desc: str):
    """Счётчик итераций бустинга без общего числа.

    Полоса с процентами здесь была бы враньём: обучение останавливает ранняя
    остановка, а не потолок в 20 000 раундов, и реально уходит 140-400. Поэтому
    счётчик со скоростью и текущей метрикой — честнее.
    """
    return tqdm(desc=desc, unit="итер", disable=None, leave=False, dynamic_ncols=True,
                mininterval=0.5,
                bar_format="    {desc}: {n_fmt} итераций [{elapsed}, {rate_fmt}{postfix}]")


def fit_member(Xtr, ytr_log, Xva, yva_log, feats, params, rounds, desc):
    """Один участник состава. Обучение повторяет путь GBM для lgbm/reg.

    Вызывается не через `models.GBM` намеренно: тому нельзя передать свой
    callback, а без него полтора часа прогона идут без единого признака жизни.
    Параметры берутся из того же `LGB_REG`, поэтому разойтись с рабочей моделью
    они не могут — локально только сам вызов обучения.
    """
    base = {**LGB_REG, **params}
    dtrain = lgb.Dataset(Xtr, label=ytr_log, feature_name=feats)
    dvalid = lgb.Dataset(Xva, label=yva_log, reference=dtrain)

    prog = iter_bar(desc)

    def on_iter(env):
        prog.update(1)
        if env.evaluation_result_list:
            prog.set_postfix_str(f"rmse {env.evaluation_result_list[0][2]:.5f}")

    on_iter.before_iteration = False
    on_iter.order = 30
    model = lgb.train(base, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(200, verbose=False), on_iter])
    prog.close()
    best = model.best_iteration or rounds
    return model.predict(Xva, num_iteration=best), best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13,21,99",
                    help="сиды через запятую; участников будет "
                         "len(конфигураций) x len(сидов)")
    ap.add_argument("--val-cutoffs", default="2026-01-15,2025-12-16")
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = distinct_configs()
    # Чередование по конфигурациям: любой префикс сбалансирован по составам.
    plan = [(f"{n}_s{sd}", kind, {**params, "seed": sd})
            for sd in seeds for n, kind, params in configs]
    print(f"конфигураций {len(configs)}, сидов {len(seeds)} -> участников {len(plan)}")
    print("  " + ", ".join(n for n, _, _ in configs))

    for raw in args.val_cutoffs.split(","):
        val_cut = dt.date.fromisoformat(raw.strip())
        n_cut = args.cutoffs if val_cut == train_cutoffs(1)[0] else args.cutoffs + 1
        print(f"\n=== валидация {val_cut} ===")
        train, val, feats, cuts = load_split(n_cut, val_cutoff=val_cut)
        Xtr, ytr = to_xy(train, feats)
        Xva, yva = to_xy(val, feats)
        del train
        ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)

        preds, curve = [], []
        outer = tqdm(total=len(plan), desc="  участники", unit="модель", disable=None,
                     leave=False, dynamic_ncols=True,
                     bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}{postfix}]")
        for i, (name, _, params) in enumerate(plan, 1):
            p, best = fit_member(Xtr, ytr_log, Xva, yva_log, feats, params,
                                 args.rounds, f"[{i}/{len(plan)}] {name}")
            preds.append(p)
            score = rmse_log(yva_log, leveled(yva_log, np.mean(preds, axis=0)))
            curve.append(score)
            outer.update(1)
            outer.set_postfix_str(f"состав {score:.5f}")
            # tqdm.write, а не print: обычная печать разорвала бы полосу.
            tqdm.write(f"  [{i:>2}] {name:<22} итераций {best:>4} | "
                       f"состав из {i:>2}: {score:.5f}")
        outer.close()

        print(f"\n  кривая по размеру состава (выровненный уровень):")
        base5 = curve[4] if len(curve) >= 5 else None
        for i in (1, 2, 4, 8, 12, 16, 20):
            if i <= len(curve):
                delta = "" if base5 is None else f" | к пяти {base5 - curve[i - 1]:+.5f}"
                print(f"    {i:>2} участников: {curve[i - 1]:.5f}{delta}")
        report(yva, np.expm1(np.clip(leveled(yva_log, np.mean(preds, axis=0)), 0, None)),
               f"состав из {len(preds)}")

        append_csv(MODELS / "experiments.csv", FIELDS, {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(), "feat_ver": "ens_size", "blocks": "all",
            "name": f"ens{len(plan)}_{val_cut:%m%d}", "model": "lgbm",
            "cutoffs": len(cuts) - 1, "n_features": len(feats),
            "rmsle_single": round(curve[0], 5), "rmsle_two_stage": "",
            "rmsle_blend": round(curve[-1], 5), "blend_w": "", "gini_blend": "",
            "sum_bias_blend": "", "best_iter_single": "", "stride": 30, "halflife": "",
            "val_cutoff": str(val_cut),
            "train_cutoffs": " ".join(str(c) for c in cuts[1:]),
            "note": (args.note + "; " if args.note else "")
                    + f"размер состава: 1 -> {curve[0]:.5f}, "
                      f"5 -> {curve[4]:.5f}, {len(curve)} -> {curve[-1]:.5f} "
                      f"[выровненный уровень]"})


if __name__ == "__main__":
    main()
