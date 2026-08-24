"""Ансамбль разнородных LightGBM с проверкой сразу на двух валидационных срезах.

Зачем ансамбль. Перебор гиперпараметров дал три конфигурации, различающиеся
на 0.00007 — выбрать из них одну нельзя (проверено: выбор не переносится на
другой срез), но усреднить их предсказания можно, и разнородность как раз
и делает усреднение полезным.

Зачем два среза. Расхождение оценок между 2026-01-15 и 2025-12-16 — порядка
0.001, поэтому эффект меньше 0.002, измеренный на одном срезе, ничего не значит.
Ансамбль принимаем, только если он выигрывает на обоих.

Усреднение идёт в log1p-шкале: там живёт метрика, и там же складываются
предсказания моделей.

    python src/ensemble.py --val-cutoffs 2026-01-15,2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from config import MODELS, SEED
from datasets import parse_blocks
from metrics import report, rmse_log
from models import GBM
from train import load_split, to_xy
from utils import append_csv, git_commit

# Участники: (имя, семейство, параметры). Семейство важнее гиперпараметров —
# CatBoost на этих данных равен LightGBM по качеству (1.68632 против 1.68612),
# но устроен иначе и ошибается на других объектах, а ансамбль зарабатывает
# именно на различии ошибок, а не на силе отдельного участника.
LGB_MEMBERS = [
    ("lgb_default_s42", "lgbm",
     dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
          feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0, max_bin=255, seed=SEED)),
    ("lgb_default_s7", "lgbm",
     dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
          feature_fraction=0.75, bagging_fraction=0.85, lambda_l2=5.0, max_bin=255, seed=7)),
    ("lgb_deep511_lr02", "lgbm",
     dict(learning_rate=0.02, num_leaves=511, min_data_in_leaf=500,
          feature_fraction=0.65, bagging_fraction=0.7, lambda_l2=20.0, max_bin=127, seed=SEED)),
    ("lgb_deep1023_ff04", "lgbm",
     dict(learning_rate=0.02, num_leaves=1023, min_data_in_leaf=200,
          feature_fraction=0.4, bagging_fraction=0.7, lambda_l2=20.0, max_bin=255, seed=SEED)),
    ("lgb_deep511_lr05", "lgbm",
     dict(learning_rate=0.05, num_leaves=511, min_data_in_leaf=1000,
          feature_fraction=0.65, bagging_fraction=0.85, lambda_l2=20.0, max_bin=255, seed=SEED)),
]

CAT_MEMBERS = [
    ("cat_d8_lr05", "cat",
     dict(depth=8, learning_rate=0.05, l2_leaf_reg=5.0, random_seed=SEED, verbose=0)),
    ("cat_d6_lr03", "cat",
     dict(depth=6, learning_rate=0.03, l2_leaf_reg=10.0, random_seed=7, verbose=0)),
]

# Рабочий состав — только LightGBM. Добавление CatBoost даёт +0.0002 и +0.0001
# на двух срезах и +0.0006 под сдвигом уровня: знак верный во всех трёх замерах,
# но величина втрое ниже порога различимости (0.002), поэтому в рабочий путь
# не берём. Состав mixed оставлен проверенным — он пригодится как заведомо
# непохожая вторая модель при выборе двух финальных решений.
# Состав по итогам перебора 40 конфигураций на АКТУАЛЬНОЙ модели — с блоком
# ranks и с признаком сети (models/tuning.csv, 22.08). Прежний состав LGB_MEMBERS
# подбирался, когда признаков было 161 и стекинга не существовало: тогда глубокие
# деревья добирали взаимодействия сами. Свежий перебор говорит обратное — вся
# верхушка это 31-63 листа, а 255 листьев стабильно внизу:
#
#     лучшие  1.66953 ... 1.66992   листьев 31-63
#     худшие  1.67107 ... 1.67118   листьев 255
#
# Участники подобраны не по одному только качеству, а по РАЗЛИЧИЮ: темп от 0.01
# до 0.03, доля признаков от 0.5 до 0.8, подвыборка от 0.7 до 1.0, регуляризация
# от 1 до 20. Ансамбль усредняет, поэтому от участников нужна непохожесть
# при сопоставимой силе — то же соотношение, что решает в блендах.
TUNED_MEMBERS = [
    ("tuned_l63_lr03_ff05", "lgbm",
     dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=500,
          feature_fraction=0.5, bagging_fraction=0.7, lambda_l2=20.0,
          max_bin=255, seed=SEED)),
    ("tuned_l31_lr03_ff08", "lgbm",
     dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
          feature_fraction=0.8, bagging_fraction=1.0, lambda_l2=20.0,
          max_bin=255, seed=SEED)),
    ("tuned_l63_lr02_bin127", "lgbm",
     dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=1000,
          feature_fraction=0.8, bagging_fraction=0.7, lambda_l2=1.0,
          max_bin=127, seed=SEED)),
    ("tuned_l31_lr03_md1000", "lgbm",
     dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=1000,
          feature_fraction=0.8, bagging_fraction=1.0, lambda_l2=20.0,
          max_bin=127, seed=7)),
    ("tuned_l63_lr01_ff05", "lgbm",
     dict(learning_rate=0.01, num_leaves=63, min_data_in_leaf=1000,
          feature_fraction=0.5, bagging_fraction=1.0, lambda_l2=1.0,
          max_bin=127, seed=SEED)),
]

MEMBER_SETS = {
    "tuned": TUNED_MEMBERS,
    "lgb": LGB_MEMBERS,
    "cat": CAT_MEMBERS,
    "mixed": LGB_MEMBERS + CAT_MEMBERS,
}
MEMBERS = LGB_MEMBERS


def run_cutoff(val_cutoff: dt.date | None, n_cutoffs: int, blocks, rounds: int,
               early_stopping: int, explicit_train: list[dt.date] | None = None,
               members: list | None = None) -> dict:
    members = members or MEMBERS
    train, val, feats, cuts = load_split(n_cutoffs, blocks=blocks, val_cutoff=val_cutoff,
                                         explicit_train=explicit_train)
    Xtr, ytr = to_xy(train, feats)
    Xva, yva = to_xy(val, feats)
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    del train, ytr

    preds, scores, by_kind = [], {}, {}
    for name, kind, params in members:
        m = GBM(kind, "reg", "cpu", n_estimators=rounds,
                early_stopping=early_stopping, params=params, log_period=0)
        m.fit(Xtr, ytr_log, Xva, yva_log, feature_names=feats)
        p = m.predict(Xva)
        preds.append(p)
        by_kind.setdefault(kind, []).append(p)
        scores[name] = rmse_log(yva_log, p)
        print(f"  {name:<20} RMSLE {scores[name]:.5f} | итераций {m.best_iter}")

    ens = np.mean(preds, axis=0)
    out = {
        "val_cutoff": str(cuts[0]),
        "members": scores,
        "best_member": min(scores.values()),
        "ensemble": rmse_log(yva_log, ens),
    }
    # Подансамбли по семействам: показывают, добавляет ли смешение семейств
    # что-то сверх усреднения внутри одного.
    for kind, ps in by_kind.items():
        if len(ps) > 1 or len(by_kind) > 1:
            out[f"ensemble_{kind}"] = rmse_log(yva_log, np.mean(ps, axis=0))
            print(f"  {'только ' + kind:<20} RMSLE {out[f'ensemble_{kind}']:.5f} "
                  f"({len(ps)} участников)")
    print(f"  {'среднее участников':<20} RMSLE {np.mean(list(scores.values())):.5f}")
    print(f"  {'АНСАМБЛЬ':<20} RMSLE {out['ensemble']:.5f} "
          f"(лучший участник {out['best_member']:.5f})")
    report(yva, np.expm1(ens), "ансамбль")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cutoffs", default="2026-01-15,2025-12-16",
                    help="через запятую; ансамбль принимаем только при выигрыше на всех")
    ap.add_argument("--cutoffs", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--train-cutoffs", default=None,
                    help="явный список обучающих срезов — например для стресс-теста уровня")
    ap.add_argument("--members", default="lgb", choices=sorted(MEMBER_SETS),
                    help="состав: lgb (рабочий), cat, mixed")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    members = MEMBER_SETS[args.members]
    explicit_train = ([dt.date.fromisoformat(d.strip()) for d in args.train_cutoffs.split(",")]
                      if args.train_cutoffs else None)

    results = []
    for raw in args.val_cutoffs.split(","):
        cut = dt.date.fromisoformat(raw.strip())
        print(f"\n=== валидация {cut} ===")
        results.append(run_cutoff(cut, args.cutoffs, parse_blocks(args.blocks),
                                  args.rounds, args.early_stopping, explicit_train, members))

    print("\n=== итог ===")
    verdict = True
    for r in results:
        gain = r["best_member"] - r["ensemble"]
        verdict &= gain > 0
        print(f"{r['val_cutoff']}: ансамбль {r['ensemble']:.5f} против лучшего "
              f"участника {r['best_member']:.5f} — выигрыш {gain:+.5f}")
        append_csv(MODELS / "ensemble.csv",
                   ["created", "commit", "val_cutoff", "n_members", "ensemble",
                    "best_member", "mean_member", "gain", "note"],
                   {"created": dt.datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit(), "val_cutoff": r["val_cutoff"],
                    "n_members": len(MEMBERS), "ensemble": round(r["ensemble"], 5),
                    "best_member": round(r["best_member"], 5),
                    "mean_member": round(float(np.mean(list(r["members"].values()))), 5),
                    "gain": round(gain, 5), "note": args.note})
    print("\nвердикт: " + ("ансамбль выигрывает на всех срезах — можно принимать"
                          if verdict else
                          "выигрыш не на всех срезах — принимать нельзя"))


if __name__ == "__main__":
    main()
