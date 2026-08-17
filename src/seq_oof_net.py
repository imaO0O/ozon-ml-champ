"""Out-of-fold предсказания СЕТИ по всем срезам — вход для стекинга у трека A.

Зачем. Сейчас сеть и бустинг соединяются блендом: взвешенная сумма с одним
весом на всех клиентов. Стекинг устроен иначе — бустинг получает предсказание
сети обычным признаком и сам учится, где ему доверять. Взвешенная сумма так
не умеет.

Критично, как именно получены предсказания. Если дать бустингу предсказания
сети на тех же срезах, на которых сеть обучалась, он выучит утечку: на
валидации будет прекрасно, на лидерборде провал. Поэтому здесь та же схема
walk-forward, что в `seq_oof.py` для бустинга: для среза `c` сеть обучается
на срезах **строго старше** `c` и предсказывает `c`, ни разу его не увидев.

Тест устроен так же, как в рабочем сабмите: сеть, обученная на всех срезах,
предсказывает TEST_CUTOFF. Этот файл берётся готовым из усреднённых сабмитов
(`--test-submission`), потому что там уже усреднены сиды.

    python -u src/seq_oof_net.py --lookback 90 --static rk_ --name netoof
    python -u src/seq_oof_net.py --name netoof --test-submission gru_w90_avg3.csv

Результат: `models/<name>_<cutoff>.npz` с полями user_id, pred_log, target —
тот же формат, что у `seq_oof.py`, чтобы трек A читал оба одинаково.
"""
from __future__ import annotations

import argparse
import datetime as dt

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from config import HORIZON, MODELS, SUBMISSIONS, TEST_CUTOFF, train_cutoffs

SRC = Path(__file__).resolve().parent


def older_count(cut: dt.date, want: int) -> int:
    """Сколько срезов запросить у train_cutoffs, чтобы у `cut` было `want` предшественников.

    Срезы идут с шагом 30 дней от 2026-01-15, поэтому позиция `cut` в списке
    считается напрямую, а к ней добавляется нужное число более старых.
    """
    newest = train_cutoffs(1)[0]
    pos = (newest - cut).days // HORIZON
    return pos + 1 + want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="netoof", help="префикс выходных .npz")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--arch", default="gru")
    # PowerShell выбрасывает пустую строку из аргументов нативной программы,
    # поэтому «без статических признаков» задаётся словом, а не `--static ""`.
    # Та же причина, по которой в tune.py появился разбор пар вместо JSON.
    ap.add_argument("--static", default="rk_",
                    help="префикс статических признаков; none или - означает «без них»")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="42",
                    help="сиды через запятую; предсказания усредняются в log1p-шкале. "
                         "Обучающий признак должен быть такой же силы, что и тестовый, "
                         "иначе бустинг учится доверять более слабой версии, чем получит")
    ap.add_argument("--cutoffs", type=int, default=6,
                    help="сколько срезов покрыть, начиная со свежего")
    ap.add_argument("--train-cutoffs", type=int, default=5,
                    help="сколько более старых срезов давать сети на каждый прогон")
    ap.add_argument("--hidden", action="store_true",
                    help="выгружать скрытые состояния вместо предсказаний. "
                         "Работает только с одним сидом: у разных сидов оси скрытого "
                         "пространства свои, и усреднять их — всё равно что складывать "
                         "координаты в разных системах отсчёта")
    ap.add_argument("--test-submission", default=None,
                    help="готовый сабмит сети (усреднённый по сидам) -> <name>_test.npz")
    args = ap.parse_args()

    cuts = train_cutoffs(args.cutoffs)
    print(f"срезы для выгрузки: {', '.join(str(c) for c in cuts)}")

    if args.static and args.static.strip().lower() in ("none", "-", "нет"):
        args.static = ""
    print(f"статические признаки: {args.static or 'не подаются'}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.hidden and len(seeds) > 1:
        raise SystemExit("--hidden работает с одним сидом: скрытые пространства "
                         "разных сидов повёрнуты друг относительно друга, и их "
                         "усреднение не имеет смысла")
    print(f"сидов на срез: {len(seeds)} ({', '.join(map(str, seeds))})"
          + (" | выгружаем скрытые состояния" if args.hidden else ""))

    for cut in cuts:
        out = MODELS / f"{args.name}_{cut}.npz"
        if out.exists():
            print(f"\n{cut}: уже есть {out.name}")
            continue
        n = older_count(cut, args.train_cutoffs)
        print(f"\n=== {cut}: обучение на {args.train_cutoffs} более старых срезах ===")
        t0 = time.time()
        parts = []
        for sd in seeds:
            tag = f"{args.name}_s{sd}_{cut}"
            kind = "hidden" if args.hidden else "valpred"
            src = MODELS / f"{tag}_{kind}_{cut}.npz"
            if not src.exists():
                cmd = [sys.executable, "-u", str(SRC / "seq_train.py"),
                       "--arch", args.arch, "--epochs", str(args.epochs),
                       "--lookback", str(args.lookback), "--seed", str(sd),
                       "--cutoffs", str(n), "--val-cutoff", str(cut),
                       "--save-val-pred", "--name", tag,
                       "--note", f"walk-forward для стекинга, срез {cut}, сид {sd}"]
                if args.hidden:
                    cmd += ["--save-hidden"]
                if args.static:
                    cmd += ["--static", args.static]
                if subprocess.run(cmd, cwd=SRC.parent).returncode != 0:
                    raise SystemExit(f"прогон на {cut}, сид {sd} завершился с ошибкой")
            if not src.exists():
                raise SystemExit(f"не найден {src.name}")
            parts.append(np.load(src))

        base = parts[0]
        for q in parts[1:]:
            if not np.array_equal(q["user_id"], base["user_id"]):
                raise SystemExit(f"{cut}: наборы пользователей у сидов различаются")
        if args.hidden:
            np.savez(out, user_id=base["user_id"], hidden=base["hidden"],
                     target=base["target"])
            z = base["hidden"]
            print(f"  -> {out.name} | {z.shape[0]:,} x {z.shape[1]} | "
                  f"{out.stat().st_size / 1024 ** 2:.0f} МБ | {time.time() - t0:.0f}s")
            continue
        # Усреднение в log1p-шкале — там же, где складываются участники ансамбля.
        pred = np.mean([q["pred_log"] for q in parts], axis=0)
        np.savez(out, user_id=base["user_id"], pred_log=pred, target=base["target"])
        spread = float(np.std([q["pred_log"].mean() for q in parts]))
        print(f"  -> {out.name} | {len(pred):,} пользователей | сидов {len(parts)} | "
              f"разброс средних по сидам {spread:.5f} | {time.time() - t0:.0f}s")

    if args.test_submission:
        import polars as pl

        sub = pl.read_csv(SUBMISSIONS / args.test_submission)
        p = np.log1p(sub["predict"].to_numpy().astype(np.float64))
        out = MODELS / f"{args.name}_test.npz"
        np.savez(out, user_id=sub["user_id"].to_numpy(), pred_log=p)
        print(f"\nтест ({TEST_CUTOFF}) из {args.test_submission} -> {out.name} | "
              f"{len(p):,} пользователей | mean log1p {p.mean():.5f}")

    print("\nГотово. Формат тот же, что у seq_oof.py: user_id, pred_log, target.")
    print("ВАЖНО для стекинга: каждое предсказание получено сетью, которая этот срез")
    print("не видела. Предсказания с обучающих срезов дали бы утечку.")


if __name__ == "__main__":
    main()
