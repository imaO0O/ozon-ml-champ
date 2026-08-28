"""Аудит утечки цели — механическая проверка, а не чтение кода.

Зачем отдельный скрипт. Утечка будущего в признаки — единственный класс ошибки,
который **улучшает** метрику и потому не может быть найден метрикой. Он виден
только прямой проверкой, и проверка должна быть механической: «в коде стоит
фильтр по дате» — это прочтение, а не доказательство. Фильтр мог оказаться
не на том уровне, обойтись джойном, потеряться в кеше выборки.

Три вектора, по одному на каждый путь данных в решении.

**1. Табличные признаки.** Собрать их дважды — из полного лога и из лога,
обрезанного по cutoff'у, — и сверить побитово. Если хоть один из 242 признаков
различается, туда попадает будущее.

**2. Вход сетей.** Тензор `пользователи x дни x каналы` строится один раз
на весь период, а окно нарезается в `gather`. Стереть в копии тензора все дни
начиная с cutoff'а и сверить нарезку. Заодно проверяется, что окно кончается
ДО cutoff'а, а не включает его.

**3. Порядок срезов во всех прогонах.** Стекинг — место, где утечка живёт чаще
всего: предсказание сети идёт признаком бустингу, и если сеть на срезе C
обучалась на данных, доставших до C, бустинг получает подсмотренный ответ.
Проверяется не путь в коде, а журнал: у каждого прогона записаны `val_cutoff`
и `train_cutoffs`. Условие строже очевидного — мало чтобы обучающий срез был
старше, надо чтобы и его ОКНО ЦЕЛИ `[c, c + HORIZON)` не доставало
до валидационного cutoff'а.

Чего этот аудит НЕ проверяет и о чём надо помнить отдельно: константы
калибровки (`TEST_LEVEL`, `TARGET_VAR`) восстановлены по публичному
лидерборду. Это не утечка цели из данных, а подгонка под паблик, она измерена
и оценена отдельно (`docs/jury/A2_final_selection.md`, экспозиция 0.000071).

    python -u src/leak_audit.py                # все три проверки
    python -u src/leak_audit.py --skip-seq     # без тензора сетей (он тяжёлый)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io

import numpy as np
import polars as pl

from config import DATA_START, HORIZON, MODELS, TEST_CUTOFF

LOG = MODELS / "experiments.csv"
# Путь ВНУТРИ репозитория: `git show ref:path` абсолютный не принимает.
LOG_IN_REPO = "models/experiments.csv"


def check_tabular(cutoff: dt.date) -> bool:
    """242 признака из полного лога против признаков из обрезанного."""
    from features import build_features, scan_log

    lf = scan_log()
    full = build_features(cutoff, lf).sort("user_id")
    trunc = build_features(cutoff, lf.filter(pl.col("event_date") < cutoff)).sort("user_id")
    if full.columns != trunc.columns:
        print("  РАЗНЫЕ КОЛОНКИ — сравнение невозможно")
        return False
    bad = []
    for c in full.columns:
        if c == "user_id":
            continue
        a, b = full[c].to_numpy(), trunc[c].to_numpy()
        na, nb = np.isnan(a), np.isnan(b)
        same = (na == nb).all() and np.array_equal(a[~na], b[~nb])
        if not same:
            bad.append((c, float(np.nanmax(np.abs(a - b)))))
    print(f"  признаков {full.width - 1}, строк {full.height:,}")
    for c, d in sorted(bad, key=lambda t: -t[1])[:10]:
        print(f"    РАЗЛИЧАЕТСЯ {c:<32} max |разница| {d:.6g}")
    print("  все признаки совпадают побитово" if not bad
          else f"  РАЗЛИЧАЮТСЯ {len(bad)} признаков")
    return not bad


def check_seq(cutoffs: list[dt.date], lookback: int, sample: int) -> bool:
    """Нарезка окна из тензора с затёртым будущим против нарезки из целого."""
    from seq_data import build, gather

    seq, users, _, _ = build()
    print(f"  тензор {seq.shape} ({seq.dtype})")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(users), size=min(sample, len(users)), replace=False))
    ok = True
    for cut in cutoffs:
        d_cut = (cut - DATA_START).days
        a = gather(seq, idx, cut, lookback)
        seq2 = seq.copy()
        seq2[:, d_cut:, :] = 0            # стереть ВСЁ начиная с дня cutoff
        b = gather(seq2, idx, cut, lookback)
        same = np.array_equal(a, b)
        ok &= same
        print(f"  {cut}: окно {a.shape[1]} дней, день cutoff {d_cut}/{seq.shape[1]} — "
              + ("будущее не читается" if same
                 else f"ЧИТАЕТСЯ, позиций {int((a != b).sum()):,}"))
    return ok


def journal_rows(refs: list[str]) -> list[dict]:
    """Строки журнала из рабочего дерева ПЛЮС из указанных веток.

    Зачем ветки. Половина команды работает в своей ветке, и её прогоны
    попадают в main не сразу. Аудит только по `main` покрывает не всю работу
    команды, а её ствол — и утверждение «ни один прогон за всю историю»
    оказывается шире измерения. Именно так и вышло 28.08: 82 прогона трека C
    жили в `track-c-self-norm` и в аудит не входили, хотя документы жюри
    ссылались на их числа.
    """
    import subprocess  # noqa: PLC0415  (нужен только здесь)

    rows = list(csv.DictReader(io.open(LOG, encoding="utf-8")))
    seen = {(r.get("name"), r.get("val_cutoff"), r.get("rmsle_single")) for r in rows}
    for ref in refs:
        r = subprocess.run(["git", "show", f"{ref}:{LOG_IN_REPO}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"  ветка {ref} недоступна, пропущена")
            continue
        extra = [x for x in csv.DictReader(io.StringIO(r.stdout))
                 if (x.get("name"), x.get("val_cutoff"), x.get("rmsle_single")) not in seen]
        print(f"  из {ref}: +{len(extra)} прогонов, которых нет в рабочем дереве")
        rows += extra
    return rows


def check_cutoff_order(refs: list[str]) -> bool:
    """Журнал прогонов: окно цели обучения не достаёт до валидационного среза."""
    rows = journal_rows(refs)
    bad_order, bad_overlap, noinfo = [], [], 0
    for r in rows:
        v = (r.get("val_cutoff") or "").strip()
        t = (r.get("train_cutoffs") or "").strip()
        if not v or not t:
            noinfo += 1
            continue
        try:
            vc = dt.date.fromisoformat(v)
            newest = max(dt.date.fromisoformat(x) for x in t.split())
        except ValueError:
            noinfo += 1
            continue
        if newest >= vc:
            bad_order.append((r["name"], v, str(newest)))
        if newest + dt.timedelta(days=HORIZON) > vc:
            bad_overlap.append((r["name"], v, str(newest)))
    print(f"  прогонов {len(rows)}, без записи о срезах {noinfo} "
          f"(ранние базовые линии до появления колонок и бленды без обучения)")
    print(f"  обучающий срез не старше валидационного: {len(bad_order)}")
    print(f"  окно цели достаёт до валидационного cutoff: {len(bad_overlap)}")
    for n, v, t in (bad_order + bad_overlap)[:10]:
        print(f"    {n}  вал {v}  самый свежий обучающий {t}")
    return not bad_order and not bad_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-01-15", help="срез для табличной проверки")
    ap.add_argument("--lookback", type=int, default=364)
    ap.add_argument("--sample", type=int, default=4000,
                    help="сколько пользователей брать в проверке тензора")
    ap.add_argument("--skip-tab", action="store_true")
    ap.add_argument("--skip-seq", action="store_true")
    ap.add_argument("--refs", default="origin/track-c-self-norm",
                    help="ветки, чьи журналы тоже проверить (через запятую); "
                         "работа второй половины команды живёт вне main")
    args = ap.parse_args()

    cut = dt.date.fromisoformat(args.cutoff)
    results = {}

    if not args.skip_tab:
        print("--- 1. табличные признаки: полный лог против обрезанного ---")
        results["табличные признаки"] = check_tabular(cut)
    if not args.skip_seq:
        print("\n--- 2. вход сетей: нарезка окна из тензора с затёртым будущим ---")
        results["вход сетей"] = check_seq(
            [cut, cut - dt.timedelta(days=HORIZON), TEST_CUTOFF], args.lookback, args.sample)
    print("\n--- 3. порядок срезов во всех прогонах журнала ---")
    results["порядок срезов"] = check_cutoff_order(
        [r.strip() for r in args.refs.split(",") if r.strip()])

    print("\n=== ИТОГ ===")
    for k, v in results.items():
        print(f"  {'чисто ' if v else 'УТЕЧКА'} {k}")
    if not all(results.values()):
        raise SystemExit("аудит НЕ пройден")
    print("\nУтечки цели не обнаружено ни на одном из проверенных путей.")


if __name__ == "__main__":
    main()
