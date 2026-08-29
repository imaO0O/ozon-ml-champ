"""Задним числом заполнить колонку `rmsle_aligned` из сохранённых предсказаний.

Зачем это нужно и почему это не косметика. Уровень предсказания правится
на сабмите бесплатно (`TEST_LEVEL` известен), поэтому конфигурации сравнивают
по ФОРМЕ — по выровненной величине. Колонка `rmsle_blend`, по которой команда
сравнивала строки журнала, содержит **сырую**: у бустинга это одно и то же,
у сетей — нет. Замер трека C по 86 январским конфигурациям:

    перевёрнутых пар      1075 из 3655 (29.4%)
    корреляция рангов     0.454
    максимальный разрыв   0.02134
    лучшая по сырой       stack_y_final
    лучшая по выровненной gru_w52

То есть колонка сравнения называла другого чемпиона. Ошибка 28.08, стоившая
команде суток, была не оплошностью: в колонке лежала не та величина, и любой
взял бы оттуда то же самое. `seq_train.py` теперь пишет `rmsle_aligned` сам,
но это не лечит 600 уже записанных строк.

Лечит их этот скрипт: для каждой строки ищется файл предсказаний
`{name}_valpred_{val_cutoff}.npz` и величина считается заново.

## Предсказания лежат в ДВУХ разных единицах, и в файле это не записано

Найдено при этой работе, и это причина обоих докладов, а не мелочь рядом:

* `train.py` сохраняет бустинг **уже выровненным** (`p - p.mean() + y.mean()`,
  строка 299, с объяснением: иначе состав сравнивать нельзя);
* `seq_train.py` сохраняет сеть **сырой**.

Файлы лежат в одной папке, называются одинаково, и внутри нет ни одного поля,
по которому единицу можно узнать. Census на 29.08: **53 выровненных против
225 сырых.** Отсюда же вывод «у бустинга сырая и выровненная совпадают»:
они совпадают у сохранённого ФАЙЛА, потому что он выровнен, а журнальная
строка бустинга — сырая, и расходится с ним на 0.0121 у `stk2_jan`. Бустинговая
половина проекта была не иммунна, у неё просто разрыв жил между файлом
и журналом, а не внутри журнала.

Проверено, что главные измерительные инструменты этим не задеты:
`partner.py` и `compose.py` выравнивают обе стороны явно, `rebuild_comp.py`
вычитает среднее у каждой руки, `net_value.py` держит константу в матрице
плана. То есть замеры направления стоят.

## Правило привязки — почему нельзя просто взять файл по имени

Имя прогона не идентифицирует прогон: `lgbm_ens` стоит в журнале трижды
на трёх разных версиях признаков, а `epcap_12` и `nostfin` — один прогон
под двумя именами. Файл на диске один, строк несколько, и приписать его
не той строке — значит подделать число.

Поэтому привязка проверяется, а не предполагается, и проверяется тем, что
в данной единице осмысленно:

* файл сырой — сверяется **сырая** величина с журнальной `rmsle_single`;
* файл выровненный — сырую сверить не с чем, поэтому сверяется **Gini**:
  он не меняется от сдвига уровня и потому годится там, где RMSLE не годится.

Совпало — файл принадлежит строке. Не совпало — строка остаётся пустой
и попадает в отчёт. Пустая клетка честнее правдоподобного числа.

## Почему запись сделана с защитой от гонки

Добавить колонку нельзя иначе, как переписав файл целиком, — а журнал общий,
и в него дописывают строки пять скриптов, в том числе долгий `seq_oof_net.py`,
который добавляет строку после каждой сети. Запись поверх дописывания теряет
чужую строку молча.

Поэтому: содержимое хешируется при чтении и перечитывается прямо перед
записью; если хеш изменился, значит кто-то дописал, и заполнение делается
заново на свежем содержимом (замеры кешированы, это мгновенно). Сама запись
идёт во временный файл и подменяется `os.replace` — атомарно, то есть
оборванный процесс не оставит обрезанного журнала.

Дефект нашёл трек C до того, как он что-то стоил: отказался писать поверх
идущего прогона OOF, сославшись на прежний инцидент с восстановлением
`log.csv` из HEAD.

    python -u src/journal_align.py            # только отчёт
    python -u src/journal_align.py --write    # заполнить колонку
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os

import numpy as np

from config import MODELS
from metrics import gini_norm, rmse_log

LOG = MODELS / "experiments.csv"
COL = "rmsle_aligned"
# Журнал округляет до пятого знака, значит расхождение привязки не может быть
# больше половины последнего разряда. Допуск взят с запасом вдвое.
TOL = 1e-5
# Gini журнал округляет до четвёртого знака; допуск взят с запасом вчетверо,
# потому что пересчёт идёт по другому пути, чем при записи строки.
TOL_GINI = 2e-4
# Ниже этого сдвига уровня файл считается сохранённым уже выровненным.
# Порог не машинный намеренно: часть выровненных файлов несёт остаточный сдвиг
# 1e-8..1e-7 (среднее вычиталось на другом этапе, чем сохранялась колонка).
# Осмысленный сдвиг уровня — это 0.01..0.2; при 1e-4 сырая и выровненная
# расходятся на 3e-9, то есть на девять знаков ниже последнего записанного.
EPS_LEVEL = 1e-4


def measure(path) -> "tuple[float, float, float, bool]":
    """сырая, выровненная, Gini, «файл сохранён выровненным»."""
    d = np.load(path)
    y_raw = d["target"].astype(np.float64)
    y = np.log1p(y_raw)
    p = d["pred_log"].astype(np.float64)
    ali = rmse_log(y, p - p.mean() + y.mean())
    gini = gini_norm(y_raw, np.expm1(np.clip(p, 0, None)))
    return rmse_log(y, p), ali, gini, abs(p.mean() - y.mean()) < EPS_LEVEL


def read_log() -> "tuple[str, list[str], list[dict]]":
    """Содержимое журнала вместе с хешем — чтобы заметить чужую дозапись."""
    raw = LOG.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows = list(reader)
    fields = list(reader.fieldnames or [])
    if COL not in fields:
        fields.append(COL)
    return hashlib.sha256(raw).hexdigest(), fields, rows


def write_log(fields: list[str], rows: list[dict]) -> None:
    """Запись через временный файл: оборванный процесс не оставит обрезка."""
    tmp = LOG.with_name(LOG.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, LOG)


def fill(rows: list[dict], memo: dict) -> dict:
    """Проставить колонку во все строки, где привязка проверяется.

    Замеры кешируются в `memo`: повтор на свежепрочитанном журнале обходится
    без единого обращения к диску, и окно гонки при перезаписи почти нулевое.
    """
    filled = already = nofile = mismatch = by_gini = 0
    gaps, bad = [], []
    for r in rows:
        if (r.get(COL) or "").strip():
            already += 1
            continue
        name, cut = (r.get("name") or "").strip(), (r.get("val_cutoff") or "").strip()
        raw_j = (r.get("rmsle_single") or "").strip()
        if not name or not cut or not raw_j:
            nofile += 1
            continue
        path = MODELS / f"{name}_valpred_{cut}.npz"
        if not path.exists():
            nofile += 1
            continue
        if path not in memo:
            memo[path] = measure(path)
        raw, ali, gini, prealigned = memo[path]
        if prealigned:
            # Сырой величины в таком файле нет вовсе, сверять RMSLE не с чем.
            g_j = (r.get("gini_blend") or "").strip()
            if not g_j or abs(gini - float(g_j)) > TOL_GINI:
                mismatch += 1
                bad.append((name, cut, "Gini", float(g_j) if g_j else float("nan"), gini))
                continue
            by_gini += 1
        elif abs(raw - float(raw_j)) > TOL:
            mismatch += 1
            bad.append((name, cut, "сырая", float(raw_j), raw))
            continue
        r[COL] = round(ali, 5)
        filled += 1
        gaps.append((float(raw_j) - ali, name, cut, float(raw_j), ali))

    # Один файл предсказаний на несколько строк с тем же именем и срезом —
    # привязка неоднозначна в принципе. Так вышло после прогона трека B:
    # его строка и наша легли рядом под одним именем и одним срезом, а колонки,
    # различающей машину, в журнале нет (устройство пишется в `note` текстом).
    # Сверка чисел развела их — но только потому, что железо разное; на одной
    # машине два прогона совпали бы, и различить их стало бы нечем.
    dup: dict[tuple, list] = {}
    for r in rows:
        dup.setdefault(((r.get("name") or "").strip(), (r.get("val_cutoff") or "").strip()),
                       []).append(r)
    ambiguous = {k: v for k, v in dup.items() if len(v) > 1 and k[0] and k[1]}
    return {"filled": filled, "already": already, "nofile": nofile, "mismatch": mismatch,
            "by_gini": by_gini, "gaps": gaps, "bad": bad, "ambiguous": ambiguous}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="записать колонку в журнал")
    ap.add_argument("--show", type=int, default=12, help="сколько расхождений печатать")
    args = ap.parse_args()

    memo: dict = {}
    digest, fields, rows = read_log()
    st = fill(rows, memo)
    filled, already, nofile = st["filled"], st["already"], st["nofile"]
    mismatch, by_gini = st["mismatch"], st["by_gini"]
    gaps, bad, ambiguous = st["gaps"], st["bad"], st["ambiguous"]

    print(f"строк в журнале {len(rows)}")
    print(f"  заполнено сейчас        {filled}")
    print(f"  уже было заполнено      {already}")
    print(f"  нет файла предсказаний  {nofile}")
    print(f"  из них по Gini          {by_gini}  (файл сохранён уже выровненным)")
    print(f"  файл есть, но НЕ ТОТ    {mismatch}  (проверка не сошлась — не приписываем)")

    if ambiguous:
        print(f"\n--- одно имя + один срез в нескольких строках: {len(ambiguous)} ---")
        print("  привязка держится только на том, что числа разошлись")
        for (n, c), v in sorted(ambiguous.items())[:args.show]:
            nums = ", ".join(sorted({(x.get("rmsle_single") or "?") for x in v}))
            print(f"  {n:<28} {c}  строк {len(v)}  числа: {nums}")

    if bad:
        print(f"\n--- привязка отвергнута (первые {args.show}) ---")
        for n, c, what, j, r_ in bad[:args.show]:
            print(f"  {n:<28} {c}  по {what:<6} журнал {j:.5f}  файл {r_:.5f}  "
                  f"разница {abs(j - r_):.5f}")

    if gaps:
        gaps.sort(key=lambda t: -abs(t[0]))
        print(f"\n--- где сырая и выровненная расходятся сильнее всего ---")
        print(f"  {'прогон':<28} {'срез':<12} {'сырая':>9} {'выровн':>9} {'разрыв':>9}")
        for d, n, c, j, a in gaps[:args.show]:
            print(f"  {n:<28} {c:<12} {j:>9.5f} {a:>9.5f} {d:>+9.5f}")
        big = sum(1 for d, *_ in gaps if abs(d) > 0.001)
        print(f"\n  разрыв больше 0.001: {big} из {len(gaps)} — "
              "столько строк сравнивать по сырой нельзя")

    if not args.write:
        print("\nотчёт; чтобы записать — --write")
        return

    # Перечитать, заполнить заново и убедиться, что за это время файл не
    # изменился. Порядок именно такой: пишется ровно то содержимое, чей хеш
    # только что проверен, иначе чужая дописанная строка пропала бы молча.
    for attempt in range(1, 4):
        digest, fields, rows = read_log()
        filled = fill(rows, memo)["filled"]
        if hashlib.sha256(LOG.read_bytes()).hexdigest() == digest:
            write_log(fields, rows)
            print(f"\nжурнал переписан, колонка {COL} заполнена в {filled} строках")
            return
        print(f"\nжурнал дописали, пока шёл счёт (попытка {attempt}) — перечитываю")
    raise SystemExit(
        "журнал переписывают прямо сейчас — запись отменена, ничего не потеряно.\n"
        "Дождитесь конца чужого прогона (обычно это seq_oof_net.py, он пишет "
        "строку после каждой сети) и повторите.")


if __name__ == "__main__":
    main()
