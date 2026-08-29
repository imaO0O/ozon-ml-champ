"""Аудит документации против источников истины — от расхождения текста с кодом.

Зачем. За один день нашлись три дефекта одного класса: `A3` описывал
вероятностной **выбывшую** сеть; `README` и `B1` приводили одну сетевую руку
из четырёх и давали ориентир предшественницы; `README` предлагал команду
`net_value.py --names`, которой у скрипта нет. Ни один не виден при чтении —
текст всюду выглядит осмысленным. Все три видны сверкой с источником истины.

Источников истины три, и все машинные:

* `submissions/log.csv` — что реально отправлялось и какой счёт получило;
* вывод `--help` каждого скрипта — какие флаги существуют на самом деле;
* `models/experiments.csv` — какие прогоны были и на какой версии признаков.

Отсюда три проверки.

**1. Числа.** Все величины вида `1.6xxxxx` в документах жюри сверяются
со счетами лидерборда. Совпадение не требуется: большинство таких чисел —
валидационные величины и вычисленные ожидания, они законны. Требуется
**объяснимость**: скрипт печатает несовпавшие, чтобы человек посмотрел
на каждое и сказал, откуда оно. Именно так нашлось, что 1.67265 принадлежит
сети-предшественнице.

**3. Ссылки на журнал прогонов.** Документ, называющий прогон по имени
и приводящий его число, обязан указывать на **действующую** строку. Дефект,
который эта проверка ловит, найден треком B 29.08: чек-лист воспроизводимости
велел сверяться со строкой `lgbm_ens` (1.68402) — версией признаков
`e0184ed7`, выбывшей 11.08. Прогон честно дал 1.68209, расхождение вдвое выше
порога, и читатель объявил бы репозиторий невоспроизводимым.

Имя прогон не идентифицирует: `lgbm_ens` стоит в журнале трижды на трёх
версиях признаков. Поэтому проверка ищет сочетание, которое законным
не бывает: имя есть в журнале, число совпало с **устаревшей** строкой,
а у более поздней строки того же имени число другое. Это ссылка на выбывший
артефакт, и вердикт здесь однозначен, как у команд.

**2. Команды.** Из документов вынимаются все строки `python -u src/*.py ...`,
и каждый флаг сверяется с выводом `--help` того самого скрипта. Здесь
расхождение — всегда дефект: команда просто не запустится.

Почему проверка «числа» намеренно не падает при несовпадении. Автоматический
вердикт здесь невозможен — законных несовпадений большинство. Задача скрипта
не решить за человека, а **сузить просмотр** вдвое-втрое и не дать
пропустить ни одного. Команды — наоборот, там вердикт однозначен.

    python -u src/doc_audit.py            # обе проверки
    python -u src/doc_audit.py --quiet    # только вердикт по командам
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import re
import subprocess
import sys

SUB_LOG = pathlib.Path("submissions/log.csv")
# Область у проверок РАЗНАЯ, и это не небрежность.
# Числа сверяются только по документам для читателя: `TASKS.md` — рабочий
# журнал, его числа законно не совпадают со счетами лидерборда сотнями,
# и включение его превращает полезный список из тридцати строк в шум из ста.
# Команды сверяются везде: незапускаемая команда вредна в любом файле.
DOC_NUM = sorted(pathlib.Path("docs/jury").glob("*.md")) + [pathlib.Path("README.md")]
DOC_CMD = DOC_NUM + [pathlib.Path("TASKS.md"), pathlib.Path("PLAN.md")]

NUM_RE = re.compile(r"\b1\.6[0-9]{3,7}\b")
JOURNAL = pathlib.Path("models/experiments.csv")
# `имя` ... число — в пределах одной-двух строк текста; дальше связь между
# именем и числом уже не гарантирована, и проверка начала бы выдумывать.
REF_RE = re.compile(r"`([a-z][a-z0-9_]{2,})`(?:[^\n]|\n(?!\n)){0,120}?\b(1\.[0-9]{4,5})\b")
MARK = "<!-- выбывшая строка приведена намеренно -->"
CMD_RE = re.compile(r"python -u (src/[a-z_]+\.py)([^\n`]*)")
FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")

# Числа не из наших отправок, но законные и постоянные.
KNOWN = {1.6452460: "результат лидера"}

_help_cache: dict[str, set[str] | None] = {}


def script_flags(script: str) -> set[str] | None:
    """Флаги скрипта по его же --help. None, если argparse не отвечает."""
    if script not in _help_cache:
        r = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        out = r.stdout or ""
        _help_cache[script] = set(FLAG_RE.findall(out)) if "usage:" in out else None
    return _help_cache[script]


def check_numbers(quiet: bool) -> None:
    if not SUB_LOG.exists():
        print("  журнала отправок нет, проверка чисел пропущена")
        return
    truth = {}
    for r in csv.DictReader(io.open(SUB_LOG, encoding="utf-8")):
        v = (r.get("lb_score") or "").strip()
        if v:
            try:
                truth[round(float(v), 7)] = r["file"]
            except ValueError:
                pass
    unknown, total = [], 0
    for d in DOC_NUM:
        t = io.open(d, encoding="utf-8").read()
        for m in NUM_RE.finditer(t):
            total += 1
            x = float(m.group())
            hit = any(abs(x - k) < 5e-7 or f"{k:.7f}".startswith(m.group()) for k in truth) \
                or any(abs(x - k) < 5e-7 for k in KNOWN)
            if not hit:
                line = t[:m.start()].count("\n") + 1
                unknown.append((d.name, line, m.group(), t.split("\n")[line - 1].strip()[:88]))
    print(f"  счетов лидерборда в журнале: {len(truth)} | чисел 1.6xxxxx в документах: {total}")
    print(f"  не совпало ни с одним счётом: {len(unknown)} — каждое должно быть объяснимо")
    if not quiet:
        for f, l, x, ctx in unknown:
            print(f"    {f}:{l}  {x}\n        {ctx}")


def check_journal_refs() -> bool:
    """Ссылки документов на строки журнала прогонов: не выбыла ли строка."""
    if not JOURNAL.exists():
        print("  журнала прогонов нет, проверка пропущена")
        return True
    runs: dict[str, list[dict]] = {}
    for r in csv.DictReader(io.open(JOURNAL, encoding="utf-8")):
        runs.setdefault((r.get("name") or "").strip(), []).append(r)

    def values(row: dict) -> set[str]:
        out = set()
        for c in ("rmsle_single", "rmsle_two_stage", "rmsle_blend", "rmsle_aligned"):
            v = (row.get(c) or "").strip()
            if v:
                out.add(v)
        return out

    stale, checked = [], 0
    for d in DOC_NUM:
        t = io.open(d, encoding="utf-8").read()
        for m in REF_RE.finditer(t):
            name, num = m.group(1), m.group(2)
            rows = runs.get(name)
            if not rows:
                continue
            checked += 1
            hit = [r for r in rows if any(v.startswith(num) or num.startswith(v)
                                          for v in values(r))]
            if not hit or len(rows) == 1:
                continue
            newest = max(rows, key=lambda r: r.get("created") or "")
            if newest in hit:
                continue
            # Число принадлежит не самой поздней строке этого имени, и у поздней
            # оно другое: документ показывает на выбывший прогон.
            line = t[:m.start()].count("\n") + 1
            # Разбор собственной ошибки обязан привести выбывшее число целиком,
            # иначе рассказывать не о чем. Исключение помечается в тексте
            # и потому видно человеку, а не спрятано в списке внутри скрипта.
            para = t[t.rfind("\n\n", 0, m.start()) + 2:
                     (t.find("\n\n", m.end()) + 1 or len(t))]
            if MARK in para:
                continue
            stale.append((d.name, line, name, num,
                          (hit[0].get("feat_ver") or "?"), (newest.get("feat_ver") or "?"),
                          sorted(values(newest))[:1]))
    print(f"  ссылок «имя + число» с известным прогоном: {checked} | "
          f"устаревших: {len(stale)}")
    for f, l, n, num, old_v, new_v, cur in stale:
        print(f"    {f}:{l}  `{n}` {num} — версия признаков {old_v} выбыла; "
              f"действующая {new_v}" + (f", число {cur[0]}" if cur else ""))
    return not stale


def check_commands() -> bool:
    bad, total = [], 0
    for d in DOC_CMD:
        t = io.open(d, encoding="utf-8").read()
        for m in CMD_RE.finditer(t):
            script, tail = m.group(1), m.group(2)
            line = t[:m.start()].count("\n") + 1
            if not pathlib.Path(script).exists():
                bad.append((d.name, line, script, "СКРИПТА НЕТ"))
                continue
            known = script_flags(script)
            if known is None:
                continue
            total += 1
            for f in FLAG_RE.findall(tail):
                if f not in known:
                    bad.append((d.name, line, script, f"флага {f} у скрипта нет"))
    print(f"  команд проверено: {total} | расхождений: {len(bad)}")
    for f, l, sc, why in bad:
        print(f"    {f}:{l}  {sc} — {why}")
    return not bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="не перечислять несовпавшие числа, только счётчики")
    args = ap.parse_args()

    print("--- 1. числа документов против журнала отправок ---")
    check_numbers(args.quiet)
    print("\n--- 2. команды документов против --help скриптов ---")
    ok = check_commands()
    print("\n--- 3. ссылки документов на строки журнала прогонов ---")
    ok_refs = check_journal_refs()
    print("\n=== ИТОГ ===")
    print("  команды: " + ("все запускаются" if ok else "ЕСТЬ НЕЗАПУСКАЕМЫЕ"))
    print("  ссылки на журнал: " + ("действующие" if ok_refs else "ЕСТЬ ВЫБЫВШИЕ"))
    print("  числа: вердикт за человеком, список выше сужает просмотр")
    ok = ok and ok_refs
    if not ok:
        raise SystemExit("аудит документации НЕ пройден")


if __name__ == "__main__":
    main()
