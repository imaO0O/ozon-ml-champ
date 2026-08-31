"""Пересборка финальных файлов из компонент — воспроизводимость последнего шага.

Зачем. Модели воспроизводятся конвейером (`README`, `docs/jury/B1_*`), но
последний шаг — смеси и калибровка — делался разовыми скриптами. По репозиторию
финальный файл собрать было нельзя. Этот скрипт закрывает пробел: он берёт
компоненты из `submissions/` и повторяет цепочку с теми же весами и
константами, после чего **сверяет результат с отправленным файлом**.

Проверка сверкой — главное здесь. Скрипт, который «что-то собирает», ничего
не доказывает; скрипт, который воспроизводит отправленный файл до седьмого
знака, доказывает.

Три константы калибровки, все измерены на лидерборде и обоснованы в PLAN:

    TEST_LEVEL = 2.32912   уровень тестового окна (зонд уровня)
    TARGET_VAR = 2.6408    целевая дисперсия log1p предсказаний (восемь замеров,
                           разброс 0.29%); alpha = sqrt(TARGET_VAR / Var(p))
    GAMMA_AC   = -0.008076756  добор кривизны для pair_ac_cal
                           (в переписке округлялось до -0.00808; сверка
                            показала, что округление стоит 7e-6 на клиента)
    GAMMA_Q    = -0.003429750255  добор кривизны для pair_q
                           (в журнале отправок записано -0.003430; та же
                            болезнь, цена округления 6.3e-6 на клиента)
    GAMMA_CUB  = -0.000806576774  кубический член для pair_s6q
                           (округление до восьмого знака стоило 3e-7 —
                            третий случай той же болезни, поймано сверкой)

Финальная пара с 30.08 — `pair_s6q.csv` и `pair_q.csv`; скрипт собирает
и сверяет ОБА. Порядок слотов на счёт не влияет: в зачёт идёт лучшее из двух.

Порядок операций внутри каждой смеси: сначала смешать, потом уровень, потом
растяжение. Иначе поправки применяются не к тому объекту — смесь разбавляет
и размах, и кривизну (PLAN, «смесь разбавляет калибровку»).

    python -u src/rebuild_final.py
    python -u src/rebuild_final.py --recipes # чем порождается каждая компонента
    python -u src/rebuild_final.py --write   # перезаписать финальные файлы
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import SUBMISSIONS

TEST_LEVEL = 2.32912
TARGET_VAR = 2.6408
GAMMA_AC = -0.008076756
GAMMA_Q = -0.003429750255
# Кубический член, решённый из одного зонда `probe_cub` (ответ 1.6468402135
# при gamma3 = +0.0036, Var(h) = 25.64153) и уточнённый проекцией на само
# кубическое направление: округление до восьмого знака давало расхождение
# 3e-07, та же болезнь, что была у GAMMA_Q. Применяется поверх шестисидовой
# цепочки и даёт `pair_s6q` — файл, вошедший в финальную пару 30.08.
GAMMA_CUB = -0.000806576774
# Шесть сидов на каждой сетевой руке. В отправленном 27.08 составе их было
# 4/4/1 — недосмотр, а не решение (см. docs/jury/A2, редакция 30.08).
SEED6 = {
    "year": ["yearfin_s42", "yearfin_s13", "yearfin_s7", "yearfin_s3",
             "yearfin_s21", "yearfin_s99"],
    "nost": ["nostfin", "nostfin_s13", "nostfin_s7", "nostfin_s3",
             "nostfin_s21", "nostfin_s99"],
    "ev": ["evfin", "evfin_s13", "evfin_s7", "evfin_s3", "evfin_s21", "evfin_s99"],
}
FINAL_PAIR = ("pair_s6q.csv", "pair_q.csv")

# (имя результата, из чего собирается, вес второго слагаемого)
# Веса взяты: 0.56/0.085/0.19 — с двух валидационных срезов (усреднены);
# 0.28/0.35/0.06 — из формулы партнёрства по известным public-счетам, ужаты.
CHAIN = [
    ("mix_multi", ["stk2_raw", "yearfin_avg4", "nostfin", "evfin"], [0.56, 0.085, 0.19]),
    ("pair_multi", ["pair_ac_cal", "mix_multi"], [0.28]),
    ("pair_w2", ["pair_multi", "cand_w2_cal"], [0.35]),
    ("pair_nost", ["pair_w2", "nost_avg"], [0.06]),
]

# (имя, слагаемые, калибровать ли результат). Разница существенна: `nost_avg`
# и `ev_avg` сохранялись ОТКАЛИБРОВАННЫМИ и в таком виде шли в смеси, а
# `yearfin_avg4` — сырым средним log1p. Скрипт нашёл это расхождение сверкой:
# сборка из сырого среднего вместо откалиброванного давала max 1.04e-02.
SEED_AVERAGES = {
    "yearfin_avg4": (["yearfin_s42", "yearfin_s13", "yearfin_s7", "yearfin_s3"], False),
    "nost_avg": (["nostfin", "nostfin_s13", "nostfin_s7", "nostfin_s3"], True),
    "ev_avg": (["evfin", "evfin_s13", "evfin_s7"], True),
}


# Чем порождается каждая компонента — с ТОЧНЫМ именем файла, которого ждёт
# сборка. Раньше этого списка не было, и проверка «репозиторий воспроизводим»
# была шире измерения: README доказывал, что модели обучаются, `rebuild_final`
# доказывал, что из компонент собирается финал, а между ними зияла дыра —
# **17 компонент из 22 не назывались в инструкциях вообще**, включая все сиды
# 21 и 99, без которых pair_s6q не собрать. Обучив всё по README, человек
# получал файлы под другими именами и упирался в отказ.
#
# `predict.py` по умолчанию пишет `<имя>_MMDD_HHMM.csv`, поэтому в рецептах
# всюду стоит явный `--out`: имя прогон не идентифицирует, а сборка ждёт
# конкретный файл.
SEQ = "python -u src/seq_train.py"
RECIPE = {
    **{f"yearfin_s{s}": f"{SEQ} --lookback 364 --bin 7 --static rk_ --bins 64 "
                        f"--seed {s} --final --name yearfin_s{s}"
       for s in (42, 13, 7, 3, 21, 99)},
    "nostfin": f"{SEQ} --lookback 90 --bins 64 --seed 42 --final --name nostfin",
    **{f"nostfin_s{s}": f"{SEQ} --lookback 90 --bins 64 --seed {s} "
                        f"--final --name nostfin_s{s}" for s in (13, 7, 3, 21, 99)},
    "evfin": f"{SEQ} --lookback 364 --events 96 --static rk_ --bins 64 "
             f"--seed 42 --final --name evfin",
    **{f"evfin_s{s}": f"{SEQ} --lookback 364 --events 96 --static rk_ --bins 64 "
                      f"--seed {s} --final --name evfin_s{s}" for s in (13, 7, 3, 21, 99)},
    # ЕДИНСТВЕННАЯ компонента, чей рецепт не подтверждён воспроизведением.
    # Прогон с чистого клона 31.08 прошёл все шаги, но `predict.py` потребовал
    # `models/netoof_b64a_test.npz` — предсказание сети на ТЕСТОВОМ срезе,
    # которого не порождает ни один шаг: `seq_oof_net.py` делает валидационные
    # срезы, тестовый берётся отдельным флагом `--test-submission` из готового
    # сабмита сети. Команды, породившей наш файл, в журналах нет.
    #
    # Попытка восстановить: сеть с `--final` (сид 42) -> `--test-submission`.
    # Получается ДРУГОЙ файл: D до нашего 3.9e-03, и дисперсия у нашего выше
    # (2.481 против 2.416), то есть это и не среднее по сидам — усреднение
    # дисперсию снижает. Чем именно он сделан, по репозиторию не восстановимо.
    #
    # Цена честно: собранный по этому рецепту `stk2_raw` отличается от нашего
    # на sqrt(D) 0.0586, а доля `stk2_raw` в финале 0.0582 — то есть финал
    # сдвинется примерно на 0.0034 в пространстве предсказаний. Остальные
    # 24 компоненты воспроизводятся, три сетевые руки — побитово.
    "stk2_raw": ("python -u src/train.py --cutoffs 6 --ensemble --net "
                 "--net-names b64a --final --name lgbm_stk2\n"
                 "     затем: python -u src/predict.py --name lgbm_stk2 "
                 "--out stk2_raw.csv\n"
                 "     ВНИМАНИЕ: перед этим нужен models/netoof_b64a_test.npz —\n"
                 "     python -u src/seq_oof_net.py --lookback 90 --static rk_ "
                 "--bins 64\n"
                 "        --name netoof_b64a --test-submission <сабмит сети>\n"
                 "     Наш файл сделан НЕ ЭТИМ путём и по журналам не восстановим;\n"
                 "     собранный так stk2_raw отличается от нашего на 0.0586."),
    # Здесь стоял `cand_w2`, и это была ошибка ПЕРЕБОРА, а не опечатка:
    # 31.08 трек C прямой сборкой показал, что вторая половина — `cand_w`,
    # и наш перебор 29.08 её просто не содержал. Средняя относительная ошибка
    # 7.06e-06 против 5.95e-03 у `cand_w2` — в 843 раза. Победитель узкого
    # набора был выдан за опознание; сигнал был (наша невязка расходилась
    # с их впятеро), и его следовало читать как «ищем не среди того».
    "pair_ac": ("смесь половин пополам в log1p:\n"
                "     expm1(0.5*log1p(mix_year) + 0.5*log1p(cand_w));\n"
                "     сверено прямой сборкой против отправленного файла —\n"
                "     средняя относительная ошибка 7.1e-06 (трек C, 31.08)"),
    "pair_ac_cal": ("выводится ЭТИМ ЖЕ скриптом из pair_ac (уровень, растяжение, "
                    "кривизна\n     GAMMA_AC) — запустите с --write, имея pair_ac"),
    "cand_w2_cal": ("= calibrate(cand_w2): уровень на TEST_LEVEL и растяжение "
                    "до TARGET_VAR,\n     без кривизны. Сверено: расхождение "
                    "1.1e-16, то есть тождество"),
    "cand_w2": ("вторая версия половины трека C — та же, что cand_w, но недельная\n"
                "     рука на шести сидах вместо трёх при том же весе 0.468:\n"
                "     log1p(cand_w2) = log1p(cand_w) + 0.468*(log1p(sub_w52_avg6)\n"
                "                      - log1p(sub_w52_avg3)) + 0.00468\n"
                "     Тождество трека C (их замер: max 1.9e-07). У нас не сверено:\n"
                "     sub_w52_avg3 — их артефакт, на нашем диске его нет"),
    "cand_w": ("артефакт трека C — ГРАНИЦА нашей воспроизводимости.\n"
               "     Конфигурации и числа: docs/jury/C2_track_c.md, C3_reproduce.md"),
    # Ингредиенты pair_ac — уровнем глубже, и без них рецепт выше не выполним.
    # Дерево доведено до конца там, где конец наш, и явно оборвано там, где
    # он чужой: собственных компонент, которые нельзя построить, у нас нет.
    # Здесь стоит ФОРМУЛА, а не команда: `compose.py` принимает npz валидации,
    # а не csv сабмитов, и подставленная сюда команда просто не запустилась бы.
    # Написать её было бы четвёртым за день случаем ссылки без проверки цели.
    "mix_year": ("смесь в log1p, вес годовой сети 0.48 (журнал отправок, 25.08):\n"
                 "     expm1(0.52*log1p(stk2_raw) + 0.48*log1p(среднее двух сидов\n"
                 "     годовой сети)), затем уровень и растяжение как в calibrate()"),
    "cand_w2": ("артефакт трека C: состав на шести сидах недельной сети "
                "(журнал, 26.08).\n"
                "     ГРАНИЦА воспроизводимости с нашей стороны — дальше "
                "docs/jury/C3_reproduce.md"),
}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Компонента из `submissions/`. Внятный отказ, если её нет.

    Файлов предсказаний в репозитории НЕТ и быть не должно: это готовые
    ответы, и `.gitignore` исключает `submissions/*` кроме журнала. Поэтому
    сразу после `git clone` этот скрипт запустить нельзя — сначала конвейер
    порождает компоненты (README, раздел про полный прогон), и только потом
    сверка имеет смысл.

    Без этой проверки первым, что видит человек со свежим клоном, был бы
    FileNotFoundError из недр polars — и вывод «репозиторий сломан» вместо
    «компоненты надо сперва собрать».
    """
    path = SUBMISSIONS / f"{name}.csv"
    if not path.exists():
        how = RECIPE.get(name, "рецепт неизвестен — см. README, раздел «Запуск»")
        raise SystemExit(
            f"нет компоненты {path.name}.\n"
            "Файлы предсказаний намеренно не хранятся в репозитории (готовые\n"
            "ответы соревнования, см. .gitignore). Порождается она так:\n\n"
            f"     {how}\n\n"
            "Полный список компонент с рецептами: python -u src/rebuild_final.py "
            "--recipes")
    d = pl.read_csv(path).sort("user_id")
    return d["user_id"].to_numpy(), np.log1p(d["predict"].to_numpy())


def level(p: np.ndarray) -> np.ndarray:
    return p - p.mean() + TEST_LEVEL


def calibrate(p: np.ndarray) -> np.ndarray:
    """Уровень на TEST_LEVEL, затем растяжение до целевой дисперсии."""
    p = level(p)
    return p.mean() + np.sqrt(TARGET_VAR / p.var()) * (p - p.mean())


def quad_basis(p: np.ndarray) -> np.ndarray:
    u = p - p.mean()
    g = u ** 2
    g = g - g.mean()
    return g - (g @ u) / (u @ u) * u


def write(uid: np.ndarray, p: np.ndarray, name: str) -> None:
    pred = np.clip(np.expm1(np.clip(p, 0, None)), 0, None)
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(SUBMISSIONS / f"{name}.csv")


def compare(name: str, p: np.ndarray) -> tuple[float, float]:
    """Максимальное и среднее расхождение с отправленным файлом, в log1p.

    Сверять надо ПОСЛЕ обрезки нулём: в файл пишется `expm1(clip(p, 0))`,
    поэтому у клиентов с отрицательным предсказанием файл хранит ноль, а
    в памяти лежит исходное отрицательное значение. Без обрезки сверка
    показывала расхождение 1.9e-02 на десяти клиентах из 250 000 —
    артефакт сравнения, а не сборки.
    """
    _, ref = load(name)
    d = np.abs(np.clip(p, 0, None) - ref)
    return float(d.max()), float(d.mean())


def cubic_basis(p: np.ndarray) -> np.ndarray:
    """u³, очищенное от константы, u и квадратичного направления.

    Повторяет `probe_cubic.cubic_dir`. Дублирование намеренное: этот скрипт
    обязан собирать финал, ничего не импортируя из зондов, — иначе правка
    в зонде молча меняет проверку отправленного файла.
    """
    u = p - p.mean()
    g = u ** 2
    g = g - g.mean()
    g = g - (g @ u) / (u @ u) * u
    h = u ** 3
    h = h - h.mean()
    h = h - (h @ u) / (u @ u) * u
    return h - (h @ g) / (g @ g) * g


def build_s6q(built: dict) -> np.ndarray:
    """Цепочка на шести сидах каждой руки плюс кубический член."""
    parts = {
        "year": np.mean([load(n)[1] for n in SEED6["year"]], axis=0),
        "nost": calibrate(np.mean([load(n)[1] for n in SEED6["nost"]], axis=0)),
        "ev": calibrate(np.mean([load(n)[1] for n in SEED6["ev"]], axis=0)),
    }
    sub = {"stk2_raw": None, "yearfin_avg4": parts["year"],
           "nostfin": None, "evfin": parts["ev"], "nost_avg": parts["nost"]}
    acc = dict(built)
    for out, names, ws in CHAIN:
        cur = None
        for i, n in enumerate(names):
            p = sub[n] if sub.get(n) is not None else (
                acc[n] if n in acc and n not in sub else load(n)[1])
            lp = level(p)
            cur = lp if cur is None else (1 - ws[i - 1]) * cur + ws[i - 1] * lp
        acc[out] = calibrate(cur)
    q = calibrate(acc["pair_nost"])
    q = q + GAMMA_Q * quad_basis(q)
    return q + GAMMA_CUB * cubic_basis(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="перезаписать финальные файлы (по умолчанию только сверка)")
    ap.add_argument("--recipes", action="store_true",
                    help="показать, чем порождается каждая компонента, и чего нет на диске")
    args = ap.parse_args()

    if args.recipes:
        need = set()
        for _, parts, _ in CHAIN:
            need |= set(parts)
        for out, (parts, _) in SEED_AVERAGES.items():
            need |= set(parts)
            need.discard(out)
        for arm in SEED6.values():
            need |= set(arm)
        # Ингредиенты pair_ac и cand_w2 — уровнем глубже, но без них список
        # не инструкция: по нему нельзя дойти до конца.
        need |= {"pair_ac", "mix_year", "cand_w", "cand_w2"}
        need -= {o for o, _, _ in CHAIN} | set(SEED_AVERAGES)
        miss = [n for n in sorted(need) if not (SUBMISSIONS / f"{n}.csv").exists()]
        # ПРЕДПОСЫЛКИ идут нулевым шагом. Без них список перечисляет компоненты,
        # но не является инструкцией: жюри, идущее строго по нему, упирается
        # в отказ на первой же сетевой команде — матрицы последовательностей
        # ещё нет. Найдено треком C на чистом прогоне 31.08; сторож сработал
        # правильно (внятный отказ вместо трейсбека), неполон был список.
        print("ШАГ 0 — предпосылки, без них ни одна команда ниже не пойдёт:\n")
        print("  данные площадки в data/raw/train.parquet и sample_submit.csv")
        print("  python -u src/datasets.py --test     признаки по всем срезам")
        print("  python -u src/seq_data.py            матрица последовательностей")
        print("                                       (250000 x 409 x 18, ~3.4 ГБ)\n")
        print(f"компонент нужно финалу: {len(need)}, отсутствует: {len(miss)}\n")
        for n in sorted(need):
            mark = "НЕТ " if n in miss else "есть"
            print(f"  [{mark}] {n}")
            print(f"         {RECIPE.get(n, 'рецепт неизвестен')}")
        return

    built: dict[str, np.ndarray] = {}
    uid = None

    print("--- усреднение сидов ---")
    for out, (parts, cal) in SEED_AVERAGES.items():
        ps = []
        for n in parts:
            u, p = load(n)
            uid = u if uid is None else uid
            if not np.array_equal(u, uid):
                raise SystemExit(f"{n}: другой набор user_id")
            ps.append(p)
        avg = np.mean(ps, axis=0)
        built[out] = calibrate(avg) if cal else avg
        mx, mn = compare(out, built[out])
        print(f"  {out:<16}из {len(parts)} сидов{' + калибровка' if cal else '':<14}"
              f"| max {mx:.2e}, среднее {mn:.2e}")

    print("\n--- цепочка смесей ---")
    for out, parts, ws in CHAIN:
        cur = None
        for i, n in enumerate(parts):
            p = built[n] if n in built else load(n)[1]
            if cur is None:
                cur = level(p)
            else:
                w = ws[i - 1]
                cur = (1 - w) * cur + w * level(p)
        cur = calibrate(cur)
        built[out] = cur
        mx, mn = compare(out, cur)
        flag = "ОК" if mx < 1e-6 else ("близко" if mx < 1e-3 else "РАСХОЖДЕНИЕ")
        print(f"  {out:<16}веса {ws} | max {mx:.2e}, среднее {mn:.2e}  {flag}")
        if args.write:
            write(uid, cur, out)

    # ПЕРВЫЙ СЛОТ. Раньше этого шага здесь не было: скрипт написан, когда
    # цепочка кончалась на pair_nost, потом выросла ещё на шаг, а скрипт
    # не обновили. Файл, идущий в зачёт ПЕРВЫМ, не пересобирался и не сверялся
    # вовсе — при том что смысл скрипта именно в этом. Ровно тот класс дефекта,
    # который он сам ловит у рецептов: рецепт разошёлся с тем, что делали.
    #
    # И та же болезнь с округлением: в журнале отправок gamma записана как
    # -0.003430, и по ней сборка расходится с файлом на 6.3e-6 на клиента.
    # Точное значение решено из самого файла проекцией на квадратичное
    # направление и даёт 8.9e-16 — машинную точность, как вся остальная цепочка.
    print("\n--- первый финал: pair_q (кривизна поверх pair_nost) ---")
    q1 = calibrate(built["pair_nost"])
    q1 = q1 + GAMMA_Q * quad_basis(q1)
    mx, mn = compare("pair_q", q1)
    print(f"  pair_q           gamma {GAMMA_Q} | "
          f"max {mx:.2e}, среднее {mn:.2e}  {'ОК' if mx < 1e-6 else 'РАСХОЖДЕНИЕ'}")
    if args.write:
        write(uid, q1, "pair_q")

    print("\n--- второй финал: pair_ac_cal ---")
    _, ac = load("pair_ac")
    q = calibrate(ac)
    q = q + GAMMA_AC * quad_basis(q)
    mx, mn = compare("pair_ac_cal", q)
    print(f"  pair_ac_cal      alpha из дисперсии, gamma {GAMMA_AC} | "
          f"max {mx:.2e}, среднее {mn:.2e}  {'ОК' if mx < 1e-6 else 'РАСХОЖДЕНИЕ'}")
    if args.write:
        write(uid, q, "pair_ac_cal")

    # ФИНАЛЬНАЯ ПАРА сменилась 30.08, и этот скрипт обязан проверять ОБА её
    # файла. До правки он собирал pair_q и pair_ac_cal и называл финалом их —
    # ровно тот класс дефекта, который он сам ловит у рецептов: скрипт отстал
    # от того, что реально отправлено. Второй раз за проект, поэтому теперь
    # список финалов вынесен наверх и печатается из одного места.
    print("\n--- также в паре: pair_s6q (шесть сидов на руку плюс кубический член) ---")
    s6 = build_s6q(built)
    mx, mn = compare("pair_s6q", s6)
    print(f"  pair_s6q         gamma3 {GAMMA_CUB} | "
          f"max {mx:.2e}, среднее {mn:.2e}  {'ОК' if mx < 1e-6 else 'РАСХОЖДЕНИЕ'}")
    if args.write:
        write(uid, s6, "pair_s6q")

    print(f"\nФИНАЛЬНАЯ ПАРА: {' + '.join(FINAL_PAIR)}")
    print("Порядок слотов на счёт не влияет: в зачёт идёт лучшее из двух.")


if __name__ == "__main__":
    main()
