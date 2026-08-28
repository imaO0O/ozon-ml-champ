"""Пересборка ВАЛИДАЦИОННОГО состава — основания всех замеров направления.

Зачем это отдельный скрипт и почему он важнее, чем кажется. Каждый вывод вида
«партнёр не проходит», «состав в оптимуме», «направление даёт ноль» измерен
против `comp_jan` / `comp_dec`. Эти файлы собирались разовым скриптом, который
не сохранился: ни строки в журнале, ни ссылки в коде. **Основание всех
направлений было невоспроизводимым** — тот же класс дефекта, который
`rebuild_final` закрывает для сабмитов.

Рецепт восстановлен точной регрессией по 123 сохранённым предсказаниям января
и 82 декабря: невязка **8·10⁻¹⁵**, то есть это не подгонка, а тождество.
Веса совпали с весами отправленной цепочки, пересчитанными в симплекс, и
совпали на обоих срезах:

    стекинг            0.3261
    годовая сеть       0.4150   среднее ДВУХ сидов (42 и 13)
    сеть без рангов    0.0689
    событийная сеть    0.1900

## Ловушка, на которой я застрял на полчаса

Четвёртой рукой в январе оказался файл с именем `epcap_12` — из опыта
про потолок эпох. Это выглядело ошибкой сборки, пока не сверились числа:
у `epcap_12` сырой RMSLE 1.71160 и `feat_ver` seq18x90b1, ровно как
у `nostfin`. **Это один и тот же прогон, сохранённый под двумя именами.**
А `nost_fp32`, которым я сначала пробовал собрать, — его fp32-двойник,
другой прогон, и он отличается на 1.38.

Отсюда практическое: имя файла не идентифицирует прогон, а числа
идентифицируют. Сверять надо по `rmsle_single` и `feat_ver`, а не по тому,
как файл назвали в тот день.

## Чем валидационный состав ОТЛИЧАЕТСЯ от отправленной цепочки

Отличается, и это надо знать при чтении любого замера направления:

* цепочка в сабмите калибрует уровень и растяжение **после каждого шага**,
  здесь калибровки нет вовсе — только выравнивание уровня в конце;
* цепочка включает половину трека C (`pair_ac_cal`, `cand_w2_cal`), здесь
  только наши четыре руки;
* годовая рука в сабмите усреднена по четырём сидам, здесь по двум.

Поэтому `comp_*` — **линейный аналог нашей половины**, а не копия сабмита.
Для замера направления этого достаточно (партнёрство определяется формой
ошибки, а не калибровкой), но выдавать одно за другое нельзя.

    python -u src/rebuild_comp.py            # сверка с сохранёнными файлами
    python -u src/rebuild_comp.py --write    # перезаписать comp_jan/comp_dec
"""
from __future__ import annotations

import argparse

import numpy as np

from config import MODELS

# (срез, {роль: имена файлов}); годовая рука — среднее перечисленных сидов.
SLICES = {
    "2026-01-15": {
        "out": "comp_jan",
        "стекинг": ["stk2_jan"],
        "годовая": ["yearw", "yearw_s13"],
        "без рангов": ["epcap_12"],
        "событийная": ["ev_jan"],
    },
    "2025-12-16": {
        "out": "comp_dec",
        "стекинг": ["stk2_dec"],
        "годовая": ["yearw_dec", "yearoof_s13_2025-12-16"],
        "без рангов": ["nost_dec"],
        "событийная": ["ev_dec"],
    },
}

# Веса НЕ подобраны, а выведены: это попарные веса цепочки (0.56 / 0.085 / 0.19
# из `rebuild_final.CHAIN`), пересчитанные в симплекс «стик-брейкингом».
# Проверка: точная регрессия по 123 предсказаниям января и 82 декабря даёт
# ровно эти числа до девятого знака, одинаковые на ОБОИХ срезах, сумма 1.0.
PAIRWISE = (0.56, 0.085, 0.19)


def simplex_weights(ws: "tuple[float, ...]") -> "dict[str, float]":
    w1, w2, w3 = ws
    return {"стекинг": (1 - w1) * (1 - w2) * (1 - w3),
            "годовая": w1 * (1 - w2) * (1 - w3),
            "без рангов": w2 * (1 - w3),
            "событийная": w3}


WEIGHTS = simplex_weights(PAIRWISE)


def load(name: str, cut: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = MODELS / f"{name}_valpred_{cut}.npz"
    if not path.exists():
        raise SystemExit(
            f"нет {path.name}.\nПредсказания валидации не хранятся в репозитории; "
            "сначала нужен прогон, который их создаёт (README, раздел про сети).")
    d = np.load(path)
    o = np.argsort(d["user_id"])
    return d["user_id"][o], d["pred_log"].astype(np.float64)[o], d["target"][o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="перезаписать comp_jan/comp_dec (по умолчанию только сверка)")
    args = ap.parse_args()

    ok = True
    for cut, spec in SLICES.items():
        out = spec["out"]
        uid = tgt = None
        acc = None
        print(f"\n--- {out} (срез {cut}) ---")
        for role, weight in WEIGHTS.items():
            parts = []
            for n in spec[role]:
                u, p, t = load(n, cut)
                if uid is None:
                    uid, tgt = u, t
                elif not np.array_equal(u, uid):
                    raise SystemExit(f"{n}: другой набор клиентов")
                parts.append(p - p.mean())
            v = np.mean(parts, axis=0)
            acc = weight * v if acc is None else acc + weight * v
            names = " + ".join(spec[role])
            print(f"  {role:<14}{weight:.4f}  {names}"
                  + (f"  (среднее {len(parts)} сидов)" if len(parts) > 1 else ""))

        ref_u, ref, _ = load(out, cut)
        d = np.abs(acc - (ref - ref.mean()))
        mx, mn = float(d.max()), float(d.mean())
        good = mx < 1e-9
        ok &= good
        print(f"  сверка с сохранённым: max {mx:.2e}, среднее {mn:.2e}  "
              f"{'ОК' if good else 'РАСХОЖДЕНИЕ'}")
        if args.write:
            y = np.log1p(tgt.astype(np.float64))
            np.savez(MODELS / f"{out}_valpred_{cut}.npz",
                     user_id=uid, pred_log=acc - acc.mean() + y.mean(), target=tgt)
            print(f"  записан {out}_valpred_{cut}.npz")

    print("\nОснование замеров направления воспроизводимо."
          if ok else "\nСБОРКА НЕ СХОДИТСЯ — замеры направления под вопросом.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
