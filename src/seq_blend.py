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
from metrics import gini_norm, report, rmse_log
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


def solve_weight(base: str, blend: str, mse0: float, mse1: float,
                 weight: float, shift: float) -> None:
    """Оптимальный вес бленда из двух уже полученных ответов лидерборда.

    Если оба сабмита стоят на одном уровне (а они стоят: сдвиг выводится так,
    чтобы mean log1p равнялся измеренному уровню окна), то их разность
    d = p_blend - p_base имеет нулевое среднее, и всё семейство

        p(t) = p_base + t * d

    сохраняет правильный уровень при любом t. Для такого направления работает
    та же арифметика, что и у зондов сдвига и размаха:

        MSE(t)  = MSE(0) - 2t*E[r*d] + t^2*E[d^2],   r = log1p(y) - p_base
        E[r*d]  = (MSE(0) - MSE(1) + E[d^2]) / 2
        t*      = E[r*d] / E[d^2],   выигрыш = E[r*d]^2 / E[d^2]

    E[d^2] считается офлайн, поэтому двух уже потраченных сабмитов достаточно:
    ни одного нового зонда на подбор веса не нужно. t = 1 отвечает тому весу
    и сдвигу, с которыми собран `blend`.
    """
    import polars as pl

    from config import SUBMISSIONS

    def lg(f):
        return np.log1p(pl.read_csv(SUBMISSIONS / f)["predict"].to_numpy().astype(np.float64))

    d = lg(blend) - lg(base)
    ed2 = float((d ** 2).mean())
    print(f"  среднее направления d: {d.mean():+.6f} (должно быть ~0, иначе уровни файлов разные)")
    print(f"  E[d^2] = {ed2:.6f}")
    if abs(d.mean()) > 0.005:
        print("  ВНИМАНИЕ: уровни файлов заметно различаются, семейство p_base + t*d "
              "меняет уровень — результат будет смещён")

    m0, m1 = mse0 ** 2, mse1 ** 2
    erd = (m0 - m1 + ed2) / 2
    t = erd / ed2
    best = max(m0 - erd ** 2 / ed2, 0.0) ** 0.5
    print(f"\n  E[r*d] = {erd:+.6f}")
    print(f"  оптимальное t = {t:.4f}")
    print(f"    вес сети = {weight:.4f} * {t:.4f} = {weight * t:.4f}")
    print(f"    сдвиг    = {shift:+.5f} * {t:.4f} = {shift * t:+.5f}")
    print(f"  ожидаемый RMSLE {best:.7f} "
          f"(выигрыш {mse0 - best:+.5f} к базе, {mse1 - best:+.5f} к текущему бленду)")
    print("\n  сверка формулы на известных точках:")
    for tt, lab, known in ((0.0, base, mse0), (1.0, blend, mse1)):
        pred = (m0 - 2 * tt * erd + tt ** 2 * ed2) ** 0.5
        print(f"    t={tt:.1f} {lab:<28} предсказано {pred:.7f} | ответ {known:.7f}")


def expand(patterns: list[str]) -> list[str]:
    """Раскрыть маски вида `gru_final_s7_*.csv` в имена файлов из submissions/.

    Имена сабмитов содержат время создания, поэтому набирать их руками — верный
    способ ошибиться. Маска раскрывается по алфавиту, дубликаты убираются.
    """
    from config import SUBMISSIONS

    out: list[str] = []
    for p in patterns:
        if any(ch in p for ch in "*?["):
            found = sorted(f.name for f in SUBMISSIONS.glob(p))
            if not found:
                raise SystemExit(f"под маску {p!r} в submissions/ ничего не подошло")
            if len(found) > 1:
                print(f"  маска {p} -> {len(found)} файлов: {', '.join(found)}")
            out.extend(found)
        else:
            out.append(p)
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def derive_shift(base: str, new: str, b_base: float) -> None:
    """Перенести измеренный зондом сдвиг с одной модели на другую — без зонда.

    Сдвиг это среднее остатка: b = E[log1p(y)] - E[log1p(p)]. Первое слагаемое
    зависит только от тестового окна и одинаково для всех моделей, поэтому

        E[log1p(y)] = mean log1p(p_base) + b_base
        b_new       = E[log1p(y)] - mean log1p(p_new)

    Равенство точное. Один зонд, потраченный когда-то на одну модель, задаёт
    уровень тестового окна раз и навсегда — новой модели свой зонд уровня
    больше не нужен, нужен только зонд размаха.

    Осторожно: b_base должен относиться именно к файлу `base`. Если зонд мерили
    на другой сборке той же модели, добавляется неточность порядка разницы
    средних между сборками.
    """
    import polars as pl

    from config import SUBMISSIONS

    logs = {}
    for tag, f in (("база", base), ("новая", new)):
        p = np.log1p(pl.read_csv(SUBMISSIONS / f)["predict"].to_numpy().astype(np.float64))
        logs[tag] = p
        print(f"  {tag:<7} {f:<32} mean log1p {p.mean():.5f} | var {p.var():.5f}")

    y_mean = logs["база"].mean() + b_base
    b_new = y_mean - logs["новая"].mean()
    print(f"\n  уровень тестового окна E[log1p(y)] = {logs['база'].mean():.5f} "
          f"{b_base:+.4f} = {y_mean:.5f}")
    print(f"  оптимальный сдвиг для новой модели: {b_new:+.5f}")
    print(f"  ожидаемый выигрыш от него: {b_new ** 2:.5f} в шкале MSE")
    if abs(b_new) < 0.02:
        print("\n  вывод: уровень новой модели уже верный, сдвигать почти нечего")
    else:
        print(f"\n  применить: python -u src/probe_shift.py --source {new} "
              f"--delta {b_new:.4f} --out <имя>.csv")
    print("\n  Растяжение так не переносится: оптимальное alpha = 1 + Cov(остаток, p)/Var(p),")
    print(f"  а Cov без ответа лидерборда не восстановить. Отношение дисперсий "
          f"новая/база = {logs['новая'].var() / logs['база'].var():.4f}.")


def average_submissions(sources: list[str], out: str | None) -> None:
    """Усреднить несколько сабмитов одной модели в log1p-шкале (сеть по сидам).

    Зачем отдельно от blend_submissions. На валидации сеть проверялась средним
    трёх сидов (1.68446), а один сид даёт 1.68266…1.68804 — разброс 0.0054,
    больше, чем весь выигрыш бленда. Сабмит из одного сида поэтому слабее того,
    что измерено, и с вероятностью около трети попадёт на худший сид.
    """
    import polars as pl

    from config import SUBMISSIONS

    sources = expand(sources)
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

    sources = expand(sources)
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


def leveled(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Оптимальный сдвиг перебором: аналитический оптимум смещён обрезкой нуля."""
    grid = np.arange(-0.40, 0.26, 0.0025)
    return p + grid[int(np.argmin([rmse_log(y, p + d) for d in grid]))]


def compose_row(spec: str, name: str, val_cut: dt.date, note: str) -> None:
    """Метрики совместного состава на валидации со строкой в общий журнал.

    Зачем отдельно от `compose.py`. Тот считает и печатает, но строки в журнал
    не пишет и берёт ровно две половины. У отправленных файлов команды строк в
    `experiments.csv` нет вовсе, поэтому валидационный Gini у лучшего файла —
    пустая клетка в таблице tie-breaker'ов, которую смотрит жюри. Здесь состав
    задаётся произвольным числом участников с весами, а результат попадает
    в журнал, откуда его берут наравне с обычными прогонами.

    Уровень каждого участника выравнивается перед смешиванием: разница уровней
    иначе подмешалась бы в состав постоянным сдвигом. Уровень самого состава
    выравнивается тоже — в сабмите он и так правится бесплатно, и засчитывать
    его исправление модели нельзя.
    """
    parts = []
    for item in spec.split(","):
        path, _, w = item.rpartition("=")
        if not path:
            raise SystemExit(f"нужно вида файл.npz=вес, получено {item!r}")
        d = np.load(ROOT / path if not str(path).startswith(("/", "E:", "C:")) else path)
        parts.append((path.strip(), float(w), d))

    base = parts[0][2]
    uid = base["user_id"]
    tgt = next((p[2]["target"] for p in parts if "target" in p[2]), None)
    if tgt is None:
        raise SystemExit("ни в одном файле нет таргета")
    y = np.log1p(tgt)

    total = np.zeros(len(uid), dtype=np.float64)
    print(f"{'участник':<44}{'вес':>7}{'RMSLE':>10}{'Gini':>9}")
    for path, w, d in parts:
        p = d["pred_log"]
        if not np.array_equal(d["user_id"], uid):
            pos = np.searchsorted(np.sort(d["user_id"]), uid)
            order = np.argsort(d["user_id"])
            if not np.array_equal(d["user_id"][order][pos], uid):
                raise SystemExit(f"{path}: другой набор пользователей")
            p = p[order][pos]
        p = leveled(y, p)
        total += w * p
        e = np.expm1(np.clip(p, 0, None))
        print(f"{path.split('/')[-1]:<44}{w:>7.3f}{rmse_log(y, p):>10.5f}"
              f"{gini_norm(tgt, e):>9.4f}")

    total = leveled(y, total)
    res = report(tgt, np.expm1(np.clip(total, 0, None)), f"состав {name}")
    append_csv(MODELS / "experiments.csv", EXPERIMENT_FIELDS, {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
        "feat_ver": "compose", "blocks": "all+seq", "name": name, "model": "blend",
        "cutoffs": "", "n_features": len(parts),
        "rmsle_single": "", "rmsle_two_stage": "", "rmsle_blend": round(res["rmsle"], 5),
        "blend_w": "", "gini_blend": round(res["gini"], 4),
        "sum_bias_blend": round(res["sum_bias"], 4), "best_iter_single": "",
        "stride": 30, "halflife": "", "val_cutoff": str(val_cut), "train_cutoffs": "",
        "note": note or ("состав: " + ", ".join(
            f"{p.split('/')[-1]} {w:.3f}" for p, w, _ in parts)
            + " [уровень выровнен у участников и у состава]")})


def sweep_weights(spec: str, at: str | None, tol: float = 0.0001) -> None:
    """Насколько плоская кривая веса — точным решением, а не перебором.

    Зачем. Веса состава подобраны на одном срезе валидации, и трек A ставит
    правильный вопрос: не подогнаны ли они под шум. У бленда голов веса скакали
    от 0.2 до 0.9 между прогонами — верный признак плоской кривой, на которой
    оптимум ничего не значит. Перебор на это не отвечает: он показывает, где
    минимум, но не показывает, сколько стоит от него отойти.

    Перебирать и не надо. После выравнивания уровня остаток каждого участника
    имеет нулевое среднее, и при сумме весов 1 остаток состава — это просто
    взвешенная сумма остатков. Значит MSE(w) = w'Cw, где C — ковариация
    остатков: **точная квадратичная форма**, а не приближение. Отсюда сразу

        оптимум      w* = C^-1 1 / (1' C^-1 1)
        цена отхода  (w - w*)' C (w - w*)

    Второе и есть ответ на вопрос о подгонке. Если отход к равным весам стоит
    меньше шума лидерборда, веса не значимы, и на private безопаснее равные:
    там выборка вчетверо больше, а подгонка под январский шум не переносится.

    Радиус — насколько далеко веса можно сдвинуть в худшую сторону, оставаясь
    дешевле `tol` по RMSLE. Считается по наибольшему собственному числу C на
    подпространстве сумма-ноль, то есть это гарантия для любого направления.
    """
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    if len(paths) < 2:
        raise SystemExit("нужно хотя бы два участника")

    base = np.load(ROOT / paths[0])
    uid = base["user_id"]
    tgt = base["target"]
    y = np.log1p(tgt)

    resid = np.empty((len(uid), len(paths)), dtype=np.float64)
    print(f"{'участник':<44}{'RMSLE':>10}{'Gini':>9}")
    for j, path in enumerate(paths):
        d = np.load(ROOT / path)
        p = d["pred_log"]
        if not np.array_equal(d["user_id"], uid):
            order = np.argsort(d["user_id"])
            pos = np.searchsorted(d["user_id"][order], uid)
            if not np.array_equal(d["user_id"][order][pos], uid):
                raise SystemExit(f"{path}: другой набор пользователей")
            p = p[order][pos]
        p = leveled(y, p)
        resid[:, j] = p - y
        print(f"{path.split('/')[-1]:<44}{rmse_log(y, p):>10.5f}"
              f"{gini_norm(tgt, np.expm1(np.clip(p, 0, None))):>9.4f}")

    k = len(paths)
    cov = resid.T @ resid / len(uid)
    inv1 = np.linalg.solve(cov, np.ones(k))
    w_opt = inv1 / inv1.sum()

    def mse(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    points = [("оптимум на этом срезе", w_opt),
              ("равные веса", np.full(k, 1.0 / k))]
    if at:
        w_at = np.array([float(v) for v in at.split(",")], dtype=np.float64)
        if len(w_at) != k:
            raise SystemExit(f"--at: нужно {k} весов, дано {len(w_at)}")
        if abs(w_at.sum() - 1.0) > 1e-6:
            raise SystemExit(f"--at: веса должны давать в сумме 1, дано {w_at.sum():.4f}")
        points.insert(1, ("веса состава", w_at))

    best = mse(w_opt)
    print(f"\n{'точка':<24}{'веса':<34}{'RMSLE':>10}{'цена':>10}")
    for label, w in points:
        cur = mse(w)
        # Цена в шкале RMSLE, а не MSE: сравнивать её будут с ответами лидерборда.
        print(f"{label:<24}{' '.join(f'{v:.3f}' for v in w):<34}"
              f"{cur ** 0.5:>10.5f}{cur ** 0.5 - best ** 0.5:>+10.5f}")

    # Худшее направление сдвига весов: сумма-ноль, наибольшее собственное число.
    basis = np.eye(k)[:, 1:] - np.eye(k)[:, :1]
    basis /= np.linalg.norm(basis, axis=0)
    lam = float(np.linalg.eigvalsh(basis.T @ cov @ basis).max())
    radius = (2 * tol * best ** 0.5 / lam) ** 0.5
    print(f"\nрадиус: веса можно сдвинуть на {radius:.3f} в любую сторону "
          f"ценой не больше {tol} RMSLE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None,
                    help="файлы состава через запятую без весов: считает оптимум, "
                         "цену равных весов и радиус плоскости")
    ap.add_argument("--at", default=None,
                    help="веса, которыми состав собран на самом деле (для --sweep)")
    ap.add_argument("--compose", default=None,
                    help="состав через запятую: файл.npz=вес,файл.npz=вес. "
                         "Считает метрики и пишет строку в experiments.csv")
    ap.add_argument("--name", default="compose", help="имя строки в журнале")
    ap.add_argument("--blend-submissions", default=None,
                    help="смешать два готовых сабмита через запятую: бустинг,сеть "
                         "(вместо проверки на валидации)")
    ap.add_argument("--average-submissions", default=None,
                    help="усреднить несколько сабмитов одной модели в log1p (сеть по сидам)")
    ap.add_argument("--solve-weight", default=None,
                    help="оптимальный вес бленда по двум ответам лидерборда: "
                         "база.csv,бленд.csv (нужны --mse0, --mse1, --weight, --shift)")
    ap.add_argument("--mse0", type=float, default=None, help="RMSLE базового файла")
    ap.add_argument("--mse1", type=float, default=None, help="RMSLE бленда")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="сдвиг, с которым собран бленд (для пересчёта на оптимальный вес)")
    ap.add_argument("--derive-shift", default=None,
                    help="перенести измеренный сдвиг с одной модели на другую без зонда: "
                         "база.csv,новая.csv (нужен --b-base)")
    ap.add_argument("--b-base", type=float, default=0.0578,
                    help="сдвиг, измеренный зондом для базового файла")
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

    if args.sweep:
        sweep_weights(args.sweep, args.at)
        return

    if args.compose:
        compose_row(args.compose, args.name,
                    dt.date.fromisoformat(args.val_cutoff), args.note)
        return

    if args.solve_weight:
        parts = [s.strip() for s in args.solve_weight.split(",") if s.strip()]
        if len(parts) != 2 or args.mse0 is None or args.mse1 is None:
            raise SystemExit("нужно база.csv,бленд.csv плюс --mse0 и --mse1")
        solve_weight(parts[0], parts[1], args.mse0, args.mse1, args.weight, args.shift)
        return
    if args.derive_shift:
        parts = [s.strip() for s in args.derive_shift.split(",") if s.strip()]
        if len(parts) != 2:
            raise SystemExit("нужно два файла через запятую: база.csv,новая.csv")
        derive_shift(parts[0], parts[1], args.b_base)
        return
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
