"""Последовательностная модель (GRU / трансформер) на дневном логе — трек C.

Зачем. Все существующие признаки — свёрнутые в суммы оконные агрегаты, и
измерения показали, что этот класс исчерпан: ансамбли, гиперпараметры и новые
блоки дают меньше порога различимости 0.002. Сеть читает не объём, а форму
поведения по дням, ошибается принципиально иначе, чем деревья, и потому ценна
даже будучи слабее их: ансамбль здесь зарабатывает на непохожести ошибок
(см. PLAN.md, раздел 5), а для финала нужны два **непохожих** решения (раздел 8).

Протокол — тот же, что у бустинга, иначе числа несопоставимы:

* обучение на старых cutoff'ах, валидация на самом свежем;
* карантин: обучающий срез берётся, только если его окно таргета целиком
  раньше валидационного (`cutoff <= val - HORIZON`);
* таргет — `log1p(сумма gmv за [cutoff, cutoff + 30))`, лосс L2, то есть
  оптимизируется ровно RMSLE соревнования;
* пользователи — те, у кого есть хотя бы одно событие до cutoff'а;
* строка результата пишется в тот же `models/experiments.csv`.

Порог приёмки — тоже общий: эффект меньше 0.002 проверяется на двух срезах
(2026-01-15 и 2025-12-16) и принимается только при положительном знаке на обоих.

    python -u src/seq_data.py                                   # один раз
    python -u src/seq_train.py --arch gru --epochs 12
    python -u src/seq_train.py --arch gru --val-cutoff 2025-12-16 --cutoffs 7
    python -u src/seq_train.py --arch gru --epochs 12 --final --name gru_v1

`--save-val-pred` кладёт предсказания на валидации в `models/` — без них
бленд сети с бустингом не подобрать, а сам бустинг свои валидационные
предсказания на диск не сохраняет.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Triton ищет установленный CUDA Toolkit (nvcc), которого при обычной установке
# torch нет и не нужно: torch.compile мы не используем, а сама CUDA-рантайм
# внутри колеса torch работает. Предупреждение безвредно и только мешает читать.
warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*")

from config import (HORIZON, MODELS, SAMPLE_SUBMIT, SEED, SUBMISSIONS, TEST_CUTOFF,
                    train_cutoffs)
from datasets import get_dataset
from features import build_target, scan_log

# Границы подокон внутри горизонта для вспомогательного надзора. Последнее окно
# длиннее: 30 дней на четыре ровные недели не делятся, а плодить пятую голову
# ради двух дней смысла нет.
AUX_EDGES = [7, 14, 21, HORIZON]
from metrics import report, rmse_log
from seq_data import (CHANNELS, MEAN_CHANNELS, SUM_CHANNELS, events_window, gather,
                      history_mask, open_seq, self_norm, window_bounds)
from utils import append_csv, git_commit

# Колонки и их порядок — ровно как в самом models/experiments.csv, а не как в
# списке внутри train.py: они разошлись исторически, и append_csv при любом
# несовпадении переписывает файл целиком. Совпадение бережёт общий журнал от
# лишних перезаписей в чужих коммитах.
EXPERIMENT_FIELDS = [
    "created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs", "n_features",
    "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
    "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
    "train_cutoffs", "stride", "halflife",
]
SUBMIT_FIELDS = ["file", "created", "commit", "name", "model", "blend_w", "val_rmsle",
                 "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"]


# bfloat16 включается флагом: у него 8 бит мантиссы, а рекуррентная сеть
# прогоняет через них 180 шагов подряд, накапливая ошибку округления. Для
# трансформера это безопасно, для GRU/LSTM — повод проверить, не в точности ли
# дело, когда валидация скачет. Выключается ключом --no-amp.
_AMP = True


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=_AMP and device.type == "cuda")


def bar(total: int, desc: str, unit: str = "батч"):
    """Индикатор хода работы с общими для всего модуля настройками.

    `disable=None` — ключевая настройка: tqdm сам выключается, когда вывод не
    в терминал. Иначе лог фонового прогона превращается в одну бесконечную
    строку с возвратами каретки, а именно так мы запускаем длинные выгрузки.

    `leave=False` — по завершении строка стирается: итог печатается отдельной
    строкой, и две записи об одном и том же в логе не нужны.
    """
    return tqdm(total=max(int(total), 1), desc=desc, unit=unit, disable=None,
                leave=False, dynamic_ncols=True, mininterval=0.3,
                bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}{postfix}]")


# --------------------------------------------------------------------------- данные

def cutoff_split(n_cutoffs: int, val_cutoff: dt.date | None = None,
                 explicit_train: list[dt.date] | None = None):
    """Те же срезы и тот же карантин, что в train.load_split.

    Скопировано, а не импортировано намеренно: `load_split` поднимает выборки
    признаков (по 400 МБ на срез), которые сети не нужны совсем. Правило одно,
    и при его изменении в train.py эту функцию надо поправить следом.
    """
    cuts = train_cutoffs(n_cutoffs)
    val_cut = val_cutoff or cuts[0]
    train_cuts = explicit_train if explicit_train else [c for c in cuts if c < val_cut]
    latest_ok = val_cut - dt.timedelta(days=HORIZON)
    dropped = [c for c in train_cuts if c > latest_ok]
    train_cuts = [c for c in train_cuts if c <= latest_ok]
    if dropped:
        print(f"карантин: отброшено срезов {len(dropped)} "
              f"({', '.join(str(c) for c in dropped)}) — их таргет пересекается с валидацией")
    if not train_cuts:
        raise SystemExit(f"нет обучающих срезов раньше {latest_ok}: увеличьте --cutoffs")
    return val_cut, train_cuts


def targets_for(cutoff: dt.date, users: np.ndarray, first_day: np.ndarray):
    """(строки матрицы, таргет) для пользователей с историей до cutoff'а.

    Таргет берётся тем же `features.build_target`, что и у бустинга, поэтому
    расхождению взяться неоткуда. Отсутствие строки в таргете означает ноль
    покупок за окно, а не пропуск.
    """
    rows = np.flatnonzero(history_mask(first_day, cutoff))
    y = np.zeros(len(users), dtype=np.float64)
    tgt = build_target(cutoff)
    pos = np.searchsorted(users, tgt["user_id"].to_numpy())
    y[pos] = tgt["target"].to_numpy()
    return rows, y[rows]


def aux_targets(cutoff: dt.date, users: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Суммы gmv по подокнам горизонта — вспомогательный надзор для сети.

    Зачем. Все наши сети предсказывают одно число: сумму за 30 дней. Бустинг
    при этом раскладывает задачу на две (вероятность покупки и условный log1p),
    и это единственная постановочная развилка, которую у сети мы не пробовали —
    перебирали только представление входа и ёмкость.

    Подокна дают сети различать «когда» внутри горизонта, а не только «сколько».
    Основная голова остаётся та же, дополнительные головы работают только на
    обучении: на выходе берётся первая. Такой надзор меняет не объём информации,
    а то, какую структуру сеть вынуждена выучить, — а именно на различии
    структуры ошибок и зарабатывает ансамбль.

    Утечки нет: подокна лежат внутри того же `[cutoff, cutoff + HORIZON)`,
    который и так целиком известен на обучающих срезах.

    Возвращает сырые суммы, как и `build_target`. В log1p их переводит вызывающий
    код — там же, где и основную цель, чтобы шкалы не разъехались.
    """
    day = (pl.col("event_date") - pl.lit(cutoff)).dt.total_days()
    bin_expr = pl.lit(len(AUX_EDGES) - 1)
    for i, e in enumerate(AUX_EDGES[:-1][::-1]):
        bin_expr = pl.when(day < e).then(len(AUX_EDGES) - 2 - i).otherwise(bin_expr)
    end = cutoff + dt.timedelta(days=HORIZON)
    df = (
        scan_log()
        .filter((pl.col("event_date") >= cutoff) & (pl.col("event_date") < end))
        .with_columns(bin_expr.alias("_bin"))
        .group_by(["user_id", "_bin"])
        .agg(pl.col("gmv").sum().alias("g"))
        .collect(engine="streaming")
    )
    out = np.zeros((len(users), len(AUX_EDGES)), dtype=np.float64)
    pos = np.searchsorted(users, df["user_id"].to_numpy())
    out[pos, df["_bin"].to_numpy()] = df["g"].to_numpy()
    return out[rows]


def rate_base(seq, rows: np.ndarray, cutoff: dt.date, lookback: int,
              chunk: int = 4000) -> np.ndarray:
    """Наивная экстраполяция темпа: log1p(средний дневной gmv в окне * HORIZON).

    Пара к `--self-norm`, без которой та нормировка неполна. Деля каналы на
    собственное среднее клиента, мы забираем у сети масштаб; если при этом цель
    остаётся абсолютной, сеть обязана масштаб восстановить — из 27 рангов, то
    есть решить лишнюю и чужую для неё задачу. Именно так был поставлен первый
    замер, и он провалился (см. `self_norm` в seq_data.py).

    Здесь масштаб не восстанавливается, а возвращается умножением: цель тоже
    делится на собственный уровень клиента. Сеть учит чистое отношение «сколько
    он купит за 30 дней относительно своего обычного темпа», а уровень
    приходит обратно готовым числом.

    Считается тем же механизмом, что `--residual`: база вычитается из цели на
    обучении и прибавляется к предсказанию везде, включая сабмит.

    Клиент без покупок в окне даёт базу log1p(0) = 0, и цель остаётся исходной —
    отдельной ветки не нужно.

    Утечки нет: окно обрывается на дне перед cutoff'ом.

    ПРОВЕРЕНО И ОТВЕРГНУТО, вместе с --self-norm --static rk_ на обоих срезах:

        январь (выровнено)  сетка 1.67292 -> вход+цель 1.68160   -0.00868
        декабрь             сетка 1.73735 -> вход+цель 1.74766   -0.01031

    Но пара действительно нужна была: вес в оптимальном составе поднялся с
    **-0.197** у неполной версии (делили только вход) до **-0.021** у полной.
    Постановка была верной, величины не хватило — ноль вместо вреда.

    Почему не хватило. Делитель оценивается по тем же разреженным данным, что
    и всё остальное: средний дневной gmv за 90 дней у клиента с одной покупкой
    — оценка почти без точности. Деление на неё не снимает шум масштаба,
    а вносит его в цель, причём ровно у тех клиентов, которые и так труднее
    всех. Видно по остатку: std 1.9653 против 2.09355 RMSLE у самой наивной
    экстраполяции — цель стала не проще, а шумнее.

    Отсюда уточнение правила из `self_norm`: убирать информацию нормировкой
    можно только на делитель, оценённый надёжно. Ранги работают именно поэтому
    — это порядковые статистики по 250 000 клиентов, а не отношение двух
    разреженных сумм.

    Побочно: Gini у этой сети 0.7351 — лучший среди всех наших сетей (0.7345
    у плотной, 0.7337 у рангов). По величине она хуже, а упорядочивает лучше.
    В RMSLE-оптимальный состав она не входит, но для tie-breaker'а это
    единственная сеть, которая тянет Gini вверх, а не вниз.
    """
    out = np.empty(len(rows), dtype=np.float64)
    for i in range(0, len(rows), chunk):
        win = gather(seq, rows[i:i + chunk], cutoff, lookback)
        out[i:i + chunk] = np.log1p(np.expm1(win[:, :, 0]).mean(axis=1) * HORIZON)
    return out


def load_base(name: str, tag: str, users: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Предсказание бустинга из seq_oof, выровненное по строкам матрицы.

    В режиме остатка сеть учит `log1p(y) - base`, поэтому её выход по построению
    ортогонален тому, что бустинг уже умеет: параллельным направлением он быть
    не может, а именно эта параллельность и упёрлась в потолок.
    """
    path = MODELS / f"{name}_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"нет {path.name} — сначала посчитайте: "
                         f"python -u src/seq_oof.py --name {name}")
    d = np.load(path)
    order = np.argsort(d["user_id"])
    su, sp = d["user_id"][order], d["pred_log"][order]
    want = users[rows]
    pos = np.clip(np.searchsorted(su, want), 0, len(su) - 1)
    ok = su[pos] == want
    out = np.zeros(len(want), dtype=np.float64)
    out[ok] = sp[pos[ok]]
    if (~ok).sum():
        print(f"  {path.name}: нет предсказания для {(~ok).sum():,} пользователей — беру ноль")
    return out


def load_static(cutoff: dt.date, users: np.ndarray, rows: np.ndarray,
                prefix: str = "rk_", with_target: bool = True):
    """Статические признаки из выборки бустинга, выровненные по строкам матрицы.

    Берутся только ранги: они уже лежат в [0, 1], не требуют масштабирования и
    не зависят от уровня площадки. Абсолютные признаки сюда намеренно не идут —
    иначе сеть превратится в медленную имитацию бустинга и потеряет то, ради
    чего она в ансамбле, то есть непохожесть ошибок.

    Пропуски (например ранг рецентности покупки у никогда не покупавшего)
    кодируются как -1: ранги лежат в (0, 1], поэтому значение вне диапазона
    сеть отличит от «самого низкого места».
    """
    df = get_dataset(cutoff, with_target=with_target)
    cols = sorted(c for c in df.columns if c.startswith(prefix))
    if not cols:
        raise SystemExit(
            f"в выборке на {cutoff} нет колонок с префиксом {prefix!r}. "
            f"Блок ranks появился в main — пересоберите кэш: "
            f"python -u src/datasets.py --test")
    src_u = df["user_id"].to_numpy()
    order = np.argsort(src_u)
    su = src_u[order]
    vals = df.select(cols).to_numpy().astype(np.float32)[order]
    want = users[rows]
    pos = np.clip(np.searchsorted(su, want), 0, len(su) - 1)
    ok = su[pos] == want
    out = np.full((len(want), len(cols)), -1.0, dtype=np.float32)
    out[ok] = vals[pos[ok]]
    out = np.nan_to_num(out, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if (~ok).sum():
        print(f"  {cutoff}: нет рангов для {(~ok).sum():,} пользователей — беру -1")
    return out, cols


def channel_stats(seq, rows: np.ndarray, cutoff: dt.date, lookback: int,
                  sample: int = 20000, seed: int = SEED, bin_days: int = 1,
                  events: int = 0, norm: bool = False):
    """Среднее и разброс по каналам на выборке окон — только по обучающим срезам.

    Считать по всем данным нельзя: в выборку попал бы валидационный срез.
    Масштабируются только 12 каналов лога, `observed` остаётся 0/1.

    В событийном режиме `age` и `gap` тоже остаются несмасштабированными: обе
    величины уже в log1p и лежат в разумном диапазоне, а вычитать из них среднее
    нельзя — у добивки они честно нулевые, и сдвиг сделал бы её значимой.
    """
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(rows, size=min(sample, len(rows)), replace=False))
    x = prep_window(gather(seq, sel, cutoff, lookback), bin_days, events,
                    norm)[:, :, :len(CHANNELS)]
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# Индексы каналов в массиве gather: сначала CHANNELS, затем добавленный observed.
SUM_IDX = [CHANNELS.index(c) for c in SUM_CHANNELS]
MEAN_IDX = [CHANNELS.index(c) for c in MEAN_CHANNELS] + [len(CHANNELS)]


def bin_window(x: np.ndarray, bin_days: int) -> np.ndarray:
    """Свернуть окно в шаги по `bin_days` дней.

    Обучение упирается в одну и ту же величину train RMSE с первой эпохи при
    любых точности, скорости обучения и сиде — так выглядит предел
    представления, а не оптимизации: 180 шагов вектора, у которого ненулевые
    значения встречаются в паре процентов дней. Неделя как шаг делает каждый
    шаг плотным и укорачивает рекуррентность в семь раз.

    Величины складываются в исходной шкале (log1p снимается и надевается
    обратно), иначе сумма логарифмов подменила бы логарифм суммы.
    """
    if bin_days <= 1:
        return x
    n, t, c = x.shape
    if t % bin_days:
        raise SystemExit(f"окно {t} не делится на шаг {bin_days} — "
                         f"главный цикл должен был подрезать его заранее")
    g = x.reshape(n, t // bin_days, bin_days, c)
    out = np.empty((n, t // bin_days, c), dtype=np.float32)
    out[:, :, SUM_IDX] = np.log1p(np.expm1(g[:, :, :, SUM_IDX]).sum(axis=2))
    out[:, :, MEAN_IDX] = g[:, :, :, MEAN_IDX].mean(axis=2)
    return out


def prep_window(x: np.ndarray, bin_days: int, events: int,
                norm: bool = False) -> np.ndarray:
    """Представление окна: дневная сетка, укрупнённые шаги или события.

    Взаимоисключающие ветки — оба преобразования читают ось дней, и применять
    их подряд бессмысленно: после укрупнения «активный день» уже не день.

    Нормировка на собственную историю применяется до них: она работает по дням,
    и её среднее должно считаться по исходной сетке, а не по укрупнённой.
    """
    if norm:
        x = self_norm(x)
    if events:
        return events_window(x, events)
    return bin_window(x, bin_days)


def n_input_channels(events: int) -> int:
    """Сколько каналов увидит сеть: у событий их на два больше (`age`, `gap`)."""
    return len(CHANNELS) + (3 if events else 1)


def batches(seq, rows: np.ndarray, cutoff: dt.date, lookback: int, batch_size: int,
            y: np.ndarray | None = None, shuffle: bool = False, rng=None,
            bin_days: int = 1, static: np.ndarray | None = None,
            aux: np.ndarray | None = None, events: int = 0, norm: bool = False):
    """Батчи окон. Индексы внутри батча сортируются — memmap читается локальнее."""
    order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(order), batch_size):
        take = np.sort(order[i:i + batch_size])
        sel = rows[take]
        x = prep_window(gather(seq, sel, cutoff, lookback), bin_days, events, norm)
        s = None if static is None else static[take]
        a = None if aux is None else aux[take]
        yield x, s, (None if y is None else y[take]), a, take


# --------------------------------------------------------------------------- модель

class SeqNet(nn.Module):
    """Кодировщик последовательности -> одно число в log1p-шкале.

    Голова смотрит на два пула сразу: последнее состояние (что происходит прямо
    перед cutoff'ом — рецентность решает в этой задаче больше всего) и среднее
    по всему окну (общий уровень клиента). Одного последнего состояния мало:
    у половины пользователей последние дни пустые.

    Статическая ветка (`n_static > 0`) принимает процентильные ранги из блока
    `ranks`. Для сети это не «ещё несколько признаков»: у неё до сих пор не было
    никакой нормировки входа, устойчивой к сдвигу уровня площадки, а ранг такую
    нормировку даёт даром — «верхние 5% по GMV» означают одно и то же на любом
    срезе. Именно на этой привязке к абсолютным величинам сеть промахивалась
    по уровню тестового окна на +0.20 против -0.058 у бустинга.
    """

    def __init__(self, n_ch: int, hidden: int = 128, layers: int = 1, arch: str = "gru",
                 dropout: float = 0.1, lookback: int = 180, heads: int = 4,
                 n_static: int = 0, n_aux: int = 0):
        super().__init__()
        self.arch = arch
        self.n_static = n_static
        self.inp = nn.Sequential(nn.Linear(n_ch, hidden), nn.LayerNorm(hidden), nn.GELU())
        if arch in ("gru", "lstm"):
            cls = nn.GRU if arch == "gru" else nn.LSTM
            self.enc = cls(hidden, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        elif arch == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, lookback, hidden))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            raise ValueError(f"неизвестная архитектура: {arch}")
        head_in = hidden * 2
        if n_static:
            self.static = nn.Sequential(
                nn.Linear(n_static, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
            )
            head_in += hidden
        # Выходов 1 + n_aux: первый — сумма за горизонт, остальные — подокна.
        # Дополнительные головы работают только на обучении.
        self.n_aux = n_aux
        # Голова разделена намеренно: `head_body` даёт скрытое состояние
        # размерности hidden — то самое, которое организаторы предлагали отдать
        # бустингу признаками. Предсказание есть его одномерная проекция, то
        # есть строго беднее: всё, что не легло на эту ось, теряется.
        self.head_body = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_out = nn.Linear(hidden, 1 + n_aux)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(x)
        if self.arch == "transformer":
            h = self.enc(h + self.pos[:, -h.shape[1]:])
        else:
            h, _ = self.enc(h)
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=1)
        if self.n_static:
            if s is None:
                raise ValueError("модель ждёт статические признаки, а их не подали")
            pooled = torch.cat([pooled, self.static(s)], dim=1)
        return self.head_out(self.head_body(pooled))

    def hidden(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        """Скрытое состояние перед последним слоем — вход для стекинга.

        Тот же путь, что и в `forward`, но без проекции в число. Дублирование
        кода здесь дешевле, чем флаг в `forward`: тот вызывается на каждом
        батче обучения, и лишняя ветка в нём — лишний риск.
        """
        h = self.inp(x)
        if self.arch == "transformer":
            h = self.enc(h + self.pos[:, -h.shape[1]:])
        else:
            h, _ = self.enc(h)
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=1)
        if self.n_static:
            if s is None:
                raise ValueError("модель ждёт статические признаки, а их не подали")
            pooled = torch.cat([pooled, self.static(s)], dim=1)
        return self.head_body(pooled)


# --------------------------------------------------------------------------- обучение

def evaluate(model, seq, rows, cutoff, lookback, batch_size, mean, std, device, y=None,
             desc: str = "предсказание", bin_days: int = 1, static=None, events: int = 0,
             norm: bool = False):
    """Предсказания в log1p-шкале (без обрезки — обрезает метрика, как лидерборд)."""
    model.eval()
    out = np.empty(len(rows), dtype=np.float64)
    prog = bar(-(-len(rows) // batch_size), desc)
    with torch.no_grad():
        for x, s, _, _, take in batches(seq, rows, cutoff, lookback, batch_size,
                                        bin_days=bin_days, static=static, events=events,
                                        norm=norm):
            xb = to_device(x, mean, std, device)
            sb = None if s is None else torch.from_numpy(s).float().to(device)
            with autocast(device):
                p = model(xb, sb)
            # Первый выход — основная голова. Остальные, если они есть, живут
            # только на обучении и в предсказание не идут.
            out[take] = p[:, 0].float().cpu().numpy()
            prog.update(1)
    prog.close()
    return out


def extract_hidden(model, seq, rows, cutoff, lookback, batch_size, mean, std, device,
                   bin_days: int = 1, static=None, events: int = 0,
                   norm: bool = False) -> np.ndarray:
    """Скрытые состояния для всех строк — матрица (len(rows), hidden).

    Хранится во float16: точность здесь не критична (бустинг всё равно режет
    признаки на 255 корзин), а размер вчетверо меньше — 250 000 x 128 это
    64 МБ вместо 256, и такие файлы ещё можно передать сокоманднику.
    """
    model.eval()
    out = None
    prog = bar(-(-len(rows) // batch_size), "скрытые состояния")
    with torch.no_grad():
        for x, s, _, _, take in batches(seq, rows, cutoff, lookback, batch_size,
                                        bin_days=bin_days, static=static, events=events):
            xb = to_device(x, mean, std, device)
            sb = None if s is None else torch.from_numpy(s).float().to(device)
            with autocast(device):
                z = model.hidden(xb, sb)
            z = z.float().cpu().numpy()
            if out is None:
                out = np.empty((len(rows), z.shape[1]), dtype=np.float16)
            out[take] = z.astype(np.float16)
            prog.update(1)
    prog.close()
    return out


def to_device(x: np.ndarray, mean, std, device) -> torch.Tensor:
    xb = torch.from_numpy(x).to(device, non_blocking=True)
    xb[:, :, :len(CHANNELS)] = (xb[:, :, :len(CHANNELS)] - mean) / std
    return xb


def train_model(seq, train_rows, train_y, train_cuts, val_rows, val_y, val_cut, args,
                mean, std, device, epochs: int | None = None,
                train_base=None, val_base=None, train_static=None, val_static=None,
                train_aux=None):
    """Обучение с ранней остановкой по валидации; возвращает (модель, эпохи, история).

    Если `val_rows` пуст — это финальное дообучение: остановка по числу эпох,
    найденному на валидации (та же логика, что `--final` в train.py, где число
    итераций фиксируется заранее).
    """
    torch.manual_seed(args.seed)
    steps = args.events or args.lookback // args.bin
    n_ch = n_input_channels(args.events)
    n_static = 0 if train_static is None else train_static[0].shape[1]
    n_aux = 0 if train_aux is None else train_aux[0].shape[1]
    model = SeqNet(n_ch, args.hidden, args.layers, args.arch,
                   args.dropout, steps, args.heads, n_static, n_aux).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    shape = (f"= {steps} событий из {args.lookback} дней" if args.events
             else f"= {steps} шагов по {args.bin} дн.")
    print(f"{args.arch}: {n_par:,} параметров | окно {args.lookback} дней "
          f"{shape} | {n_ch} каналов"
          f"{f' + {n_static} рангов' if n_static else ''}"
          f"{f' | + {n_aux} вспомогательных голов, вес {args.aux_weight}' if n_aux else ''}"
          f" | устройство {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Шаги считаются посрезово: батчи не смешивают срезы, поэтому хвостовой
    # неполный батч есть у каждого, и общая сумма больше, чем total/batch_size.
    steps = max(1, sum(-(-len(r) // args.batch_size) for r in train_rows))
    n_epochs = epochs or args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps * n_epochs, pct_start=0.15)
    rng = np.random.default_rng(args.seed)

    best = (float("inf"), None, 0)
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, run, seen, done = time.time(), 0.0, 0, 0
        prog = bar(steps, f"эпоха {epoch}/{n_epochs}")
        # Срезы перемешиваются целиком, а не построчно: батч из одного среза —
        # это и локальность memmap, и корректные окна (у срезов разные границы).
        for ci in rng.permutation(len(train_cuts)):
            cut, rows, y = train_cuts[ci], train_rows[ci], train_y[ci]
            # В режиме остатка целью становится то, чего бустингу не хватило.
            ylog = np.log1p(y) if train_base is None else np.log1p(y) - train_base[ci]
            st = None if train_static is None else train_static[ci]
            ax = None if train_aux is None else train_aux[ci]
            for x, s, yb, ab, _ in batches(seq, rows, cut, args.lookback, args.batch_size,
                                           ylog, shuffle=True, rng=rng, bin_days=args.bin, events=args.events, norm=args.self_norm,
                                           static=st, aux=ax):
                xb = to_device(x, mean, std, device)
                sb = None if s is None else torch.from_numpy(s).float().to(device)
                tb = torch.from_numpy(yb).float().to(device)
                with autocast(device):
                    p = model(xb, sb)
                    loss = nn.functional.mse_loss(p[:, 0], tb)
                    main = float(loss)
                    if ab is not None:
                        # Вспомогательные головы влияют только на градиент:
                        # в отчётный RMSE идёт основная, иначе числа стали бы
                        # несравнимы с прежними прогонами.
                        aux_t = torch.from_numpy(ab).float().to(device)
                        loss = loss + args.aux_weight * nn.functional.mse_loss(
                            p[:, 1:], aux_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                run += main * len(yb)
                seen += len(yb)
                done += 1
                prog.update(1)
                if done % 20 == 0:
                    prog.set_postfix_str(f"RMSE {np.sqrt(run / seen):.4f}")
        prog.close()

        line = f"эпоха {epoch:>2}/{n_epochs} | train RMSE {np.sqrt(run / seen):.5f}"
        if len(val_rows):
            p = evaluate(model, seq, val_rows, val_cut, args.lookback, args.batch_size,
                         mean, std, device, desc=f"эпоха {epoch}/{n_epochs} валидация",
                         bin_days=args.bin, events=args.events, static=val_static)
            if val_base is not None:
                p = val_base + p          # метрика считается по полному предсказанию
            score = rmse_log(np.log1p(val_y), p)
            history.append(score)
            mark = ""
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
                mark = "  <- лучшая"
            print(f"{line} | val RMSLE {score:.5f} | {time.time() - t0:.0f}s{mark}")
            if epoch - best[2] >= args.patience:
                print(f"ранняя остановка: {args.patience} эпох без улучшения")
                break
        else:
            print(f"{line} | {time.time() - t0:.0f}s")

    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, (best[2] if best[1] is not None else n_epochs), history


# --------------------------------------------------------------------------- сабмит

def predict_empty(model, args, mean, std, device) -> float:
    """Предсказание для пользователя без единого события: окно наблюдалось, но пусто.

    `observed = 1` намеренно: дни существовали, активности в них не было —
    это не то же самое, что «данных за эти дни нет».

    В событийном режиме такой клиент даёт пустую последовательность: ни одного
    токена, все `valid` нулевые. Это ровно тот вход, который событийное
    представление и обязано отдавать в этом случае, — добивать его нечем.
    """
    steps = args.events or args.lookback // args.bin
    x = np.zeros((1, steps, n_input_channels(args.events)), dtype=np.float32)
    if not args.events:
        x[:, :, len(CHANNELS)] = 1.0
    model.eval()
    with torch.no_grad():
        xb = to_device(x, mean, std, device)
        # Ранги такого клиента нулевые: он в самом низу любого распределения.
        sb = (None if not model.n_static else
              torch.zeros((1, model.n_static), dtype=torch.float32, device=device))
        with autocast(device):
            return float(model(xb, sb)[:, 0].float().cpu().numpy()[0])


def make_submission(model, seq, users, first_day, args, mean, std, device, meta: dict,
                    static=None) -> None:
    """Сабмит на 250k пользователей из sample_submit тем же форматом, что predict.py."""
    sub_users = pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy()
    pos = np.searchsorted(users, sub_users)
    pos = np.clip(pos, 0, len(users) - 1)
    known = users[pos] == sub_users
    if (~known).sum():
        print(f"внимание: {(~known).sum():,} пользователей сабмита нет в логе — "
              f"им уйдёт предсказание по пустой истории")

    rows = pos.copy()
    p_log = evaluate(model, seq, rows, TEST_CUTOFF, args.lookback, args.batch_size,
                     mean, std, device, bin_days=args.bin, events=args.events, static=static)
    # У неизвестных пользователей searchsorted указал на чужую строку — им
    # положено предсказание по пустой истории, а не по соседу из индекса.
    if (~known).sum():
        p_log[~known] = predict_empty(model, args, mean, std, device)
    if args.norm_target:
        base = rate_base(seq, rows, TEST_CUTOFF, args.lookback)
        print(f"  наивный темп добавляется обратно: mean base {base.mean():.4f}, "
              f"mean остатка {p_log.mean():+.4f}")
        p_log = p_log + base
    if args.residual:
        # Тот же объект, что и в обучении: бустинг, обученный на всех срезах.
        base = load_base(args.residual, "test", users, rows)
        print(f"  остаток добавляется к бустингу: mean base {base.mean():.4f}, "
              f"mean остатка {p_log.mean():+.4f}")
        p_log = base + p_log
    pred = np.clip(np.expm1(np.clip(p_log, 0, None)), 0, None)

    out = args.out or f"{args.name}_{dt.datetime.now():%m%d_%H%M}.csv"
    path = SUBMISSIONS / out
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out")
    pl.DataFrame({"user_id": sub_users, "predict": pred.astype(np.float32)}).write_csv(path)

    print(f"\n{path}")
    print(f"строк: {len(pred):,} | нулевых: {(pred < 1e-6).mean():.2%} | "
          f"среднее: {pred.mean():.2f} | медиана: {np.median(pred):.2f} | max: {pred.max():,.0f}")
    print(f"суммарный предсказанный GMV: {pred.sum():,.0f}")
    print("\nПоправки 0.0578 и 1.0545 к этой модели НЕ относятся: они измерены\n"
          "зондами для ансамбля бустингов. Для сети нужны свои зонды (PLAN.md, раздел 3).")
    append_csv(SUBMISSIONS / "log.csv", SUBMIT_FIELDS, {
        "file": out, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(), "name": args.name, "model": args.arch, "blend_w": "",
        "val_rmsle": round(meta["rmsle"], 5), "val_gini": round(meta["gini"], 4),
        "val_sum_err": round(meta["rmspe_total"], 4),
        "pred_sum": round(float(pred.sum())), "pred_zeros": f"{(pred < 1e-6).mean():.4f}",
        "note": f"сеть {args.arch}, окно {args.lookback}, hidden {args.hidden}; "
                f"поправки калибровки не применялись",
    })


# --------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gru", choices=["gru", "lstm", "transformer"])
    ap.add_argument("--lookback", type=int, default=180, help="длина окна в днях")
    ap.add_argument("--bin", type=int, default=1,
                    help="сколько дней в одном шаге последовательности: 1 — по дням, "
                         "7 — по неделям (180 дней превращаются в 26 плотных шагов). "
                         "Длина окна должна делиться на это число")
    ap.add_argument("--events", type=int, default=0,
                    help="событийное представление: сколько последних активных дней "
                         "подавать токенами вместо дневной сетки (0 — сетка). "
                         "Промежутки в днях идут отдельными каналами age и gap")
    ap.add_argument("--norm-target", action="store_true",
                    help="делить и цель на собственный уровень клиента: сеть учит "
                         "отношение к наивной экстраполяции темпа. Пара к --self-norm")
    ap.add_argument("--self-norm", action="store_true",
                    help="делить каналы клиента на его же среднее по окну: сеть "
                         "видит форму и тайминг, масштаб приходит рангами")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--heads", type=int, default=4, help="только для transformer")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None, help="валидационный срез, ГГГГ-ММ-ДД")
    ap.add_argument("--train-cutoffs", default=None, help="явный список обучающих срезов")
    ap.add_argument("--subsample", type=int, default=0,
                    help="не больше N пользователей на ОБУЧАЮЩИЙ срез (быстрая проверка); "
                         "валидация всегда полная, иначе число несопоставимо с журналом")
    ap.add_argument("--val-subsample", type=int, default=0,
                    help="урезать и валидацию — только для отладки скорости: "
                         "две случайные половины валидации расходятся на 0.007 RMSLE, "
                         "поэтому такое число нельзя сравнивать ни с чем")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Разница между bf16 и fp32 измерена и оказалась НЕразличимой: три сида в
    # fp32 дали 1.68266 / 1.68798 / 1.68804 (среднее 1.6862, разброс 0.0054),
    # единственный прогон в bf16 — 1.68979, то есть внутри того же разброса.
    # Ранний вывод «fp32 выигрывает 0.007» был сравнением двух одиночных
    # прогонов и не пережил проверки сидами. По правилу PLAN.md (раздел 2)
    # подпороговый эффект принимается, только если ничего не стоит; fp32 стоит
    # 45% времени эпохи (57с против 39с), поэтому умолчание — bf16.
    ap.add_argument("--amp", dest="amp", action="store_true", default=None,
                    help="считать в bfloat16 (умолчание)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="считать в float32: медленнее на 45%%, разницы в качестве "
                         "на GRU не обнаружено (см. комментарий в коде)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", default="seq")
    ap.add_argument("--note", default="")
    ap.add_argument("--final", action="store_true",
                    help="дообучить на train+val и собрать сабмит")
    ap.add_argument("--out", default=None, help="имя файла сабмита")
    ap.add_argument("--multitask", action="store_true",
                    help="дополнительно предсказывать суммы по подокнам горизонта. "
                         "На выходе по-прежнему одна голова — подокна работают "
                         "только на обучении и меняют структуру ошибок")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="вес вспомогательного лосса относительно основного")
    ap.add_argument("--static", default=None,
                    help="подать сети статические признаки по префиксу колонок "
                         "(например rk_ — процентильные ранги из блока ranks). "
                         "Для сети это первая нормировка входа, устойчивая к сдвигу "
                         "уровня площадки; нужен кэш признаков из datasets.py")
    ap.add_argument("--residual", default=None,
                    help="учить остаток бустинга: имя набора из seq_oof.py (например oof). "
                         "Выход сети по построению ортогонален тому, что бустинг уже умеет")
    ap.add_argument("--save-hidden", action="store_true",
                    help="сохранить скрытое состояние сети на валидации — вход для "
                         "стекинга. Предсказание есть его одномерная проекция, то есть "
                         "строго беднее: всё, что не легло на эту ось, теряется")
    ap.add_argument("--save-val-pred", action="store_true",
                    help="сохранить предсказания на валидации для бленда с бустингом")
    args = ap.parse_args()

    global _AMP
    if args.amp is None:
        args.amp = True
    _AMP = args.amp
    device = torch.device(args.device)
    precision = "bf16" if _AMP and device.type == "cuda" else "fp32"
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} ГБ | "
              f"точность {precision}")

    seq, users, first_day, _ = open_seq()
    # Бины выравниваются по cutoff'у: последний шаг это дни, непосредственно
    # предшествующие срезу. Поэтому лишние дни отрезаются с дальнего конца,
    # где они стоят дешевле всего — рецентность решает в этой задаче больше всего.
    if args.events and args.bin > 1:
        raise SystemExit("--events и --bin взаимоисключающие: после укрупнения "
                         "шага «активный день» перестаёт быть днём")
    if args.lookback % args.bin:
        trimmed = args.lookback - args.lookback % args.bin
        print(f"окно {args.lookback} не делится на шаг {args.bin} — "
              f"беру {trimmed} дней ({trimmed // args.bin} шагов), лишние отрезаны с дальнего конца")
        args.lookback = trimmed
    parse_date = dt.date.fromisoformat
    val_cut, train_cuts = cutoff_split(
        args.cutoffs,
        parse_date(args.val_cutoff) if args.val_cutoff else None,
        [parse_date(d.strip()) for d in args.train_cutoffs.split(",")] if args.train_cutoffs else None,
    )
    d0, _ = window_bounds(min(train_cuts), args.lookback)
    if d0 < 0:
        print(f"окно {args.lookback} дней длиннее истории самого старого среза "
              f"({min(train_cuts)}): {-d0} дней будут добиты нулями с observed=0")

    rng = np.random.default_rng(args.seed)
    train_rows, train_y = [], []
    print(f"считаю таргеты: {len(train_cuts) + 1} проходов по логу "
          f"(на каждый срез — свой полный проход)")
    for c in train_cuts:
        t0 = time.time()
        r, y = targets_for(c, users, first_day)
        if args.subsample and len(r) > args.subsample:
            keep = np.sort(rng.choice(len(r), args.subsample, replace=False))
            r, y = r[keep], y[keep]
        train_rows.append(r)
        train_y.append(y)
        print(f"  срез {c}: {len(r):,} пользователей | покупателей {(y > 0).mean():.2%} "
              f"| {time.time() - t0:.0f}s")
    t0 = time.time()
    val_rows, val_y = targets_for(val_cut, users, first_day)
    print(f"  (валидационный таргет за {time.time() - t0:.0f}s)")
    if args.val_subsample and len(val_rows) > args.val_subsample:
        keep = np.sort(rng.choice(len(val_rows), args.val_subsample, replace=False))
        val_rows, val_y = val_rows[keep], val_y[keep]
        print("ВНИМАНИЕ: валидация урезана — это число несравнимо с experiments.csv")
    print(f"  валидация {val_cut}: {len(val_rows):,} пользователей | "
          f"покупателей {(val_y > 0).mean():.2%}")

    train_static = val_static = None
    if args.static:
        train_static, cols = [], None
        for c, r in zip(train_cuts, train_rows):
            s, cols = load_static(c, users, r, args.static)
            train_static.append(s)
        val_static, _ = load_static(val_cut, users, val_rows, args.static)
        print(f"статические признаки: {len(cols)} колонок по префиксу '{args.static}' "
              f"({', '.join(cols[:4])}, ...)")

    train_base = val_base = None
    if args.norm_target:
        if args.residual:
            raise SystemExit("--norm-target и --residual оба задают базу остатка, "
                             "вместе они бессмысленны")
        print("цель делится на собственный уровень: "
              "цель = log1p(y) - log1p(средний дневной gmv в окне * 30)")
        train_base = [rate_base(seq, r, c, args.lookback)
                      for c, r in zip(train_cuts, train_rows)]
        val_base = rate_base(seq, val_rows, val_cut, args.lookback)
        print(f"  наивный темп сам по себе: RMSLE {rmse_log(np.log1p(val_y), val_base):.5f}")
        print(f"  остаток: среднее {(np.log1p(val_y) - val_base).mean():+.4f} | "
              f"std {(np.log1p(val_y) - val_base).std():.4f}")
    if args.residual:
        print(f"режим остатка: цель = log1p(y) - предсказание бустинга ({args.residual})")
        train_base = [load_base(args.residual, c.isoformat(), users, r)
                      for c, r in zip(train_cuts, train_rows)]
        val_base = load_base(args.residual, val_cut.isoformat(), users, val_rows)
        base_rmsle = rmse_log(np.log1p(val_y), val_base)
        print(f"  бустинг на валидации сам по себе: RMSLE {base_rmsle:.5f}")
        print(f"  остаток: среднее {(np.log1p(val_y) - val_base).mean():+.4f} | "
              f"std {(np.log1p(val_y) - val_base).std():.4f}")

    t0 = time.time()
    print("считаю масштаб каналов по выборке обучающих окон...", flush=True)
    mean_np, std_np = channel_stats(seq, train_rows[0], train_cuts[0], args.lookback,
                                    seed=args.seed, bin_days=args.bin, events=args.events)
    print(f"  готово за {time.time() - t0:.0f}s")
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)

    train_aux = None
    if args.multitask:
        t0 = time.time()
        # log1p, как и основная цель: сеть предсказывает в этой шкале, и лосс
        # между рублями и логарифмами был бы просто несопоставимыми величинами.
        train_aux = [np.log1p(aux_targets(c, users, r))
                     for c, r in zip(train_cuts, train_rows)]
        share = np.mean([(a > 0).mean() for a in train_aux])
        print(f"вспомогательные головы: {len(AUX_EDGES)} подокон "
              f"{AUX_EDGES}, доля ненулевых {share:.1%}, за {time.time() - t0:.0f}s")

    model, best_epoch, _ = train_model(seq, train_rows, train_y, train_cuts,
                                       val_rows, val_y, val_cut, args, mean, std, device,
                                       train_base=train_base, val_base=val_base,
                                       train_static=train_static, val_static=val_static,
                                       train_aux=train_aux)

    p_log = evaluate(model, seq, val_rows, val_cut, args.lookback, args.batch_size,
                     mean, std, device, bin_days=args.bin, events=args.events, static=val_static)
    print(f"\n--- валидация (cutoff {val_cut}) ---")
    if val_base is not None:
        report(val_y, np.expm1(np.clip(val_base, 0, None)), "бустинг")
        p_log = val_base + p_log
    res = report(val_y, np.expm1(np.clip(p_log, 0, None)),
                 args.arch + ("+бустинг" if val_base is not None else ""))

    if args.save_val_pred:
        np.savez(MODELS / f"{args.name}_valpred_{val_cut}.npz",
                 user_id=users[val_rows], pred_log=p_log, target=val_y)
        print(f"предсказания валидации: {MODELS / f'{args.name}_valpred_{val_cut}.npz'}")

    if args.save_hidden:
        z = extract_hidden(model, seq, val_rows, val_cut, args.lookback, args.batch_size,
                           mean, std, device, bin_days=args.bin, events=args.events, static=val_static)
        path = MODELS / f"{args.name}_hidden_{val_cut}.npz"
        np.savez(path, user_id=users[val_rows], hidden=z, target=val_y)
        print(f"скрытые состояния: {path} | {z.shape[0]:,} x {z.shape[1]} "
              f"| {path.stat().st_size / 1024 ** 2:.0f} МБ")

    append_csv(MODELS / "experiments.csv", EXPERIMENT_FIELDS, {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
        "feat_ver": (f"seq{len(CHANNELS)}x{args.lookback}"
                     + (f"e{args.events}" if args.events else f"b{args.bin}")
                     + ("n" if args.self_norm else "")
                     + ("t" if args.norm_target else "")),
        "blocks": "seq",
        "name": args.name, "model": args.arch, "cutoffs": len(train_cuts),
        "n_features": n_input_channels(args.events),
        "rmsle_single": round(res["rmsle"], 5), "rmsle_two_stage": "",
        # rmsle_blend дублирует rmsle_single: у сети одна голова, а команда
        # сравнивает строки журнала именно по этой колонке.
        "rmsle_blend": round(res["rmsle"], 5), "blend_w": "",
        "gini_blend": round(res["gini"], 4), "sum_bias_blend": round(res["sum_bias"], 4),
        "best_iter_single": best_epoch, "stride": 30, "halflife": "",
        "val_cutoff": str(val_cut), "train_cutoffs": " ".join(str(c) for c in train_cuts),
        # Отметка о подвыборке обязательна: строка с урезанным обучением или
        # урезанной валидацией стоит в журнале рядом с полными и без пометки
        # была бы неотличима от них.
        # Устройство и точность — в примечании обязательно: в общем журнале нет
        # своей колонки под них, а bf16 против fp32 стоит 0.007 RMSLE, то есть
        # две строки без такой пометки выглядели бы одинаково и были бы
        # несравнимы (то же касается CPU против GPU у бустингов).
        "note": ((f"ПОДВЫБОРКА train={args.subsample or 'полн'} "
                  f"val={args.val_subsample or 'полн'}; " if args.subsample or args.val_subsample else "")
                 + (args.note or f"{args.arch} hidden={args.hidden} layers={args.layers} "
                                 f"lookback={args.lookback} "
                                 + (f"events={args.events} " if args.events
                                    else f"bin={args.bin} ")
                                 + f"ch={len(CHANNELS)} bs={args.batch_size} lr={args.lr}")
                 + (" +self-norm" if args.self_norm else "")
                 + (" +norm-target" if args.norm_target else "")
                 + (f" +static:{args.static}" if args.static else "")
                 + (f" +residual:{args.residual}" if args.residual else "")
                 + f" [{device.type}/{precision} seed={args.seed}]"),
    })

    if not args.final:
        print("прогон без --final: веса не сохранялись, результат — строкой в experiments.csv")
        return

    # Финал: валидационный срез уходит в обучение, эпох столько же, сколько
    # оказалось лучшим (данных больше на 1/n — рост эпох не нужен, шагов и так
    # станет больше пропорционально).
    print(f"\n--- финальное обучение на всех {len(train_cuts) + 1} срезах, "
          f"{best_epoch} эпох ---")
    all_cuts = [*train_cuts, val_cut]
    all_rows = [*train_rows, val_rows]
    all_y = [*train_y, val_y]
    all_base = None if train_base is None else [*train_base, val_base]
    all_static = None if train_static is None else [*train_static, val_static]
    all_aux = (None if train_aux is None
               else [*train_aux, np.log1p(aux_targets(val_cut, users, val_rows))])
    final, _, _ = train_model(seq, all_rows, all_y, all_cuts, np.array([], dtype=int),
                              np.array([]), val_cut, args, mean, std, device,
                              epochs=best_epoch, train_base=all_base,
                              train_static=all_static, train_aux=all_aux)

    ckpt = MODELS / f"{args.name}_{args.arch}.pt"
    torch.save({"state_dict": final.state_dict(), "mean": mean_np, "std": std_np,
                "args": vars(args), "channels": CHANNELS, "val_metrics": res}, ckpt)
    (MODELS / f"{args.name}_meta.json").write_text(
        json.dumps({"name": args.name, "model": args.arch, "final": True,
                    "lookback": args.lookback, "epochs": best_epoch,
                    "val_cutoff": str(val_cut), "metrics": res}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"веса: {ckpt}")

    test_static = None
    if args.static:
        sub_users = pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy()
        rows_test = np.clip(np.searchsorted(users, sub_users), 0, len(users) - 1)
        test_static, _ = load_static(TEST_CUTOFF, users, rows_test, args.static,
                                     with_target=False)
    make_submission(final, seq, users, first_day, args, mean, std, device, res,
                    static=test_static)


if __name__ == "__main__":
    main()
