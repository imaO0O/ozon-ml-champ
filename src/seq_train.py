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
import sys
import time
import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn

# Triton ищет установленный CUDA Toolkit (nvcc), которого при обычной установке
# torch нет и не нужно: torch.compile мы не используем, а сама CUDA-рантайм
# внутри колеса torch работает. Предупреждение безвредно и только мешает читать.
warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*")

from config import (HORIZON, MODELS, SAMPLE_SUBMIT, SEED, SUBMISSIONS, TEST_CUTOFF,
                    train_cutoffs)
from features import build_target
from metrics import report, rmse_log
from seq_data import (CHANNELS, gather, history_mask, open_seq, window_bounds)
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


class Progress:
    """Однострочный индикатор хода работы.

    В терминале строка обновляется на месте через `\\r`, а при перенаправлении
    вывода в файл печатается редкими обычными строками: иначе лог фонового
    прогона превращается в одну бесконечную строку без переводов.
    """

    def __init__(self, total: int, desc: str, width: int = 22):
        self.total, self.desc, self.width = max(int(total), 1), desc, width
        self.t0 = self.last = time.time()
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.printed = 0

    @staticmethod
    def _hms(sec: float) -> str:
        sec = int(max(sec, 0))
        return f"{sec // 60}м{sec % 60:02d}с" if sec >= 60 else f"{sec}с"

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        # В терминале — до пяти раз в секунду, в файл — раз в 20 секунд.
        if done < self.total and now - self.last < (0.2 if self.tty else 20.0):
            return
        self.last = now
        frac = min(done / self.total, 1.0)
        el = now - self.t0
        eta = el / frac - el if frac > 1e-9 else 0.0
        full = int(frac * self.width)
        msg = (f"  {self.desc} [{'█' * full}{'·' * (self.width - full)}] "
               f"{done}/{self.total} {self._hms(el)}<{self._hms(eta)}{extra}")
        if self.tty:
            print("\r" + msg + " " * max(0, self.printed - len(msg)), end="", flush=True)
            self.printed = len(msg)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self.tty and self.printed:
            print("\r" + " " * self.printed + "\r", end="", flush=True)
        self.printed = 0


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


def channel_stats(seq, rows: np.ndarray, cutoff: dt.date, lookback: int,
                  sample: int = 20000, seed: int = SEED):
    """Среднее и разброс по каналам на выборке окон — только по обучающим срезам.

    Считать по всем данным нельзя: в выборку попал бы валидационный срез.
    Масштабируются только 12 каналов лога, `observed` остаётся 0/1.
    """
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(rows, size=min(sample, len(rows)), replace=False))
    x = gather(seq, sel, cutoff, lookback)[:, :, :len(CHANNELS)]
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def batches(seq, rows: np.ndarray, cutoff: dt.date, lookback: int, batch_size: int,
            y: np.ndarray | None = None, shuffle: bool = False, rng=None):
    """Батчи окон. Индексы внутри батча сортируются — memmap читается локальнее."""
    order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(order), batch_size):
        take = np.sort(order[i:i + batch_size])
        sel = rows[take]
        x = gather(seq, sel, cutoff, lookback)
        yield x, (None if y is None else y[take]), take


# --------------------------------------------------------------------------- модель

class SeqNet(nn.Module):
    """Кодировщик последовательности -> одно число в log1p-шкале.

    Голова смотрит на два пула сразу: последнее состояние (что происходит прямо
    перед cutoff'ом — рецентность решает в этой задаче больше всего) и среднее
    по всему окну (общий уровень клиента). Одного последнего состояния мало:
    у половины пользователей последние дни пустые.
    """

    def __init__(self, n_ch: int, hidden: int = 128, layers: int = 1, arch: str = "gru",
                 dropout: float = 0.1, lookback: int = 180, heads: int = 4):
        super().__init__()
        self.arch = arch
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
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        if self.arch == "transformer":
            h = self.enc(h + self.pos[:, -h.shape[1]:])
        else:
            h, _ = self.enc(h)
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------- обучение

def evaluate(model, seq, rows, cutoff, lookback, batch_size, mean, std, device, y=None,
             desc: str = "предсказание"):
    """Предсказания в log1p-шкале (без обрезки — обрезает метрика, как лидерборд)."""
    model.eval()
    out = np.empty(len(rows), dtype=np.float64)
    prog = Progress(-(-len(rows) // batch_size), desc)
    with torch.no_grad():
        for i, (x, _, take) in enumerate(batches(seq, rows, cutoff, lookback, batch_size), 1):
            xb = to_device(x, mean, std, device)
            with autocast(device):
                p = model(xb)
            out[take] = p.float().cpu().numpy()
            prog.update(i)
    prog.close()
    return out


def to_device(x: np.ndarray, mean, std, device) -> torch.Tensor:
    xb = torch.from_numpy(x).to(device, non_blocking=True)
    xb[:, :, :len(CHANNELS)] = (xb[:, :, :len(CHANNELS)] - mean) / std
    return xb


def train_model(seq, train_rows, train_y, train_cuts, val_rows, val_y, val_cut, args,
                mean, std, device, epochs: int | None = None):
    """Обучение с ранней остановкой по валидации; возвращает (модель, эпохи, история).

    Если `val_rows` пуст — это финальное дообучение: остановка по числу эпох,
    найденному на валидации (та же логика, что `--final` в train.py, где число
    итераций фиксируется заранее).
    """
    torch.manual_seed(args.seed)
    model = SeqNet(len(CHANNELS) + 1, args.hidden, args.layers, args.arch,
                   args.dropout, args.lookback).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_par:,} параметров | окно {args.lookback} дней | "
          f"устройство {device}")

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
        prog = Progress(steps, f"эпоха {epoch}/{n_epochs}")
        # Срезы перемешиваются целиком, а не построчно: батч из одного среза —
        # это и локальность memmap, и корректные окна (у срезов разные границы).
        for ci in rng.permutation(len(train_cuts)):
            cut, rows, y = train_cuts[ci], train_rows[ci], train_y[ci]
            ylog = np.log1p(y)
            for x, yb, _ in batches(seq, rows, cut, args.lookback, args.batch_size,
                                    ylog, shuffle=True, rng=rng):
                xb = to_device(x, mean, std, device)
                tb = torch.from_numpy(yb).float().to(device)
                with autocast(device):
                    loss = nn.functional.mse_loss(model(xb), tb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                run += float(loss) * len(yb)
                seen += len(yb)
                done += 1
                prog.update(done, f" | RMSE {np.sqrt(run / seen):.4f}")
        prog.close()

        line = f"эпоха {epoch:>2}/{n_epochs} | train RMSE {np.sqrt(run / seen):.5f}"
        if len(val_rows):
            p = evaluate(model, seq, val_rows, val_cut, args.lookback, args.batch_size,
                         mean, std, device, desc=f"эпоха {epoch}/{n_epochs} валидация")
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
    """
    x = np.zeros((1, args.lookback, len(CHANNELS) + 1), dtype=np.float32)
    x[:, :, len(CHANNELS)] = 1.0
    model.eval()
    with torch.no_grad():
        xb = to_device(x, mean, std, device)
        with autocast(device):
            return float(model(xb).float().cpu().numpy()[0])


def make_submission(model, seq, users, first_day, args, mean, std, device, meta: dict) -> None:
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
                     mean, std, device)
    # У неизвестных пользователей searchsorted указал на чужую строку — им
    # положено предсказание по пустой истории, а не по соседу из индекса.
    if (~known).sum():
        p_log[~known] = predict_empty(model, args, mean, std, device)
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

    t0 = time.time()
    print("считаю масштаб каналов по выборке обучающих окон...", flush=True)
    mean_np, std_np = channel_stats(seq, train_rows[0], train_cuts[0], args.lookback,
                                    seed=args.seed)
    print(f"  готово за {time.time() - t0:.0f}s")
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)

    model, best_epoch, _ = train_model(seq, train_rows, train_y, train_cuts,
                                       val_rows, val_y, val_cut, args, mean, std, device)

    p_log = evaluate(model, seq, val_rows, val_cut, args.lookback, args.batch_size,
                     mean, std, device)
    print(f"\n--- валидация (cutoff {val_cut}) ---")
    res = report(val_y, np.expm1(np.clip(p_log, 0, None)), args.arch)

    if args.save_val_pred:
        np.savez(MODELS / f"{args.name}_valpred_{val_cut}.npz",
                 user_id=users[val_rows], pred_log=p_log, target=val_y)
        print(f"предсказания валидации: {MODELS / f'{args.name}_valpred_{val_cut}.npz'}")

    append_csv(MODELS / "experiments.csv", EXPERIMENT_FIELDS, {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
        "feat_ver": f"seq{len(CHANNELS)}x{args.lookback}", "blocks": "seq",
        "name": args.name, "model": args.arch, "cutoffs": len(train_cuts),
        "n_features": len(CHANNELS) + 1,
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
                                 f"lookback={args.lookback} bs={args.batch_size} lr={args.lr}")
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
    final, _, _ = train_model(seq, all_rows, all_y, all_cuts, np.array([], dtype=int),
                              np.array([]), val_cut, args, mean, std, device,
                              epochs=best_epoch)

    ckpt = MODELS / f"{args.name}_{args.arch}.pt"
    torch.save({"state_dict": final.state_dict(), "mean": mean_np, "std": std_np,
                "args": vars(args), "channels": CHANNELS, "val_metrics": res}, ckpt)
    (MODELS / f"{args.name}_meta.json").write_text(
        json.dumps({"name": args.name, "model": args.arch, "final": True,
                    "lookback": args.lookback, "epochs": best_epoch,
                    "val_cutoff": str(val_cut), "metrics": res}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"веса: {ckpt}")

    make_submission(final, seq, users, first_day, args, mean, std, device, res)


if __name__ == "__main__":
    main()
