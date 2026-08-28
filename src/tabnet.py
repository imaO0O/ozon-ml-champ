"""Нейросеть на ТАБЛИЧНЫХ признаках — последнее непроверенное семейство моделей.

Зачем. У нас два семейства: бустинг на 242 агрегатах (кусочно-постоянные
функции) и рекуррентные сети на последовательностях (дневной, недельный
годовой, событийный вход). Нейросети на самих агрегатах нет ни одной.

Это та же информация, что у бустинга, но принципиально другое индуктивное
смещение: гладкие функции вместо ступенчатых, глобальная параметризация
вместо локальных разбиений. По закону партнёрства (PLAN, «расстояние ничего
не значит, значит направление») такой кандидат осмыслен: полезен не тот,
кто непохож, а тот, чьё отличие направлено против нашей ошибки, — и разная
форма гипотез на одних данных ровно это может дать.

Ожидание при этом скромное и его стоит назвать заранее: на табличных данных
такой размерности бустинг обычно сильнее нейросети, поэтому рука будет
отставать. Вопрос не в её качестве, а в направлении.

Остановка по ВЫРОВНЕННОМУ RMSLE — уровень на сабмите правится бесплатно
(PLAN, раздел 4а). Это тот самый дефект, который у нас обесценил 79 прогонов
сети, и повторять его в новом скрипте нельзя.

    python -u src/tabnet.py --cutoffs 6
    python -u src/tabnet.py --cutoffs 7 --val-cutoff 2025-12-16
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np

from config import MODELS, SEED, train_cutoffs
from datasets import feature_names, features_version, get_dataset, parse_blocks
from metrics import gini_norm, rmse_log
from utils import append_csv, git_commit

LOG = MODELS / "experiments.csv"
FIELDS = ["created", "commit", "feat_ver", "blocks", "name", "model", "cutoffs",
          "n_features", "rmsle_single", "rmsle_two_stage", "rmsle_blend", "blend_w",
          "gini_blend", "sum_bias_blend", "best_iter_single", "note", "val_cutoff",
          "train_cutoffs", "stride", "halflife"]


def aligned(y: np.ndarray, p: np.ndarray) -> float:
    return rmse_log(y, p - p.mean() + y.mean())


def prepare(X: np.ndarray, mean=None, std=None):
    """Нормировка входа. Без неё сеть на 242 разномасштабных колонках не учится.

    Медиана и межквартильный размах вместо среднего и дисперсии: у оконных
    сумм тяжёлые хвосты, и стандартизация по дисперсии сжала бы девяносто
    процентов клиентов в точку.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if mean is None:
        mean = np.median(X, axis=0)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        std = np.maximum(q3 - q1, 1e-3)
    Z = (X - mean) / std
    return np.clip(Z, -8.0, 8.0).astype(np.float32), mean, std


def build(n_in: int, hidden: int, depth: int, dropout: float):
    import torch.nn as nn
    layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                   nn.Dropout(dropout)]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--val-cutoff", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--name", default="tab")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cuts = train_cutoffs(args.cutoffs)
    val_cut = dt.date.fromisoformat(args.val_cutoff) if args.val_cutoff else cuts[0]
    train_cuts = [c for c in cuts if c < val_cut]
    blocks = parse_blocks(args.blocks)

    dfv = get_dataset(val_cut, blocks=blocks)
    feats = feature_names(dfv)
    ids = dfv["user_id"].to_numpy()
    order = np.argsort(ids)
    Xv_raw = dfv.select(feats).to_numpy().astype(np.float64)
    yv = dfv["target"].to_numpy().astype(np.float64)
    parts = [get_dataset(c, blocks=blocks) for c in train_cuts]
    Xt_raw = np.vstack([d.select(feats).to_numpy().astype(np.float64) for d in parts])
    yt = np.concatenate([d["target"].to_numpy().astype(np.float64) for d in parts])
    del parts

    Xt, mean, std = prepare(Xt_raw)
    Xv, _, _ = prepare(Xv_raw, mean, std)
    ytl, yvl = np.log1p(yt), np.log1p(yv)
    print(f"валидация {val_cut} | обучение {len(ytl):,} x {len(feats)} | устройство {dev}")

    model = build(len(feats), args.hidden, args.depth, args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"параметров {n_par:,} | скрытый слой {args.hidden} x {args.depth}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = int(np.ceil(len(ytl) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps * args.epochs,
                                                pct_start=0.15)
    Xt_t = torch.from_numpy(Xt)
    yt_t = torch.from_numpy(ytl.astype(np.float32))
    Xv_t = torch.from_numpy(Xv).to(dev)

    best = (float("inf"), None, 0)
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(ytl))
        run = 0.0
        for i in range(steps):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = Xt_t[idx].to(dev, non_blocking=True)
            yb = yt_t[idx].to(dev, non_blocking=True)
            p = model(xb)[:, 0]
            loss = nn.functional.mse_loss(p, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(Xv_t[i:i + 8192])[:, 0].cpu().numpy()
                                 for i in range(0, len(Xv_t), 8192)]).astype(np.float64)
        # Остановка по ВЫРОВНЕННОЙ величине: уровень правится бесплатно.
        score = aligned(yvl, pv)
        mark = ""
        if score < best[0]:
            best = (score, pv.copy(), ep)
            mark = "  <- лучшая"
        print(f"эпоха {ep:>2}/{args.epochs} | train MSE {run / len(ytl):.5f} | "
              f"val выровн. {score:.5f} | {time.time() - t0:.0f}s{mark}")
        if ep - best[2] >= args.patience:
            print(f"ранняя остановка: {args.patience} эпох без улучшения")
            break

    pv = best[1]
    a = aligned(yvl, pv)
    print(f"\nлучшая эпоха {best[2]} | выровненный {a:.5f} | "
          f"сырой {rmse_log(yvl, pv):.5f} | Gini {gini_norm(yv, np.expm1(pv)):.4f}")

    append_csv(LOG, FIELDS, {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(), "feat_ver": features_version(),
        "blocks": args.blocks or "all", "name": args.name, "model": "mlp",
        "cutoffs": len(train_cuts), "n_features": len(feats),
        "rmsle_single": round(float(rmse_log(yvl, pv)), 5),
        "rmsle_blend": round(float(rmse_log(yvl, pv)), 5),
        "gini_blend": round(float(gini_norm(yv, np.expm1(pv))), 4),
        "best_iter_single": best[2], "val_cutoff": str(val_cut),
        "train_cutoffs": " ".join(str(c) for c in train_cuts), "stride": 30,
        "note": f"{args.note} [MLP на табличных признаках, {args.hidden}x{args.depth}, "
                f"выровненный {a:.5f}]",
    })
    np.savez_compressed(MODELS / f"{args.name}_valpred_{val_cut}.npz",
                        user_id=ids[order], pred_log=pv[order], target=yv[order])
    print(f"записано в {LOG}, предсказания сохранены")


if __name__ == "__main__":
    main()
