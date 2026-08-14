"""Сборка сабмита: признаки на TEST_CUTOFF -> предсказание -> csv для всех 250k."""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import polars as pl

from config import MODELS, SAMPLE_SUBMIT, SUBMISSIONS, TEST_CUTOFF
from datasets import get_dataset
from models import GBM
from utils import append_csv, git_commit

# Признаки-счётчики у «спящего» пользователя равны нулю, а не «неизвестны».
ZERO_FILL_HINTS = ("_sum_", "_days_", "active_days", "ord_days", "lt_")


def build_test_frame(features: list[str], rebuild: bool = False,
                     blocks: list[str] | None = None) -> tuple[pl.Series, np.ndarray]:
    users = pl.read_csv(SAMPLE_SUBMIT).select("user_id")
    feats = get_dataset(TEST_CUTOFF, with_target=False, rebuild=rebuild, blocks=blocks)
    df = users.join(feats, on="user_id", how="left")
    missing = df[features[0]].null_count()
    if missing:
        print(f"внимание: {missing:,} пользователей без истории до {TEST_CUTOFF} — заполняю нулями")
    fills = [pl.col(c).fill_null(0.0) for c in features if any(h in c for h in ZERO_FILL_HINTS)]
    if fills:
        df = df.with_columns(fills)
    X = df.select(features).to_numpy().astype(np.float32)
    return df["user_id"], X


def log_submission(row: dict) -> None:
    """Журнал сабмитов: какой файл каким кодом и с какой валидацией получен.

    Нужен и команде (5 сабмитов в день на всех), и жюри — воспроизводимость.
    Поле lb_score заполняется руками после ответа лидерборда.
    """
    fields = ["file", "created", "commit", "name", "model", "blend_w",
              "val_rmsle", "val_gini", "val_sum_err", "pred_sum", "pred_zeros", "lb_score", "note"]
    append_csv(SUBMISSIONS / "log.csv", fields, row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm", help="имя артефактов из models/")
    ap.add_argument("--out", default=None, help="имя файла сабмита")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--note", default="", help="комментарий для журнала сабмитов")
    args = ap.parse_args()

    # Мета создаётся только вместе с весами, то есть под --final. Без неё
    # прежняя проверка была недостижима: чтение падало трейсбеком строкой выше.
    meta_path = MODELS / f"{args.name}_meta.json"
    if not meta_path.exists():
        have = sorted(p.stem[:-5] for p in MODELS.glob("*_meta.json"))
        raise SystemExit(
            f"нет моделей с именем '{args.name}': обучите их командой "
            f"train.py --final --name {args.name}"
            + (f"\nготовые имена: {', '.join(have)}" if have else ""))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("final"):
        raise SystemExit(f"модели '{args.name}' обучены без --final, для сабмита переобучите train.py --final")
    features, w = meta["features"], meta["blend_w"]
    # Тестовые признаки собираем тем же набором блоков, что и обучающие.
    blocks = meta.get("blocks") if meta.get("blocks") != "all" else None

    user_id, X = build_test_frame(features, rebuild=args.rebuild, blocks=blocks)
    single = GBM.load(MODELS / f"{args.name}_single.pkl")
    clf = GBM.load(MODELS / f"{args.name}_clf.pkl")
    reg = GBM.load(MODELS / f"{args.name}_reg.pkl")

    pred_log = (1 - w) * single.predict(X) + w * (clf.predict(X) * reg.predict(X))
    pred = np.clip(np.expm1(np.clip(pred_log, 0, None)), 0, None)

    # Время в имени обязательно: сабмитов до пяти в день на команду, и файл
    # без часов-минут молча затирает предыдущий вместе со строкой журнала.
    out = args.out or f"{args.name}_{dt.datetime.now():%m%d_%H%M}.csv"
    path = SUBMISSIONS / out
    if path.exists():
        raise SystemExit(f"{path.name} уже существует — задайте --out, чтобы не затереть прошлый сабмит")
    pl.DataFrame({"user_id": user_id, "predict": pred.astype(np.float32)}).write_csv(path)

    print(f"\n{path}")
    print(f"строк: {len(pred):,} | нулевых: {(pred < 1e-6).mean():.2%} | "
          f"среднее: {pred.mean():.2f} | медиана: {np.median(pred):.2f} | max: {pred.max():,.0f}")
    print(f"суммарный предсказанный GMV: {pred.sum():,.0f}")

    val = meta.get("metrics", {}).get("blend", {})
    log_submission({
        "file": out, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(), "name": args.name, "model": meta.get("model"),
        "blend_w": round(w, 2), "val_rmsle": round(val.get("rmsle", float("nan")), 5),
        "val_gini": round(val.get("gini", float("nan")), 4),
        "val_sum_err": round(val.get("rmspe_total", float("nan")), 4),
        "pred_sum": round(float(pred.sum())), "pred_zeros": f"{(pred < 1e-6).mean():.4f}",
        "note": args.note,
    })


if __name__ == "__main__":
    main()
