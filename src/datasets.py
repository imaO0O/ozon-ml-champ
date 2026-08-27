"""Сборка и кэширование обучающих выборок по cutoff'ам."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import time
from pathlib import Path

import polars as pl

from config import DATA_PROC, HORIZON, TEST_CUTOFF, TRAIN_PARQUET, train_cutoffs
from features import BLOCKS, build_dataset

ID = "user_id"
NON_FEATURES = {ID, "target", "cutoff"}


def features_version(blocks: list[str] | None = None) -> str:
    """Хеш кода признаков: любое изменение блоков или config.py обесценивает кэш.

    Без этого сокомандник, добавивший признак, молча обучится на старом кэше
    и решит, что его идея не работает. Пересборка стоит ~20 секунд на срез.

    Переводы строк нормализуются: git на Windows подменяет LF на CRLF при
    выгрузке, и без нормализации хеш менялся бы после каждого `git checkout`
    или у сокомандника с другими настройками autocrlf — кэш пересобирался бы
    впустую, а `feat_ver` в журналах выглядел бы разным при одинаковом коде.
    """
    def norm(path: Path) -> bytes:
        return path.read_bytes().replace(b"\r\n", b"\n")

    pkg = Path(__file__).with_name("features")
    src = b"".join(norm(p) for p in sorted(pkg.glob("*.py")))
    src += norm(Path(__file__).with_name("config.py"))
    src += ("|".join(blocks) if blocks else "all").encode()
    return hashlib.md5(src).hexdigest()[:8]


def dataset_path(cutoff: dt.date, blocks: list[str] | None = None, history: int | None = None,
                 net: bool = False, net_feats: str = "rank_centered",
                 horizon: int | None = None):
    # В имени — источник данных (синтетика/реальные), версия кода признаков,
    # глубина обрезки истории и наличие признаков сети: это разные выборки.
    tail = f"_h{history}" if history else ""
    # Горизонт таргета — часть ключа: выборка с 15-дневным таргетом и с
    # 30-дневным различаются только колонкой target, и перепутать их нельзя.
    if horizon and horizon != HORIZON:
        tail += f"_hz{horizon}"
    if net:
        # Имена сетей входят в ключ: выборка с одной сетью и с тремя — разные.
        tail += "_net" if net is True else "_net-" + "-".join(net)
        # Набор признаков сети тоже: с уровнем и без — разные выборки.
        if net_feats != "rank_centered":
            tail += f"-{net_feats}"
    return (DATA_PROC /
            f"ds_{TRAIN_PARQUET.stem}_{cutoff.isoformat()}_{features_version(blocks)}{tail}.parquet")


def get_dataset(cutoff: dt.date, with_target: bool = True, rebuild: bool = False,
                blocks: list[str] | None = None, history: int | None = None,
                net: bool = False, net_feats: str = "rank_centered",
                horizon: int | None = None) -> pl.DataFrame:
    path = dataset_path(cutoff, blocks, history, net, net_feats, horizon)
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    t0 = time.time()
    if horizon and horizon != HORIZON:
        raise SystemExit(
            f"выборки с горизонтом {horizon} нет в кэше: соберите её через "
            f"src/fresh.py, здесь она только читается")
    df = build_dataset(cutoff, with_target=with_target, blocks=blocks, history=history,
                       net=net, net_feats=net_feats)
    df.write_parquet(path)
    print(f"[{cutoff}] {df.height:,} строк x {df.width} колонок за {time.time() - t0:.1f}s -> {path.name}")
    return df


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def clean_stale_cache(keep: str) -> None:
    """Удалить выборки прошлых версий признаков — каждая версия весит ~640 МБ.

    Версия ищется в любом месте имени, а не только в конце: у выборок,
    собранных с --history, после хеша стоит суффикс вида `_h229`, и проверка
    по окончанию имени сносила их всегда, сколько бы раз их ни пересобирали.
    """
    freed = 0
    for p in DATA_PROC.glob("ds_*.parquet"):
        if keep in p.stem:
            continue
        freed += p.stat().st_size
        p.unlink()
    print(f"освобождено {freed / 1024 ** 2:.0f} МБ, оставлена версия {keep}")


def parse_blocks(value: str | None) -> list[str] | None:
    """--blocks windows,lifetime — для замера вклада отдельных блоков."""
    return [b.strip() for b in value.split(",") if b.strip()] if value else None


def parse_net(flag: bool, names: str | None):
    """True = одна безымянная сеть, список имён = стекинг на нескольких."""
    if names:
        return [n.strip() for n in names.split(",") if n.strip()]
    return bool(flag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, default=None, help="сколько обучающих cutoff'ов собрать")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--test", action="store_true", help="собрать также выборку на TEST_CUTOFF (без таргета)")
    ap.add_argument("--blocks", default=None,
                    help=f"подмножество блоков через запятую (есть: {','.join(sorted(BLOCKS))})")
    ap.add_argument("--clean", action="store_true", help="удалить кэш прошлых версий признаков")
    ap.add_argument("--stride", type=int, default=None, help="шаг между срезами в днях")
    ap.add_argument("--net", action="store_true",
                    help="признак предсказания сети (features/net.py)")
    ap.add_argument("--net-names", default=None,
                    help="имена сетей через запятую для стекинга на нескольких, например r180,ch180,w90")
    ap.add_argument("--history", type=int, default=None,
                    help="обрезать историю до K дней на всех срезах (одинаковая глубина)")
    args = ap.parse_args()

    blocks = parse_blocks(args.blocks)
    if args.clean:
        clean_stale_cache(features_version(blocks))
    n = args.cutoffs if args.cutoffs is not None else None
    cuts = train_cutoffs(stride=args.stride) if n is None else train_cutoffs(n, args.stride)
    for c in cuts:
        get_dataset(c, with_target=True, rebuild=args.rebuild, blocks=blocks,
                    history=args.history, net=parse_net(args.net, args.net_names))
    if args.test:
        get_dataset(TEST_CUTOFF, with_target=False, rebuild=args.rebuild, blocks=blocks,
                    history=args.history, net=parse_net(args.net, args.net_names))


if __name__ == "__main__":
    main()
