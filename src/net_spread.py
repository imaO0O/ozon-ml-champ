"""Неуверенность сети как признак стекинга: расхождение сидов, а не только среднее.

Что бустинг видит сейчас. Признаки `net_rank` и `net_centered` — это ТОЧКА:
усреднённое по трём сидам предсказание сети. Где сеть уверена, а где гадает,
бустинг не знает и узнать не может.

Что добавляется. Те же три сида, но их РАСХОЖДЕНИЕ на каждом клиенте. Там,
где три независимо обученные сети сходятся, предсказанию можно верить; там,
где они расходятся вдвое, бустингу стоит опереться на собственные признаки.
Это ровно та условная на клиенте информация, ради которой стекинг и заводился,
и её у него до сих пор не было.

Чем это отличается от отвергнутого скрытого состояния. Там подавались 128
сырых осей, и они дали −0.00065 сверх предсказания: масштаб плывёт между
срезами, а размерность съедает ёмкость. Здесь **одна** колонка,
интерпретируемая, и по построению устойчивая к уровню — расхождение сидов
не зависит от того, где стоит их среднее.

Стоимость: ноль обучений. Предсказания всех сидов по всем срезам уже лежат
на диске после сборки OOF, тестовые — это три файла сабмитов, из которых
собиралось усреднение.

    python -u src/net_spread.py --name b64 --seeds 42,13,7 \\
        --test gru_b64_f42.csv,gru_b64_f13.csv,gru_b64_f7.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from config import MODELS, SAMPLE_SUBMIT, SUBMISSIONS, train_cutoffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="b64", help="префикс OOF, из которого берём сиды")
    ap.add_argument("--seeds", default="42,13,7")
    ap.add_argument("--out", default="spread", help="имя новой «сети» для --net-names")
    ap.add_argument("--test", default=None,
                    help="файлы сабмитов по сидам через запятую — для тестового среза")
    ap.add_argument("--cutoffs", type=int, default=7)
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",")]
    made = 0
    for cut in train_cutoffs(args.cutoffs):
        parts, users, target = [], None, None
        for sd in seeds:
            path = MODELS / f"netoof_{args.name}_s{sd}_{cut}_valpred_{cut}.npz"
            if not path.exists():
                parts = []
                break
            z = np.load(path)
            order = np.argsort(z["user_id"])
            if users is None:
                users, target = z["user_id"][order], z["target"][order]
            elif not np.array_equal(z["user_id"][order], users):
                raise SystemExit(f"{path.name}: другой состав клиентов")
            parts.append(z["pred_log"].astype(np.float64)[order])
        if not parts:
            print(f"  {cut}: нет полного набора сидов, пропуск")
            continue
        spread = np.std(parts, axis=0)
        out = MODELS / f"netoof_{args.out}_{cut}.npz"
        np.savez_compressed(out, user_id=users, pred_log=spread, target=target)
        print(f"  {cut}: расхождение сидов — среднее {spread.mean():.4f}, "
              f"медиана {np.median(spread):.4f}, макс {spread.max():.4f}")
        made += 1

    if args.test:
        ref = pl.read_csv(SAMPLE_SUBMIT)["user_id"].to_numpy()
        cols = []
        for f in args.test.split(","):
            df = pl.read_csv(SUBMISSIONS / f.strip())
            if not np.array_equal(df["user_id"].to_numpy(), ref):
                raise SystemExit(f"{f}: другой порядок user_id")
            cols.append(np.log1p(df["predict"].to_numpy().astype(np.float64)))
        spread = np.std(cols, axis=0)
        np.savez_compressed(MODELS / f"netoof_{args.out}_test.npz",
                            user_id=ref, pred_log=spread,
                            target=np.zeros(len(ref)))
        print(f"  тест: расхождение сидов — среднее {spread.mean():.4f}, "
              f"медиана {np.median(spread):.4f}")
        made += 1

    print(f"\nготово файлов: {made}. Подключается как ещё одна «сеть»:")
    print(f"  python -u src/train.py --cutoffs 6 --ensemble --net "
          f"--net-names {args.name},{args.out}")


if __name__ == "__main__":
    main()
