"""Метрика соревнования и tie-breaker'ы жюри."""
from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE в исходной шкале (предсказания клипаются снизу нулём, как на лидерборде)."""
    p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(p)) ** 2)))


def rmse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """То же самое, но когда обе величины уже в log1p-шкале."""
    p = np.clip(y_pred_log, 0, None)
    return float(np.sqrt(np.mean((y_true_log - p) ** 2)))


def _gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    order = np.argsort(-y_pred, kind="mergesort")
    y = y_true[order]
    n = len(y)
    cum = np.cumsum(y) / (y.sum() + 1e-12)
    return float((cum.sum() / n - (n + 1) / (2 * n)))


def gini_norm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Нормированный Gini: 1.0 = идеальное ранжирование клиентов по GMV."""
    denom = _gini(y_true, y_true)
    return float(_gini(y_true, y_pred) / denom) if denom > 0 else 0.0


def sum_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Смещение суммарного GMV со знаком: <0 — недооценка, >0 — переоценка."""
    t = y_true.sum()
    return float((np.clip(y_pred, 0, None).sum() - t) / (t + 1e-12))


def rmspe_total(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Модуль относительной ошибки суммарного GMV — tie-breaker жюри."""
    return abs(sum_bias(y_true, y_pred))


def report(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    m = {
        "rmsle": rmsle(y_true, y_pred),
        "gini": gini_norm(y_true, y_pred),
        "rmspe_total": rmspe_total(y_true, y_pred),
        "sum_bias": sum_bias(y_true, y_pred),
    }
    tag = f"{name:>14} | " if name else ""
    print(f"{tag}RMSLE {m['rmsle']:.5f} | Gini {m['gini']:.4f} | сумма {m['sum_bias']:+.2%}")
    return m
