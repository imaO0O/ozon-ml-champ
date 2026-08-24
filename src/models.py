"""Единый интерфейс к бустингам: LightGBM (CPU) и CatBoost/XGBoost (GPU у сокомандников).

Обучение всегда идёт в log1p-шкале таргета, поэтому обычный RMSE-лосс = метрика
соревнования, а ранняя остановка по 'rmse' валидна напрямую.
"""
from __future__ import annotations

import numpy as np

from config import SEED

LGB_REG = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.75,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=255,
    verbosity=-1,
    seed=SEED,
)
LGB_BIN = {**LGB_REG, "objective": "binary", "metric": "auc"}


def aligned_rmsle(preds, data):
    """Ранняя остановка по ВЫРОВНЕННОМУ RMSLE — той величине, что решает у нас.

    Умолчание `rmse` считает ошибку вместе с уровнем предсказаний. Но уровень
    на сабмите правится бесплатным сдвигом из TEST_LEVEL, поэтому в зачёт идёт
    только форма. С ростом итераций уровень уезжает, сырая метрика перестаёт
    улучшаться и обрывает обучение — а форма продолжает улучшаться, и после
    выравнивания это видно:

        срез        оптимум по сырому   оптимум по выровненному   цена
        2026-01-15        128                    ~200            +0.00072
        2025-12-16        128                     260            +0.00095

    То есть ловушка уровня, которая четырежды искажала наши ВЫВОДЫ, сидела
    ещё и внутри самого обучения, и обрывала его вдвое раньше нужного.
    """
    y = data.get_label()
    p = np.asarray(preds, dtype=np.float64)
    p = p - p.mean() + y.mean()
    return "rmsle_aligned", float(np.sqrt(np.mean((p - y) ** 2))), False


class Ensemble:
    """Среднее нескольких моделей в log1p-шкале — там же, где живёт метрика.

    Интерфейс совпадает с GBM (predict/save/feature_importance), поэтому
    predict.py не отличает ансамбль от одиночной модели и менять его не нужно.

    Почему усреднение, а не выбор лучшей: конфигурации-победители не переносятся
    между валидационными срезами (проверено на двух), а среднее выигрывает
    у рабочего умолчания на обоих.
    """

    def __init__(self, models: list["GBM"], names: list[str] | None = None):
        self.models = models
        self.names = names or [f"m{i}" for i in range(len(models))]

    @property
    def best_iter(self) -> int:
        return max(m.best_iter for m in self.models)

    def predict(self, X) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def feature_importance(self, names: list[str]) -> list[tuple[str, float]]:
        total: dict[str, float] = {}
        for m in self.models:
            for f, g in m.feature_importance(names):
                total[f] = total.get(f, 0.0) + g / len(self.models)
        return sorted(total.items(), key=lambda t: -t[1])

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)


class GBM:
    """kind: lgbm | cat | xgb; task: reg | bin; device: cpu | gpu."""

    def __init__(self, kind: str = "lgbm", task: str = "reg", device: str = "cpu",
                 n_estimators: int = 6000, early_stopping: int = 200, params: dict | None = None,
                 log_period: int = 200):
        self.kind, self.task, self.device = kind, task, device
        self.n_estimators, self.early_stopping = n_estimators, early_stopping
        self.log_period = log_period  # 0 — молча, нужно при переборе гиперпараметров
        self.params = dict(params or {})
        # Остановка по выровненному RMSLE вместо сырого. Умолчание True:
        # выровненная величина — единственная, по которой мы вообще сравниваем
        # модели (PLAN.md, раздел 2), и останавливаться по другой значит
        # обрывать обучение по чужому критерию.
        self.align_stop = bool(self.params.pop("align_stop", True))
        self.model = None
        self.best_iter = n_estimators

    def fit(self, X, y, X_val=None, y_val=None, feature_names=None, sample_weight=None,
            init_score=None, init_score_val=None):
        """sample_weight — веса обучающих примеров; валидация всегда без весов,
        иначе метрика перестанет совпадать с лидербордом.

        init_score — смещение: модель учит остаток поверх него. Нужно, чтобы
        уровень пользователя переносился напрямую, а не выучивался деревьями
        в том диапазоне, который был в обучении. Предсказание возвращается уже
        без смещения, прибавлять его обратно должен вызывающий код.
        """
        self.has_init_score = init_score is not None
        has_val = X_val is not None and self.early_stopping > 0
        if self.kind == "lgbm":
            import lightgbm as lgb

            base = dict(LGB_BIN if self.task == "bin" else LGB_REG)
            base.update(self.params)
            if self.device == "gpu":
                base["device_type"] = "gpu"
            dtrain = lgb.Dataset(X, label=y, weight=sample_weight, init_score=init_score,
                                 feature_name=feature_names or "auto")
            valid, cbs = [], [lgb.log_evaluation(self.log_period)]
            if has_val:
                valid = [lgb.Dataset(X_val, label=y_val, init_score=init_score_val,
                                     reference=dtrain)]
                cbs.append(lgb.early_stopping(self.early_stopping, verbose=False))
            feval = None
            if has_val and self.align_stop:
                # Остановка по выровненному RMSLE: см. aligned_rmsle выше.
                base = {**base, "metric": "None"}
                feval = aligned_rmsle
            self.model = lgb.train(base, dtrain, num_boost_round=self.n_estimators,
                                   valid_sets=valid, feval=feval, callbacks=cbs)
            self.best_iter = self.model.best_iteration or self.n_estimators

        elif self.kind == "cat":
            from catboost import CatBoostClassifier, CatBoostRegressor

            kw = dict(iterations=self.n_estimators, learning_rate=0.03, depth=8,
                      l2_leaf_reg=5.0, random_seed=SEED, verbose=200,
                      task_type="GPU" if self.device == "gpu" else "CPU")
            if has_val:
                kw["early_stopping_rounds"] = self.early_stopping
            kw.update(self.params)
            Model = CatBoostClassifier if self.task == "bin" else CatBoostRegressor
            if self.task == "reg":
                kw.setdefault("loss_function", "RMSE")
            self.model = Model(**kw)
            self.model.fit(X, y, sample_weight=sample_weight,
                           eval_set=(X_val, y_val) if has_val else None)
            self.best_iter = int(self.model.get_best_iteration() or self.n_estimators) + 1

        elif self.kind == "xgb":
            import xgboost as xgb

            kw = dict(n_estimators=self.n_estimators, learning_rate=0.03, max_depth=8,
                      min_child_weight=20, subsample=0.85, colsample_bytree=0.75,
                      reg_lambda=5.0, random_state=SEED, tree_method="hist",
                      device="cuda" if self.device == "gpu" else "cpu",
                      objective="binary:logistic" if self.task == "bin" else "reg:squarederror")
            if has_val:
                kw["early_stopping_rounds"] = self.early_stopping
            kw.update(self.params)
            Model = xgb.XGBClassifier if self.task == "bin" else xgb.XGBRegressor
            self.model = Model(**kw)
            self.model.fit(X, y, sample_weight=sample_weight,
                           eval_set=[(X_val, y_val)] if has_val else None, verbose=200)
            self.best_iter = int(getattr(self.model, "best_iteration", self.n_estimators) or self.n_estimators) + 1
        else:
            raise ValueError(f"unknown kind: {self.kind}")
        return self

    def predict(self, X) -> np.ndarray:
        if self.kind == "lgbm":
            return self.model.predict(X, num_iteration=self.best_iter)
        if self.task == "bin":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def feature_importance(self, names: list[str]) -> list[tuple[str, float]]:
        if self.kind == "lgbm":
            imp = self.model.feature_importance("gain")
        elif self.kind == "cat":
            imp = self.model.get_feature_importance()
        else:
            imp = self.model.feature_importances_
        return sorted(zip(names, map(float, imp)), key=lambda t: -t[1])

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "GBM":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
