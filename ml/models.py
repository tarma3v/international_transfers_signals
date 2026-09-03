"""Классический ML. Масштабирование внутри Pipeline — иначе оно подсмотрит тест."""
from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_classifiers() -> dict[str, Pipeline]:
    return {
        "логистическая регрессия": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)),
            ]
        ),
        "случайный лес": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=40,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "градиентный бустинг": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        max_iter=200,
                        learning_rate=0.05,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "CatBoost": Pipeline(
            [
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=400,
                        depth=4,
                        learning_rate=0.04,
                        l2_leaf_reg=6.0,
                        random_seed=RANDOM_STATE,
                        verbose=0,
                        allow_writing_files=False,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=3.0,
                        min_child_weight=20,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),
                )
            ]
        ),
    }


def make_regressor() -> Pipeline:
    """Регрессия на выгоду в бп — метрика, которую чувствует клиент."""
    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))])
