"""Candidate models for lift benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class Scorer(Protocol):
    def fit(self, frame: pd.DataFrame, target: str, features: list[str]) -> "Scorer": ...
    def score(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray: ...


@dataclass
class SklearnModel:
    """Wrapper exposing a unified score API for sklearn classifiers."""

    estimator: object

    def fit(self, frame: pd.DataFrame, target: str, features: list[str]) -> "SklearnModel":
        self.estimator.fit(frame[features], frame[target].astype(int))
        return self

    def score(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(frame[features])[:, 1]
        if hasattr(self.estimator, "decision_function"):
            values = self.estimator.decision_function(frame[features])
            return 1.0 / (1.0 + np.exp(-values))
        return np.asarray(self.estimator.predict(frame[features]), dtype=float)


@dataclass
class RuleModel:
    """Transparent indicator scorer: no fitting, only past-only engineered features."""

    name: str

    def fit(self, frame: pd.DataFrame, target: str, features: list[str]) -> "RuleModel":
        return self

    def score(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        if self.name == "level_low_percentile":
            return (
                (100.0 - frame["pct_range_90"].clip(0, 100)) * 0.45
                + frame["days_beaten_90"].clip(0, 100) * 0.55
            ).to_numpy() / 100.0
        if self.name == "momentum_down":
            return (
                frame["streak_down"].clip(0, 5) / 5.0 * 0.55
                + np.maximum(frame["ret_5"], 0).clip(0, 300) / 300.0 * 0.45
            ).to_numpy()
        if self.name == "reversal_from_low":
            low_component = (100.0 - frame["pct_range_30"].clip(0, 100)) / 100.0
            bounce_component = np.maximum(-frame["ret_1"], 0).clip(0, 150) / 150.0
            return (low_component * 0.65 + bounce_component * 0.35).to_numpy()
        raise ValueError(f"unknown rule model: {self.name}")


def make_model(name: str) -> Scorer:
    """Create a fresh model instance for each walk-forward fold."""
    if name == "random_baseline":
        return SklearnModel(DummyClassifier(strategy="prior"))
    if name == "logistic_regression":
        return SklearnModel(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced")),
                ]
            )
        )
    if name == "random_forest":
        return SklearnModel(
            RandomForestClassifier(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=30,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        )
    if name == "hist_gradient_boosting":
        return SklearnModel(
            HistGradientBoostingClassifier(
                max_iter=160,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        )
    if name in {
        "level_low_percentile",
        "momentum_down",
        "reversal_from_low",
    }:
        return RuleModel(name)
    raise ValueError(f"unknown model: {name}")


def default_model_names() -> list[str]:
    """Models intentionally kept installable with only scikit-learn."""
    return [
        "level_low_percentile",
        "momentum_down",
        "reversal_from_low",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
