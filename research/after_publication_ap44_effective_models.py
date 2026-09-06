"""Quarterly mature-only meta-CatBoost for AP44."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ml.data import CORRIDORS
from research.after_publication_ap1 import SEED


EXTRA_FEATURE_NAMES = tuple('currency_' + value for value in CORRIDORS) + (
    'annual_sin', 'annual_cos', 'post_2022_02_24')


def augment_features(base, dates, currencies):
    base = np.asarray(base, dtype=float)
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    annual = np.array([
        2 * np.pi * (day.timetuple().tm_yday - 1) / 365.25 for day in dates])
    extra = np.column_stack([
        *[(currencies == value).astype(float) for value in CORRIDORS],
        np.sin(annual), np.cos(annual),
        np.array([day >= dt.date(2022, 2, 24) for day in dates], dtype=float),
    ])
    return np.column_stack([base, extra])


def fit_quarterly_meta_cat(features, target, maturity, dates, eligible):
    features = np.asarray(features, dtype=float).copy()
    features[~np.isfinite(features)] = np.nan
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    eligible = np.asarray(eligible, dtype=bool)
    prediction = np.full(len(dates), np.nan)
    logs = []
    importances = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2022, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
        if dt.date(year, month, 1) >= dt.date(2022, 7, 1)]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        mature = np.array([value is not None and value < cutoff
                           for value in maturity])
        train = (eligible & np.isfinite(target) & (dates < origin) & mature)
        ids = np.flatnonzero(train)
        constant = len(ids) < 200 or np.unique(target[train]).size < 2
        if constant:
            value = float(target[train].mean()) if train.any() else .5
            prediction[query] = value
            importance = np.zeros(features.shape[1])
        else:
            age = np.array([(origin - dates[i]).days for i in ids], dtype=float)
            weight = np.exp2(-age / 730.)
            model = CatBoostClassifier(
                iterations=240, depth=5, learning_rate=.03,
                l2_leaf_reg=10., random_strength=.5,
                bootstrap_type='Bernoulli', subsample=.8,
                auto_class_weights='Balanced', random_seed=SEED,
                thread_count=2, verbose=False, allow_writing_files=False,
                loss_function='Logloss')
            model.fit(features[train], target[train].astype(int),
                      sample_weight=weight)
            prediction[query] = model.predict_proba(features[query])[:, 1]
            importance = np.asarray(model.get_feature_importance(), dtype=float)
        importances.append(importance)
        logs.append({
            'origin': str(origin),
            'n_train': int(train.sum()),
            'n_positive': int(target[train].sum()) if train.any() else 0,
            'n_query': int(query.sum()),
            'constant_prediction': bool(constant),
            'last_target_maturity': str(max(maturity[train])) if train.any() else '',
        })
    mean_importance = (np.mean(importances, axis=0) if importances
                       else np.zeros(features.shape[1]))
    return prediction, logs, mean_importance
