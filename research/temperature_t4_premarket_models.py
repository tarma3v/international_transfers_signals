"""Frozen premarket feature selection and quarterly history-only model."""
from __future__ import annotations

import datetime as dt

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


SEED = 20260904
MIN_TRAIN_DATE = dt.date(2022, 2, 24)
FEATURES = (
    'pct_range_30', 'pct_range_90', 'pct_range_180',
    'days_beaten_30', 'days_beaten_90', 'days_beaten_180',
    'rank_level_20', 'rank_level_60', 'rank_level_120', 'rank_level_250',
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_60',
    'vol_10', 'vol_30', 'vol_90', 'vol_ratio', 'vol_ratio_5_60',
    'vol_ratio_20_120', 'gap_days', 'is_after_gap',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'dom_sin', 'dom_cos',
    'in_payday_window', 'pre_holiday_14d', 'pre_new_year_14',
    'pre_sep_first_14', 'first_week_month', 'last_week_month',
    'cny_raw_ret_1', 'cny_raw_ret_5', 'cny_raw_ret_20',
    'usd_raw_ret_1', 'usd_raw_ret_5', 'usd_raw_ret_20',
    'eur_raw_ret_1', 'eur_raw_ret_5', 'eur_raw_ret_20',
    'peer_ret_5_mean', 'peer_dispersion_5', 'rel_to_peers_5',
    'currency_AMD', 'currency_KGS', 'currency_KZT', 'currency_TJS',
    'currency_UZS',
)


def compact_features(matrix, names):
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in FEATURES if name not in lookup]
    if missing:
        raise KeyError(missing)
    return np.asarray(matrix[:, [lookup[name] for name in FEATURES]], dtype=float)


def fit_quarterly_hist(features, target, maturity, dates):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    probability = np.full(len(dates), np.nan)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]
    for origin in origins:
        end_month = origin.month + 3
        end = (dt.date(origin.year + 1, end_month - 12, 1)
               if end_month > 12 else dt.date(origin.year, end_month, 1))
        query = ((dates >= origin) & (dates < end)
                 & np.all(np.isfinite(features), axis=1))
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        train = ((dates >= MIN_TRAIN_DATE) & (dates < origin)
                 & np.array([value < cutoff for value in maturity])
                 & np.isfinite(target) & np.all(np.isfinite(features), axis=1))
        ids = np.flatnonzero(train)
        n_train[query] = len(ids)
        if len(ids) < 500 or np.unique(target[train]).size < 2:
            logs.append({'origin': str(origin), 'n_train': int(len(ids)),
                         'n_query': int(query.sum()), 'model_fit': False,
                         'last_target_maturity': ''})
            continue
        model = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=42, l2_regularization=15., random_state=SEED)
        age = np.array([(origin - dates[i]).days for i in ids], dtype=float)
        weight = np.exp2(-age / 730.)
        model.fit(features[train], target[train].astype(int),
                  sample_weight=weight)
        probability[query] = model.predict_proba(features[query])[:, 1]
        logs.append({
            'origin': str(origin), 'n_train': int(len(ids)),
            'n_positive': int(target[train].sum()), 'n_query': int(query.sum()),
            'model_fit': True,
            'last_target_maturity': str(max(maturity[train])),
        })
    return probability, n_train, logs
