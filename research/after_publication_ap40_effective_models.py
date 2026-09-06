"""Causal weekly optimal-stopping target, features, logistic fit and router."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS


FEATURE_NAMES = (
    'rank_ap26', 'rank_ridge', 'rank_distributional',
    'rank_mean', 'rank_min', 'rank_max', 'rank_std', 'support_count',
    'known_change', 'weekday_sin', 'weekday_cos', 'days_to_friday',
    'core_flag', 'fallback_flag', 'fallback_precision_margin',
    'ap37_trailing_rate', 'log_days_since', 'ap37_week_used',
    'ap37_month_empty', 'day_of_month_scaled',
)


def frozen_leader_state(dates, currencies, leader_signal):
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    leader_signal = np.asarray(leader_signal, dtype=bool)
    week_used = np.zeros(len(dates), dtype=np.int8)
    month_empty = np.ones(len(dates), dtype=bool)
    for currency in CORRIDORS:
        week, used, month, month_used = None, 0, None, 0
        for i in np.flatnonzero(currencies == currency):
            iso = dates[i].isocalendar()[:2]
            current_month = (dates[i].year, dates[i].month)
            if iso != week:
                week, used = iso, 0
            if current_month != month:
                month, month_used = current_month, 0
            week_used[i] = used
            month_empty[i] = month_used == 0
            if leader_signal[i]:
                used += 1
                month_used += 1
    return week_used, month_empty


def build_stopping_features(panel, expert_ranks, core, fallback,
                            local_precision, overall_precision,
                            trailing_rate, days_since, leader_signal):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    rank_matrix = np.column_stack([np.asarray(value, dtype=float)
                                   for value in expert_ranks.values()])
    finite = np.isfinite(rank_matrix)
    safe = np.where(finite, rank_matrix, np.nan)
    with np.errstate(invalid='ignore'):
        rank_mean = np.nanmean(safe, axis=1)
        rank_min = np.nanmin(safe, axis=1)
        rank_max = np.nanmax(safe, axis=1)
        rank_std = np.nanstd(safe, axis=1)
    support = np.where(finite.all(axis=1), (rank_matrix >= .70).sum(axis=1), np.nan)
    known_change = (panel.announced_price.to_numpy(dtype=float)
                    / panel.current_price.to_numpy(dtype=float) - 1.)
    weekday = np.array([day.isoweekday() for day in dates], dtype=float)
    angle = 2 * np.pi * (weekday - 1.) / 7.
    week_used, month_empty = frozen_leader_state(
        dates, currencies, leader_signal)
    gap = np.asarray(days_since, dtype=float)
    gap = np.where(np.isfinite(gap), np.minimum(gap, 365.), 365.)
    matrix = np.column_stack([
        rank_matrix,
        rank_mean, rank_min, rank_max, rank_std, support,
        known_change,
        np.sin(angle), np.cos(angle), np.maximum(5. - weekday, 0.),
        np.asarray(core, dtype=float), np.asarray(fallback, dtype=float),
        np.asarray(local_precision, dtype=float) - np.asarray(overall_precision, dtype=float),
        np.asarray(trailing_rate, dtype=float), np.log1p(gap),
        week_used.astype(float), month_empty.astype(float),
        np.array([day.day / 31. for day in dates]),
    ])
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((matrix.shape, len(FEATURE_NAMES)))
    return matrix


def build_weekly_take_target(dates, currencies, opportunity, outcomes, mature20):
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    opportunity = np.asarray(opportunity, dtype=bool)
    ys = np.column_stack([np.asarray(outcomes['y' + str(h)], dtype=float)
                          for h in (3, 5, 10, 20)])
    valid = np.isfinite(ys).all(axis=1)
    utility = np.full(len(dates), np.nan)
    utility[valid] = ys[valid].mean(axis=1)
    mature20 = np.asarray(mature20, dtype=object)
    target = np.full(len(dates), np.nan)
    target_maturity = np.full(len(dates), None, dtype=object)
    future_count = np.zeros(len(dates), dtype=np.int16)
    wait_gain = np.full(len(dates), np.nan)
    groups = {}
    for i in np.flatnonzero(opportunity):
        groups.setdefault((currencies[i], dates[i].isocalendar()[:2]), []).append(i)
    for indices in groups.values():
        for position, i in enumerate(indices):
            remaining = np.asarray(indices[position:], dtype=int)
            if not valid[remaining].all():
                continue
            maturities = mature20[remaining]
            if any(value is None for value in maturities):
                continue
            later = remaining[1:]
            future_count[i] = len(later)
            if len(later):
                best_later = float(np.max(utility[later]))
                target[i] = float(utility[i] >= best_later)
                wait_gain[i] = best_later - utility[i]
            else:
                target[i] = 1.
                wait_gain[i] = 0.
            target_maturity[i] = max(maturities)
    return target, target_maturity, utility, future_count, wait_gain


def fit_quarterly_logit(features, target, target_maturity, dates, opportunity):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    target_maturity = np.asarray(target_maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    opportunity = np.asarray(opportunity, dtype=bool)
    prediction = np.full(len(dates), np.nan)
    logs = []
    origins = [dt.date(y, m, 1)
               for y in range(2022, max(day.year for day in dates) + 1)
               for m in (1, 4, 7, 10)
               if dt.date(y, m, 1) >= dt.date(2022, 7, 1)]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        mature = np.array([value is not None and value < cutoff
                           for value in target_maturity])
        train = (opportunity & np.isfinite(target) & (dates < origin) & mature)
        train_ids = np.flatnonzero(train)
        query_ids = np.flatnonzero(query)
        median = np.nanmedian(features[train], axis=0) if len(train_ids) else np.zeros(features.shape[1])
        median[~np.isfinite(median)] = 0.
        X_train = np.where(np.isfinite(features[train]), features[train], median)
        X_query = np.where(np.isfinite(features[query]), features[query], median)
        constant = len(train_ids) < 100 or np.unique(target[train]).size < 2
        if constant:
            prediction[query] = .5
            weight_sum = 0.
            positive = int(np.nansum(target[train]))
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_query = scaler.transform(X_query)
            age = np.array([(origin - dates[i]).days for i in train_ids], dtype=float)
            weight = np.exp2(-age / 730.)
            model = LogisticRegression(
                C=.1, solver='lbfgs', class_weight='balanced', max_iter=1000)
            model.fit(X_train, target[train].astype(int), sample_weight=weight)
            prediction[query] = model.predict_proba(X_query)[:, 1]
            weight_sum = float(weight.sum())
            positive = int(target[train].sum())
        logs.append({
            'origin': str(origin),
            'n_train': int(len(train_ids)),
            'n_positive': positive,
            'n_query': int(query.sum()),
            'constant_prediction': bool(constant),
            'weight_sum': weight_sum,
            'last_target_maturity': str(max(target_maturity[train])) if train.any() else '',
        })
    return prediction, logs


def optimal_stopping_router(core, fallback, take_probability, fallback_quality,
                            dates, currencies, eligible, rate_floor=1.):
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    core = np.asarray(core, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
    take_probability = np.asarray(take_probability, dtype=float)
    fallback_quality = np.asarray(fallback_quality, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    signal = np.zeros(len(core), dtype=bool)
    trailing_rate = np.full(len(core), np.nan)
    days_since = np.full(len(core), np.nan)
    reason = np.zeros(len(core), dtype=np.int8)
    core_veto = np.zeros(len(core), dtype=bool)
    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        available = indices[eligible[indices]]
        first_eligible = dates[available[0]] if len(available) else dates[indices[0]]
        week, used, month, month_used = None, 0, None, 0
        chosen_days = []
        for i in indices:
            iso = dates[i].isocalendar()[:2]
            current_month = (dates[i].year, dates[i].month)
            if iso != week:
                week, used = iso, 0
            if current_month != month:
                month, month_used = current_month, 0
            if dates[i] < first_eligible:
                continue
            start = max(first_eligible, dates[i] - dt.timedelta(days=365))
            weeks = max((dates[i] - start).days / 7., 1.)
            rate = sum(start <= chosen < dates[i]
                       for chosen in chosen_days) / weeks
            trailing_rate[i] = rate
            gap = ((dates[i] - chosen_days[-1]).days
                   if chosen_days else np.inf)
            days_since[i] = gap
            warmed = (dates[i] - first_eligible).days >= 84
            late_week = dates[i].isoweekday() >= 4 and used == 0
            silence = gap >= 10
            guard_reason = 0
            if dates[i].isoweekday() >= 5:
                guard_reason = 4
            elif rate < rate_floor:
                guard_reason = 5
            elif silence:
                guard_reason = 6
            elif dates[i].day >= 24 and month_used == 0:
                guard_reason = 7
            model_take = (not np.isfinite(take_probability[i])
                          or take_probability[i] >= .5)
            use_core = core[i] and (model_take or guard_reason > 0)
            core_veto[i] = eligible[i] and core[i] and not use_core
            precision_late_week = late_week and fallback_quality[i]
            use_fallback = (not core[i] and warmed and rate < rate_floor
                            and fallback[i]
                            and (silence or precision_late_week))
            if eligible[i] and used < 2 and (use_core or use_fallback):
                signal[i] = True
                if use_core:
                    reason[i] = 1 if model_take else guard_reason
                elif precision_late_week:
                    reason[i] = 2
                else:
                    reason[i] = 3
                chosen_days.append(dates[i])
                used += 1
                month_used += 1
    return signal, trailing_rate, days_since, reason, core_veto
