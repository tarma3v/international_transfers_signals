"""Frozen quarterly mature-only convex gate for magnitude predictions."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


WEIGHTS = np.asarray([0., .25, .5, .75, 1.])
MIN_TRAIN_ROWS = 500
WINDOW_DAYS = 730
HALF_LIFE_DAYS = 730.


def fit_quarterly_gate(model, prior, target, maturity, dates):
    model = np.asarray(model, dtype=float)
    prior = np.asarray(prior, dtype=float)
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    prediction = np.full(len(dates), np.nan)
    selected_weight = np.full(len(dates), np.nan)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period("Q") + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        start = origin - dt.timedelta(days=WINDOW_DAYS)
        train = (
            (dates >= start) & (dates < origin)
            & np.asarray([value < cutoff for value in maturity])
            & np.isfinite(target) & np.isfinite(model) & np.isfinite(prior)
        )
        ids = np.flatnonzero(train)
        n_train[query] = len(ids)
        if len(ids) < MIN_TRAIN_ROWS:
            weight = 0.
            losses = [np.nan] * len(WEIGHTS)
        else:
            age = np.asarray([(origin - dates[i]).days for i in ids], float)
            sample_weight = np.exp2(-age / HALF_LIFE_DAYS)
            losses = []
            for weight_value in WEIGHTS:
                estimate = (prior[train]
                            + weight_value * (model[train] - prior[train]))
                losses.append(float(np.average(
                    np.abs(estimate - target[train]), weights=sample_weight)))
            weight = float(WEIGHTS[int(np.argmin(losses))])
        prediction[query] = (
            prior[query] + weight * (model[query] - prior[query]))
        selected_weight[query] = weight
        logs.append({
            "origin": str(origin), "n_train": int(len(ids)),
            "n_query": int(query.sum()), "selected_weight": weight,
            **{f"mae_weight_{value:g}": loss
               for value, loss in zip(WEIGHTS, losses)},
            "last_target_maturity": (
                str(max(maturity[train])) if len(ids) else ""),
        })
    return prediction, selected_weight, n_train, logs
