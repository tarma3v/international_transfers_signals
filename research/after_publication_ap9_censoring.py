"""Training followup reconstructed from available CBR prefixes, never full future paths."""
import datetime as dt
import hashlib

import numpy as np

from ml.targets import HORIZONS
from research.after_publication_ap1 import factory

KINDS = ('direct_full20', 'direct_mature5', 'coarse_full20', 'coarse_partial',
         'fine_full20', 'fine_partial')


def observed_followup(series, panel, origin):
    cutoff = origin - dt.timedelta(days=2)
    counts = np.zeros(len(panel), dtype=np.uint8)
    failures = np.zeros(len(panel), dtype=np.uint8)
    receipts = {c: s.dates - dt.timedelta(days=1) for c, s in series.items()}
    last = {c: np.searchsorted(days, cutoff, side='left') - 1 for c, days in receipts.items()}
    for row, event in enumerate(panel.itertuples()):
        if not dt.date(2022, 1, 1) <= event.date < cutoff:
            continue
        i = int(event.announced_index)
        n = min(20, max(0, int(last[event.currency]) - i))
        counts[row] = n
        if n:
            values = series[event.currency].values
            bad = np.flatnonzero(values[i + 1:i + n + 1] < values[i])
            failures[row] = int(bad[0] + 1) if len(bad) else 0
    return counts, failures


def records(counts, failures, endpoints, full_only=False):
    counts, failures = np.asarray(counts), np.asarray(failures)
    endpoints = np.asarray(endpoints)
    if counts.shape != failures.shape or counts.ndim != 1:
        raise ValueError('One count and observed firstfailure per event required')
    if ((counts < 0) | (counts > 20) | (failures < 0) | (failures > counts)).any():
        raise ValueError('Cannot observe a failure beyond observed followup')
    if len(endpoints) == 0 or (np.diff(endpoints) <= 0).any() or endpoints[0] < 1 or endpoints[-1] > 20:
        raise ValueError('Increasing observation endpoints1..20 required')
    start = np.r_[0, endpoints[:-1]]
    complete = counts[:, None] >= endpoints
    risk = (failures[:, None] == 0) | (failures[:, None] > start)
    usable = complete & risk
    if full_only:
        usable &= counts[:, None] == 20
    rows, intervals = np.where(usable)
    target = ((failures[rows] > 0) & (failures[rows] <= endpoints[intervals])).astype(float)
    return rows, intervals, target


def record_hash(rows, intervals, target):
    return hashlib.sha256(np.column_stack([rows, intervals, target]).astype('<i4').tobytes()).hexdigest()


def design(X, intervals, rows=None, n_intervals=5):
    if rows is None:
        rows = np.repeat(np.arange(len(X)), n_intervals)
        intervals = np.tile(np.arange(n_intervals), len(X))
    return np.column_stack([X[rows], np.eye(n_intervals)[intervals]])


def fit_one(X, counts, failures, kind, query):
    if kind not in KINDS:
        raise ValueError(kind)
    direct = kind.startswith('direct')
    endpoints = np.array([5]) if direct else np.array(HORIZONS if kind.startswith('coarse') else range(1, 21))
    if direct:
        rows = np.flatnonzero(counts >= (20 if kind.endswith('full20') else 5))
        intervals = np.zeros(len(rows), dtype=int)
        target = ((failures[rows] == 0) | (failures[rows] > 5)).astype(float)
        train_x, test_x = X[rows], X[query]
    else:
        rows, intervals, target = records(counts, failures, endpoints, kind.endswith('full20'))
        train_x = design(X, intervals, rows, len(endpoints))
        test_x = design(X[query], None, n_intervals=len(endpoints))
    unique = np.unique(rows)
    if len(unique) < 400 or len(np.unique(target)) < 2:
        means = np.array([target[intervals == j].mean() if (intervals == j).any() else 0.
                          for j in range(len(endpoints))])
        pred = np.tile(means, (len(query), 1))
    else:
        model = factory('hist7y').set_params(early_stopping=False)
        model.fit(train_x, target)
        pred = model.predict_proba(test_x)[:, 1].reshape(len(query), len(endpoints))
    curve = pred if direct else np.cumprod(1 - pred, axis=1)
    summary = {'n_events': len(unique), 'n_records': len(rows), 'n_positive_labels': int(target.sum()),
        'n_partial_events': int((counts[unique] < 20).sum()),
        'records_sha256': record_hash(rows, intervals, target)}
    return curve, rows, intervals, target, endpoints, summary
