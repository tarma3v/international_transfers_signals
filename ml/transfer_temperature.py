"""Latest-valid after-publication transfer-temperature lookup."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MOSCOW = ZoneInfo('Europe/Moscow')
HORIZONS = (1, 3, 5, 10, 20)


def _freshness(age_minutes):
    if age_minutes <= 12 * 60:
        return 'fresh'
    if age_minutes <= 36 * 60:
        return 'aging'
    return 'stale'


def _phase(as_of, source_at):
    local = as_of.astimezone(MOSCOW)
    if local.date() == source_at.astimezone(MOSCOW).date():
        return 'after_new_cbr'
    if local.weekday() >= 5:
        return 'weekend_or_holiday'
    return 'before_new_cbr'


def score_as_of(panel, arrays, currency, as_of, horizon=5,
                push_candidate=None):
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError('as_of must be timezone-aware')
    if horizon not in (3, 5, 10, 20):
        raise ValueError('horizon must be one of 3, 5, 10, 20 publications')
    frame = panel.copy()
    received = pd.to_datetime(frame.decision_at, utc=True)
    as_utc = pd.Timestamp(as_of).tz_convert('UTC')
    candidates = np.flatnonzero(
        frame.currency.eq(currency).to_numpy()
        & (received <= as_utc).to_numpy()
        & np.isfinite(arrays['calibrated_probability_' + str(horizon)]))
    if not len(candidates):
        return None
    i = int(candidates[np.argmax(received.iloc[candidates].astype('int64'))])
    source_at = received.iloc[i].to_pydatetime().astimezone(MOSCOW)
    age = max((as_of.astimezone(MOSCOW) - source_at).total_seconds() / 60., 0.)
    probability = float(arrays['calibrated_probability_' + str(horizon)][i])
    push = (bool(arrays['signal__' + push_candidate][i])
            if push_candidate is not None else False)
    benefit_key = 'expected_future_bps_' + str(horizon)
    expected_bps = (float(arrays[benefit_key][i])
                    if benefit_key in arrays and np.isfinite(arrays[benefit_key][i])
                    else None)
    return {
        'currency': currency,
        'horizon_publications': horizon,
        'temperature_0_100': 100. * probability,
        'probability_now_best_h': probability,
        'expected_future_cbr_bps_h': expected_bps,
        'raw_probability': float(arrays['head_probability_' + str(horizon)][i]),
        'composite_temperature_0_100': 100. * float(
            arrays['composite_calibrated_probability'][i]),
        'score_as_of': as_of.isoformat(),
        'last_source_at': source_at.isoformat(),
        'age_minutes': age,
        'freshness': _freshness(age),
        'phase': _phase(as_of, source_at),
        'confidence': ('mature_history' if arrays[
            'calibration_n_train_' + str(horizon)][i] >= 500 else 'limited'),
        'availability_evidence': str(frame.iloc[i].availability_evidence),
        'push_now': push,
    }


def load_temperature_artifact(folder):
    folder = Path(folder)
    panel = pd.read_csv(folder / 'announcement_panel.csv')
    with np.load(folder / 'outputs.npz') as source:
        arrays = {key: source[key] for key in source.files}
    return panel, arrays


def _snapshot_freshness(age_minutes, source_kind):
    if source_kind in {'moex_perpetual_prefix', 'moex_early_prefix', 'moex_prefix',
                       'post_window_market'}:
        fresh, aging = 90., 4 * 60.
    elif source_kind == 'cbr_history':
        fresh, aging = 36 * 60., 72 * 60.
    else:
        fresh, aging = 12 * 60., 36 * 60.
    if age_minutes <= fresh:
        return 'fresh'
    if age_minutes <= aging:
        return 'aging'
    return 'stale'


def _temperature_label(temperature, freshness, confidence):
    if freshness == 'stale' or confidence == 'limited':
        qualifier = 'оценка с ограниченной уверенностью'
    else:
        qualifier = None
    if temperature >= 70:
        label = 'момент выглядит выгоднее обычного'
    elif temperature >= 58:
        label = 'скорее выгодный момент'
    elif temperature >= 38:
        label = 'нейтральный момент'
    else:
        label = 'вероятно, лучше подождать'
    return f'{label}; {qualifier}' if qualifier else label


def _horizon_provenance(row, field, horizon, default):
    key = f'{field}_h{horizon}'
    if key in row.index and pd.notna(row[key]) and str(row[key]) != '':
        return row[key]
    return default


def score_snapshot_as_of(snapshots, currency, as_of, horizon=5):
    """Select the latest admissible phase snapshot for a product widget.

    `valid_from`/`valid_until` prevent yesterday's after-receipt reference from
    silently becoming today's pre-receipt forecast. All timestamps must include
    a timezone and the returned source can never be later than `as_of`.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError('as_of must be timezone-aware')
    if horizon not in HORIZONS:
        raise ValueError('horizon must be one of 1, 3, 5, 10, 20 publications')
    frame = snapshots.copy()
    if frame.empty:
        return None
    as_utc = pd.Timestamp(as_of).tz_convert('UTC')
    valid_from = pd.to_datetime(frame.valid_from, utc=True)
    if 'valid_until' in frame:
        valid_until = pd.to_datetime(frame.valid_until, utc=True, errors='coerce')
        within = valid_until.isna() | (as_utc < valid_until)
    else:
        within = np.ones(len(frame), dtype=bool)
    probability_key = 'probability_h' + str(horizon)
    if probability_key not in frame:
        return None
    probability_values = pd.to_numeric(frame[probability_key], errors='coerce')
    candidates = np.flatnonzero(
        frame.currency.eq(currency).to_numpy()
        & (valid_from <= as_utc).to_numpy()
        & np.asarray(within, dtype=bool)
        & np.isfinite(probability_values.to_numpy()))
    if not len(candidates):
        return None
    i = int(candidates[np.argmax(valid_from.iloc[candidates].astype('int64'))])
    row = frame.iloc[i]
    source_at = pd.to_datetime(_horizon_provenance(
        row, 'source_at', horizon, row.source_at), utc=True)
    if source_at > as_utc:
        raise AssertionError('future source selected')
    age = max((as_utc - source_at).total_seconds() / 60., 0.)
    source_kind = str(_horizon_provenance(
        row, 'source_kind', horizon, row.source_kind))
    freshness = _snapshot_freshness(age, source_kind)
    confidence = str(_horizon_provenance(
        row, 'confidence', horizon, row.get('confidence', 'limited')))
    probability = float(probability_values.iloc[i])
    expected_key = 'expected_future_bps_h' + str(horizon)
    expected = None
    if expected_key in frame and np.isfinite(pd.to_numeric(
            pd.Series([frame.iloc[i][expected_key]]), errors='coerce').iloc[0]):
        expected = float(frame.iloc[i][expected_key])
    temperature = 100. * probability
    phase = str(_horizon_provenance(row, 'phase', horizon, row.phase))
    availability_evidence = str(_horizon_provenance(
        row, 'availability_evidence', horizon,
        row.get('availability_evidence', 'unknown')))
    return {
        'currency': currency,
        'horizon_publications': horizon,
        'temperature_0_100': temperature,
        'probability_now_best_h': probability,
        'expected_future_cbr_bps_h': expected,
        'label': _temperature_label(temperature, freshness, confidence),
        'score_as_of': as_of.isoformat(),
        'last_source_at': source_at.tz_convert(MOSCOW).isoformat(),
        'age_minutes': age,
        'freshness': freshness,
        'phase': phase,
        'confidence': confidence,
        'source_kind': source_kind,
        'availability_evidence': availability_evidence,
        'push_now': bool(row.get('push_now', False)),
    }
