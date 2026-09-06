"""Independently rebuild partial feedback from each step's receipt and audit AP9."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_ap9 import OUT, BASE, INCUMBENT, AP4_CONTROL, policies
from research.after_publication_ap9_censoring import KINDS, record_hash
from research.after_publication_panel import build_outcomes


def step_tables(series, panel):
    """Audit-only full labels with per-step receipts, not training features."""
    n = len(panel)
    survival = np.full((n, 20), np.nan)
    receipts = np.full((n, 20), dt.date.max, dtype=object)
    for row, event in enumerate(panel.itertuples()):
        s, i = series[event.currency], int(event.announced_index)
        count = min(20, len(s.values) - i - 1)
        if count:
            survival[row, :count] = np.minimum.accumulate(s.values[i + 1:i + 1 + count] >= s.values[i])
            receipts[row, :count] = s.dates[i + 1:i + 1 + count] - dt.timedelta(days=1)
    return survival, receipts


def independent_records(panel, survival, receipts, origin, kind):
    cutoff = origin - dt.timedelta(days=2)
    eligible = ((panel.date.to_numpy() >= dt.date(2022, 1, 1)) & (panel.date.to_numpy() < cutoff))
    endpoints = np.array([5]) if kind.startswith('direct') else np.array(HORIZONS if kind.startswith('coarse') else range(1, 21))
    allowed = (receipts[:, endpoints - 1] < cutoff) & eligible[:, None]
    if kind.endswith('full20'):
        allowed &= receipts[:, 19:20] < cutoff
    if kind.startswith('direct'):
        rows = np.flatnonzero(allowed[:, 0])
        intervals = np.zeros(len(rows), dtype=int)
        target = survival[rows, 4]
    else:
        previous = np.column_stack([np.ones(len(panel)), survival[:, endpoints[:-1] - 1]])
        rows, intervals = np.where(allowed & (previous == 1))
        target = 1 - survival[rows, endpoints[intervals] - 1]
    return rows, intervals, target, endpoints


def main():
    for path, digest in json.loads((OUT / 'metadata.json').read_text())['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'publication')
    survival, receipts = step_tables(series, panel)
    training = pd.read_csv(OUT / 'training_log.csv')
    with np.load(OUT / 'outputs.npz') as packet, np.load(BASE / 'outputs.npz') as base:
        saved, previous = ({k: obj[k] for k in obj.files} for obj in (packet, base))
    for key in ('dates', 'currencies', 'early', 'later', 'groups'):
        np.testing.assert_array_equal(saved[key], previous[key])
    np.testing.assert_array_equal(saved['features'], previous['features__1830'])
    np.testing.assert_array_equal(saved['hazard_features'], previous['hazard_features__1830'])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])
        np.testing.assert_array_equal(survival[:, h - 1], saved[f'y{h}'])
    counts_by_origin, exposure_rows = {}, []
    for origin_text in sorted(training.origin.unique()):
        origin = dt.date.fromisoformat(origin_text)
        cutoff = origin - dt.timedelta(days=2)
        eligible = (panel.date.to_numpy() >= dt.date(2022, 1, 1)) & (panel.date.to_numpy() < cutoff)
        available = (receipts < cutoff) & eligible[:, None]
        counts = available.sum(axis=1)
        bad = available & (survival == 0)
        failure = np.where(bad.any(axis=1), np.argmax(bad, axis=1) + 1, 0)
        np.testing.assert_array_equal(saved['observed_count__' + origin_text], counts)
        np.testing.assert_array_equal(saved['observed_failure__' + origin_text], failure)
        counts_by_origin[origin_text] = counts
        for row in training[training.origin == origin_text].itertuples():
            rows, intervals, target, endpoints = independent_records(panel, survival, receipts, origin, row.kind)
            ids = np.unique(rows)
            assert (len(ids), len(rows), int(target.sum())) == (row.n_events, row.n_records, row.n_positive_labels)
            assert row.records_sha256 == record_hash(rows, intervals, target)
            assert row.n_partial_events == int((counts[ids] < 20).sum())
            assert row.n_full20_available == int((counts == 20).sum())
            assert row.n_observed_events == int((counts > 0).sum())
            assert row.n_observed_failures == int((failure > 0).sum())
            label_receipts = receipts[rows, endpoints[intervals] - 1]
            assert (label_receipts < cutoff).all()
            assert str(max(label_receipts)) == row.last_label_receipt
            assert str(max(panel.date.to_numpy()[rows])) == row.last_event_date
            for j, endpoint in enumerate(endpoints):
                mask = intervals == j
                exposure_rows.append({'origin': origin_text, 'cadence': row.cadence, 'kind': row.kind,
                    'endpoint': int(endpoint), 'n_risk': int(mask.sum()),
                    'positive_labels': int(target[mask].sum()),
                    'n_recent_records': int((counts[rows[mask]] < 20).sum())})
        print(f'AP9 audit {origin_text}: complete and censored training records verified', flush=True)
    forecasts = {key.removeprefix('curve__'): value for key, value in saved.items() if key.startswith('curve__')}
    raw, signals = policies(panel, forecasts, previous)
    for key, value in raw.items():
        np.testing.assert_array_equal(value, saved['score__' + key])
    for key, value in signals.items():
        np.testing.assert_array_equal(value, saved['signal__' + key])
    np.testing.assert_array_equal(forecasts['quarter_direct_full20'][:, 0], previous['hist__1830'])
    np.testing.assert_array_equal(forecasts['quarter_coarse_full20'], previous['survival__1830'])
    accuracy = []
    for name, curve in forecasts.items():
        finite = np.isfinite(curve).all(axis=1)
        assert ((curve[finite] >= 0) & (curve[finite] <= 1)).all()
        assert (np.diff(curve[finite], axis=1) <= 1e-12).all()
        horizons = [5] if curve.shape[1] == 1 else HORIZONS
        pred = curve if curve.shape[1] != 20 else curve[:, np.array(HORIZONS) - 1]
        for scope in ('early', 'later'):
            for j, h in enumerate(horizons):
                mask = saved[scope] & np.isfinite(saved[f'y{h}'])
                accuracy.append({'model': name, 'period': scope, 'h': h, 'n': int(mask.sum()),
                    'brier': float(np.mean((pred[mask, j] - saved[f'y{h}'][mask]) ** 2)),
                    'mean_probability': float(pred[mask, j].mean()), 'event_rate': float(saved[f'y{h}'][mask].mean())})
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)
    pd.DataFrame(exposure_rows).to_csv(OUT / 'training_exposure.csv', index=False)
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    pd.concat([selected_bootstrap(panel, saved, selected, b) for b in (20, 50)]).to_csv(
        OUT / 'selected_block_sensitivity.csv', index=False)
    detail = []
    years = np.array([d.year for d in panel.date])
    for key in dict.fromkeys([selected, INCUMBENT, AP4_CONTROL]):
        for year in (2024, 2025, 2026):
            for c in sorted(panel.currency.unique()):
                scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == c)
                detail.extend({'candidate': key, 'year': year, 'currency': c, **row} for row in
                    scorecard(panel, outcomes, signals[key], scope, saved['groups']))
    pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
    pairs = [(AP4_CONTROL, list(signals))]
    def variants(prefix, kind):
        names = [prefix + kind + '_h5', 'cny50_' + prefix + kind]
        if not kind.startswith('direct'):
            names.append(prefix + kind + '_mean')
        return [name + '_urgent_cap2' for name in names]
    for cadence in ('quarter_', 'month_'):
        for full, fresh in (('direct_full20', 'direct_mature5'), ('coarse_full20', 'coarse_partial'), ('fine_full20', 'fine_partial')):
            pairs.extend((left, [right]) for left, right in zip(variants(cadence, full), variants(cadence, fresh)))
    for kind in KINDS:
        pairs.extend((left, [right]) for left, right in zip(variants('quarter_', kind), variants('month_', kind)))
    for cadence in ('quarter_', 'month_'):
        for full, fine in (('coarse_full20', 'fine_full20'), ('coarse_partial', 'fine_partial')):
            pairs.extend((left, [right]) for left, right in zip(variants(cadence, full), variants(cadence, fine)))
    common_audit(OUT, list(signals), pairs)
    (OUT / 'audit_checks.json').write_text(json.dumps({'source_hashes_verified': True,
        'fit_records_independently_rebuilt': len(training), 'observed_prefix_snapshots_rebuilt': len(counts_by_origin),
        'record_hashes_all_match': True, 'same_features_outcomes_support': True,
        'all_label_receipts_before_origin_embargo': True, 'all_controls_exact': True,
        'all_scores_and_signals_rebuilt': True, 'all_survival_curves_valid': True,
        'all_censoring_counts_and_failures_rebuilt': True, 'historical_receipts_certified': False}, indent=2))


if __name__ == '__main__':
    main()
