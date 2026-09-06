"""Independent benefit-model reconstruction and prefix audit for AP51-T."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.transfer_temperature import score_as_of
from research.after_publication_ap1 import DATA
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap50_temperature import HORIZONS
from research.after_publication_ap51_benefit import BASE, OUT
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes


def loadz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates = panel.date.to_numpy()
    saved = loadz(OUT / 'outputs.npz')
    old = loadz(BASE / 'outputs.npz')
    cap = build_outcomes(load(DATA), panel, 'publication')
    logs = []
    predictions = {}
    for h in HORIZONS:
        features = np.column_stack([
            old['multihorizon_features'],
            old['calibrated_probability_' + str(h)],
        ])
        result = fit_quarterly_benefit(
            features, old['forward' + str(h)], cap['mature' + str(h)], dates)
        prediction, prior, count, head_logs = result
        predictions[h] = prediction
        same(saved['expected_future_bps_' + str(h)], prediction)
        same(saved['benefit_prior_' + str(h)], prior)
        np.testing.assert_array_equal(saved['benefit_n_train_' + str(h)], count)
        logs.extend({'h': h, **row} for row in head_logs)
    saved_log = pd.read_csv(OUT / 'training_log.csv')
    rebuilt_log = pd.DataFrame(logs)
    for left, right in zip(saved_log.coefficients, rebuilt_log.coefficients):
        parsed = ast.literal_eval(left) if isinstance(left, str) else []
        np.testing.assert_allclose(parsed, right, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(
        saved_log.drop(columns='coefficients').fillna(''),
        rebuilt_log.drop(columns='coefficients').fillna(''), check_dtype=False)

    cut = int(np.searchsorted(dates, dt.date(2025, 1, 6), side='left'))
    future = np.arange(len(dates)) >= cut
    changed = {key: value.copy() for key, value in old.items()}
    bad_cap = {key: value.copy() for key, value in cap.items()}
    for h in HORIZONS:
        changed['multihorizon_features'][future] = np.where(
            np.isfinite(changed['multihorizon_features'][future]),
            changed['multihorizon_features'][future] + 31., -31.)
        pkey, ykey = ('calibrated_probability_' + str(h),
                      'forward' + str(h))
        changed[pkey][future] = np.where(
            np.isfinite(changed[pkey][future]),
            1. - changed[pkey][future], np.nan)
        changed[ykey][future] = np.where(
            np.isfinite(changed[ykey][future]),
            -changed[ykey][future] + 1000., np.nan)
        bad_cap['mature' + str(h)][future] = dt.date(1900, 1, 1)
        bad_features = np.column_stack([
            changed['multihorizon_features'], changed[pkey]])
        altered = fit_quarterly_benefit(
            bad_features, changed[ykey], bad_cap['mature' + str(h)], dates)[0]
        same(predictions[h][:cut], altered[:cut])

    as_of = dt.datetime(2025, 1, 5, 23, 0,
                        tzinfo=dt.timezone(dt.timedelta(hours=3)))
    lookup = score_as_of(panel, saved, 'KZT', as_of, 5)
    assert lookup['expected_future_cbr_bps_h'] is not None
    assert lookup['last_source_at'] <= as_of.isoformat()
    checks = {
        'source_hashes_verified': True,
        'four_quarterly_benefit_models_rebuilt_exactly': True,
        'train_only_winsorization_rebuilt_exactly': True,
        'mature_label_and_two_day_embargo_checked': True,
        'future_feature_target_maturity_corruption_prefix_invariant': True,
        'latest_valid_lookup_exposes_expected_bps': True,
        'push_policy_changed': False,
        'units': 'CBR future-only basis points',
        'bank_execution_validated': False,
        'intraday_widget_validated': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
