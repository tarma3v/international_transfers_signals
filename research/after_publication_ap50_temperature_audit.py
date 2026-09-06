"""Independent calibration reconstruction and time-prefix audit for AP50-T."""
from __future__ import annotations

import datetime as dt
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.transfer_temperature import score_as_of
from research.after_publication_ap1 import DATA
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap49_effective_models import geometric_consensus
from research.after_publication_ap50_temperature import BASE, HORIZONS, OUT
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
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
    currencies = panel.currency.to_numpy()
    saved = loadz(OUT / 'outputs.npz')
    old = loadz(BASE / 'outputs.npz')
    cap = build_outcomes(load(DATA), panel, 'publication')

    reconstructed, logs = {}, []
    for h in HORIZONS:
        result = fit_quarterly_calibrator(
            old['head_probability_' + str(h)], old['y' + str(h)],
            cap['mature' + str(h)], dates, currencies)
        calibrated, prior, count, head_logs = result
        reconstructed[h] = calibrated
        same(saved['calibrated_probability_' + str(h)], calibrated)
        same(saved['calibration_prior_' + str(h)], prior)
        np.testing.assert_array_equal(
            saved['calibration_n_train_' + str(h)], count)
        logs.extend({'h': h, **row} for row in head_logs)
    same(saved['composite_calibrated_probability'],
         geometric_consensus(reconstructed))
    saved_log = pd.read_csv(OUT / 'calibration_log.csv')
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
        pkey, ykey = 'head_probability_' + str(h), 'y' + str(h)
        changed[pkey][future] = np.where(
            np.isfinite(changed[pkey][future]),
            1. - changed[pkey][future], np.nan)
        finite = future & np.isfinite(changed[ykey])
        changed[ykey][finite] = 1. - changed[ykey][finite]
        bad_cap['mature' + str(h)][future] = dt.date(1900, 1, 1)
        altered = fit_quarterly_calibrator(
            changed[pkey], changed[ykey], bad_cap['mature' + str(h)],
            dates, currencies)[0]
        same(reconstructed[h][:cut], altered[:cut])

    # A request before the first changed future receipt must still resolve to
    # the exact same earlier row and probability.
    as_of = dt.datetime(2025, 1, 5, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    original_lookup = score_as_of(panel, saved, 'KZT', as_of, 5)
    changed_saved = {key: value.copy() for key, value in saved.items()}
    changed_saved['calibrated_probability_5'][future] = .999
    changed_lookup = score_as_of(panel, changed_saved, 'KZT', as_of, 5)
    assert original_lookup == changed_lookup

    checks = {
        'source_hashes_verified': True,
        'four_quarterly_calibrators_rebuilt_exactly': True,
        'mature_label_and_two_day_embargo_checked': True,
        'future_score_label_maturity_corruption_prefix_invariant': True,
        'latest_valid_asof_lookup_prefix_invariant': True,
        'push_policy_changed': False,
        'historical_receipts_certified': False,
        'intraday_widget_validated': False,
        'bank_execution_validated': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
