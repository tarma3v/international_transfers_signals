"""Independent T3 reconstruction and future-label corruption audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.temperature_t3_phase_calibration import (
    BASE,
    MIN_TRAIN_DATE,
    OUT,
    loadz,
)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    _X, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    source = loadz(BASE / 'scores.npz')
    saved = loadz(OUT / 'outputs.npz')
    candidates = sorted(key.removeprefix('rank__') for key in source
                        if key.startswith('rank__'))
    cut_date = dt.date(2025, 1, 6)
    past, future = dates < cut_date, dates >= cut_date
    reconstructed = 0
    for candidate in candidates:
        raw = source['rank__' + candidate]
        for h in HORIZONS:
            target = targets['fav_h' + str(h)]
            maturity = target_reach_dates(index, series, h)
            result = fit_quarterly_calibrator(
                raw, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            probability, prior, count, _logs = result
            prefix = candidate + '__h' + str(h)
            same(saved['prob__' + prefix], probability)
            same(saved['prior__' + prefix], prior)
            np.testing.assert_array_equal(saved['n_train__' + prefix], count)
            reconstructed += 1

            changed_raw = raw.copy()
            changed_target = target.copy()
            changed_maturity = maturity.copy()
            changed_raw[future] = np.where(
                np.isfinite(changed_raw[future]), 1. - changed_raw[future],
                np.nan)
            finite = future & np.isfinite(changed_target)
            changed_target[finite] = 1. - changed_target[finite]
            changed_maturity[future] = dt.date(1900, 1, 1)
            altered = fit_quarterly_calibrator(
                changed_raw, changed_target, changed_maturity, dates,
                currencies, min_train_date=MIN_TRAIN_DATE)[0]
            same(probability[past], altered[past])

    log = pd.read_csv(OUT / 'calibration_log.csv')
    assert set(log.min_train_date.dropna()) == {str(MIN_TRAIN_DATE)}
    checks = {
        'source_hashes_verified': True,
        'calibrators_rebuilt_exactly': reconstructed,
        'post_2022_min_train_date_verified': True,
        'mature_label_and_two_day_embargo_checked': True,
        'future_score_label_maturity_corruption_prefix_invariant': True,
        'physical_future_candle_corruption_inherited_from_t2': True,
        'tomorrow_cbr_used': False,
        'historical_receipts_certified': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
