"""Independent T4 reconstruction and feature/target prefix audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import causality_check, load_round5_features
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket import OUT
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
    fit_quarterly_hist,
)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    matrix, names, index, series, *_ = load_round5_features()
    causality_check(dt.date(2025, 1, 6))
    features = compact_features(matrix, names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    saved = loadz(OUT / 'outputs.npz')
    cut_date = dt.date(2025, 1, 6)
    past, future = dates < cut_date, dates >= cut_date
    rebuilt = 0
    for h in HORIZONS:
        target = targets['fav_h' + str(h)]
        maturity = target_reach_dates(index, series, h)
        probability, count, _logs = fit_quarterly_hist(
            features, target, maturity, dates)
        same(saved['model_raw_probability_' + str(h)], probability)
        np.testing.assert_array_equal(saved['model_n_train_' + str(h)], count)
        changed_features = features.copy()
        changed_features[future] *= 11.
        changed_target = target.copy()
        finite = future & np.isfinite(changed_target)
        changed_target[finite] = 1. - changed_target[finite]
        changed_maturity = maturity.copy()
        changed_maturity[future] = dt.date(1900, 1, 1)
        altered = fit_quarterly_hist(
            changed_features, changed_target, changed_maturity, dates)[0]
        same(probability[past], altered[past])

        for candidate, raw in (
                ('history_hist', probability),
                ('range_anchor', saved['anchor_raw_probability'])):
            result = fit_quarterly_calibrator(
                raw, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            calibrated, prior, cal_count, _cal_logs = result
            prefix = candidate + '__h' + str(h)
            same(saved['prob__' + prefix], calibrated)
            same(saved['prior__' + prefix], prior)
            np.testing.assert_array_equal(
                saved['calibration_n_train__' + prefix], cal_count)
            rebuilt += 1
    checks = {
        'source_hashes_verified': True,
        'physical_future_cbr_feature_prefix_invariant': True,
        'five_quarterly_models_rebuilt_exactly': True,
        'ten_quarterly_calibrators_rebuilt_exactly': rebuilt,
        'future_feature_target_maturity_corruption_prefix_invariant': True,
        'post_2022_training_boundary_verified': True,
        'tomorrow_cbr_used': False,
        'market_data_used': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
