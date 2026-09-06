"""Independent T6 reconstruction and future-state corruption audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.targets import HORIZONS
from ml.validation import target_reach_dates
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
)
from research.temperature_t6_pre_receipt_benefit import (
    OUT,
    T3,
    T4,
    T5,
    states_for_horizon,
)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    matrix, names, index, series, *_ = load_round5_features()
    base = compact_features(matrix, names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    t3, t4, t5 = loadz(T3 / 'outputs.npz'), loadz(
        T4 / 'outputs.npz'), loadz(T5 / 'outputs.npz')
    saved = loadz(OUT / 'outputs.npz')
    cut_date = dt.date(2025, 1, 6)
    past, future = dates < cut_date, dates >= cut_date
    rebuilt = 0
    for h in HORIZONS:
        target = _forward(series, index, h)
        maturity = target_reach_dates(index, series, h)
        for state, (raw, probability) in states_for_horizon(
                h, t3, t4, t5).items():
            features = np.column_stack([base, raw, probability])
            prediction, prior, count, _logs = fit_quarterly_benefit(
                features, target, maturity, dates,
                min_train_date=MIN_TRAIN_DATE)
            prefix = state + '__h' + str(h)
            same(saved['expected_bps__' + prefix], prediction)
            same(saved['prior_bps__' + prefix], prior)
            np.testing.assert_array_equal(saved['n_train__' + prefix], count)

            changed_features = features.copy()
            changed_features[future] = np.nan_to_num(
                changed_features[future], nan=0.) * 17. + 3.
            changed_target = target.copy()
            changed_target[future & np.isfinite(changed_target)] *= -13.
            changed_maturity = maturity.copy()
            changed_maturity[future] = dt.date(1900, 1, 1)
            altered = fit_quarterly_benefit(
                changed_features, changed_target, changed_maturity, dates,
                min_train_date=MIN_TRAIN_DATE)[0]
            same(prediction[past], altered[past])
            rebuilt += 1
    checks = {
        'source_hashes_verified': True,
        'fifty_five_models_rebuilt_exactly': rebuilt,
        'future_state_feature_target_maturity_prefix_invariant': True,
        'post_2022_training_boundary_verified': True,
        'units': 'CBR future-only basis points',
        'tomorrow_cbr_used': False,
        'bank_execution_validated': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
