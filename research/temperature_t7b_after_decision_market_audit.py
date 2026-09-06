"""Independent T7B reconstruction and future-prefix audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE
from research.temperature_t7_after_receipt_market_models import (
    fit_quarterly_delta_calibrator,
)
from research.temperature_t7b_after_decision_market import (
    BASE,
    CLOCKS,
    HORIZONS,
    OUT,
    clock_name,
    delayed_market_delta,
    physical_causality_check,
)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    arrays, saved = loadz(BASE / 'outputs.npz'), loadz(OUT / 'outputs.npz')
    series = load(DATA)
    cap = build_outcomes(series, panel, 'publication')
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    history, _digest = load_spot_1530_history()
    cut_date = dt.date(2025, 1, 6)
    past, future = dates < cut_date, dates >= cut_date
    probability_models = benefit_models = 0
    for clock in CLOCKS:
        physical_causality_check(panel, history, clock, cut_date)
        name = clock_name(clock)
        delta, available = delayed_market_delta(panel, history, clock)
        same(saved['delta_bps_' + name], delta)
        np.testing.assert_array_equal(saved['available_' + name], available)
        for h in HORIZONS:
            target, maturity = arrays['y' + str(h)], cap['mature' + str(h)]
            probability, count, _logs = fit_quarterly_delta_calibrator(
                arrays['head_probability_' + str(h)], delta, available,
                target, maturity, dates, currencies)
            prefix = name + '__h' + str(h)
            same(saved['probability__' + prefix], probability)
            np.testing.assert_array_equal(
                saved['probability_n_train__' + prefix], count)
            changed_delta = delta.copy()
            changed_available = available.copy()
            changed_target = target.copy()
            changed_maturity = maturity.copy()
            changed_delta[future] = changed_delta[future] * -19. + 7.
            changed_available[future] = ~changed_available[future]
            changed_target[future & np.isfinite(changed_target)] = 1. - (
                changed_target[future & np.isfinite(changed_target)])
            changed_maturity[future] = dt.date(1900, 1, 1)
            altered = fit_quarterly_delta_calibrator(
                arrays['head_probability_' + str(h)], changed_delta,
                changed_available, changed_target, changed_maturity, dates,
                currencies)[0]
            same(probability[past], altered[past])
            probability_models += 1
            features = np.column_stack([
                arrays['multihorizon_features'],
                arrays['calibrated_probability_' + str(h)],
                delta / 100., np.abs(delta) / 100., available.astype(float),
            ])
            prediction, prior, benefit_count, _logs = fit_quarterly_benefit(
                features, arrays['forward' + str(h)], maturity, dates,
                min_train_date=MIN_TRAIN_DATE)
            same(saved['expected_bps__' + prefix], prediction)
            same(saved['benefit_prior__' + prefix], prior)
            np.testing.assert_array_equal(
                saved['benefit_n_train__' + prefix], benefit_count)
            changed_features = features.copy()
            changed_features[future] = np.nan_to_num(
                changed_features[future], nan=0.) * 23. - 2.
            changed_benefit = arrays['forward' + str(h)].copy()
            changed_benefit[future & np.isfinite(changed_benefit)] *= -11.
            altered_benefit = fit_quarterly_benefit(
                changed_features, changed_benefit, changed_maturity, dates,
                min_train_date=MIN_TRAIN_DATE)[0]
            same(prediction[past], altered_benefit[past])
            benefit_models += 1
    checks = {
        'source_hashes_verified': True,
        'two_physical_future_candle_checks_passed': True,
        'probability_models_rebuilt_exactly': probability_models,
        'benefit_models_rebuilt_exactly': benefit_models,
        'new_information_starts_after_1830_delayed_boundary': True,
        'future_market_target_maturity_prefix_invariant': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
