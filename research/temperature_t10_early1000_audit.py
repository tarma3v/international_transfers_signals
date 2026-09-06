"""Exact rebuild and prefix audit for T10."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.validation import target_reach_dates
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE, compact_features
from research.temperature_t10_early1000 import (
    CLOCK,
    OUT,
    SELECTED,
    T4,
    T6,
    T9,
    run_experiment,
    selected_keys,
)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    persisted = loadz(OUT / "outputs.npz")
    rebuilt = run_experiment()["arrays"]
    assert set(persisted) == set(rebuilt)
    for key, value in rebuilt.items():
        np.testing.assert_array_equal(persisted[key], value)

    t4, t6, t9 = (loadz(path / "outputs.npz") for path in (T4, T6, T9))
    available = persisted["available"].astype(bool)
    for h, model in SELECTED.items():
        _raw_key, probability_key = selected_keys(h)
        np.testing.assert_array_equal(
            persisted[f"routed_probability_h{h}"][available],
            t9[probability_key][available])
        np.testing.assert_array_equal(
            persisted[f"routed_probability_h{h}"][~available],
            t4[f"prob__history_hist__h{h}"][~available])
        np.testing.assert_array_equal(
            persisted[f"routed_expected_bps_h{h}"][~available],
            t6[f"expected_bps__premarket__h{h}"][~available])

    matrix, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    target = _forward(series, index, 5)
    maturity = target_reach_dates(index, series, 5)
    raw_key, probability_key = selected_keys(5)
    features = np.column_stack((
        compact_features(matrix, names), t9[f"market__{CLOCK}"],
        t9[raw_key], t9[probability_key]))
    original = fit_quarterly_benefit(
        features, target, maturity, dates,
        min_train_date=MIN_TRAIN_DATE)[0]
    boundary = dt.date(2025, 1, 1)
    future = dates >= boundary
    changed_target = target.copy()
    changed_target[future & np.isfinite(changed_target)] += 7777.
    changed_maturity = maturity.copy()
    changed_maturity[future] = dt.date(2099, 1, 1)
    changed = fit_quarterly_benefit(
        features, changed_target, changed_maturity, dates,
        min_train_date=MIN_TRAIN_DATE)[0]
    np.testing.assert_array_equal(original[dates < boundary],
                                  changed[dates < boundary])
    checks = {
        "source_hashes_verified": True,
        "all_persisted_arrays_exactly_rebuilt": True,
        "selected_probability_identity_on_available_rows": True,
        "exact_t4_t6_fallback_on_unavailable_rows": True,
        "future_benefit_target_maturity_prefix_invariant": True,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
