"""Independent rebuild and causality audit for T9."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.targets import build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
    fit_quarterly_hist,
)
from research.temperature_t9_early_market import OUT, run_experiment
from research.temperature_t9_early_market_models import (
    CLOCKS,
    build_early_market_features,
    physical_causality_check,
)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    persisted = np.load(OUT / "outputs.npz", allow_pickle=True)
    rebuilt = run_experiment()["arrays"]
    assert set(persisted.files) == set(rebuilt)
    for key, value in rebuilt.items():
        np.testing.assert_array_equal(persisted[key], value)

    matrix, names, index, series, *_ = load_round5_features()
    base = compact_features(matrix, names)
    _broad, _broad_names, references = load_broad_features(index, series)
    history, _digest = load_spot_1530_history()
    for clock in CLOCKS:
        assert physical_causality_check(index, history, references, clock)

    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    target = build_targets(series, index)["fav_h5"]
    maturity = target_reach_dates(index, series, 5)
    state = build_early_market_features(
        index, history, references, CLOCKS[3])[0]
    features = np.column_stack((base, state))
    raw, _, _ = fit_quarterly_hist(features, target, maturity, dates)
    calibrated = fit_quarterly_calibrator(
        raw, target, maturity, dates, currencies,
        min_train_date=MIN_TRAIN_DATE)[0]
    boundary = dt.date(2025, 1, 1)
    changed_target = target.copy()
    future = dates >= boundary
    finite = future & np.isfinite(changed_target)
    changed_target[finite] = 1. - changed_target[finite]
    changed_maturity = maturity.copy()
    changed_maturity[future] = dt.date(2099, 1, 1)
    changed_raw, _, _ = fit_quarterly_hist(
        features, changed_target, changed_maturity, dates)
    changed_calibrated = fit_quarterly_calibrator(
        changed_raw, changed_target, changed_maturity, dates, currencies,
        min_train_date=MIN_TRAIN_DATE)[0]
    past = dates < boundary
    np.testing.assert_array_equal(raw[past], changed_raw[past])
    np.testing.assert_array_equal(
        calibrated[past], changed_calibrated[past])
    checks = {
        "source_hashes_verified": True,
        "all_persisted_arrays_exactly_rebuilt": True,
        "physical_future_candle_corruption_all_clocks": True,
        "future_target_and_maturity_prefix_invariant": True,
        "tomorrow_cbr_used": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
