"""Independent exact-rebuild and prefix audit for T13."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_moex_perpetual_hourly_features import load_hourly_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
)
from research.temperature_t13_early_perpetual import (
    CANDIDATES,
    CLOCKS,
    OUT,
    control_probability,
    paired_bootstrap,
    run_experiment,
    select_candidates,
)
from research.temperature_t13_early_perpetual_models import (
    build_perpetual_prefix_features,
    clock_name,
    fit_quarterly_available_hgb,
    physical_causality_check,
)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    persisted = loadz(OUT / "outputs.npz")
    rebuilt_result = run_experiment()
    rebuilt = rebuilt_result["arrays"]
    assert set(persisted) == set(rebuilt)
    for key, value in rebuilt.items():
        np.testing.assert_array_equal(persisted[key], value)

    matrix, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    _broad, _broad_names, references = load_broad_features(index, series)
    history, _digest = load_hourly_history()
    t4 = loadz(Path("results/research/temperature/t4_premarket/outputs.npz"))
    t10 = loadz(Path("results/research/temperature/t10_early1000/outputs.npz"))

    for clock in CLOCKS:
        physical_causality_check(index, history, references, clock)
        state_name = clock_name(clock)
        available = persisted[f"available__{state_name}"].astype(bool)
        sources = persisted[f"source_at__{state_name}"]
        cutoff = np.asarray([
            dt.datetime.combine(day, clock) for day in dates], dtype=object)
        for i in np.flatnonzero(available):
            assert sources[i] is not None
            assert sources[i].date() == dates[i]
            assert sources[i] < cutoff[i]
        for h in (1, 3, 5, 10, 20):
            control = control_probability(clock, h, t4, t10)
            np.testing.assert_array_equal(
                persisted[f"control__{state_name}__h{h}"], control)
            for candidate in CANDIDATES:
                prefix = f"{state_name}__{candidate}__h{h}"
                usable = persisted[f"usable__{prefix}"].astype(bool)
                routed = persisted[f"routed__{prefix}"]
                probability = persisted[f"prob__{prefix}"]
                np.testing.assert_array_equal(routed[~usable], control[~usable])
                np.testing.assert_array_equal(routed[usable], probability[usable])
                assert np.all(usable <= available)

    metrics = pd.read_csv(OUT / "probability_metrics.csv")
    bootstrap = pd.read_csv(OUT / "paired_bootstrap.csv")
    selection = pd.read_csv(OUT / "selection.csv")
    pd.testing.assert_frame_equal(
        metrics.reset_index(drop=True),
        rebuilt_result["metrics"].reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    rebuilt_bootstrap = paired_bootstrap(persisted)
    pd.testing.assert_frame_equal(
        bootstrap.reset_index(drop=True),
        rebuilt_bootstrap.reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(
        selection.reset_index(drop=True),
        select_candidates(rebuilt_result["metrics"], rebuilt_bootstrap)
        .reset_index(drop=True),
        check_dtype=False)

    clock = dt.time(9, 0)
    state, _state_names, available, _source = (
        build_perpetual_prefix_features(index, history, references, clock))
    features = np.column_stack((compact_features(matrix, names), state))
    target = build_targets(series, index)["fav_h5"]
    maturity = target_reach_dates(index, series, 5)
    original_raw = fit_quarterly_available_hgb(
        features, target, maturity, dates, available)[0]
    original_probability = fit_quarterly_calibrator(
        original_raw, target, maturity, dates, currencies,
        min_train_date=MIN_TRAIN_DATE)[0]
    boundary = dt.date(2025, 1, 1)
    future = dates >= boundary
    changed_target = target.copy()
    changed_target[future & np.isfinite(changed_target)] = (
        1. - changed_target[future & np.isfinite(changed_target)])
    changed_maturity = maturity.copy()
    changed_maturity[future] = dt.date(2099, 1, 1)
    changed_raw = fit_quarterly_available_hgb(
        features, changed_target, changed_maturity, dates, available)[0]
    changed_probability = fit_quarterly_calibrator(
        changed_raw, changed_target, changed_maturity, dates, currencies,
        min_train_date=MIN_TRAIN_DATE)[0]
    np.testing.assert_array_equal(
        original_raw[dates < boundary], changed_raw[dates < boundary])
    np.testing.assert_array_equal(
        original_probability[dates < boundary],
        changed_probability[dates < boundary])

    selected = selection.set_index(["clock", "h"])["selected"].to_dict()
    assert selected[("perp_0900", 1)] == "perp_basis_rank"
    assert selected[("perp_0900", 3)] == "perp_basis_rank"
    assert all(selected[("perp_1000", h)] == "control"
               for h in (1, 3, 5, 10, 20))
    assert metadata["tomorrow_cbr_used"] is False
    checks = {
        "source_hashes_verified": True,
        "all_persisted_arrays_exactly_rebuilt": True,
        "metric_tables_rebuilt": True,
        "source_at_strictly_before_as_of": True,
        "same_day_nominally_complete_candles_only": True,
        "future_market_corruption_prefix_invariant": True,
        "future_target_maturity_prefix_invariant": True,
        "exact_control_fallback": True,
        "paired_bootstrap_exactly_rebuilt": True,
        "selection_exactly_rebuilt_from_screen_2024": True,
        "adopted_0900_horizons": [1, 3],
        "no_1000_perpetual_replacement": True,
        "tomorrow_cbr_used": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
