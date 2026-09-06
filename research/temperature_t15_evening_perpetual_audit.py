"""Independent exact rebuild and prefix audit for T15."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA as CBR_DATA
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes
from research.round6_moex_perpetual_hourly_features import load_hourly_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE
from research.temperature_t15_evening_perpetual import (
    BASE,
    CANDIDATES,
    CLOCKS,
    HORIZONS,
    OUT,
    T7B,
    paired_bootstrap,
    run_experiment,
    select_candidates,
)
from research.temperature_t15_evening_perpetual_models import (
    build_evening_features,
    candidate_features,
    clock_name,
    fit_quarterly_classifier,
    physical_causality_check,
)


def same(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype == object or right.dtype == object:
        np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_allclose(
            left, right, rtol=1e-13, atol=1e-13, equal_nan=True)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    persisted = loadz(OUT / "outputs.npz")
    rebuilt_result = run_experiment()
    rebuilt = rebuilt_result["arrays"]
    assert set(persisted) == set(rebuilt)
    for key, value in rebuilt.items():
        same(persisted[key], value)

    panel = pd.read_csv(BASE / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    base = loadz(BASE / "outputs.npz")
    t7b = loadz(T7B / "outputs.npz")
    history, _digest = load_hourly_history()
    cap = build_outcomes(load(CBR_DATA), panel, "publication")
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    boundary = dt.date(2025, 1, 6)
    past, future = dates < boundary, dates >= boundary
    probability_prefix_checks = benefit_prefix_checks = 0

    same(persisted["known_probability_h1"], t7b["known_probability_h1"])
    same(persisted["known_future_bps_h1"], t7b["known_future_bps_h1"])

    for clock in CLOCKS:
        physical_causality_check(panel, history, clock, boundary)
        state = clock_name(clock)
        market, cny_available, dual_available, cny_source, dual_source = (
            build_evening_features(panel, history, clock))
        same(persisted[f"market__{state}"], market)
        same(persisted[f"cny_available__{state}"], cny_available)
        same(persisted[f"dual_available__{state}"], dual_available)
        same(persisted[f"cny_source_at__{state}"], cny_source)
        same(persisted[f"dual_source_at__{state}"], dual_source)
        cutoff = dt.datetime.combine(dt.date.max, clock).time()
        for i in np.flatnonzero(cny_available):
            assert cny_source[i].date() == dates[i]
            assert cny_source[i].time() < cutoff
        for i in np.flatnonzero(dual_available):
            assert dual_source[i].date() == dates[i]
            assert dual_source[i].time() < cutoff

        for h in HORIZONS:
            control_probability = t7b[f"probability__2000__h{h}"]
            control_benefit = t7b[f"expected_bps__2000__h{h}"]
            same(persisted[f"control_probability__{state}__h{h}"],
                 control_probability)
            same(persisted[f"control_benefit__{state}__h{h}"],
                 control_benefit)
            target = np.asarray(base[f"y{h}"], dtype=float)
            maturity = np.asarray(cap[f"mature{h}"], dtype=object)
            for candidate in CANDIDATES:
                available = (cny_available if candidate == "perp_cny_logit"
                             else dual_available)
                features = candidate_features(
                    control_probability, market, currencies, candidate)
                kind = "hgb" if candidate.endswith("hgb") else "logit"
                raw, count, _logs = fit_quarterly_classifier(
                    features, target, maturity, dates, available, kind)
                if kind == "hgb":
                    predicted, prior, cal_count, _logs = (
                        fit_quarterly_calibrator(
                            raw, target, maturity, dates, currencies,
                            min_train_date=MIN_TRAIN_DATE))
                else:
                    predicted, prior, cal_count = (
                        raw, np.full(len(raw), np.nan), count)
                prefix = f"{state}__{candidate}__h{h}"
                same(persisted[f"raw__{prefix}"], raw)
                same(persisted[f"probability__{prefix}"], predicted)
                same(persisted[f"prior__{prefix}"], prior)
                same(persisted[f"n_train__{prefix}"], cal_count)
                usable = available & np.isfinite(predicted)
                routed = np.where(usable, predicted, control_probability)
                same(persisted[f"usable__{prefix}"], usable)
                same(persisted[f"routed_probability__{prefix}"], routed)
                same(routed[~usable], control_probability[~usable])

                changed_features = features.copy()
                changed_features[future] = changed_features[future] * -17. + 3.
                changed_available = available.copy()
                changed_available[future] = ~changed_available[future]
                changed_target = target.copy()
                changed_target[future & np.isfinite(changed_target)] = 1. - (
                    changed_target[future & np.isfinite(changed_target)])
                changed_maturity = maturity.copy()
                changed_maturity[future] = dt.date(1900, 1, 1)
                altered_raw = fit_quarterly_classifier(
                    changed_features, changed_target, changed_maturity, dates,
                    changed_available, kind)[0]
                same(raw[past], altered_raw[past])
                if kind == "hgb":
                    altered = fit_quarterly_calibrator(
                        altered_raw, changed_target, changed_maturity, dates,
                        currencies, min_train_date=MIN_TRAIN_DATE)[0]
                    same(predicted[past], altered[past])
                probability_prefix_checks += 1

            benefit_features = np.column_stack([
                base["multihorizon_features"], control_probability,
                market, dual_available.astype(float),
            ])
            forward = np.asarray(base[f"forward{h}"], dtype=float)
            prediction, prior, count, _logs = fit_quarterly_benefit(
                benefit_features, forward, maturity, dates,
                min_train_date=MIN_TRAIN_DATE)
            usable = dual_available & np.isfinite(prediction)
            routed = np.where(usable, prediction, control_benefit)
            same(persisted[f"benefit__{state}__h{h}"], prediction)
            same(persisted[f"benefit_prior__{state}__h{h}"], prior)
            same(persisted[f"benefit_n_train__{state}__h{h}"], count)
            same(persisted[f"benefit_usable__{state}__h{h}"], usable)
            same(persisted[f"routed_benefit__{state}__h{h}"], routed)
            same(routed[~usable], control_benefit[~usable])

            changed_features = benefit_features.copy()
            changed_features[future] = changed_features[future] * 13. - 5.
            changed_forward = forward.copy()
            changed_forward[future & np.isfinite(changed_forward)] *= -19.
            changed_maturity = maturity.copy()
            changed_maturity[future] = dt.date(1900, 1, 1)
            altered = fit_quarterly_benefit(
                changed_features, changed_forward, changed_maturity, dates,
                min_train_date=MIN_TRAIN_DATE)[0]
            same(prediction[past], altered[past])
            benefit_prefix_checks += 1

    bootstrap = paired_bootstrap(persisted)
    persisted_bootstrap = pd.read_csv(OUT / "paired_bootstrap.csv")
    pd.testing.assert_frame_equal(
        persisted_bootstrap, bootstrap, check_exact=False,
        rtol=1e-13, atol=1e-13)
    selection = select_candidates(
        rebuilt_result["probability_metrics"],
        rebuilt_result["benefit_metrics"], bootstrap)
    persisted_selection = pd.read_csv(OUT / "selection.csv")
    pd.testing.assert_frame_equal(
        persisted_selection, selection, check_exact=False,
        rtol=1e-13, atol=1e-13)
    selected_probability = persisted_selection.query(
        "kind == 'probability' and adopted")
    assert selected_probability.empty
    selected_benefit = persisted_selection.query(
        "kind == 'benefit' and adopted")
    assert set(zip(selected_benefit.clock, selected_benefit.h)) == {
        ("perp_2000", 3), ("perp_2100", 3), ("perp_2100", 5),
        ("perp_2200", 3), ("perp_2200", 5),
        ("perp_2300", 3), ("perp_2300", 5),
    }

    checks = {
        "source_hashes_verified": True,
        "exact_arrays_rebuilt": True,
        "four_physical_future_candle_checks_passed": True,
        "probability_prefix_checks": probability_prefix_checks,
        "benefit_prefix_checks": benefit_prefix_checks,
        "future_market_target_maturity_prefix_invariant": True,
        "unavailable_fallback_exact": True,
        "h1_known_values_copied_exactly": True,
        "paired_bootstrap_rebuilt": True,
        "selection_rebuilt": True,
        "probability_replacements": 0,
        "benefit_replacements": int(len(selected_benefit)),
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
