"""T13: completed perpetual-FX prefixes at 09:00 and 10:00 Moscow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import (
    probability_metrics,
    reliability_rows,
)
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_moex_perpetual_hourly_features import load_hourly_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
)
from research.temperature_t13_early_perpetual_models import (
    CLOCKS,
    build_perpetual_prefix_features,
    clock_name,
    fit_quarterly_available_hgb,
    physical_causality_check,
)


OUT = Path("results/research/temperature/t13_early_perpetual")
DATA = Path("data/moex_perpetual_fx_hourly_2022_2026.json")
T4 = Path("results/research/temperature/t4_premarket")
T10 = Path("results/research/temperature/t10_early1000")
CANDIDATES = ("perp_basis_rank", "perp_open_rank", "perp_hybrid_hgb")
BOOTSTRAP_DRAWS = 1000


def control_probability(clock, h, t4, t10):
    if clock.hour == 9:
        return np.asarray(t4[f"prob__history_hist__h{h}"], dtype=float)
    return np.asarray(t10[f"routed_probability_h{h}"], dtype=float)


def run_experiment():
    matrix, names, index, series, *_ = load_round5_features()
    base = compact_features(matrix, names)
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_hourly_history()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    t4, t10 = (loadz(path / "outputs.npz") for path in (T4, T10))
    arrays = {"dates": dates, "currencies": currencies}
    metrics, reliability, model_logs, calibration_logs = [], [], [], []
    availability_rows = []

    for clock in CLOCKS:
        physical_causality_check(index, history, references, clock)
        state, state_names, available, source_at = (
            build_perpetual_prefix_features(index, history, references, clock))
        state_name = clock_name(clock)
        arrays[f"available__{state_name}"] = available
        arrays[f"source_at__{state_name}"] = source_at
        arrays[f"market__{state_name}"] = state
        basis_raw = np.where(
            available, state[:, state_names.index("cnyrubf_mean_cbr_basis")],
            np.nan)
        open_raw = np.where(
            available,
            state[:, state_names.index("cnyrubf_open_to_cutoff_return")],
            np.nan)
        scalar_raw = {
            "perp_basis_rank": causal_percentiles(
                basis_raw, dates, currencies, 250, 20),
            "perp_open_rank": causal_percentiles(
                open_raw, dates, currencies, 250, 20),
        }
        for candidate, raw in scalar_raw.items():
            raw[~available] = np.nan
            arrays[f"raw__{state_name}__{candidate}"] = raw

        for year in range(2022, max(day.year for day in dates) + 1):
            scope = np.asarray([day.year == year for day in dates])
            availability_rows.append({
                "clock": state_name, "year": year,
                "rows": int(scope.sum()),
                "physical_cny_share": float(available[scope].mean()),
                "physical_cny_rows": int(available[scope].sum()),
            })

        hybrid_features = np.column_stack((base, state))
        for h in HORIZONS:
            target = targets[f"fav_h{h}"]
            maturity = target_reach_dates(index, series, h)
            hybrid_raw, train_count, logs = fit_quarterly_available_hgb(
                hybrid_features, target, maturity, dates, available)
            arrays[f"raw__{state_name}__perp_hybrid_hgb__h{h}"] = hybrid_raw
            arrays[f"model_n_train__{state_name}__h{h}"] = train_count
            model_logs.extend({"clock": state_name, "h": h, **row}
                              for row in logs)
            raw_candidates = {
                **scalar_raw,
                "perp_hybrid_hgb": hybrid_raw,
            }
            control = control_probability(clock, h, t4, t10)
            arrays[f"control__{state_name}__h{h}"] = control
            for candidate, raw in raw_candidates.items():
                calibrated, prior, count, logs = fit_quarterly_calibrator(
                    raw, target, maturity, dates, currencies,
                    min_train_date=MIN_TRAIN_DATE)
                usable = available & np.isfinite(calibrated)
                routed = np.where(usable, calibrated, control)
                prefix = f"{state_name}__{candidate}__h{h}"
                arrays[f"prob__{prefix}"] = calibrated
                arrays[f"prior__{prefix}"] = prior
                arrays[f"cal_n_train__{prefix}"] = count
                arrays[f"usable__{prefix}"] = usable
                arrays[f"routed__{prefix}"] = routed
                calibration_logs.extend({
                    "clock": state_name, "candidate": candidate, "h": h,
                    **row,
                } for row in logs)

                slices = [
                    ("screen_2024", "ALL", "ALL",
                     np.asarray([day.year == 2024 for day in dates])),
                    ("opened_2025_2026", "ALL", "ALL",
                     np.asarray([day.year in (2025, 2026) for day in dates])),
                ]
                for period, years in (("screen_2024", (2024,)),
                                      ("opened_2025_2026", (2025, 2026))):
                    period_scope = np.asarray([day.year in years for day in dates])
                    for currency in sorted(set(currencies)):
                        slices.append((
                            period, "currency", currency,
                            period_scope & (currencies == currency)))
                    for year in years:
                        slices.append((
                            period, "year", str(year),
                            np.asarray([day.year == year for day in dates])))
                for period, slice_name, group, scope in slices:
                    if not scope.any():
                        continue
                    metrics.append({
                        "clock": state_name, "candidate": candidate, "h": h,
                        "period": period, "slice": slice_name, "group": group,
                        "physical_availability": float(available[scope].mean()),
                        "usable_share": float(usable[scope].mean()),
                        **probability_metrics(
                            target[scope], routed[scope], routed[scope],
                            control[scope]),
                    })
                for period, years in (("screen_2024", (2024,)),
                                      ("opened_2025_2026", (2025, 2026))):
                    scope = np.asarray([day.year in years for day in dates])
                    reliability.extend({
                        "clock": state_name, "candidate": candidate,
                        "period": period, **row,
                    } for row in reliability_rows(
                        target[scope], routed[scope], h))
    return {
        "arrays": arrays,
        "metrics": pd.DataFrame(metrics),
        "reliability": pd.DataFrame(reliability),
        "availability": pd.DataFrame(availability_rows),
        "model_logs": pd.DataFrame(model_logs),
        "calibration_logs": pd.DataFrame(calibration_logs),
        "digest": digest,
    }


def paired_bootstrap(arrays):
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    targets = build_targets(series, index)
    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("opened_2025_2026", (2025, 2026))):
        period_scope = np.asarray([day.year in years for day in dates])
        for clock_i, clock in enumerate(CLOCKS):
            state_name = clock_name(clock)
            for h in HORIZONS:
                target = targets[f"fav_h{h}"]
                control = arrays[f"control__{state_name}__h{h}"]
                for candidate_i, candidate in enumerate(CANDIDATES):
                    routed = arrays[f"routed__{state_name}__{candidate}__h{h}"]
                    valid = (period_scope & np.isfinite(target)
                             & np.isfinite(routed) & np.isfinite(control))
                    day_values = np.asarray(sorted(set(dates[valid])), dtype=object)
                    daily_delta = np.asarray([
                        np.mean((routed[valid & (dates == day)]
                                 - target[valid & (dates == day)]) ** 2
                                - (control[valid & (dates == day)]
                                   - target[valid & (dates == day)]) ** 2)
                        for day in day_values], dtype=float)
                    for block in (20, 50):
                        rng = np.random.default_rng(
                            20260906 + h * 101 + block + clock_i * 1000
                            + candidate_i * 10000
                            + (0 if period == "screen_2024" else 100000))
                        draws = []
                        blocks = int(np.ceil(len(daily_delta) / block))
                        offsets = np.arange(block)
                        for _ in range(BOOTSTRAP_DRAWS):
                            starts = rng.integers(
                                0, len(daily_delta), size=blocks)
                            ids = ((starts[:, None] + offsets[None, :])
                                   % len(daily_delta)).ravel()[:len(daily_delta)]
                            draws.append(float(np.mean(daily_delta[ids])))
                        draws = np.asarray(draws)
                        rows.append({
                            "period": period, "clock": state_name,
                            "candidate": candidate, "h": h,
                            "block_dates": block, "n_dates": len(day_values),
                            "mean_brier_delta": float(np.mean(daily_delta)),
                            "ci_low": float(np.quantile(draws, .025)),
                            "ci_high": float(np.quantile(draws, .975)),
                            "probability_candidate_better": float(
                                np.mean(draws < 0.)),
                        })
    return pd.DataFrame(rows)


def select_candidates(metrics, bootstrap):
    rows = []
    screen = metrics[
        (metrics.period == "screen_2024")
        & (metrics.slice == "ALL")
        & (metrics.group == "ALL")]
    for clock in map(clock_name, CLOCKS):
        for h in HORIZONS:
            part = screen[(screen.clock == clock) & (screen.h == h)]
            best = part.sort_values(
                ["brier_calibrated", "candidate"]).iloc[0]
            intervals = bootstrap[
                (bootstrap.period == "screen_2024")
                & (bootstrap.clock == clock)
                & (bootstrap.h == h)
                & (bootstrap.candidate == best.candidate)]
            adopted = (set(intervals.block_dates) == {20, 50}
                       and bool((intervals.ci_high < 0.).all()))
            rows.append({
                "clock": clock, "h": int(h),
                "best_screen_candidate": str(best.candidate),
                "screen_brier": float(best.brier_calibrated),
                "control_brier": float(best.brier_prior),
                "adopted": adopted,
                "selected": str(best.candidate) if adopted else "control",
                "selection_period": "screen_2024",
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    np.savez_compressed(OUT / "outputs.npz", **result["arrays"])
    result["metrics"].to_csv(OUT / "probability_metrics.csv", index=False)
    result["reliability"].to_csv(OUT / "reliability_bins.csv", index=False)
    result["availability"].to_csv(OUT / "availability.csv", index=False)
    result["model_logs"].to_csv(OUT / "model_training_log.csv", index=False)
    result["calibration_logs"].to_csv(
        OUT / "calibration_log.csv", index=False)
    bootstrap = paired_bootstrap(result["arrays"])
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    selection = select_candidates(result["metrics"], bootstrap)
    selection.to_csv(OUT / "selection.csv", index=False)
    sources = [
        DATA, T4 / "metadata.json", T4 / "outputs.npz",
        T10 / "metadata.json", T10 / "outputs.npz",
        Path("research/temperature_t13_early_perpetual_registered.md"),
        Path("research/temperature_t13_early_perpetual.py"),
        Path("research/temperature_t13_early_perpetual_models.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T13",
        "clocks": [clock.isoformat() for clock in CLOCKS],
        "candidates": list(CANDIDATES),
        "screen": "2024 only",
        "opened_diagnostic": "2025-2026",
        "controls": {"perp_0900": "T4", "perp_1000": "T10"},
        "tomorrow_cbr_used": False,
        "historical_exchange_availability_certified": False,
        "bank_execution_validated": False,
        "payload_sha256": result["digest"],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    opened = result["metrics"][(result["metrics"].period == "opened_2025_2026")
                               & (result["metrics"].slice == "ALL")]
    print(result["availability"].to_string(index=False), flush=True)
    print(selection.to_string(index=False), flush=True)
    print(opened[["clock", "candidate", "h", "usable_share",
                  "brier_calibrated", "brier_prior", "auc_calibrated",
                  "auc_prior", "ece_calibrated"]].to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
