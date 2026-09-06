"""T15: honest completed perpetual-FX updates at 20:00-23:00 Moscow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA as CBR_DATA
from research.after_publication_ap50_temperature import (
    probability_metrics,
    reliability_rows,
)
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.after_publication_ap51_benefit import magnitude_metrics
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes
from research.round6_moex_perpetual_hourly_features import load_hourly_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE
from research.temperature_t15_evening_perpetual_models import (
    CLOCKS,
    build_evening_features,
    candidate_features,
    clock_name,
    fit_quarterly_classifier,
    physical_causality_check,
)


OUT = Path("results/research/temperature/t15_evening_perpetual")
BASE = Path("results/research/after_publication/ap51_benefit")
T7B = Path("results/research/temperature/t7b_after_decision_market")
MARKET_DATA = Path("data/moex_perpetual_fx_hourly_2022_2026.json")
HORIZONS = (3, 5, 10, 20)
CANDIDATES = ("perp_cny_logit", "perp_dual_logit", "perp_dual_hgb")
BOOTSTRAP_DRAWS = 1000


def run_experiment():
    panel = pd.read_csv(BASE / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    base = loadz(BASE / "outputs.npz")
    t7b = loadz(T7B / "outputs.npz")
    history, digest = load_hourly_history()
    cap = build_outcomes(load(CBR_DATA), panel, "publication")
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    arrays = {
        "dates": dates, "currencies": currencies,
        "known_probability_h1": t7b["known_probability_h1"],
        "known_future_bps_h1": t7b["known_future_bps_h1"],
    }
    probability_rows, benefit_rows, reliability = [], [], []
    model_logs, calibration_logs, benefit_logs, availability_rows = [], [], [], []

    for clock in CLOCKS:
        physical_causality_check(panel, history, clock)
        state = clock_name(clock)
        market, cny_available, dual_available, cny_source, dual_source = (
            build_evening_features(panel, history, clock))
        arrays[f"market__{state}"] = market
        arrays[f"cny_available__{state}"] = cny_available
        arrays[f"dual_available__{state}"] = dual_available
        arrays[f"cny_source_at__{state}"] = cny_source
        arrays[f"dual_source_at__{state}"] = dual_source
        for year in range(2022, max(day.year for day in dates) + 1):
            scope = np.asarray([day.year == year for day in dates])
            availability_rows.append({
                "clock": state, "year": year, "rows": int(scope.sum()),
                "cny_share": float(cny_available[scope].mean()),
                "dual_share": float(dual_available[scope].mean()),
                "dual_rows": int(dual_available[scope].sum()),
            })

        for h in HORIZONS:
            control_probability = np.asarray(
                t7b[f"probability__2000__h{h}"], dtype=float)
            control_benefit = np.asarray(
                t7b[f"expected_bps__2000__h{h}"], dtype=float)
            target = np.asarray(base[f"y{h}"], dtype=float)
            forward = np.asarray(base[f"forward{h}"], dtype=float)
            maturity = cap[f"mature{h}"]
            arrays[f"control_probability__{state}__h{h}"] = control_probability
            arrays[f"control_benefit__{state}__h{h}"] = control_benefit

            for candidate in CANDIDATES:
                available = (cny_available if candidate == "perp_cny_logit"
                             else dual_available)
                features = candidate_features(
                    control_probability, market, currencies, candidate)
                kind = "hgb" if candidate.endswith("hgb") else "logit"
                raw, count, logs = fit_quarterly_classifier(
                    features, target, maturity, dates, available, kind)
                model_logs.extend({
                    "clock": state, "candidate": candidate, "h": h, **row,
                } for row in logs)
                if kind == "hgb":
                    predicted, prior, cal_count, logs = fit_quarterly_calibrator(
                        raw, target, maturity, dates, currencies,
                        min_train_date=MIN_TRAIN_DATE)
                    calibration_logs.extend({
                        "clock": state, "candidate": candidate, "h": h,
                        **row,
                    } for row in logs)
                else:
                    predicted = raw
                    prior = np.full(len(raw), np.nan)
                    cal_count = count
                usable = available & np.isfinite(predicted)
                routed = np.where(usable, predicted, control_probability)
                prefix = f"{state}__{candidate}__h{h}"
                arrays[f"raw__{prefix}"] = raw
                arrays[f"probability__{prefix}"] = predicted
                arrays[f"prior__{prefix}"] = prior
                arrays[f"n_train__{prefix}"] = cal_count
                arrays[f"usable__{prefix}"] = usable
                arrays[f"routed_probability__{prefix}"] = routed
                for period, years in (("screen_2024", (2024,)),
                                      ("opened_2025_2026", (2025, 2026))):
                    period_scope = np.asarray([day.year in years for day in dates])
                    slices = [("ALL", "ALL", period_scope)]
                    slices.extend(("currency", currency,
                                   period_scope & (currencies == currency))
                                  for currency in sorted(set(currencies)))
                    slices.extend(("year", str(year),
                                   np.asarray([day.year == year for day in dates]))
                                  for year in years)
                    for slice_name, group, scope in slices:
                        if not scope.any():
                            continue
                        probability_rows.append({
                            "clock": state, "candidate": candidate, "h": h,
                            "period": period, "slice": slice_name,
                            "group": group,
                            "physical_availability": float(available[scope].mean()),
                            "usable_share": float(usable[scope].mean()),
                            **probability_metrics(
                                target[scope], routed[scope], routed[scope],
                                control_probability[scope]),
                        })
                    reliability.extend({
                        "clock": state, "candidate": candidate,
                        "period": period, **row,
                    } for row in reliability_rows(
                        target[period_scope], routed[period_scope], h))

            benefit_features = np.column_stack([
                base["multihorizon_features"], control_probability,
                market, dual_available.astype(float),
            ])
            prediction, prior, count, logs = fit_quarterly_benefit(
                benefit_features, forward, maturity, dates,
                min_train_date=MIN_TRAIN_DATE)
            benefit_logs.extend({
                "clock": state, "candidate": "perp_dual_ridge", "h": h,
                **row,
            } for row in logs)
            usable = dual_available & np.isfinite(prediction)
            routed = np.where(usable, prediction, control_benefit)
            arrays[f"benefit__{state}__h{h}"] = prediction
            arrays[f"benefit_prior__{state}__h{h}"] = prior
            arrays[f"benefit_n_train__{state}__h{h}"] = count
            arrays[f"benefit_usable__{state}__h{h}"] = usable
            arrays[f"routed_benefit__{state}__h{h}"] = routed
            for period, years in (("screen_2024", (2024,)),
                                  ("opened_2025_2026", (2025, 2026))):
                period_scope = np.asarray([day.year in years for day in dates])
                slices = [("ALL", "ALL", period_scope)]
                slices.extend(("currency", currency,
                               period_scope & (currencies == currency))
                              for currency in sorted(set(currencies)))
                slices.extend(("year", str(year),
                               np.asarray([day.year == year for day in dates]))
                              for year in years)
                for slice_name, group, scope in slices:
                    if not scope.any():
                        continue
                    benefit_rows.append({
                        "clock": state, "candidate": "perp_dual_ridge",
                        "h": h, "period": period, "slice": slice_name,
                        "group": group,
                        "physical_availability": float(
                            dual_available[scope].mean()),
                        "usable_share": float(usable[scope].mean()),
                        **magnitude_metrics(
                            forward[scope], routed[scope],
                            control_benefit[scope]),
                    })
    return {
        "arrays": arrays,
        "probability_metrics": pd.DataFrame(probability_rows),
        "benefit_metrics": pd.DataFrame(benefit_rows),
        "reliability": pd.DataFrame(reliability),
        "availability": pd.DataFrame(availability_rows),
        "model_logs": pd.DataFrame(model_logs),
        "calibration_logs": pd.DataFrame(calibration_logs),
        "benefit_logs": pd.DataFrame(benefit_logs),
        "digest": digest,
    }


def _bootstrap_loss_delta(dates, valid, candidate_loss, control_loss, seed):
    day_values = np.asarray(sorted(set(dates[valid])), dtype=object)
    daily_delta = np.asarray([
        np.mean(candidate_loss[valid & (dates == day)]
                - control_loss[valid & (dates == day)])
        for day in day_values
    ], dtype=float)
    rows = []
    for block in (20, 50):
        rng = np.random.default_rng(seed + block)
        blocks = int(np.ceil(len(daily_delta) / block))
        offsets = np.arange(block)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            starts = rng.integers(0, len(daily_delta), size=blocks)
            ids = ((starts[:, None] + offsets[None, :])
                   % len(daily_delta)).ravel()[:len(daily_delta)]
            draws.append(float(np.mean(daily_delta[ids])))
        draws = np.asarray(draws)
        rows.append({
            "block_dates": block, "n_dates": len(day_values),
            "mean_loss_delta": float(np.mean(daily_delta)),
            "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)),
            "probability_candidate_better": float(np.mean(draws < 0.)),
        })
    return rows


def paired_bootstrap(arrays):
    panel = pd.read_csv(BASE / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    base = loadz(BASE / "outputs.npz")
    dates = panel.date.to_numpy()
    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("opened_2025_2026", (2025, 2026))):
        scope = np.asarray([day.year in years for day in dates])
        for clock_i, clock in enumerate(CLOCKS):
            state = clock_name(clock)
            for h in HORIZONS:
                target = np.asarray(base[f"y{h}"], dtype=float)
                control = arrays[f"control_probability__{state}__h{h}"]
                for candidate_i, candidate in enumerate(CANDIDATES):
                    routed = arrays[
                        f"routed_probability__{state}__{candidate}__h{h}"]
                    valid = (scope & np.isfinite(target) & np.isfinite(routed)
                             & np.isfinite(control))
                    generated = _bootstrap_loss_delta(
                        dates, valid, (routed - target) ** 2,
                        (control - target) ** 2,
                        20260906 + clock_i * 10000 + h * 101
                        + candidate_i * 1000
                        + (0 if period == "screen_2024" else 100000))
                    rows.extend({
                        "period": period, "metric": "brier", "clock": state,
                        "candidate": candidate, "h": h, **row,
                    } for row in generated)
                forward = np.asarray(base[f"forward{h}"], dtype=float)
                control_benefit = arrays[f"control_benefit__{state}__h{h}"]
                routed_benefit = arrays[f"routed_benefit__{state}__h{h}"]
                valid = (scope & np.isfinite(forward)
                         & np.isfinite(routed_benefit)
                         & np.isfinite(control_benefit))
                generated = _bootstrap_loss_delta(
                    dates, valid, np.abs(routed_benefit - forward),
                    np.abs(control_benefit - forward),
                    20270906 + clock_i * 10000 + h * 101
                    + (0 if period == "screen_2024" else 100000))
                rows.extend({
                    "period": period, "metric": "mae", "clock": state,
                    "candidate": "perp_dual_ridge", "h": h, **row,
                } for row in generated)
    return pd.DataFrame(rows)


def select_candidates(probability_metrics_frame, benefit_metrics_frame,
                      bootstrap):
    rows = []
    probability_screen = probability_metrics_frame[
        (probability_metrics_frame.period == "screen_2024")
        & (probability_metrics_frame.slice == "ALL")
        & (probability_metrics_frame.group == "ALL")]
    benefit_screen = benefit_metrics_frame[
        (benefit_metrics_frame.period == "screen_2024")
        & (benefit_metrics_frame.slice == "ALL")
        & (benefit_metrics_frame.group == "ALL")]
    for clock in map(clock_name, CLOCKS):
        for h in HORIZONS:
            part = probability_screen[
                (probability_screen.clock == clock)
                & (probability_screen.h == h)]
            best = part.sort_values(
                ["brier_calibrated", "candidate"]).iloc[0]
            intervals = bootstrap[
                (bootstrap.period == "screen_2024")
                & (bootstrap.metric == "brier")
                & (bootstrap.clock == clock) & (bootstrap.h == h)
                & (bootstrap.candidate == best.candidate)]
            adopted = (set(intervals.block_dates) == {20, 50}
                       and bool((intervals.ci_high < 0.).all()))
            rows.append({
                "kind": "probability", "clock": clock, "h": h,
                "best_screen_candidate": best.candidate,
                "screen_loss": float(best.brier_calibrated),
                "control_loss": float(best.brier_prior),
                "adopted": adopted,
                "selected": best.candidate if adopted else "control",
            })
            benefit = benefit_screen[
                (benefit_screen.clock == clock)
                & (benefit_screen.h == h)].iloc[0]
            intervals = bootstrap[
                (bootstrap.period == "screen_2024")
                & (bootstrap.metric == "mae")
                & (bootstrap.clock == clock) & (bootstrap.h == h)]
            adopted = (set(intervals.block_dates) == {20, 50}
                       and bool((intervals.ci_high < 0.).all()))
            rows.append({
                "kind": "benefit", "clock": clock, "h": h,
                "best_screen_candidate": "perp_dual_ridge",
                "screen_loss": float(benefit.mae_model),
                "control_loss": float(benefit.mae_prior),
                "adopted": adopted,
                "selected": "perp_dual_ridge" if adopted else "control",
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    np.savez_compressed(OUT / "outputs.npz", **result["arrays"])
    for key, filename in (
        ("probability_metrics", "probability_metrics.csv"),
        ("benefit_metrics", "benefit_metrics.csv"),
        ("reliability", "reliability_bins.csv"),
        ("availability", "availability.csv"),
        ("model_logs", "model_training_log.csv"),
        ("calibration_logs", "calibration_log.csv"),
        ("benefit_logs", "benefit_training_log.csv"),
    ):
        result[key].to_csv(OUT / filename, index=False)
    bootstrap = paired_bootstrap(result["arrays"])
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    selection = select_candidates(
        result["probability_metrics"], result["benefit_metrics"], bootstrap)
    selection.to_csv(OUT / "selection.csv", index=False)
    sources = [
        MARKET_DATA, BASE / "metadata.json", BASE / "outputs.npz",
        T7B / "metadata.json", T7B / "outputs.npz",
        Path("research/temperature_t15_evening_perpetual_registered.md"),
        Path("research/temperature_t15_evening_perpetual.py"),
        Path("research/temperature_t15_evening_perpetual_models.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T15",
        "clocks": [clock.isoformat() for clock in CLOCKS],
        "probability_candidates": list(CANDIDATES),
        "benefit_candidate": "perp_dual_ridge",
        "control": "T7B 20:00 state carried forward",
        "screen": "2024 only",
        "opened_diagnostic": "2025-2026",
        "h1_is_known_not_forecast": True,
        "historical_exchange_availability_certified": False,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "payload_sha256": result["digest"],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }, indent=2))
    opened_p = result["probability_metrics"].query(
        "period == 'opened_2025_2026' and slice == 'ALL'")
    opened_b = result["benefit_metrics"].query(
        "period == 'opened_2025_2026' and slice == 'ALL'")
    print(result["availability"].to_string(index=False), flush=True)
    print(selection.to_string(index=False), flush=True)
    print(opened_p[[
        "clock", "candidate", "h", "usable_share", "brier_calibrated",
        "brier_prior", "auc_calibrated", "auc_prior", "ece_calibrated",
    ]].to_string(index=False), flush=True)
    print(opened_b[[
        "clock", "h", "usable_share", "mae_model", "mae_prior",
        "spearman",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
