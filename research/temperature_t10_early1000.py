"""T10: one availability-routed 10:00 probability and benefit snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import probability_metrics
from research.after_publication_ap51_benefit import magnitude_metrics
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
)


OUT = Path("results/research/temperature/t10_early1000")
T4 = Path("results/research/temperature/t4_premarket")
T6 = Path("results/research/temperature/t6_pre_receipt_benefit")
T9 = Path("results/research/temperature/t9_early_market")
CLOCK = "early_1000"
SELECTED = {1: "hybrid_hgb", 3: "cny_rank", 5: "cny_rank",
            10: "cny_rank", 20: "cny_rank"}
BOOTSTRAP_DRAWS = 1000


def selected_keys(h):
    model = SELECTED[h]
    raw = (f"raw__hybrid__{CLOCK}__h{h}" if model == "hybrid_hgb"
           else f"rank__cny__{CLOCK}")
    probability = f"prob__{model}__{CLOCK}__h{h}"
    return raw, probability


def run_experiment():
    matrix, names, index, series, *_ = load_round5_features()
    base = compact_features(matrix, names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    t4, t6, t9 = (loadz(path / "outputs.npz") for path in (T4, T6, T9))
    available = np.asarray(t9[f"available__{CLOCK}"], dtype=bool)
    market = np.asarray(t9[f"market__{CLOCK}"], dtype=float)
    arrays = {
        "dates": dates, "currencies": currencies,
        "available": available, "source_at": t9[f"source_at__{CLOCK}"],
    }
    probability_rows, benefit_rows, logs = [], [], []
    for h in HORIZONS:
        raw_key, probability_key = selected_keys(h)
        selected_raw = t9[raw_key]
        selected_probability = t9[probability_key]
        fallback_probability = t4[f"prob__history_hist__h{h}"]
        routed_probability = np.where(
            available, selected_probability, fallback_probability)
        probability_target = targets[f"fav_h{h}"]
        probability_prior = t9[f"prior__{SELECTED[h]}__{CLOCK}__h{h}"]
        arrays[f"selected_raw_h{h}"] = selected_raw
        arrays[f"selected_probability_h{h}"] = selected_probability
        arrays[f"routed_probability_h{h}"] = routed_probability

        benefit_target = _forward(series, index, h)
        maturity = target_reach_dates(index, series, h)
        benefit_features = np.column_stack((
            base, market, selected_raw, selected_probability))
        predicted, prior, count, head_logs = fit_quarterly_benefit(
            benefit_features, benefit_target, maturity, dates,
            min_train_date=MIN_TRAIN_DATE)
        fallback_benefit = t6[f"expected_bps__premarket__h{h}"]
        routed_benefit = np.where(available, predicted, fallback_benefit)
        arrays[f"market_expected_bps_h{h}"] = predicted
        arrays[f"market_prior_bps_h{h}"] = prior
        arrays[f"benefit_n_train_h{h}"] = count
        arrays[f"routed_expected_bps_h{h}"] = routed_benefit
        logs.extend({"h": h, **row} for row in head_logs)

        for period, years in (("screen_2024", (2024,)),
                              ("opened_2025_2026", (2025, 2026))):
            scope = np.asarray([day.year in years for day in dates])
            probability_rows.append({
                "h": h, "period": period,
                "selected": SELECTED[h],
                "availability": float(available[scope].mean()),
                **probability_metrics(
                    probability_target[scope], routed_probability[scope],
                    routed_probability[scope], probability_prior[scope]),
            })
            benefit_rows.append({
                "h": h, "period": period,
                "selected": SELECTED[h],
                "availability": float(available[scope].mean()),
                **magnitude_metrics(
                    benefit_target[scope], routed_benefit[scope],
                    t6[f"prior_bps__premarket__h{h}"][scope]),
            })
    return {
        "arrays": arrays,
        "probability_metrics": pd.DataFrame(probability_rows),
        "benefit_metrics": pd.DataFrame(benefit_rows),
        "training_log": pd.DataFrame(logs),
    }


def paired_bootstrap(arrays):
    """Paired circular date-block deltas; negative favours T10."""
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    targets = build_targets(series, index)
    t4, t6 = (loadz(path / "outputs.npz") for path in (T4, T6))
    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("opened_2025_2026", (2025, 2026))):
        period_scope = np.asarray([day.year in years for day in dates])
        for h in HORIZONS:
            comparisons = {
                "brier": (
                    targets[f"fav_h{h}"], arrays[f"routed_probability_h{h}"],
                    t4[f"prob__history_hist__h{h}"],
                    lambda y, prediction: (prediction - y) ** 2),
                "absolute_error_bps": (
                    _forward(series, index, h),
                    arrays[f"routed_expected_bps_h{h}"],
                    t6[f"expected_bps__premarket__h{h}"],
                    lambda y, prediction: np.abs(prediction - y)),
            }
            for metric, (target, candidate, control, loss) in comparisons.items():
                valid = (period_scope & np.isfinite(target)
                         & np.isfinite(candidate) & np.isfinite(control))
                day_values = np.asarray(sorted(set(dates[valid])), dtype=object)
                daily_delta = np.asarray([
                    np.mean(loss(target[valid & (dates == day)],
                                 candidate[valid & (dates == day)])
                            - loss(target[valid & (dates == day)],
                                   control[valid & (dates == day)]))
                    for day in day_values], dtype=float)
                for block in (20, 50):
                    rng = np.random.default_rng(
                        20260906 + h * 101 + block
                        + (0 if metric == "brier" else 10000)
                        + (0 if period == "screen_2024" else 20000))
                    draws = []
                    blocks = int(np.ceil(len(daily_delta) / block))
                    offsets = np.arange(block)
                    for _ in range(BOOTSTRAP_DRAWS):
                        starts = rng.integers(0, len(daily_delta), size=blocks)
                        ids = ((starts[:, None] + offsets[None, :])
                               % len(daily_delta)).ravel()[:len(daily_delta)]
                        draws.append(float(np.mean(daily_delta[ids])))
                    draws = np.asarray(draws)
                    rows.append({
                        "period": period, "h": h, "metric": metric,
                        "block_dates": block, "n_dates": len(day_values),
                        "mean_delta": float(np.mean(daily_delta)),
                        "ci_low": float(np.quantile(draws, .025)),
                        "ci_high": float(np.quantile(draws, .975)),
                        "probability_t10_better": float(np.mean(draws < 0.)),
                    })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    np.savez_compressed(OUT / "outputs.npz", **result["arrays"])
    result["probability_metrics"].to_csv(
        OUT / "probability_metrics.csv", index=False)
    result["benefit_metrics"].to_csv(
        OUT / "benefit_metrics.csv", index=False)
    result["training_log"].to_csv(OUT / "training_log.csv", index=False)
    bootstrap = paired_bootstrap(result["arrays"])
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    sources = [
        T4 / "metadata.json", T4 / "outputs.npz",
        T6 / "metadata.json", T6 / "outputs.npz",
        T9 / "metadata.json", T9 / "outputs.npz",
        Path("research/temperature_t10_early1000_registered.md"),
        Path("research/temperature_t10_early1000.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T10",
        "clock": "10:00:00",
        "selected_probability_heads": SELECTED,
        "fallback_when_unavailable": "T4 probability plus T6 benefit",
        "min_train_date": str(MIN_TRAIN_DATE),
        "tomorrow_cbr_used": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(result["probability_metrics"].to_string(index=False), flush=True)
    print(result["benefit_metrics"].to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
