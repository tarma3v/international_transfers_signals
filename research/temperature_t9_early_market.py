"""T9: early-session market versus history-only premarket probabilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import probability_metrics
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
    fit_quarterly_hist,
)
from research.temperature_t9_early_market_models import (
    CLOCKS,
    build_early_market_features,
    clock_name,
    physical_causality_check,
)


OUT = Path("results/research/temperature/t9_early_market")
DATA = Path("data/moex_spot_fx_10min_2022_2026.json")
T4 = Path("results/research/temperature/t4_premarket/outputs.npz")


def run_experiment():
    matrix, names, index, series, *_ = load_round5_features()
    base = compact_features(matrix, names)
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    t4 = np.load(T4, allow_pickle=True)
    arrays = {"dates": dates, "currencies": currencies}
    metrics, model_logs, calibration_logs, availability_rows = [], [], [], []
    for clock in CLOCKS:
        physical_causality_check(index, history, references, clock)
        state, state_names, available, source_at = build_early_market_features(
            index, history, references, clock)
        candidate = clock_name(clock)
        arrays[f"available__{candidate}"] = available
        arrays[f"source_at__{candidate}"] = source_at
        arrays[f"market__{candidate}"] = state
        raw_cny = state[:, state_names.index("cnyrub_tom_mean_cbr_basis")]
        cny_rank = causal_percentiles(raw_cny, dates, currencies, 250, 20)
        arrays[f"rank__cny__{candidate}"] = cny_rank
        hybrid_features = np.column_stack((base, state))
        for period, years in (("screen_2024", (2024,)),
                              ("opened_2025_2026", (2025, 2026))):
            scope = np.asarray([day.year in years for day in dates])
            availability_rows.append({
                "clock": candidate, "period": period,
                "cny_candle_share": float(available[scope].mean()),
            })
        for h in HORIZONS:
            target = targets[f"fav_h{h}"]
            maturity = target_reach_dates(index, series, h)
            hybrid_raw, train_count, logs = fit_quarterly_hist(
                hybrid_features, target, maturity, dates)
            arrays[f"raw__hybrid__{candidate}__h{h}"] = hybrid_raw
            arrays[f"n_train__hybrid__{candidate}__h{h}"] = train_count
            model_logs.extend({"clock": candidate, "h": h, **row}
                              for row in logs)
            candidates = {
                "history_hist": t4[f"model_raw_probability_{h}"],
                "cny_rank": cny_rank,
                "hybrid_hgb": hybrid_raw,
            }
            for model_name, raw in candidates.items():
                calibrated, prior, count, logs = fit_quarterly_calibrator(
                    raw, target, maturity, dates, currencies,
                    min_train_date=MIN_TRAIN_DATE)
                prefix = f"{model_name}__{candidate}__h{h}"
                arrays[f"prob__{prefix}"] = calibrated
                arrays[f"prior__{prefix}"] = prior
                arrays[f"cal_n_train__{prefix}"] = count
                calibration_logs.extend({
                    "clock": candidate, "candidate": model_name, "h": h,
                    **row,
                } for row in logs)
                for period, years in (("screen_2024", (2024,)),
                                      ("opened_2025_2026", (2025, 2026))):
                    scope = np.asarray([day.year in years for day in dates])
                    metrics.append({
                        "clock": candidate, "candidate": model_name,
                        "h": h, "period": period,
                        **probability_metrics(
                            target[scope], raw[scope], calibrated[scope],
                            prior[scope]),
                    })
    return {
        "arrays": arrays, "metrics": pd.DataFrame(metrics),
        "model_logs": pd.DataFrame(model_logs),
        "calibration_logs": pd.DataFrame(calibration_logs),
        "availability": pd.DataFrame(availability_rows),
        "digest": digest,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    result["metrics"].to_csv(OUT / "calibration_metrics.csv", index=False)
    result["model_logs"].to_csv(OUT / "model_training_log.csv", index=False)
    result["calibration_logs"].to_csv(
        OUT / "calibration_log.csv", index=False)
    result["availability"].to_csv(OUT / "availability.csv", index=False)
    np.savez_compressed(OUT / "outputs.npz", **result["arrays"])
    sources = [
        DATA, T4,
        Path("research/temperature_t9_early_market_registered.md"),
        Path("research/temperature_t9_early_market.py"),
        Path("research/temperature_t9_early_market_models.py"),
    ]
    metadata = {
        "packet": "temperature-T9",
        "clocks": [clock.isoformat() for clock in CLOCKS],
        "market_feature_count": 32,
        "min_train_date": str(MIN_TRAIN_DATE),
        "tomorrow_cbr_used": False,
        "historical_exchange_availability_certified": False,
        "payload_sha256": result["digest"],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    opened = result["metrics"][
        result["metrics"].period == "opened_2025_2026"]
    summary = opened.groupby(["clock", "candidate"]).agg(
        mean_brier=("brier_calibrated", "mean"),
        max_brier=("brier_calibrated", "max"),
        mean_auc=("auc_calibrated", "mean"),
        min_auc=("auc_calibrated", "min"),
        mean_ece=("ece_calibrated", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "opened_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(result["availability"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
