"""T11: causal error-regime gate for pre-receipt expected benefit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from ml.validation import target_reach_dates
from research.after_publication_ap51_benefit import magnitude_metrics
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t6_pre_receipt_benefit import states_for_horizon
from research.temperature_t11_causal_benefit_gate_models import (
    HALF_LIFE_DAYS,
    MIN_TRAIN_ROWS,
    WEIGHTS,
    WINDOW_DAYS,
    fit_quarterly_gate,
)


OUT = Path("results/research/temperature/t11_causal_benefit_gate")
T3 = Path("results/research/temperature/t3_phase_calibration")
T4 = Path("results/research/temperature/t4_premarket")
T5 = Path("results/research/temperature/t5_market_grid")
T6 = Path("results/research/temperature/t6_pre_receipt_benefit")
T10 = Path("results/research/temperature/t10_early1000")
BOOTSTRAP_DRAWS = 1000


def source_predictions(h, t3, t4, t5, t6, t10):
    result = {}
    for state in states_for_horizon(h, t3, t4, t5):
        result[state] = (
            t6[f"expected_bps__{state}__h{h}"],
            t6[f"prior_bps__{state}__h{h}"],
        )
    available = t10["available"].astype(bool)
    ten_model = np.where(
        available, t10[f"market_expected_bps_h{h}"],
        t6[f"expected_bps__premarket__h{h}"])
    result["early_1000"] = (
        ten_model, t6[f"prior_bps__premarket__h{h}"])
    return result


def run_experiment():
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    t3, t4, t5, t6, t10 = (
        loadz(path / "outputs.npz") for path in (T3, T4, T5, T6, T10))
    arrays = {"dates": dates, "currencies": currencies}
    metrics, logs = [], []
    for h in HORIZONS:
        target = _forward(series, index, h)
        maturity = target_reach_dates(index, series, h)
        for state, (model, prior) in source_predictions(
                h, t3, t4, t5, t6, t10).items():
            blended, weight, count, head_logs = fit_quarterly_gate(
                model, prior, target, maturity, dates)
            prefix = f"{state}__h{h}"
            arrays[f"model__{prefix}"] = model
            arrays[f"prior__{prefix}"] = prior
            arrays[f"adaptive__{prefix}"] = blended
            arrays[f"weight__{prefix}"] = weight
            arrays[f"n_train__{prefix}"] = count
            logs.extend({"state": state, "h": h, **row}
                        for row in head_logs)
            for period, years in (("screen_2024", (2024,)),
                                  ("opened_2025_2026", (2025, 2026))):
                scope = np.asarray([day.year in years for day in dates])
                for candidate, prediction in (
                        ("model", model), ("prior", prior),
                        ("adaptive", blended)):
                    metrics.append({
                        "state": state, "h": h, "period": period,
                        "candidate": candidate,
                        **magnitude_metrics(
                            target[scope], prediction[scope], prior[scope]),
                    })
    return {
        "arrays": arrays, "metrics": pd.DataFrame(metrics),
        "training_log": pd.DataFrame(logs),
    }


def paired_bootstrap(arrays):
    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("opened_2025_2026", (2025, 2026))):
        scope = np.asarray([day.year in years for day in dates])
        for h in HORIZONS:
            target = _forward(series, index, h)
            states = sorted({key.split("__h")[0].split("__", 1)[1]
                             for key in arrays if key.startswith("model__")
                             and key.endswith(f"__h{h}")})
            for state in states:
                adaptive = arrays[f"adaptive__{state}__h{h}"]
                for control_name in ("model", "prior"):
                    control = arrays[f"{control_name}__{state}__h{h}"]
                    valid = (scope & np.isfinite(target)
                             & np.isfinite(adaptive) & np.isfinite(control))
                    unique = np.asarray(sorted(set(dates[valid])), dtype=object)
                    daily = np.asarray([
                        np.mean(np.abs(adaptive[valid & (dates == day)]
                                       - target[valid & (dates == day)])
                                - np.abs(control[valid & (dates == day)]
                                         - target[valid & (dates == day)]))
                        for day in unique], dtype=float)
                    for block in (20, 50):
                        rng = np.random.default_rng(
                            20260906 + h * 101 + block
                            + (0 if control_name == "prior" else 10000)
                            + sum(ord(char) for char in state))
                        values = []
                        blocks = int(np.ceil(len(daily) / block))
                        offsets = np.arange(block)
                        for _ in range(BOOTSTRAP_DRAWS):
                            starts = rng.integers(0, len(daily), size=blocks)
                            ids = ((starts[:, None] + offsets[None, :])
                                   % len(daily)).ravel()[:len(daily)]
                            values.append(float(np.mean(daily[ids])))
                        values = np.asarray(values)
                        rows.append({
                            "period": period, "state": state, "h": h,
                            "control": control_name, "block_dates": block,
                            "n_dates": len(unique),
                            "mean_delta": float(np.mean(daily)),
                            "ci_low": float(np.quantile(values, .025)),
                            "ci_high": float(np.quantile(values, .975)),
                            "probability_adaptive_better": float(
                                np.mean(values < 0.)),
                        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment()
    np.savez_compressed(OUT / "outputs.npz", **result["arrays"])
    result["metrics"].to_csv(OUT / "benefit_metrics.csv", index=False)
    result["training_log"].to_csv(OUT / "weight_log.csv", index=False)
    bootstrap = paired_bootstrap(result["arrays"])
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    sources = [
        *(path / "metadata.json" for path in (T3, T4, T5, T6, T10)),
        *(path / "outputs.npz" for path in (T6, T10)),
        Path("research/temperature_t11_causal_benefit_gate_registered.md"),
        Path("research/temperature_t11_causal_benefit_gate.py"),
        Path("research/temperature_t11_causal_benefit_gate_models.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T11",
        "states": sorted(result["metrics"].state.unique()),
        "weights": WEIGHTS.tolist(), "window_days": WINDOW_DAYS,
        "half_life_days": HALF_LIFE_DAYS, "min_train_rows": MIN_TRAIN_ROWS,
        "tomorrow_cbr_used": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    frame = result["metrics"]
    pivot = frame.pivot_table(
        index=["period", "state", "h"], columns="candidate",
        values="mae_model").reset_index()
    print(pivot.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
