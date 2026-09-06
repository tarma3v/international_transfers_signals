"""T42: label-free training-distribution distance shrink for T40."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from research import temperature_t40_long_rolling_history as t40
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS


OUT = Path("results/research/temperature/t42_ood_history_shrink")
REGISTERED = Path("research/temperature_t42_ood_history_shrink_registered.md")
T40_OUT = t40.OUT
CANDIDATE = "ood_shrunk_annual_history_h20_shadow"
SOURCE = "t40_annual_probability"
ENERGY_FLOOR = 1e-12


def _load_input():
    frame = pd.read_csv(T40_OUT / "publication_predictions.csv.gz")
    frame["publication_date"] = pd.to_datetime(
        frame.publication_date).dt.date
    frame = frame.rename(columns={t40.CANDIDATE: SOURCE})
    if frame[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("T40 publication keys duplicated")
    return frame.sort_values(
        ["publication_date", "currency"]).reset_index(drop=True)


def _training_mask(base, year):
    fit_origin = dt.date(year - 1, 1, 1)
    cutoff_ord = (
        fit_origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    mask = (
        (base["dates"] < fit_origin)
        & np.isfinite(base["target"])
        & np.isfinite(base["maturity"])
        & (base["maturity"] < cutoff_ord)
    )
    return mask, fit_origin, cutoff_ord


def _ood_shrink(source=None, base=None):
    frame = _load_input() if source is None else source.copy()
    frame = frame.sort_values(
        ["publication_date", "currency"]).reset_index(drop=True)
    base = t40._load_base() if base is None else base
    compact = np.asarray(base["compact"], dtype=float)
    source_rows = frame.source_row.to_numpy(dtype=int)
    if np.any(source_rows < 0) or np.any(source_rows >= len(compact)):
        raise AssertionError("T42 source-row outside feature matrix")
    mapped_dates = base["dates"][source_rows]
    mapped_currencies = base["currencies"][source_rows]
    if not np.array_equal(mapped_dates, frame.publication_date.to_numpy()):
        raise AssertionError("T42 publication-date source mapping mismatch")
    if not np.array_equal(mapped_currencies, frame.currency.to_numpy()):
        raise AssertionError("T42 currency source mapping mismatch")

    energy_output = np.full(len(frame), np.nan)
    alpha_output = np.full(len(frame), np.nan)
    candidate_output = np.full(len(frame), np.nan)
    state_rows = []
    for year in t40.YEARS:
        train, fit_origin, cutoff_ord = _training_mask(base, year)
        query = frame.year.to_numpy(dtype=int) == year
        if train.sum() < 1000 or query.sum() < 500:
            raise AssertionError(f"insufficient T42 annual support for {year}")
        scaler = StandardScaler().fit(compact[train])
        query_rows = source_rows[query]
        standardized = scaler.transform(compact[query_rows])
        energy = np.mean(np.square(standardized), axis=1)
        alpha = np.minimum(1.0, 1.0 / np.maximum(energy, ENERGY_FLOOR))
        prior = frame.loc[query, "causal_prior"].to_numpy(dtype=float)
        expert = frame.loc[query, SOURCE].to_numpy(dtype=float)
        candidate = prior + alpha * (expert - prior)
        energy_output[query] = energy
        alpha_output[query] = alpha
        candidate_output[query] = np.clip(
            candidate, t40.EPSILON, 1.0 - t40.EPSILON)
        state_rows.append({
            "year": year,
            "fit_origin": fit_origin,
            "cutoff_ord": cutoff_ord,
            "training_rows": int(train.sum()),
            "query_rows": int(query.sum()),
            "latest_training_date": max(base["dates"][train]),
            "latest_training_maturity_ord": int(
                np.max(base["maturity"][train])),
            "feature_count": int(compact.shape[1]),
            "scaler_mean": json.dumps(
                [float(value) for value in scaler.mean_]),
            "scaler_scale": json.dumps(
                [float(value) for value in scaler.scale_]),
            "energy_mean": float(np.mean(energy)),
            "energy_p95": float(np.quantile(energy, 0.95)),
            "alpha_mean": float(np.mean(alpha)),
            "alpha_min": float(np.min(alpha)),
            "alpha_full_share": float(np.mean(alpha == 1.0)),
        })
    if not np.isfinite(candidate_output).all():
        raise AssertionError("T42 probability missing")
    frame["ood_energy"] = energy_output
    frame["ood_alpha"] = alpha_output
    frame[CANDIDATE] = candidate_output
    return frame, pd.DataFrame(state_rows), base


def _historical_outputs(predictions):
    aliased = predictions.copy()
    aliased[t40.CANDIDATE] = aliased[CANDIDATE]
    historical, metrics = t40._historical_metrics(aliased)
    intervals = t40._historical_bootstrap(historical)
    gates = t40._historical_gates(metrics, intervals)
    return historical, metrics, intervals, gates


def _rename_open(value):
    if not isinstance(value, pd.DataFrame):
        return value
    output = value.copy()
    output.columns = [
        str(column).replace(t40.CANDIDATE, CANDIDATE)
        for column in output.columns]
    if "model" in output:
        output["model"] = output.model.replace({t40.CANDIDATE: CANDIDATE})
    return output


def _open_outputs(predictions):
    aliased = predictions.copy()
    aliased[t40.CANDIDATE] = aliased[CANDIDATE]
    opened = t40._open_outputs(aliased)
    return {key: _rename_open(value) for key, value in opened.items()}


def _source_hashes():
    files = [
        REGISTERED,
        Path("research/temperature_t42_ood_history_shrink.py"),
        Path("research/temperature_t42_ood_history_shrink_audit.py"),
        t40.FEATURE_CACHE,
        t40.LONG_DATA,
        T40_OUT / "metadata.json",
        T40_OUT / "publication_predictions.csv.gz",
        T40_OUT / "annual_fit_log.csv",
        T40_OUT / "historical_gates.csv",
        T40_OUT / "audit_checks.json",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def run():
    predictions, states, _base = _ood_shrink()
    historical, metrics, intervals, gates = _historical_outputs(predictions)
    historical_gate_passed = bool(gates["pass"].all())
    open_outputs = _open_outputs(predictions) if historical_gate_passed else None
    checks = {
        "single_label_free_candidate": True,
        "feature_count_exact_t24": bool(states.feature_count.eq(41).all()),
        "annual_state_unique": bool(not states.year.duplicated().any()),
        "training_embargo_respected": bool((
            states.latest_training_maturity_ord < states.cutoff_ord).all()),
        "energy_positive": bool(predictions.ood_energy.gt(0.0).all()),
        "alpha_bounded": bool(predictions.ood_alpha.between(0.0, 1.0).all()),
        "publication_key_unique": bool(not predictions[[
            "publication_date", "currency"]].duplicated().any()),
        "probability_bounded": bool(predictions[CANDIDATE].between(
            0.0, 1.0).all()),
        "open_not_evaluated_if_historical_failed": bool(
            historical_gate_passed or open_outputs is None),
    }
    if open_outputs is not None:
        checks["non_history_exact_t37"] = bool(
            open_outputs["non_history_exact"])
    metadata = {
        "packet": "temperature-T42",
        "candidate": CANDIDATE,
        "source": SOURCE,
        "distance": "mean squared StandardScaler coordinate",
        "alpha_formula": "min(1, 1 / max(energy, 1e-12))",
        "historical_screen": "2019-2022",
        "historical_validation": "2023-2024",
        "open_diagnostic": "2025-2026 only after historical pass",
        "annual_states": int(len(states)),
        "historical_gate_passed": historical_gate_passed,
        "open_evaluated": open_outputs is not None,
        "open_repair_passed": bool(
            open_outputs and open_outputs["open_repair_passed"]),
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
        "changes_push_policy": False,
        "changes_expected_future_bps": False,
        "changes_runtime_router": False,
        "checks": checks,
        "source_sha256": _source_hashes(),
    }
    result = {
        "publication_predictions": predictions,
        "annual_ood_state": states,
        "historical_metrics": metrics,
        "historical_paired_bootstrap": intervals,
        "historical_gates": gates,
        "metadata": metadata,
    }
    if open_outputs is not None:
        result.update(open_outputs)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["publication_predictions"].to_csv(
        OUT / "publication_predictions.csv.gz", index=False,
        compression="gzip")
    for key in (
        "annual_ood_state", "historical_metrics",
        "historical_paired_bootstrap", "historical_gates",
    ):
        result[key].to_csv(OUT / f"{key}.csv", index=False)
    if result["metadata"]["open_evaluated"]:
        for key in (
            "predictions", "metrics", "reliability", "paired_bootstrap",
            "state_summary", "component_metrics", "pooled_metrics",
            "pooled_paired_bootstrap", "pooled_gate",
            "pairwise_state_metrics", "pairwise_pooled_metrics",
            "clock_local_metrics", "pooled_local_metrics",
            "local_paired_bootstrap", "pooled_local_gates", "failure_summary",
        ):
            suffix = ".csv.gz" if key == "predictions" else ".csv"
            result[key].to_csv(
                OUT / f"{key}{suffix}", index=False,
                compression="gzip" if suffix.endswith(".gz") else None)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "historical_gates": result["historical_gates"].to_dict("records"),
        "state_summary": result["annual_ood_state"][[
            "year", "energy_mean", "energy_p95", "alpha_mean",
            "alpha_min", "alpha_full_share",
        ]].to_dict("records"),
        "historical_gate_passed": result["metadata"][
            "historical_gate_passed"],
        "open_evaluated": result["metadata"]["open_evaluated"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
