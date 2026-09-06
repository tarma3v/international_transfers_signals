"""T41: causal quarterly Brier shrink for the T40 history expert."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import temperature_t40_long_rolling_history as t40
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS


OUT = Path("results/research/temperature/t41_mature_quarterly_shrink")
REGISTERED = Path(
    "research/temperature_t41_mature_quarterly_shrink_registered.md")
T40_OUT = t40.OUT
CANDIDATE = "mature_quarterly_history_h20_shadow"
SOURCE = "t40_annual_probability"
WINDOW_BATCHES = 125
MIN_FEEDBACK_BATCHES = 20


def _quarter_origin(day):
    return dt.date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def _load_input():
    frame = pd.read_csv(T40_OUT / "publication_predictions.csv.gz")
    frame["publication_date"] = pd.to_datetime(
        frame.publication_date).dt.date
    frame = frame.rename(columns={t40.CANDIDATE: SOURCE})
    if frame[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("T40 publication keys duplicated")
    return frame.sort_values(
        ["publication_date", "currency"]).reset_index(drop=True)


def _closed_form_alpha(prior, expert, target):
    prior = np.asarray(prior, dtype=float)
    expert = np.asarray(expert, dtype=float)
    target = np.asarray(target, dtype=float)
    direction = expert - prior
    numerator = float(np.sum((target - prior) * direction))
    denominator = float(np.sum(direction ** 2))
    raw = numerator / denominator if denominator > 0.0 else 0.0
    return float(np.clip(raw, 0.0, 1.0)), numerator, denominator, raw


def _quarterly_shrink(source=None):
    frame = _load_input() if source is None else source.copy()
    frame = frame.sort_values(
        ["publication_date", "currency"]).reset_index(drop=True)
    dates = frame.publication_date.to_numpy()
    target = frame.target.to_numpy(dtype=float)
    prior = frame.causal_prior.to_numpy(dtype=float)
    expert = frame[SOURCE].to_numpy(dtype=float)
    if not np.isfinite(prior).all() or not np.isfinite(expert).all():
        raise AssertionError("T41 source probability missing")

    batches = []
    for publication_date, positions in frame.groupby(
            "publication_date", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        finite = (
            np.isfinite(target[positions])
            & np.isfinite(frame.loc[positions, "maturity_ord"].to_numpy(
                dtype=float)))
        if finite.all():
            batches.append({
                "publication_date": pd.Timestamp(publication_date).date(),
                "maturity_ord": int(frame.loc[
                    positions, "maturity_ord"].max()),
                "positions": positions,
            })

    output = np.empty(len(frame), dtype=float)
    alpha_output = np.empty(len(frame), dtype=float)
    state_rows = []
    origins = sorted({_quarter_origin(day) for day in dates})
    for origin in origins:
        cutoff_ord = (origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        eligible = [
            batch for batch in batches
            if batch["publication_date"] < origin
            and batch["maturity_ord"] < cutoff_ord
        ]
        selected = eligible[-WINDOW_BATCHES:]
        if len(selected) < MIN_FEEDBACK_BATCHES:
            alpha, numerator, denominator, raw_alpha = 0.0, 0.0, 0.0, 0.0
            training_positions = np.asarray([], dtype=int)
        else:
            training_positions = np.concatenate(
                [batch["positions"] for batch in selected])
            alpha, numerator, denominator, raw_alpha = _closed_form_alpha(
                prior[training_positions], expert[training_positions],
                target[training_positions])
        quarter_positions = np.flatnonzero(
            np.asarray([_quarter_origin(day) == origin for day in dates]))
        output[quarter_positions] = (
            prior[quarter_positions]
            + alpha * (expert[quarter_positions] - prior[quarter_positions]))
        alpha_output[quarter_positions] = alpha
        state_rows.append({
            "quarter_origin": origin,
            "cutoff_ord": cutoff_ord,
            "eligible_feedback_batches": int(len(eligible)),
            "selected_feedback_batches": int(len(selected)),
            "training_rows": int(len(training_positions)),
            "latest_feedback_publication_date": (
                selected[-1]["publication_date"] if selected else None),
            "latest_feedback_maturity_ord": (
                selected[-1]["maturity_ord"] if selected else None),
            "numerator": numerator,
            "denominator": denominator,
            "raw_alpha": raw_alpha,
            "alpha": alpha,
            "query_rows": int(len(quarter_positions)),
        })
    frame[CANDIDATE] = np.clip(output, 1e-6, 1.0 - 1e-6)
    frame["quarter_alpha"] = alpha_output
    frame["quarter_origin"] = [
        _quarter_origin(day) for day in frame.publication_date]
    return frame, pd.DataFrame(state_rows)


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
    return {
        key: _rename_open(value)
        for key, value in opened.items()
    }


def _source_hashes():
    files = [
        REGISTERED,
        Path("research/temperature_t41_mature_quarterly_shrink.py"),
        Path("research/temperature_t41_mature_quarterly_shrink_audit.py"),
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
    predictions, states = _quarterly_shrink()
    historical, metrics, intervals, gates = _historical_outputs(predictions)
    historical_gate_passed = bool(gates["pass"].all())
    open_outputs = _open_outputs(predictions) if historical_gate_passed else None

    state_mature = states.latest_feedback_maturity_ord.isna() | (
        states.latest_feedback_maturity_ord < states.cutoff_ord)
    checks = {
        "single_candidate_no_grid": True,
        "window_inherited_t33": WINDOW_BATCHES == 125,
        "cold_start_inherited_t34": MIN_FEEDBACK_BATCHES == 20,
        "quarter_state_unique": bool(not states.quarter_origin.duplicated().any()),
        "feedback_embargo_respected": bool(state_mature.all()),
        "alpha_bounded": bool(states.alpha.between(0.0, 1.0).all()),
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
        "packet": "temperature-T41",
        "candidate": CANDIDATE,
        "source": SOURCE,
        "window_feedback_batches": WINDOW_BATCHES,
        "minimum_feedback_batches": MIN_FEEDBACK_BATCHES,
        "historical_screen": "2019-2022",
        "historical_validation": "2023-2024",
        "open_diagnostic": "2025-2026 only after historical pass",
        "quarter_states": int(len(states)),
        "active_quarters": int(states.alpha.gt(0.0).sum()),
        "alpha_min": float(states.alpha.min()),
        "alpha_max": float(states.alpha.max()),
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
        "states": states,
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
        "states", "historical_metrics", "historical_paired_bootstrap",
        "historical_gates",
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
        "active_quarters": result["metadata"]["active_quarters"],
        "alpha_range": [
            result["metadata"]["alpha_min"],
            result["metadata"]["alpha_max"],
        ],
        "historical_gate_passed": result["metadata"]["historical_gate_passed"],
        "open_evaluated": result["metadata"]["open_evaluated"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
