"""T33: quarterly frozen convex stacking of the T30 h20 experts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.temperature_t19_anytime_quality_audit import (
    EMBARGO_DAYS,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc
from research.temperature_t26_delayed_base_rate import _paired_intervals
from research.temperature_t31_mature_fixed_share import EXPERTS


T30_OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")
OUT = Path("results/research/temperature/t33_quarterly_stacking")
REGISTERED = Path("research/temperature_t33_quarterly_stacking_registered.md")
SCREEN_START = dt.date(2023, 1, 1)
VALIDATION_START = dt.date(2024, 1, 1)
EVALUATION_START = dt.date(2025, 1, 1)
WINDOWS = (60, 125, 250, None)
RIDGES = (0.00, 0.01, 0.10, 1.00)


def _candidate_name(window, ridge):
    window_name = "exp" if window is None else f"{window:03d}"
    return f"qstack_w{window_name}_r{int(ridge * 100):03d}"


SPECS = {
    _candidate_name(window, ridge): {"window": window, "ridge": ridge}
    for window in WINDOWS for ridge in RIDGES
}
CANDIDATES = tuple(SPECS)
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}
MODELS = (
    "identity_early", "t25_base", "t33_candidate", "t33_selected",
    *CANDIDATES,
)


def _quarter_origin(day):
    return dt.date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def _fit_simplex(probabilities, target, ridge):
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.asarray(target, dtype=float)
    uniform = np.full(probabilities.shape[1], 1.0 / probabilities.shape[1])

    def objective(weight):
        prediction = np.clip(probabilities @ weight, 1e-9, 1 - 1e-9)
        loss = -np.mean(
            target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction))
        return float(loss + ridge * np.sum((weight - uniform) ** 2))

    result = minimize(
        objective, uniform, method="SLSQP", bounds=[(0.0, 1.0)] * len(uniform),
        constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"quarterly simplex fit failed: {result.message}")
    weight = np.clip(result.x, 0.0, 1.0)
    weight /= weight.sum()
    return weight, float(result.fun)


def _quarterly_candidates(frame, return_states=True):
    frame = frame.sort_values(["query_date", "currency"]).reset_index(drop=True)
    dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    expert_values = frame[list(EXPERTS)].to_numpy(dtype=float)
    target = frame.target.to_numpy(dtype=float)
    if not np.isfinite(expert_values).all():
        raise AssertionError("expert probability missing")
    if not ((expert_values > 0.0) & (expert_values < 1.0)).all():
        raise AssertionError("expert probability outside open unit interval")

    publication_batches = []
    for publication_date, positions in frame.groupby("query_date", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        publication_batches.append({
            "publication_date": pd.Timestamp(publication_date).date(),
            "maturity_ord": int(frame.loc[positions, "maturity_ord"].max()),
            "positions": positions,
        })

    outputs = {name: np.empty(len(frame), dtype=float) for name in CANDIDATES}
    states = []
    origins = sorted({_quarter_origin(day) for day in dates})
    for origin in origins:
        cutoff_ord = (origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        eligible = [
            batch for batch in publication_batches
            if batch["publication_date"] < origin
            and batch["maturity_ord"] < cutoff_ord
        ]
        quarter_positions = np.flatnonzero(
            np.asarray([_quarter_origin(day) == origin for day in dates]))
        for name, spec in SPECS.items():
            selected = eligible[-spec["window"]:] if spec["window"] else eligible
            if len(selected) < 20:
                weight = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
                objective = np.nan
                training_positions = np.asarray([], dtype=int)
            else:
                training_positions = np.concatenate(
                    [batch["positions"] for batch in selected])
                weight, objective = _fit_simplex(
                    expert_values[training_positions], target[training_positions],
                    spec["ridge"])
            outputs[name][quarter_positions] = expert_values[quarter_positions] @ weight
            if return_states:
                states.append({
                    "quarter_origin": origin,
                    "candidate": name,
                    "window": "expanding" if spec["window"] is None else spec["window"],
                    "ridge": spec["ridge"],
                    "cutoff_ord": cutoff_ord,
                    "eligible_feedback_batches": int(len(eligible)),
                    "selected_feedback_batches": int(len(selected)),
                    "training_rows": int(len(training_positions)),
                    "latest_feedback_publication_date": (
                        selected[-1]["publication_date"] if selected else None),
                    "latest_feedback_maturity_ord": (
                        selected[-1]["maturity_ord"] if selected else None),
                    "weight_identity": float(weight[0]),
                    "weight_all": float(weight[1]),
                    "weight_recent2y": float(weight[2]),
                    "weight_sum": float(weight.sum()),
                    "objective": objective,
                    "query_rows": int(len(quarter_positions)),
                })
    return frame, outputs, pd.DataFrame(states)


def _candidate_metrics(frame, mask):
    base_auc = _fast_auc(frame.loc[mask, "target"], frame.loc[mask, "identity_early"])
    rows = []
    for name in CANDIDATES:
        result = _probability_metrics(
            frame.loc[mask, "target"], frame.loc[mask, name],
            frame.loc[mask, "identity_early"])
        result["model"] = name
        result["auc_delta"] = result["auc"] - base_auc
        result["ece_delta"] = result["ece"] - result["baseline_ece"]
        result["feasible"] = bool(
            result["auc_delta"] >= 0.02
            and result["brier_delta"] < 0.0
            and result["logloss_delta"] < 0.0
            and result["ece_delta"] <= 0.005)
        rows.append(result)
    return rows


def _choose(rows):
    feasible = [row for row in rows if row["feasible"]]
    return (sorted(feasible, key=lambda row: (
        row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "identity_early")


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def _metrics(frame):
    rows = []
    for name in MODELS:
        for slice_name, group, part in _slices(frame):
            identity = _probability_metrics(part.target, part[name], part.identity_early)
            t25 = _probability_metrics(part.target, part[name], part.t25_base)
            rows.append({
                "model": name, "slice": slice_name, "group": group,
                **identity,
                "auc_delta_identity": identity["auc"] - _fast_auc(
                    part.target, part.identity_early),
                "ece_delta_identity": identity["ece"] - identity["baseline_ece"],
                "brier_delta_t25": t25["brier_delta"],
                "logloss_delta_t25": t25["logloss_delta"],
                "auc_delta_t25": t25["auc"] - _fast_auc(part.target, part.t25_base),
                "ece_delta_t25": t25["ece"] - t25["baseline_ece"],
            })
    return pd.DataFrame(rows)


def _load_input():
    development = pd.read_csv(T30_OUT / "development_predictions.csv.gz")
    evaluation = pd.read_csv(T30_OUT / "predictions.csv.gz")
    development["t25_base"] = development.identity_early
    columns = [
        "query_date", "year", "currency", "target", "maturity_ord",
        "identity_early", "t25_base", "all_platt_b050", "recent2y_platt_b100",
    ]
    frame = pd.concat([development[columns], evaluation[columns]], ignore_index=True)
    frame["query_date"] = pd.to_datetime(frame.query_date).dt.date
    if frame[["query_date", "currency"]].duplicated().any():
        raise AssertionError("T30 publication keys duplicated")
    return frame


def run():
    t30_meta = json.loads((T30_OUT / "metadata.json").read_text())
    for name, expected in t30_meta["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"T30 source hash mismatch: {name}")
    frame, candidates, states = _quarterly_candidates(_load_input())
    for name, values in candidates.items():
        frame[name] = values

    dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    screen_cutoff = (VALIDATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (EVALUATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        (dates >= SCREEN_START) & (dates < VALIDATION_START)
        & (frame.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (dates >= VALIDATION_START) & (dates < EVALUATION_START)
        & (frame.maturity_ord.to_numpy() < validation_cutoff))
    evaluation = dates >= EVALUATION_START
    screen_rows = _candidate_metrics(frame, screen)
    screen_selected = _choose(screen_rows)
    validation_rows = _candidate_metrics(frame, validation)
    validation_lookup = {row["model"]: row for row in validation_rows}
    validation_passed = bool(
        screen_selected != "identity_early"
        and validation_lookup[screen_selected]["feasible"])
    final_model = screen_selected if validation_passed else "identity_early"
    frame["t33_candidate"] = (
        frame.identity_early if screen_selected == "identity_early"
        else frame[screen_selected])
    frame["t33_selected"] = (
        frame.identity_early if final_model == "identity_early"
        else frame[final_model])

    evaluated = frame[evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "t33_selected", "identity_early", "identity"),
        _paired_intervals(evaluated, "t33_selected", "t25_base", "t25"),
    ], ignore_index=True)
    primary = metrics[
        (metrics.model == "t33_selected") & metrics.slice.eq("ALL")].iloc[0]
    brier_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "brier")]
    brier_t25 = intervals[
        (intervals.comparison == "t25") & (intervals.metric == "brier")]
    auc_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "auc")]
    passed = bool(
        validation_passed and final_model != "identity_early"
        and primary.brier_delta < 0.0 and primary.logloss_delta < 0.0
        and primary.ece_delta_identity <= 0.005
        and primary.auc_delta_identity >= 0.02
        and primary.brier_delta_t25 < 0.0
        and primary.logloss_delta_t25 < 0.0
        and primary.ece_delta_t25 <= 0.005
        and primary.auc_delta_t25 >= -0.005
        and (brier_identity.ci_high < 0.0).all()
        and (brier_t25.ci_high < 0.0).all()
        and (auc_identity.ci_low > 0.0).all())

    used_states = states[states.latest_feedback_maturity_ord.notna()]
    source_files = [
        REGISTERED,
        Path("research/temperature_t33_quarterly_stacking.py"),
        Path("research/temperature_t30_presvo_rank_postsvo_map.py"),
        T30_OUT / "metadata.json",
        T30_OUT / "development_predictions.csv.gz",
        T30_OUT / "predictions.csv.gz",
    ]
    metadata = {
        "packet": "temperature-T33",
        "experts": list(EXPERTS),
        "specs": SPECS,
        "screen_selected": screen_selected,
        "validation_passed": validation_passed,
        "selected_model": final_model,
        "passed": passed,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "splits": {
            "screen": "2023",
            "validation": "2024",
            "evaluation": "2025-2026",
            "screen_rows": int(screen.sum()),
            "validation_rows": int(validation.sum()),
            "evaluation_rows": int(len(evaluated)),
            "evaluation_dates": int(evaluated.query_date.nunique()),
            "screen_cutoff_ord": screen_cutoff,
            "validation_cutoff_ord": validation_cutoff,
            "latest_screen_maturity_ord": int(frame.loc[screen, "maturity_ord"].max()),
            "latest_validation_maturity_ord": int(
                frame.loc[validation, "maturity_ord"].max()),
        },
        "screen": screen_rows,
        "validation": validation_rows,
        "checks": {
            "t30_source_hashes_match": True,
            "candidate_grid_complete": len(CANDIDATES) == 16,
            "screen_validation_disjoint": bool(not (screen & validation).any()),
            "screen_mature": bool(
                frame.loc[screen, "maturity_ord"].max() < screen_cutoff),
            "validation_mature": bool(
                frame.loc[validation, "maturity_ord"].max() < validation_cutoff),
            "feedback_before_origin": bool((
                pd.to_datetime(used_states.latest_feedback_publication_date).dt.date
                < pd.to_datetime(used_states.quarter_origin).dt.date).all()),
            "feedback_mature_before_cutoff": bool((
                used_states.latest_feedback_maturity_ord < used_states.cutoff_ord).all()),
            "weights_sum_to_one": bool(np.allclose(states.weight_sum, 1.0)),
            "weights_nonnegative": bool((states[[
                "weight_identity", "weight_all", "weight_recent2y"]] >= 0.0).all().all()),
            "publication_keys_unique": bool(not frame[
                ["query_date", "currency"]].duplicated().any()),
            "changes_push_policy": False,
            "historical_receipts_certified": False,
            "bank_execution_validated": False,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "predictions": evaluated[[
            "query_date", "year", "currency", "target", "maturity_ord",
            *MODELS, *EXPERTS[1:]]],
        "development_predictions": frame[~evaluation][[
            "query_date", "year", "currency", "target", "maturity_ord",
            "identity_early", "t33_candidate", "t33_selected", *CANDIDATES,
            *EXPERTS[1:]]],
        "states": states,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "screen_validation": pd.DataFrame([
            {"stage": "screen", **row} for row in screen_rows
        ] + [{"stage": "validation", **row} for row in validation_rows]),
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["development_predictions"].to_csv(
        OUT / "development_predictions.csv.gz", index=False, compression="gzip")
    result["states"].to_csv(OUT / "states.csv", index=False)
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["screen_validation"].to_csv(OUT / "screen_validation.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "screen_selected": result["metadata"]["screen_selected"],
        "validation_passed": result["metadata"]["validation_passed"],
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "screen": result["metadata"]["screen"],
        "validation": result["metadata"]["validation"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "auc_delta_identity",
            "brier_delta_t25", "logloss_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
