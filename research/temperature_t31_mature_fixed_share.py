"""T31: mature-only fixed-share mixture of frozen T30 regime experts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS, _probability_metrics
from research.temperature_t21_h20_curve_head import _fast_auc
from research.temperature_t26_delayed_base_rate import _paired_intervals


T30_OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")
OUT = Path("results/research/temperature/t31_mature_fixed_share")
REGISTERED = Path("research/temperature_t31_mature_fixed_share_registered.md")
SCREEN_START = dt.date(2023, 1, 1)
VALIDATION_START = dt.date(2024, 1, 1)
EVALUATION_START = dt.date(2025, 1, 1)
EXPERTS = ("identity_early", "all_platt_b050", "recent2y_platt_b100")
ETAS = (0.25, 0.50, 1.00, 2.00)
GAMMAS = (0.00, 0.01, 0.05, 0.10)


def _candidate_name(eta, gamma):
    return f"hedge_e{int(eta * 100):03d}_g{int(gamma * 100):03d}"


SPECS = {
    _candidate_name(eta, gamma): {"eta": eta, "gamma": gamma}
    for eta in ETAS for gamma in GAMMAS
}
CANDIDATES = tuple(SPECS)
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}
MODELS = ("identity_early", "t25_base", "t31_candidate", "t31_selected", *CANDIDATES)


def _bernoulli_loss(target, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    target = np.asarray(target, dtype=float)
    return -(target[:, None] * np.log(probability)
             + (1.0 - target[:, None]) * np.log(1.0 - probability))


def _online_candidates(frame, return_states=True):
    frame = frame.sort_values(["query_date", "currency"]).reset_index(drop=True)
    dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    expert_values = frame[list(EXPERTS)].to_numpy(dtype=float)
    if not np.isfinite(expert_values).all():
        raise AssertionError("expert probability missing")
    if not ((expert_values > 0.0) & (expert_values < 1.0)).all():
        raise AssertionError("expert probability outside open unit interval")

    batches = []
    for publication_date, positions in frame.groupby("query_date", sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        batch_maturity = int(frame.loc[positions, "maturity_ord"].max())
        losses = _bernoulli_loss(
            frame.loc[positions, "target"].to_numpy(dtype=float),
            expert_values[positions],
        ).mean(axis=0)
        batches.append({
            "publication_date": pd.Timestamp(publication_date).date(),
            "maturity_ord": batch_maturity,
            "rows": int(len(positions)),
            "losses": losses,
        })
    batches.sort(key=lambda row: (row["maturity_ord"], row["publication_date"]))

    weights = {name: np.full(len(EXPERTS), 1.0 / len(EXPERTS)) for name in CANDIDATES}
    outputs = {name: np.empty(len(frame), dtype=float) for name in CANDIDATES}
    states = []
    consumed = 0
    latest_maturity = None
    latest_publication = None
    for query_date in sorted(set(dates)):
        cutoff = (query_date - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        newly_consumed = []
        while consumed < len(batches):
            batch = batches[consumed]
            if not (batch["maturity_ord"] < cutoff
                    and batch["publication_date"] < query_date):
                break
            newly_consumed.append(batch)
            consumed += 1
        for batch in newly_consumed:
            for name, spec in SPECS.items():
                updated = weights[name] * np.exp(-spec["eta"] * batch["losses"])
                total = float(updated.sum())
                if not np.isfinite(total) or total <= 0.0:
                    raise AssertionError("online weights collapsed")
                updated /= total
                gamma = spec["gamma"]
                weights[name] = (1.0 - gamma) * updated + gamma / len(EXPERTS)
            latest_maturity = batch["maturity_ord"]
            latest_publication = batch["publication_date"]
        positions = np.flatnonzero(dates == query_date)
        for name in CANDIDATES:
            current = weights[name]
            outputs[name][positions] = expert_values[positions] @ current
            if return_states:
                states.append({
                    "query_date": query_date,
                    "candidate": name,
                    "eta": SPECS[name]["eta"],
                    "gamma": SPECS[name]["gamma"],
                    "cutoff_ord": cutoff,
                    "new_feedback_batches": int(len(newly_consumed)),
                    "consumed_feedback_batches": int(consumed),
                    "latest_feedback_publication_date": latest_publication,
                    "latest_feedback_maturity_ord": latest_maturity,
                    "weight_identity": float(current[0]),
                    "weight_all": float(current[1]),
                    "weight_recent2y": float(current[2]),
                    "weight_sum": float(current.sum()),
                    "query_rows": int(len(positions)),
                })
    return frame, outputs, pd.DataFrame(states), {
        "feedback_batches": int(len(batches)),
        "consumed_by_last_query": int(consumed),
    }


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
    frame, candidates, states, feedback = _online_candidates(_load_input())
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
    frame["t31_candidate"] = (
        frame.identity_early if screen_selected == "identity_early"
        else frame[screen_selected])
    frame["t31_selected"] = (
        frame.identity_early if final_model == "identity_early"
        else frame[final_model])

    evaluated = frame[evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "t31_selected", "identity_early", "identity"),
        _paired_intervals(evaluated, "t31_selected", "t25_base", "t25"),
    ], ignore_index=True)
    primary = metrics[(metrics.model == "t31_selected") & (metrics.slice == "ALL")].iloc[0]
    brier_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "brier")]
    brier_t25 = intervals[(intervals.comparison == "t25") & (intervals.metric == "brier")]
    auc_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "auc")]
    passed = bool(
        validation_passed and final_model != "identity_early"
        and primary.brier_delta < 0.0 and primary.logloss_delta < 0.0
        and primary.ece_delta_identity <= 0.005 and primary.auc_delta_identity >= 0.02
        and primary.brier_delta_t25 < 0.0 and primary.logloss_delta_t25 < 0.0
        and primary.ece_delta_t25 <= 0.005 and primary.auc_delta_t25 >= -0.005
        and (brier_identity.ci_high < 0.0).all()
        and (brier_t25.ci_high < 0.0).all()
        and (auc_identity.ci_low > 0.0).all())

    used_states = states[states.latest_feedback_maturity_ord.notna()]
    source_files = [
        REGISTERED,
        Path("research/temperature_t31_mature_fixed_share.py"),
        T30_OUT / "metadata.json",
        T30_OUT / "development_predictions.csv.gz",
        T30_OUT / "predictions.csv.gz",
    ]
    metadata = {
        "packet": "temperature-T31",
        "experts": list(EXPERTS),
        "specs": SPECS,
        "screen_selected": screen_selected,
        "validation_passed": validation_passed,
        "selected_model": final_model,
        "passed": passed,
        "selection_on_open_2025_2026": False,
        "pre2025_previously_inspected": True,
        "fresh_independent_holdout": False,
        "feedback": feedback,
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
            "latest_validation_maturity_ord": int(frame.loc[validation, "maturity_ord"].max()),
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
            "feedback_before_query": bool((
                pd.to_datetime(used_states.latest_feedback_publication_date).dt.date
                < pd.to_datetime(used_states.query_date).dt.date).all()),
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
            *MODELS]],
        "development_predictions": frame[~evaluation][[
            "query_date", "year", "currency", "target", "maturity_ord",
            "identity_early", "t31_candidate", "t31_selected", *CANDIDATES]],
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
    result["states"].to_csv(OUT / "states.csv.gz", index=False, compression="gzip")
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
