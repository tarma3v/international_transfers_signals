"""T32: mature-only Hedge with a disjoint 2022 OOS warm-up stream."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS
from research.temperature_t21_h20_curve_head import _clip, _logit
from research.temperature_t24_history_h20_anchor import _maturity
from research.temperature_t26_delayed_base_rate import _paired_intervals
from research.temperature_t30_presvo_rank_postsvo_map import (
    _fit_positive_platt,
    _fit_rank_models,
)
from research.temperature_t31_mature_fixed_share import (
    CANDIDATES,
    EXPERTS,
    MODELS as T31_MODELS,
    _candidate_metrics,
    _choose,
    _metrics as _t31_metrics,
    _online_candidates,
)
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    SEED,
    compact_features,
)


OUT = Path("results/research/temperature/t32_oos_warm_hedge")
REGISTERED = Path("research/temperature_t32_oos_warm_hedge_registered.md")
T30_OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")
MAP_START = dt.date(2022, 4, 1)
MAP_END = dt.date(2022, 8, 1)
WARM_START = dt.date(2022, 10, 1)
SCREEN_START = dt.date(2023, 1, 1)
VALIDATION_START = dt.date(2024, 1, 1)
EVALUATION_START = dt.date(2025, 1, 1)
MODELS = tuple(
    "t32_candidate" if name == "t31_candidate" else
    "t32_selected" if name == "t31_selected" else name
    for name in T31_MODELS
)


def _fit_q4_identity_anchor(X, names, dates, target, maturity):
    features = compact_features(X, names)
    origin = WARM_START
    finite = np.all(np.isfinite(features), axis=1)
    cutoff = (origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    train = (
        (dates >= MIN_TRAIN_DATE) & (dates < origin)
        & np.isfinite(target) & np.isfinite(maturity) & (maturity < cutoff)
        & finite
    )
    query = (dates >= origin) & (dates < SCREEN_START) & finite
    if train.sum() < 500 or query.sum() < 200:
        raise AssertionError("insufficient causal Q4 identity anchor rows")
    model = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=.035, max_leaf_nodes=9,
        min_samples_leaf=42, l2_regularization=15., random_state=SEED)
    ids = np.flatnonzero(train)
    age = np.asarray([(origin - dates[i]).days for i in ids], dtype=float)
    sample_weight = np.exp2(-age / 730.)
    model.fit(features[train], target[train].astype(int),
              sample_weight=sample_weight)
    output = np.full(len(dates), np.nan)
    output[query] = model.predict_proba(features[query])[:, 1]
    log = {
        "origin": str(origin),
        "end_exclusive": str(SCREEN_START),
        "cutoff_ord": cutoff,
        "training_rows": int(train.sum()),
        "query_rows": int(query.sum()),
        "latest_training_date": str(max(dates[train])),
        "latest_training_maturity_ord": int(np.max(maturity[train])),
    }
    return output, log


def _build_input():
    X, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    raw_models, fit_log, model_details = _fit_rank_models(
        X, names, dates, target, maturity)

    mapping_cutoff = (WARM_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    mapping_base = (
        (dates >= MAP_START) & (dates < MAP_END)
        & np.isfinite(target) & np.isfinite(maturity)
        & (maturity < mapping_cutoff)
    )
    experts = {}
    map_rows = []
    for family, beta, name in (
        ("all", 0.50, "all_platt_b050"),
        ("recent2y", 1.00, "recent2y_platt_b100"),
    ):
        raw = np.asarray(raw_models[family], dtype=float)
        usable = mapping_base & np.isfinite(raw)
        if usable.sum() < 200:
            raise AssertionError(f"insufficient disjoint mapping rows: {family}")
        intercept, slope, objective = _fit_positive_platt(raw[usable], target[usable])
        platt = _clip(expit(intercept + slope * _logit(raw)))
        experts[name] = _clip(expit(
            (1.0 - beta) * _logit(raw) + beta * _logit(platt)))
        map_rows.append({
            "family": family,
            "expert": name,
            "beta": beta,
            "map_start": MAP_START,
            "map_end_exclusive": MAP_END,
            "cutoff_ord": mapping_cutoff,
            "mapping_rows": int(usable.sum()),
            "latest_mapping_date": max(dates[usable]),
            "latest_mapping_maturity_ord": int(np.max(maturity[usable])),
            "intercept": intercept,
            "slope": slope,
            "objective": objective,
        })

    with np.load(
            "results/research/temperature/t4_premarket/outputs.npz",
            allow_pickle=True) as archive:
        if not np.array_equal(archive["dates"], dates):
            raise AssertionError("T4 dates no longer align")
        if not np.array_equal(
                archive["currencies"].astype(str), currencies.astype(str)):
            raise AssertionError("T4 currencies no longer align")
        identity = archive["prob__history_hist__h20"].astype(float)
    warm_identity, warm_anchor_log = _fit_q4_identity_anchor(
        X, names, dates, target, maturity)
    warm_rows = (dates >= WARM_START) & (dates < SCREEN_START)
    identity[warm_rows] = warm_identity[warm_rows]

    frame = pd.DataFrame({
        "query_date": dates,
        "year": [day.year for day in dates],
        "currency": currencies,
        "target": target,
        "maturity_ord": maturity,
        "identity_early": identity,
        **experts,
    })
    valid = (
        (dates >= WARM_START) & np.isfinite(target) & np.isfinite(maturity)
        & np.all(np.isfinite(frame[list(EXPERTS)].to_numpy()), axis=1)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    t30_open = pd.read_csv(T30_OUT / "predictions.csv.gz")[[
        "query_date", "currency", "t25_base"]]
    t30_open["query_date"] = pd.to_datetime(t30_open.query_date).dt.date
    frame = frame.merge(
        t30_open, on=["query_date", "currency"], how="left",
        validate="one_to_one")
    frame["t25_base"] = frame.t25_base.fillna(frame.identity_early)
    return frame, fit_log, pd.DataFrame(map_rows), model_details, warm_anchor_log


def _metrics(frame):
    renamed = frame.rename(columns={
        "t32_candidate": "t31_candidate",
        "t32_selected": "t31_selected",
    })
    result = _t31_metrics(renamed)
    result["model"] = result.model.replace({
        "t31_candidate": "t32_candidate",
        "t31_selected": "t32_selected",
    })
    return result


def run():
    frame, fit_log, map_log, model_details, warm_anchor_log = _build_input()
    frame, candidates, states, feedback = _online_candidates(frame)
    for name, values in candidates.items():
        frame[name] = values

    dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    screen_cutoff = (VALIDATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (EVALUATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    warmup = (dates >= WARM_START) & (dates < SCREEN_START)
    screen = (
        (dates >= SCREEN_START) & (dates < VALIDATION_START)
        & (frame.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (dates >= VALIDATION_START) & (dates < EVALUATION_START)
        & (frame.maturity_ord.to_numpy() < validation_cutoff))
    evaluation = dates >= EVALUATION_START
    if min(warmup.sum(), screen.sum(), validation.sum(), evaluation.sum()) < 200:
        raise AssertionError("chronological split unexpectedly small")

    screen_rows = _candidate_metrics(frame, screen)
    screen_selected = _choose(screen_rows)
    validation_rows = _candidate_metrics(frame, validation)
    validation_lookup = {row["model"]: row for row in validation_rows}
    validation_passed = bool(
        screen_selected != "identity_early"
        and validation_lookup[screen_selected]["feasible"])
    final_model = screen_selected if validation_passed else "identity_early"
    frame["t32_candidate"] = (
        frame.identity_early if screen_selected == "identity_early"
        else frame[screen_selected])
    frame["t32_selected"] = (
        frame.identity_early if final_model == "identity_early"
        else frame[final_model])

    evaluated = frame[evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "t32_selected", "identity_early", "identity"),
        _paired_intervals(evaluated, "t32_selected", "t25_base", "t25"),
    ], ignore_index=True)
    primary = metrics[
        (metrics.model == "t32_selected") & metrics.slice.eq("ALL")].iloc[0]
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
        Path("research/temperature_t32_oos_warm_hedge.py"),
        Path("research/temperature_t30_presvo_rank_postsvo_map.py"),
        Path("research/temperature_t31_mature_fixed_share.py"),
        Path("research/temperature_t24_history_h20_anchor.py"),
        Path("research/temperature_t4_premarket_models.py"),
        Path("results/research/temperature/t4_premarket/outputs.npz"),
        T30_OUT / "predictions.csv.gz",
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    metadata = {
        "packet": "temperature-T32",
        "experts": list(EXPERTS),
        "screen_selected": screen_selected,
        "validation_passed": validation_passed,
        "selected_model": final_model,
        "passed": passed,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "feedback": feedback,
        "splits": {
            "rank_origin": "2022-01-01",
            "mapping": "2022-04-01 to mature rows before 2022-08-01",
            "identity_anchor_fit": "2022-10-01",
            "warmup": "2022-10-01 to 2022-12-31",
            "screen": "2023",
            "validation": "2024",
            "evaluation": "2025-2026",
            "mapping_rows": int(map_log.mapping_rows.sum()),
            "warmup_rows": int(warmup.sum()),
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
        "model_details": model_details,
        "warm_anchor_log": warm_anchor_log,
        "checks": {
            "mapping_warmup_disjoint": True,
            "q4_identity_anchor_mature": bool(
                warm_anchor_log["latest_training_maturity_ord"]
                < warm_anchor_log["cutoff_ord"]),
            "mapping_mature_before_warmup": bool((
                map_log.latest_mapping_maturity_ord < map_log.cutoff_ord).all()),
            "rank_fit_causal": bool((
                fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all()),
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
            *MODELS, *EXPERTS[1:]]],
        "development_predictions": frame[screen | validation][[
            "query_date", "year", "currency", "target", "maturity_ord",
            "identity_early", "t32_candidate", "t32_selected", *CANDIDATES,
            *EXPERTS[1:]]],
        "warmup_predictions": frame[warmup][[
            "query_date", "year", "currency", "target", "maturity_ord",
            *EXPERTS, *CANDIDATES]],
        "states": states,
        "fit_log": fit_log,
        "map_log": map_log,
        "warm_anchor_log": pd.DataFrame([warm_anchor_log]),
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
    for name in ("predictions", "development_predictions", "warmup_predictions"):
        result[name].to_csv(OUT / f"{name}.csv.gz", index=False, compression="gzip")
    result["states"].to_csv(OUT / "states.csv.gz", index=False, compression="gzip")
    result["fit_log"].to_csv(OUT / "fit_log.csv", index=False)
    result["map_log"].to_csv(OUT / "map_log.csv", index=False)
    result["warm_anchor_log"].to_csv(
        OUT / "warm_anchor_log.csv", index=False)
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
