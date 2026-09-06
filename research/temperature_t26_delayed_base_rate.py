"""T26: causal delayed base-rate correction for the T25 premarket h20 rank."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    BOOTSTRAP_DRAWS,
    EMBARGO_DAYS,
    _load_snapshots,
    _probability_metrics,
    _query_grid,
    _selected_queries,
    _target_arrays,
)
from research.temperature_t21_h20_curve_head import _clip, _fast_auc, _feature_frame, _logit
from research.temperature_t24_history_h20_anchor import (
    CAL_SPLIT,
    EVAL_ORIGIN,
    FIT_ORIGIN,
    _maturity,
    _publication_lookup,
)
from research.temperature_t25_anchor_preserving_map import (
    _build_maps,
    _fit_compact,
    _select as _select_t25,
)


OUT = Path("results/research/temperature/t26_delayed_base_rate")
REGISTERED = Path("research/temperature_t26_delayed_base_rate_registered.md")
GLOBAL_RIDGE = 20.0
CURRENCY_RIDGE = 40.0
MIN_DATES = 30
GLOBAL_CLIP = 1.5
CURRENCY_CLIP = 0.75
SPECS = {
    "delayed_global_w30": {"window": 30, "hierarchical": False},
    "delayed_global_w60": {"window": 60, "hierarchical": False},
    "delayed_global_w125": {"window": 125, "hierarchical": False},
    "delayed_global_w250": {"window": 250, "hierarchical": False},
    "delayed_global_expanding": {"window": None, "hierarchical": False},
    "delayed_hier_w125": {"window": 125, "hierarchical": True},
}
CANDIDATES = tuple(SPECS)
MODELS = ("identity_early", "t25_base", "delayed_selected", *CANDIDATES)
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}


def _solve_intercept(probability, target, ridge, limit):
    probability = _clip(np.asarray(probability, dtype=float))
    target = np.asarray(target, dtype=float)
    delta = 0.0
    base_logit = _logit(probability)
    for _ in range(50):
        shifted = expit(base_logit + delta)
        gradient = float(np.sum(target - shifted) - ridge * delta)
        curvature = float(np.sum(shifted * (1.0 - shifted)) + ridge)
        step = gradient / max(curvature, 1e-12)
        delta = float(np.clip(delta + step, -limit, limit))
        if abs(step) < 1e-10 or abs(delta) >= limit:
            break
    return delta


def _window_rows(frame, eligible, window):
    eligible_dates = np.asarray(sorted(frame.loc[eligible, "publication_date"].unique()))
    if len(eligible_dates) < MIN_DATES:
        return np.zeros(len(frame), dtype=bool), len(eligible_dates)
    chosen_dates = eligible_dates if window is None else eligible_dates[-window:]
    rows = eligible & frame.publication_date.isin(chosen_dates).to_numpy()
    return rows, len(chosen_dates)


def _delayed_candidates(frame, base, return_states=True):
    """Build causal candidates; only already matured unique event rows update them."""
    frame = frame.reset_index(drop=True)
    base = _clip(np.asarray(base, dtype=float))
    query_dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    publication_dates = pd.to_datetime(frame.publication_date).dt.date.to_numpy()
    event = query_dates == publication_dates
    if frame.loc[event, ["publication_date", "currency"]].duplicated().any():
        raise AssertionError("publication feedback key is duplicated")
    output = {name: base.copy() for name in CANDIDATES}
    states = []
    for query_date in np.unique(query_dates):
        today = query_dates == query_date
        cutoff = (query_date - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        eligible = (
            event
            & (publication_dates < query_date)
            & (frame.maturity_ord.to_numpy(dtype=float) < cutoff)
        )
        for name, spec in SPECS.items():
            rows, feedback_dates = _window_rows(frame, eligible, spec["window"])
            feedback_rows = int(rows.sum())
            if feedback_dates < MIN_DATES:
                global_delta = 0.0
                latest_maturity = None
            else:
                global_delta = _solve_intercept(
                    base[rows], frame.loc[rows, "target"], GLOBAL_RIDGE, GLOBAL_CLIP)
                latest_maturity = int(frame.loc[rows, "maturity_ord"].max())
            for currency in sorted(frame.currency.unique()):
                query_rows = today & frame.currency.eq(currency).to_numpy()
                currency_rows = rows & frame.currency.eq(currency).to_numpy()
                currency_delta = 0.0
                if spec["hierarchical"] and feedback_dates >= MIN_DATES:
                    globally_shifted = _clip(expit(_logit(base[currency_rows]) + global_delta))
                    currency_delta = _solve_intercept(
                        globally_shifted,
                        frame.loc[currency_rows, "target"],
                        CURRENCY_RIDGE,
                        CURRENCY_CLIP,
                    )
                total_delta = global_delta + currency_delta
                output[name][query_rows] = _clip(expit(
                    _logit(base[query_rows]) + total_delta))
                if return_states:
                    states.append({
                        "query_date": query_date,
                        "candidate": name,
                        "currency": currency,
                        "cutoff_ord": cutoff,
                        "feedback_rows_global": feedback_rows,
                        "feedback_rows_currency": int(currency_rows.sum()),
                        "feedback_dates": int(feedback_dates),
                        "latest_feedback_maturity_ord": latest_maturity,
                        "global_delta": global_delta,
                        "currency_delta": currency_delta,
                        "total_delta": total_delta,
                    })
    state_frame = pd.DataFrame(states) if return_states else None
    return output, state_frame, {
        "event_rows": int(event.sum()),
        "event_keys_unique": bool(not frame.loc[
            event, ["publication_date", "currency"]].duplicated().any()),
    }


def _selection_metrics(frame, candidates, selection):
    base_auc = _fast_auc(frame.loc[selection, "target"], frame.loc[selection, "t25_base"])
    rows = []
    for name in CANDIDATES:
        result = _probability_metrics(
            frame.loc[selection, "target"],
            candidates[name][selection],
            frame.loc[selection, "t25_base"],
        )
        result["ece_delta"] = result["ece"] - result["baseline_ece"]
        result["auc_delta"] = result["auc"] - base_auc
        result["model"] = name
        result["feasible"] = bool(
            result["brier_delta"] < 0.0
            and result["logloss_delta"] < 0.0
            and result["ece_delta"] <= 0.005
            and result["auc_delta"] >= -0.005
        )
        rows.append(result)
    feasible = [row for row in rows if row["feasible"]]
    selected = (
        sorted(feasible, key=lambda row: (row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "t25_base"
    )
    return selected, rows


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
            result = _probability_metrics(part.target, part[name], part.t25_base)
            identity = _probability_metrics(part.target, part[name], part.identity_early)
            rows.append({
                "model": name,
                "slice": slice_name,
                "group": group,
                **result,
                "ece_delta_t25": result["ece"] - result["baseline_ece"],
                "auc_delta_t25": result["auc"] - _fast_auc(part.target, part.t25_base),
                "brier_delta_identity": identity["brier_delta"],
                "logloss_delta_identity": identity["logloss_delta"],
                "ece_delta_identity": identity["ece"] - identity["baseline_ece"],
                "auc_delta_identity": identity["auc"] - _fast_auc(part.target, part.identity_early),
            })
    return pd.DataFrame(rows)


def _paired_intervals(frame, primary, baseline, comparison):
    ordered = frame.sort_values(["query_date", "currency"])
    dates = ordered.query_date.drop_duplicates().to_numpy()
    if len(ordered) != len(dates) * ordered.currency.nunique():
        raise AssertionError("unexpected rows per date")
    width = ordered.currency.nunique()
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), width)
    base = ordered[baseline].to_numpy().reshape(len(dates), width)
    pred = ordered[primary].to_numpy().reshape(len(dates), width)
    rows = []
    for block in BLOCKS:
        rng = np.random.default_rng(20260911 + block + (0 if comparison == "t25" else 100))
        blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, blocks))
        samples = (
            (starts[:, :, None] + np.arange(block)[None, None, :]) % len(dates)
        ).reshape(BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier_delta = np.empty(BOOTSTRAP_DRAWS)
        logloss_delta = np.empty(BOOTSTRAP_DRAWS)
        auc_delta = np.empty(BOOTSTRAP_DRAWS)
        for draw_i, sample in enumerate(samples):
            ys = y[sample].ravel()
            bs = _clip(base[sample].ravel())
            ps = _clip(pred[sample].ravel())
            brier_delta[draw_i] = np.mean((ps - ys) ** 2 - (bs - ys) ** 2)
            logloss_delta[draw_i] = np.mean(
                -ys * np.log(ps) - (1 - ys) * np.log(1 - ps)
                + ys * np.log(bs) + (1 - ys) * np.log(1 - bs))
            auc_delta[draw_i] = _fast_auc(ys, ps) - _fast_auc(ys, bs)
        for metric, values in (
            ("brier", brier_delta), ("logloss", logloss_delta), ("auc", auc_delta)
        ):
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "block_dates": block,
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            })
    return pd.DataFrame(rows)


def _build_query():
    X, names, index, series, _trajectory, _trajectory_names, _paths = load_round5_features()
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    raw_publication, model_details = _fit_compact(X, names, index, target, maturity)

    snapshots = _load_snapshots()
    selected_queries = _selected_queries(_query_grid(snapshots), snapshots, "no_same_day_receipt")
    query = _feature_frame(selected_queries, series, _target_arrays(series))
    query = query[query.clock.eq("00:15") & np.isfinite(query.target)].copy().reset_index(drop=True)
    query["identity_early"] = query.identity_h20
    publication_rows, publication_dates = _publication_lookup(index, query)
    query["publication_date"] = publication_dates
    compact_score = raw_publication[publication_rows]

    dates = pd.to_datetime(query.query_date).dt.date.to_numpy()
    cal_cutoff = (CAL_SPLIT - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    eval_cutoff = (EVAL_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    calibration = (
        (dates >= FIT_ORIGIN) & (dates < CAL_SPLIT)
        & (query.maturity_ord.to_numpy() < cal_cutoff)
    )
    selection = (
        (dates >= CAL_SPLIT) & (dates < EVAL_ORIGIN)
        & (query.maturity_ord.to_numpy() < eval_cutoff)
    )
    evaluation = dates >= EVAL_ORIGIN
    t25_maps, t25_map_details = _build_maps(query, compact_score, calibration)
    t25_selected, t25_screen = _select_t25(query, t25_maps, selection)
    if t25_selected != "residual_a040":
        raise AssertionError(f"T25 selection changed: {t25_selected}")
    query["t25_base"] = t25_maps[t25_selected]
    query["year"] = pd.to_datetime(query.query_date).dt.year
    model_details.update({
        "t25_selected": t25_selected,
        "t25_map_details": t25_map_details,
        "t25_screen": t25_screen,
    })
    split = {
        "calibration_rows": int(calibration.sum()),
        "selection_rows": int(selection.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "calibration_cutoff_ord": cal_cutoff,
        "selection_cutoff_ord": eval_cutoff,
        "latest_selection_maturity_ord": int(query.loc[selection, "maturity_ord"].max()),
    }
    return query, selection, evaluation, model_details, split


def run():
    query, selection, evaluation, model_details, split = _build_query()
    candidates, states, feedback_details = _delayed_candidates(query, query.t25_base)
    selected_model, screen = _selection_metrics(query, candidates, selection)
    for name, values in candidates.items():
        query[name] = values
    query["delayed_selected"] = (
        query.t25_base if selected_model == "t25_base" else query[selected_model]
    )
    evaluated = query[evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "delayed_selected", "t25_base", "t25"),
        _paired_intervals(evaluated, "delayed_selected", "identity_early", "identity"),
    ], ignore_index=True)

    primary = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "ALL")].iloc[0]
    year_rows = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "year")]
    currency_rows = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "currency")]
    brier_intervals = intervals[(intervals.comparison == "t25") & (intervals.metric == "brier")]
    passed = bool(
        primary.brier_delta < 0.0
        and primary.logloss_delta < 0.0
        and primary.ece_delta_t25 <= 0.005
        and primary.auc_delta_t25 >= -0.005
        and (brier_intervals.ci_high < 0.0).all()
        and (year_rows.brier_delta <= 0.0).all()
        and (currency_rows.brier_delta <= 0.005).all()
    )

    used_states = states[states.feedback_rows_global > 0]
    trace_causal = bool((
        used_states.latest_feedback_maturity_ord < used_states.cutoff_ord
    ).all())
    source_files = [
        REGISTERED,
        Path("research/temperature_t26_delayed_base_rate.py"),
        Path("research/temperature_t25_anchor_preserving_map.py"),
        Path("research/temperature_t24_history_h20_anchor.py"),
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    metadata = {
        "packet": "temperature-T26",
        "selected_model": selected_model,
        "t25_base_model": model_details["t25_selected"],
        "selection_on_open_period": False,
        "passed": passed,
        "evaluation_rows": int(len(evaluated)),
        "evaluation_dates": int(evaluated.query_date.nunique()),
        "screen": screen,
        "split": split,
        "feedback_details": feedback_details,
        "parameters": {
            "global_ridge": GLOBAL_RIDGE,
            "currency_ridge": CURRENCY_RIDGE,
            "minimum_feedback_dates": MIN_DATES,
            "global_clip": GLOBAL_CLIP,
            "currency_clip": CURRENCY_CLIP,
            "specs": SPECS,
        },
        "checks": {
            "t25_selection_rebuilt": True,
            "selection_labels_mature": bool(
                split["latest_selection_maturity_ord"] < split["selection_cutoff_ord"]),
            "feedback_trace_causal": trace_causal,
            "publication_date_no_later_than_query": bool((
                pd.to_datetime(query.publication_date)
                <= pd.to_datetime(query.query_date)).all()),
            "all_horizon_sources_no_later_than_query": bool((
                pd.to_datetime(query.source_at, utc=True)
                <= pd.to_datetime(query.query_at, utc=True)).all()),
            "event_feedback_unique": feedback_details["event_keys_unique"],
            "opened_evaluation_only": True,
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
            "query_date", "year", "currency", "query_at", "source_at",
            "publication_date", "target", "maturity_ord", *MODELS,
        ]],
        "states": states,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "model_details": model_details,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["states"].to_csv(OUT / "states.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice == "ALL"]
    print(json.dumps({
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "screen": result["metadata"]["screen"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "ece_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "paired_bootstrap": result["paired_bootstrap"].to_dict("records"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
