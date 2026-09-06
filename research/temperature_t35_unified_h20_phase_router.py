"""T35: fixed any-time h20 route across early, market, and receipt phases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    BOOTSTRAP_DRAWS,
    CLOCKS,
    SCENARIOS,
    _ece,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc


T22_OUT = Path("results/research/temperature/t22_h20_rank_correction")
T34_OUT = Path("results/research/temperature/t34_cold_start_identity_gate")
OUT = Path("results/research/temperature/t35_unified_h20_phase_router")
REGISTERED = Path("research/temperature_t35_unified_h20_phase_router_registered.md")
EARLY_CLOCKS = ("00:15", "06:00", "09:15", "10:15")
AFTER_RECEIPT_CLOCKS = ("18:45", "19:15", "20:15", "21:15", "22:15", "23:15")
CANDIDATE = "unified_h20_phase_shadow"
MODELS = ("identity_h20", CANDIDATE)


def _load_inputs():
    t22 = pd.read_csv(T22_OUT / "predictions.csv.gz")
    t22["query_date"] = pd.to_datetime(t22.query_date).dt.date
    t22["query_at"] = pd.to_datetime(t22.query_at, utc=True)
    t22["source_at"] = pd.to_datetime(t22.source_at, utc=True)
    if t22[["scenario", "query_date", "clock", "currency"]].duplicated().any():
        raise AssertionError("T22 query key duplicated")

    frames = []
    for filename in ("development_predictions.csv.gz", "predictions.csv.gz"):
        frame = pd.read_csv(T34_OUT / filename)
        frames.append(frame[[
            "query_date", "currency", "target",
            "cold_identity_qstack_w125_r100",
        ]])
    t34 = pd.concat(frames, ignore_index=True)
    t34["query_date"] = pd.to_datetime(t34.query_date).dt.date
    if t34[["query_date", "currency"]].duplicated().any():
        raise AssertionError("T34 publication key duplicated")
    return t22, t34


def _asof_t34(t22, t34):
    joined = []
    for currency in CORRIDORS:
        left = t22[t22.currency.eq(currency)].copy()
        left["_row"] = left.index
        left["_join_date"] = pd.to_datetime(left.query_date)
        left = left.sort_values(["_join_date", "scenario", "clock"])
        right = t34[t34.currency.eq(currency)].copy().rename(columns={
            "query_date": "t34_publication_date",
            "target": "t34_target",
            "cold_identity_qstack_w125_r100": "t34_probability",
        })
        right["_join_date"] = pd.to_datetime(right.t34_publication_date)
        right = right.sort_values("_join_date")
        part = pd.merge_asof(
            left,
            right[["_join_date", "t34_publication_date", "t34_target",
                   "t34_probability"]],
            on="_join_date",
            direction="backward",
            allow_exact_matches=True,
        )
        part = part.drop(columns="_join_date")
        joined.append(part)
    output = pd.concat(joined, ignore_index=True).sort_values("_row")
    output = output.drop(columns="_row").reset_index(drop=True)
    if output.t34_probability.isna().any():
        raise AssertionError("T34 early publication anchor unavailable")
    if not (output.t34_publication_date <= output.query_date).all():
        raise AssertionError("future T34 publication joined")
    exact = output.t34_publication_date.eq(output.query_date)
    if not np.allclose(
            output.loc[exact, "target"], output.loc[exact, "t34_target"],
            equal_nan=True, atol=0.0):
        raise AssertionError("T22/T34 target alignment mismatch")
    return output


def _apply_phase_route(joined):
    output = joined.copy()
    early = output.clock.isin(EARLY_CLOCKS)
    after_receipt_replay = (
        output.scenario.eq("calendar_assumed_replay")
        & output.clock.isin(AFTER_RECEIPT_CLOCKS))
    output[CANDIDATE] = output.identity_h20
    output["route_source"] = "t19_market_or_hold"
    output.loc[early, CANDIDATE] = output.loc[early, "t34_probability"]
    output.loc[early, "route_source"] = "t34_early_publication"
    output.loc[after_receipt_replay, CANDIDATE] = output.loc[
        after_receipt_replay, "rank_correction_selected"]
    output.loc[after_receipt_replay, "route_source"] = "t22_after_receipt_replay"
    output["modified"] = early | after_receipt_replay
    return output


def _build_route(t22, t34):
    return _apply_phase_route(_asof_t34(t22, t34))


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def _metric_rows(predictions):
    rows = []
    reliability = []
    for (scenario, clock), state in predictions.groupby(
            ["scenario", "clock"], sort=False):
        for model in MODELS:
            for slice_name, group, part in _slices(state):
                result = _probability_metrics(
                    part.target, part[model], part.identity_h20)
                rows.append({
                    "scenario": scenario,
                    "clock": clock,
                    "h": 20,
                    "model": model,
                    "slice": slice_name,
                    "group": group,
                    **result,
                    "auc_delta_identity": result["auc"] - _fast_auc(
                        part.target, part.identity_h20),
                    "ece_delta_identity": result["ece"] - result["baseline_ece"],
                })
            y = state.target.to_numpy(dtype=float)
            p = state[model].to_numpy(dtype=float)
            edges = np.linspace(0.0, 1.0, 11)
            bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
            reliability.extend({
                "scenario": scenario,
                "clock": clock,
                "model": model,
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "n": int((bins == i).sum()),
                "predicted": float(p[bins == i].mean()),
                "actual": float(y[bins == i].mean()),
            } for i in range(10) if np.any(bins == i))
    return pd.DataFrame(rows), pd.DataFrame(reliability)


def _paired_intervals(part, scenario, clock):
    ordered = part.sort_values(["query_date", "currency"])
    dates = ordered.query_date.drop_duplicates().to_numpy()
    if len(ordered) != len(dates) * len(CORRIDORS):
        raise AssertionError("unexpected rows per state date")
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), len(CORRIDORS))
    base = ordered.identity_h20.to_numpy().reshape(len(dates), len(CORRIDORS))
    candidate = ordered[CANDIDATE].to_numpy().reshape(len(dates), len(CORRIDORS))
    rows = []
    scenario_i = SCENARIOS.index(scenario)
    clock_i = CLOCKS.index(clock)
    for block in BLOCKS:
        rng = np.random.default_rng(
            20260920 + scenario_i * 100000 + clock_i * 1000 + block)
        blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, blocks))
        offsets = np.arange(block)
        sample = ((starts[:, :, None] + offsets[None, None, :]) % len(dates))
        sample = sample.reshape(BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier = np.empty(BOOTSTRAP_DRAWS)
        logloss = np.empty(BOOTSTRAP_DRAWS)
        auc = np.empty(BOOTSTRAP_DRAWS)
        for draw_i, chosen in enumerate(sample):
            ys = y[chosen].ravel()
            bs = np.clip(base[chosen].ravel(), 1e-6, 1 - 1e-6)
            ps = np.clip(candidate[chosen].ravel(), 1e-6, 1 - 1e-6)
            brier[draw_i] = np.mean((ps - ys) ** 2 - (bs - ys) ** 2)
            logloss[draw_i] = np.mean(
                -(ys * np.log(ps) + (1 - ys) * np.log(1 - ps))
                + ys * np.log(bs) + (1 - ys) * np.log(1 - bs))
            auc[draw_i] = _fast_auc(ys, ps) - _fast_auc(ys, bs)
        for metric, values in (("brier", brier), ("logloss", logloss), ("auc", auc)):
            rows.append({
                "scenario": scenario,
                "clock": clock,
                "metric": metric,
                "block_dates": block,
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, .025)),
                "ci_high": float(np.quantile(values, .975)),
            })
    return rows


def _pooled_metrics(predictions):
    rows = []
    intervals = []
    for scenario, part in predictions.groupby("scenario", sort=False):
        for model in MODELS:
            result = _probability_metrics(
                part.target, part[model], part.identity_h20)
            rows.append({"scenario": scenario, "model": model, **result})

        daily = list(part.groupby("query_date", sort=True))
        base_y = [x.target.to_numpy(dtype=int) for _, x in daily]
        base_p = [x.identity_h20.to_numpy(dtype=float) for _, x in daily]
        cand_p = [x[CANDIDATE].to_numpy(dtype=float) for _, x in daily]
        n_dates = len(daily)
        for block in BLOCKS:
            rng = np.random.default_rng(
                20260921 + SCENARIOS.index(scenario) * 100000 + block)
            blocks = int(np.ceil(n_dates / block))
            starts = rng.integers(0, n_dates, size=(BOOTSTRAP_DRAWS, blocks))
            offsets = np.arange(block)
            samples = ((starts[:, :, None] + offsets[None, None, :]) % n_dates)
            samples = samples.reshape(BOOTSTRAP_DRAWS, -1)[:, :n_dates]
            brier = np.empty(BOOTSTRAP_DRAWS)
            logloss = np.empty(BOOTSTRAP_DRAWS)
            auc = np.empty(BOOTSTRAP_DRAWS)
            for draw_i, sample in enumerate(samples):
                ys = np.concatenate([base_y[i] for i in sample])
                bs = np.clip(np.concatenate([base_p[i] for i in sample]), 1e-6, 1 - 1e-6)
                ps = np.clip(np.concatenate([cand_p[i] for i in sample]), 1e-6, 1 - 1e-6)
                brier[draw_i] = np.mean((ps - ys) ** 2 - (bs - ys) ** 2)
                logloss[draw_i] = np.mean(
                    -(ys * np.log(ps) + (1 - ys) * np.log(1 - ps))
                    + ys * np.log(bs) + (1 - ys) * np.log(1 - bs))
                auc[draw_i] = _fast_auc(ys, ps) - _fast_auc(ys, bs)
            for metric, values in (("brier", brier), ("logloss", logloss), ("auc", auc)):
                intervals.append({
                    "scenario": scenario,
                    "metric": metric,
                    "block_dates": block,
                    "mean_delta": float(values.mean()),
                    "ci_low": float(np.quantile(values, .025)),
                    "ci_high": float(np.quantile(values, .975)),
                })
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def _state_gates(metrics, intervals, predictions):
    overall = metrics[
        metrics.slice.eq("ALL") & metrics.model.isin(MODELS)
    ].pivot(index=["scenario", "clock"], columns="model",
            values=["brier", "logloss", "ece", "auc", "average_precision"])
    overall.columns = ["_".join(column) for column in overall.columns]
    state = overall.reset_index()
    for metric in ("brier", "logloss", "ece", "auc", "average_precision"):
        state[f"{metric}_delta"] = (
            state[f"{metric}_{CANDIDATE}"] - state[f"{metric}_identity_h20"])
    for metric in ("brier", "auc"):
        bounds = intervals[intervals.metric.eq(metric)].groupby(
            ["scenario", "clock"]).agg(
                ci_low=("ci_low", "min"), ci_high=("ci_high", "max")
        ).add_prefix(f"{metric}_").reset_index()
        state = state.merge(bounds, on=["scenario", "clock"], validate="one_to_one")
    route = predictions.groupby(["scenario", "clock"], sort=False).agg(
        route_source=("route_source", "first"),
        modified=("modified", "all"),
        route_source_count=("route_source", "nunique"),
    ).reset_index()
    state = state.merge(route, on=["scenario", "clock"], validate="one_to_one")
    if not state.route_source_count.eq(1).all():
        raise AssertionError("route source varies within state")
    state["pass"] = np.where(
        state.modified,
        (state.auc_delta > 0.0) & (state.auc_ci_low > 0.0)
        & (state.brier_delta < 0.0) & (state.brier_ci_high < 0.0)
        & (state.logloss_delta < 0.0) & (state.ece_delta <= 0.01),
        np.isclose(state.auc_delta, 0.0)
        & np.isclose(state.brier_delta, 0.0)
        & np.isclose(state.logloss_delta, 0.0)
        & np.isclose(state.ece_delta, 0.0),
    )
    return state


def run():
    t22, t34 = _load_inputs()
    predictions = _build_route(t22, t34)
    if not (predictions.source_at <= predictions.query_at).all():
        raise AssertionError("source timestamp after query")
    no_receipt = predictions.scenario.eq("no_same_day_receipt")
    if predictions.loc[no_receipt, "route_source"].eq(
            "t22_after_receipt_replay").any():
        raise AssertionError("T22 used without receipt")

    metrics, reliability = _metric_rows(predictions)
    interval_rows = []
    for (scenario, clock), part in predictions.groupby(
            ["scenario", "clock"], sort=False):
        interval_rows.extend(_paired_intervals(part, scenario, clock))
    intervals = pd.DataFrame(interval_rows)
    state_gates = _state_gates(metrics, intervals, predictions)
    pooled_metrics, pooled_intervals = _pooled_metrics(predictions)
    modified = state_gates[state_gates.modified]
    unchanged = state_gates[~state_gates.modified]
    retrospective_passed = bool(
        modified["pass"].all() and unchanged["pass"].all())

    source_files = [
        REGISTERED,
        Path("research/temperature_t35_unified_h20_phase_router.py"),
        Path("research/temperature_t35_unified_h20_phase_router_audit.py"),
        T22_OUT / "metadata.json",
        T22_OUT / "predictions.csv.gz",
        T34_OUT / "metadata.json",
        T34_OUT / "development_predictions.csv.gz",
        T34_OUT / "predictions.csv.gz",
    ]
    metadata = {
        "packet": "temperature-T35",
        "candidate": CANDIDATE,
        "early_clocks": list(EARLY_CLOCKS),
        "after_receipt_replay_clocks": list(AFTER_RECEIPT_CLOCKS),
        "evaluation_period": "open 2025-2026 retrospective",
        "evaluation_rows": int(len(predictions)),
        "state_rows": int(len(state_gates)),
        "modified_states": int(state_gates.modified.sum()),
        "passing_modified_states": int(modified["pass"].sum()),
        "passing_unchanged_states": int(unchanged["pass"].sum()),
        "retrospective_route_passed": retrospective_passed,
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "historical_receipts_certified": False,
        "production_requires_verified_receipt_at": True,
        "changes_push_policy": False,
        "changes_runtime_router": False,
        "checks": {
            "one_candidate_no_grid": True,
            "t34_backward_join_only": bool(
                (predictions.t34_publication_date <= predictions.query_date).all()),
            "source_no_later_than_query": True,
            "no_receipt_never_uses_t22": True,
            "only_registered_early_clocks_use_t34": bool(
                predictions.route_source.eq("t34_early_publication").eq(
                    predictions.clock.isin(EARLY_CLOCKS)).all()),
            "only_assumed_after_receipt_states_use_t22": bool(
                predictions.route_source.eq("t22_after_receipt_replay").eq(
                    predictions.scenario.eq("calendar_assumed_replay")
                    & predictions.clock.isin(AFTER_RECEIPT_CLOCKS)).all()),
            "probability_bounded": bool(
                predictions[list(MODELS)].ge(0.0).all().all()
                and predictions[list(MODELS)].le(1.0).all().all()),
            "expected_future_bps_unchanged": True,
            "push_policy_unchanged": True,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "reliability": reliability,
        "paired_bootstrap": intervals,
        "state_gates": state_gates,
        "pooled_metrics": pooled_metrics,
        "pooled_paired_bootstrap": pooled_intervals,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["reliability"].to_csv(OUT / "reliability_bins.csv", index=False)
    result["paired_bootstrap"].to_csv(
        OUT / "paired_bootstrap.csv", index=False)
    result["state_gates"].to_csv(OUT / "state_gates.csv", index=False)
    result["pooled_metrics"].to_csv(OUT / "pooled_metrics.csv", index=False)
    result["pooled_paired_bootstrap"].to_csv(
        OUT / "pooled_paired_bootstrap.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "modified_states": result["metadata"]["modified_states"],
        "passing_modified_states": result["metadata"][
            "passing_modified_states"],
        "retrospective_route_passed": result["metadata"][
            "retrospective_route_passed"],
        "production_promoted": result["metadata"]["production_promoted"],
        "state_gates": result["state_gates"].to_dict("records"),
        "pooled_metrics": result["pooled_metrics"].to_dict("records"),
        "pooled_intervals": result["pooled_paired_bootstrap"].to_dict("records"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
