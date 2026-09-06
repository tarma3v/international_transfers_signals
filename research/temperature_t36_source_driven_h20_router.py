"""T36: h20 route driven by the actually available source kind."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import RECEIPT_DEPENDENT_SOURCE_KINDS
from research import temperature_t35_unified_h20_phase_router as t35
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    CLOCKS,
    SCENARIOS,
    _field,
    _load_snapshots,
    _probability_metrics,
    _query_grid,
    _selected_queries,
)
from research.temperature_t21_h20_curve_head import _fast_auc


T22_OUT = t35.T22_OUT
T34_OUT = t35.T34_OUT
OUT = Path("results/research/temperature/t36_source_driven_h20_router")
REGISTERED = Path("research/temperature_t36_source_driven_h20_router_registered.md")
CANDIDATE = "source_driven_h20_shadow"
MODELS = ("identity_h20", CANDIDATE)


def _source_map():
    snapshots = _load_snapshots()
    queries = _query_grid(snapshots)
    rows = []
    for scenario in SCENARIOS:
        selected = _selected_queries(queries, snapshots, scenario)
        selected = selected[pd.to_datetime(selected.query_date).dt.year >= 2025]
        selected = selected.copy()
        selected["snapshot_source_at"] = _field(
            selected, "source_at", 20, "snapshot_source_at")
        selected["snapshot_source_kind"] = _field(
            selected, "source_kind", 20, "snapshot_source_kind")
        selected["snapshot_phase"] = _field(
            selected, "phase", 20, "snapshot_phase")
        selected["snapshot_confidence"] = _field(
            selected, "confidence", 20, "snapshot_confidence")
        rows.append(selected[[
            "scenario", "query_date", "clock", "currency", "query_at",
            "snapshot_valid_from", "snapshot_source_at", "snapshot_source_kind",
            "snapshot_phase", "snapshot_confidence",
        ]])
    output = pd.concat(rows, ignore_index=True)
    output["query_date"] = pd.to_datetime(output.query_date).dt.date
    output["query_at"] = pd.to_datetime(output.query_at, utc=True)
    output["snapshot_valid_from"] = pd.to_datetime(
        output.snapshot_valid_from, utc=True)
    output["snapshot_source_at"] = pd.to_datetime(
        output.snapshot_source_at, utc=True)
    key = ["scenario", "query_date", "clock", "currency"]
    if output[key].duplicated().any():
        raise AssertionError("source map key duplicated")
    return output


def _load_inputs():
    t22, t34 = t35._load_inputs()
    source = _source_map()
    key = ["scenario", "query_date", "clock", "currency"]
    t22 = t22.merge(source, on=key, how="left", validate="one_to_one",
                    suffixes=("", "_map"))
    if t22.snapshot_source_kind.isna().any():
        raise AssertionError("T19 source mapping missing")
    if not np.array_equal(
            pd.to_datetime(t22.query_at, utc=True).to_numpy(),
            pd.to_datetime(t22.query_at_map, utc=True).to_numpy()):
        raise AssertionError("T22/T19 query timestamp mismatch")
    if not np.array_equal(
            pd.to_datetime(t22.source_at, utc=True).to_numpy(),
            pd.to_datetime(t22.snapshot_source_at, utc=True).to_numpy()):
        raise AssertionError("T22/T19 source timestamp mismatch")
    t22 = t22.drop(columns="query_at_map")
    return t22, t34


def _apply_source_route(joined):
    output = joined.copy()
    history = output.snapshot_source_kind.eq("cbr_history")
    receipt = (
        output.scenario.eq("calendar_assumed_replay")
        & output.snapshot_source_kind.isin(RECEIPT_DEPENDENT_SOURCE_KINDS))
    if (history & receipt).any():
        raise AssertionError("history and receipt route overlap")
    output[CANDIDATE] = output.identity_h20
    output["route_source"] = "t19_market_bridge_or_hold"
    output.loc[history, CANDIDATE] = output.loc[history, "t34_probability"]
    output.loc[history, "route_source"] = "t34_cbr_history"
    output.loc[receipt, CANDIDATE] = output.loc[
        receipt, "rank_correction_selected"]
    output.loc[receipt, "route_source"] = "t22_after_receipt_replay"
    output["modified"] = history | receipt
    return output


def _build_route(t22, t34):
    return _apply_source_route(t35._asof_t34(t22, t34))


def _aliased(predictions):
    output = predictions.copy()
    output[t35.CANDIDATE] = output[CANDIDATE]
    return output


def _metric_outputs(predictions):
    metrics, reliability = t35._metric_rows(_aliased(predictions))
    metrics["model"] = metrics.model.replace({t35.CANDIDATE: CANDIDATE})
    reliability["model"] = reliability.model.replace({t35.CANDIDATE: CANDIDATE})
    intervals = []
    alias = _aliased(predictions)
    for (scenario, clock), part in alias.groupby(
            ["scenario", "clock"], sort=False):
        intervals.extend(t35._paired_intervals(part, scenario, clock))
    pooled, pooled_intervals = t35._pooled_metrics(alias)
    pooled["model"] = pooled.model.replace({t35.CANDIDATE: CANDIDATE})
    return metrics, reliability, pd.DataFrame(intervals), pooled, pooled_intervals


def _state_summary(metrics, intervals, predictions):
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
        modified_rows=("modified", "sum"),
        rows=("modified", "size"),
        route_source_count=("route_source", "nunique"),
    ).reset_index()
    state = state.merge(route, on=["scenario", "clock"], validate="one_to_one")
    state["noninferior"] = (
        (state.brier_delta <= 0.001)
        & (state.logloss_delta <= 0.003)
        & (state.ece_delta <= 0.01)
        & (state.auc_delta >= -0.005)
    )
    return state


def _component_metrics(predictions):
    work = predictions[predictions.modified].copy()
    work = work.sort_values([
        "scenario", "query_date", "currency", "snapshot_source_at", "clock"])
    work = work.drop_duplicates([
        "scenario", "query_date", "currency", "snapshot_source_at",
        "route_source"], keep="last")
    rows = []
    for source, part in work.groupby("route_source", sort=True):
        result = _probability_metrics(
            part.target, part[CANDIDATE], part.identity_h20)
        rows.append({
            "route_source": source,
            **result,
            "auc_delta_identity": result["auc"] - _fast_auc(
                part.target, part.identity_h20),
            "ece_delta_identity": result["ece"] - result["baseline_ece"],
            "deduplicated_rows": int(len(part)),
            "dates": int(part.query_date.nunique()),
        })
    return pd.DataFrame(rows)


def _pooled_gate(pooled, intervals):
    pivot = pooled.pivot(index="scenario", columns="model",
                          values=["brier", "logloss", "ece", "auc"])
    pivot.columns = ["_".join(column) for column in pivot.columns]
    frame = pivot.reset_index()
    for metric in ("brier", "logloss", "ece", "auc"):
        frame[f"{metric}_delta"] = (
            frame[f"{metric}_{CANDIDATE}"]
            - frame[f"{metric}_identity_h20"])
    for metric in ("brier", "auc"):
        bounds = intervals[intervals.metric.eq(metric)].groupby("scenario").agg(
            ci_low=("ci_low", "min"), ci_high=("ci_high", "max")
        ).add_prefix(f"{metric}_").reset_index()
        frame = frame.merge(bounds, on="scenario", validate="one_to_one")
    frame["pass"] = (
        (frame.brier_delta < 0.0) & (frame.brier_ci_high < 0.0)
        & (frame.logloss_delta < 0.0)
        & (frame.auc_delta > 0.0) & (frame.auc_ci_low > 0.0)
    )
    return frame


def run():
    t22, t34 = _load_inputs()
    predictions = _build_route(t22, t34)
    if not (predictions.snapshot_source_at <= predictions.query_at).all():
        raise AssertionError("source timestamp after query")
    no_receipt = predictions.scenario.eq("no_same_day_receipt")
    if predictions.loc[no_receipt].route_source.eq(
            "t22_after_receipt_replay").any():
        raise AssertionError("T22 used without receipt")

    metrics, reliability, intervals, pooled, pooled_intervals = (
        _metric_outputs(predictions))
    state = _state_summary(metrics, intervals, predictions)
    component = _component_metrics(predictions)
    pooled_gate = _pooled_gate(pooled, pooled_intervals)
    component_pass = (
        (component.brier_delta < 0.0)
        & (component.logloss_delta < 0.0)
        & (component.auc_delta_identity > 0.0)
        & (component.ece_delta_identity <= 0.01)
    )
    component["pass"] = component_pass
    unchanged = ~predictions.modified
    unchanged_exact = bool(np.array_equal(
        predictions.loc[unchanged, CANDIDATE].to_numpy(),
        predictions.loc[unchanged, "identity_h20"].to_numpy()))
    retrospective_passed = bool(
        pooled_gate["pass"].all()
        and component["pass"].all()
        and state.noninferior.all()
        and unchanged_exact)

    source_files = [
        REGISTERED,
        Path("research/temperature_t36_source_driven_h20_router.py"),
        Path("research/temperature_t36_source_driven_h20_router_audit.py"),
        Path("research/temperature_t35_unified_h20_phase_router.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        T22_OUT / "metadata.json",
        T22_OUT / "predictions.csv.gz",
        T34_OUT / "metadata.json",
        T34_OUT / "development_predictions.csv.gz",
        T34_OUT / "predictions.csv.gz",
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    metadata = {
        "packet": "temperature-T36",
        "candidate": CANDIDATE,
        "evaluation_period": "open 2025-2026 retrospective",
        "evaluation_rows": int(len(predictions)),
        "modified_rows": int(predictions.modified.sum()),
        "state_rows": int(len(state)),
        "noninferior_states": int(state.noninferior.sum()),
        "component_passes": int(component["pass"].sum()),
        "component_count": int(len(component)),
        "pooled_scenario_passes": int(pooled_gate["pass"].sum()),
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
            "source_driven_not_clock_driven": True,
            "source_no_later_than_query": True,
            "t34_backward_join_only": bool(
                (predictions.t34_publication_date <= predictions.query_date).all()),
            "t34_only_on_cbr_history": bool(
                predictions.route_source.eq("t34_cbr_history").eq(
                    predictions.snapshot_source_kind.eq("cbr_history")).all()),
            "t22_only_on_receipt_dependent_replay": bool(
                predictions.route_source.eq("t22_after_receipt_replay").eq(
                    predictions.scenario.eq("calendar_assumed_replay")
                    & predictions.snapshot_source_kind.isin(
                        RECEIPT_DEPENDENT_SOURCE_KINDS)).all()),
            "no_receipt_never_uses_t22": True,
            "unchanged_rows_exact_identity": unchanged_exact,
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
        "state_summary": state,
        "component_metrics": component,
        "pooled_metrics": pooled,
        "pooled_paired_bootstrap": pooled_intervals,
        "pooled_gate": pooled_gate,
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
    result["state_summary"].to_csv(OUT / "state_summary.csv", index=False)
    result["component_metrics"].to_csv(
        OUT / "component_metrics.csv", index=False)
    result["pooled_metrics"].to_csv(OUT / "pooled_metrics.csv", index=False)
    result["pooled_paired_bootstrap"].to_csv(
        OUT / "pooled_paired_bootstrap.csv", index=False)
    result["pooled_gate"].to_csv(OUT / "pooled_gate.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "retrospective_route_passed": result["metadata"][
            "retrospective_route_passed"],
        "production_promoted": result["metadata"]["production_promoted"],
        "component_metrics": result["component_metrics"].to_dict("records"),
        "pooled_gate": result["pooled_gate"].to_dict("records"),
        "worst_states": result["state_summary"].sort_values([
            "noninferior", "brier_delta", "auc_delta"],
            ascending=[True, False, True]).head(8).to_dict("records"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
