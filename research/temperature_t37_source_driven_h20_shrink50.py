"""T37: fixed 50% log-odds shrink for T36's CBR-history h20 route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import RECEIPT_DEPENDENT_SOURCE_KINDS
from research import temperature_t36_source_driven_h20_router as t36


OUT = Path("results/research/temperature/t37_source_driven_h20_shrink50")
REGISTERED = Path(
    "research/temperature_t37_source_driven_h20_shrink50_registered.md")
CANDIDATE = "source_driven_h20_shrink50_shadow"
MODELS = ("identity_h20", CANDIDATE)
SHRINK_WEIGHT = 0.5
EPSILON = 1e-6


def _equal_logit_blend(identity, expert):
    identity = np.clip(np.asarray(identity, dtype=float), EPSILON, 1 - EPSILON)
    expert = np.clip(np.asarray(expert, dtype=float), EPSILON, 1 - EPSILON)
    identity_logit = np.log(identity / (1.0 - identity))
    expert_logit = np.log(expert / (1.0 - expert))
    blended_logit = ((1.0 - SHRINK_WEIGHT) * identity_logit
                     + SHRINK_WEIGHT * expert_logit)
    return 1.0 / (1.0 + np.exp(-blended_logit))


def _apply_shrink_route(joined):
    output = t36._apply_source_route(joined)
    history = output.snapshot_source_kind.eq("cbr_history")
    receipt = (
        output.scenario.eq("calendar_assumed_replay")
        & output.snapshot_source_kind.isin(RECEIPT_DEPENDENT_SOURCE_KINDS))
    output[CANDIDATE] = output[t36.CANDIDATE]
    output.loc[history, CANDIDATE] = _equal_logit_blend(
        output.loc[history, "identity_h20"],
        output.loc[history, "t34_probability"])
    output.loc[history, "route_source"] = "t34_cbr_history_shrink50"
    output.loc[receipt, "route_source"] = "t22_after_receipt_replay"
    output["modified"] = history | receipt
    return output


def _build_route(t22, t34):
    return _apply_shrink_route(t36.t35._asof_t34(t22, t34))


def _alias_for_t36(predictions):
    output = predictions.copy()
    output[t36.CANDIDATE] = output[CANDIDATE]
    return output


def _rename_candidate(frame):
    output = frame.copy()
    output.columns = [
        str(column).replace(t36.CANDIDATE, CANDIDATE)
        for column in output.columns]
    if "model" in output:
        output["model"] = output.model.replace({t36.CANDIDATE: CANDIDATE})
    return output


def _evaluate(predictions):
    aliased = _alias_for_t36(predictions)
    metrics_old, reliability_old, intervals, pooled_old, pooled_intervals = (
        t36._metric_outputs(aliased))
    state_old = t36._state_summary(metrics_old, intervals, aliased)
    component = t36._component_metrics(aliased)
    pooled_gate_old = t36._pooled_gate(pooled_old, pooled_intervals)
    return {
        "metrics": _rename_candidate(metrics_old),
        "reliability": _rename_candidate(reliability_old),
        "paired_bootstrap": intervals,
        "state_summary": _rename_candidate(state_old),
        "component_metrics": component,
        "pooled_metrics": _rename_candidate(pooled_old),
        "pooled_paired_bootstrap": pooled_intervals,
        "pooled_gate": _rename_candidate(pooled_gate_old),
    }


def run():
    t22, t34 = t36._load_inputs()
    predictions = _build_route(t22, t34)
    if not (predictions.snapshot_source_at <= predictions.query_at).all():
        raise AssertionError("source timestamp after query")
    no_receipt = predictions.scenario.eq("no_same_day_receipt")
    if predictions.loc[no_receipt].route_source.eq(
            "t22_after_receipt_replay").any():
        raise AssertionError("T22 used without receipt")

    evaluated = _evaluate(predictions)
    state = evaluated["state_summary"]
    component = evaluated["component_metrics"]
    pooled_gate = evaluated["pooled_gate"]
    component["pass"] = (
        (component.brier_delta < 0.0)
        & (component.logloss_delta < 0.0)
        & (component.auc_delta_identity > 0.0)
        & (component.ece_delta_identity <= 0.01)
    )
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
        Path("research/temperature_t37_source_driven_h20_shrink50.py"),
        Path("research/temperature_t37_source_driven_h20_shrink50_audit.py"),
        Path("research/temperature_t36_source_driven_h20_router.py"),
        Path("research/temperature_t35_unified_h20_phase_router.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        t36.T22_OUT / "metadata.json",
        t36.T22_OUT / "predictions.csv.gz",
        t36.T34_OUT / "metadata.json",
        t36.T34_OUT / "development_predictions.csv.gz",
        t36.T34_OUT / "predictions.csv.gz",
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    history = predictions.snapshot_source_kind.eq("cbr_history")
    receipt = (
        predictions.scenario.eq("calendar_assumed_replay")
        & predictions.snapshot_source_kind.isin(
            RECEIPT_DEPENDENT_SOURCE_KINDS))
    metadata = {
        "packet": "temperature-T37",
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
        "post_t36_calibration_repair": True,
        "shrink_weight": SHRINK_WEIGHT,
        "shrink_weight_grid_searched": False,
        "historical_receipts_certified": False,
        "production_requires_verified_receipt_at": True,
        "changes_push_policy": False,
        "changes_runtime_router": False,
        "checks": {
            "one_candidate_no_grid": True,
            "source_driven_not_clock_driven": True,
            "source_no_later_than_query": True,
            "t34_backward_join_only": bool(
                (predictions.t34_publication_date
                 <= predictions.query_date).all()),
            "fixed_half_logit_shrink_only_on_cbr_history": bool(
                predictions.loc[history].route_source.eq(
                    "t34_cbr_history_shrink50").all()),
            "t22_only_on_receipt_dependent_replay": bool(
                predictions.route_source.eq("t22_after_receipt_replay").eq(
                    receipt).all()),
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
    return {"predictions": predictions, **evaluated, "metadata": metadata}


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
    result["pooled_metrics"].to_csv(
        OUT / "pooled_metrics.csv", index=False)
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
