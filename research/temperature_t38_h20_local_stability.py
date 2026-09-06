"""T38: local currency/year stability audit for the frozen T37 h20 shadow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import temperature_t37_source_driven_h20_shrink50 as t37
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    CLOCKS,
    SCENARIOS,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc


T37_OUT = t37.OUT
OUT = Path("results/research/temperature/t38_h20_local_stability")
REGISTERED = Path("research/temperature_t38_h20_local_stability_registered.md")
CANDIDATE = t37.CANDIDATE
LOCAL_SLICES = ("currency", "year", "currency_year")
BOOTSTRAP_DRAWS = 500


def _load_predictions():
    frame = pd.read_csv(T37_OUT / "predictions.csv.gz")
    frame["query_date"] = pd.to_datetime(frame.query_date).dt.date
    frame["year"] = pd.to_datetime(frame.query_date).dt.year.astype(str)
    frame["currency_year"] = frame.currency.astype(str) + ":" + frame.year
    return frame


def _clock_local_metrics():
    metrics = pd.read_csv(T37_OUT / "metrics.csv")
    output = metrics[
        metrics.model.eq(CANDIDATE) & metrics.slice.isin(LOCAL_SLICES)
    ].copy()
    output["noninferior"] = (
        (output.brier_delta <= 0.001)
        & (output.logloss_delta <= 0.003)
        & (output.ece_delta_identity <= 0.01)
        & (output.auc_delta_identity >= -0.005)
    )
    return output.reset_index(drop=True)


def _pooled_local_metrics(predictions):
    rows = []
    for scenario in SCENARIOS:
        scenario_part = predictions[predictions.scenario.eq(scenario)]
        for slice_name in LOCAL_SLICES:
            for group, part in scenario_part.groupby(slice_name, sort=True):
                result = _probability_metrics(
                    part.target, part[CANDIDATE], part.identity_h20)
                rows.append({
                    "scenario": scenario,
                    "slice": slice_name,
                    "group": str(group),
                    **result,
                    "auc_delta_identity": (
                        result["auc"]
                        - _fast_auc(part.target, part.identity_h20)),
                    "ece_delta_identity": (
                        result["ece"] - result["baseline_ece"]),
                    "dates": int(part.query_date.nunique()),
                })
    return pd.DataFrame(rows)


def _bootstrap_one(part, scenario, slice_name, group):
    ordered = part.sort_values(["query_date", "clock", "currency"])
    dates = ordered.query_date.drop_duplicates().to_numpy()
    counts = ordered.groupby("query_date", sort=True).size().to_numpy()
    if len(set(counts.tolist())) != 1:
        raise AssertionError("local group does not have a rectangular date grid")
    rows_per_date = int(counts[0])
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), rows_per_date)
    base = np.clip(
        ordered.identity_h20.to_numpy(dtype=float), 1e-6, 1 - 1e-6
    ).reshape(len(dates), rows_per_date)
    candidate = np.clip(
        ordered[CANDIDATE].to_numpy(dtype=float), 1e-6, 1 - 1e-6
    ).reshape(len(dates), rows_per_date)
    daily_brier = np.mean(
        (candidate - y) ** 2 - (base - y) ** 2, axis=1)
    daily_logloss = np.mean(
        -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
        + y * np.log(base) + (1 - y) * np.log(1 - base),
        axis=1,
    )
    rows = []
    seed_key = f"{scenario}|{slice_name}|{group}".encode()
    seed_base = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "big")
    for block in BLOCKS:
        rng = np.random.default_rng(seed_base + block)
        n_blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(
            0, len(dates), size=(BOOTSTRAP_DRAWS, n_blocks))
        offsets = np.arange(block)
        samples = (
            starts[:, :, None] + offsets[None, None, :]
        ) % len(dates)
        samples = samples.reshape(BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier = daily_brier[samples].mean(axis=1)
        logloss = daily_logloss[samples].mean(axis=1)
        auc = np.empty(BOOTSTRAP_DRAWS)
        for draw_i, selected in enumerate(samples):
            ys = y[selected].ravel()
            auc[draw_i] = (
                _fast_auc(ys, candidate[selected].ravel())
                - _fast_auc(ys, base[selected].ravel()))
        for metric, values in (
            ("brier", brier), ("logloss", logloss), ("auc", auc)
        ):
            rows.append({
                "scenario": scenario,
                "slice": slice_name,
                "group": str(group),
                "metric": metric,
                "block_dates": block,
                "draws": BOOTSTRAP_DRAWS,
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            })
    return rows


def _local_bootstrap(predictions):
    rows = []
    for scenario in SCENARIOS:
        scenario_part = predictions[predictions.scenario.eq(scenario)]
        for slice_name in LOCAL_SLICES:
            for group, part in scenario_part.groupby(slice_name, sort=True):
                rows.extend(_bootstrap_one(
                    part, scenario, slice_name, str(group)))
    return pd.DataFrame(rows)


def _pooled_gates(metrics, intervals):
    brier = intervals[intervals.metric.eq("brier")].groupby(
        ["scenario", "slice", "group"], sort=False
    ).agg(brier_ci_low=("ci_low", "min"),
          brier_ci_high=("ci_high", "max")).reset_index()
    auc = intervals[intervals.metric.eq("auc")].groupby(
        ["scenario", "slice", "group"], sort=False
    ).agg(auc_ci_low=("ci_low", "min"),
          auc_ci_high=("ci_high", "max")).reset_index()
    output = metrics.merge(
        brier, on=["scenario", "slice", "group"], validate="one_to_one"
    ).merge(
        auc, on=["scenario", "slice", "group"], validate="one_to_one")
    output["pass"] = (
        (output.brier_delta < 0.0)
        & (output.logloss_delta < 0.0)
        & (output.auc_delta_identity > 0.0)
        & (output.ece_delta_identity <= 0.01)
        & (output.brier_ci_high < 0.0)
        & (output.auc_ci_low > 0.0)
    )
    return output


def _failure_summary(clock_local, pooled_gates):
    rows = []
    for slice_name in LOCAL_SLICES:
        clock = clock_local[clock_local.slice.eq(slice_name)]
        pooled = pooled_gates[pooled_gates.slice.eq(slice_name)]
        rows.append({
            "slice": slice_name,
            "clock_rows": int(len(clock)),
            "clock_noninferior": int(clock.noninferior.sum()),
            "clock_failures": int((~clock.noninferior).sum()),
            "clock_ece_failures": int((clock.ece_delta_identity > 0.01).sum()),
            "clock_auc_failures": int((clock.auc_delta_identity < -0.005).sum()),
            "pooled_groups": int(len(pooled)),
            "pooled_passes": int(pooled["pass"].sum()),
            "pooled_failures": int((~pooled["pass"]).sum()),
        })
    return pd.DataFrame(rows)


def run():
    predictions = _load_predictions()
    clock_local = _clock_local_metrics()
    pooled_metrics = _pooled_local_metrics(predictions)
    intervals = _local_bootstrap(predictions)
    pooled_gates = _pooled_gates(pooled_metrics, intervals)
    failure_summary = _failure_summary(clock_local, pooled_gates)
    local_stability_passed = bool(
        clock_local.noninferior.all() and pooled_gates["pass"].all())

    source_files = [
        REGISTERED,
        Path("research/temperature_t38_h20_local_stability.py"),
        Path("research/temperature_t38_h20_local_stability_audit.py"),
        T37_OUT / "metadata.json",
        T37_OUT / "predictions.csv.gz",
        T37_OUT / "metrics.csv",
        T37_OUT / "paired_bootstrap.csv",
    ]
    metadata = {
        "packet": "temperature-T38",
        "candidate": CANDIDATE,
        "evaluation_period": "open 2025-2026 retrospective",
        "evaluation_rows": int(len(predictions)),
        "local_slices": list(LOCAL_SLICES),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_blocks": list(BLOCKS),
        "clock_local_rows": int(len(clock_local)),
        "clock_local_noninferior": int(clock_local.noninferior.sum()),
        "pooled_local_groups": int(len(pooled_gates)),
        "pooled_local_passes": int(pooled_gates["pass"].sum()),
        "local_stability_passed": local_stability_passed,
        "production_promoted": False,
        "model_changed": False,
        "runtime_changed": False,
        "push_changed": False,
        "selection_on_open_2025_2026": False,
        "open_point_slices_inspected_before_registration": True,
        "local_bootstrap_inspected_before_registration": False,
        "fresh_independent_holdout": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "clock_local_metrics": clock_local,
        "pooled_local_metrics": pooled_metrics,
        "local_paired_bootstrap": intervals,
        "pooled_local_gates": pooled_gates,
        "failure_summary": failure_summary,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    for key in (
        "clock_local_metrics", "pooled_local_metrics",
        "local_paired_bootstrap", "pooled_local_gates", "failure_summary",
    ):
        result[key].to_csv(OUT / f"{key}.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "local_stability_passed": result["metadata"][
            "local_stability_passed"],
        "production_promoted": False,
        "failure_summary": result["failure_summary"].to_dict("records"),
        "failed_pooled_groups": result["pooled_local_gates"].loc[
            lambda frame: ~frame["pass"],
            ["scenario", "slice", "group", "brier_delta",
             "ece_delta_identity", "auc_delta_identity",
             "brier_ci_high", "auc_ci_low"],
        ].to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
