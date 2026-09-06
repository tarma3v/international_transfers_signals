"""T39: currency h20 history shrink selected only on pre-2025 OOS rows."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import temperature_t34_cold_start_identity_gate as t34
from research import temperature_t37_source_driven_h20_shrink50 as t37
from research import temperature_t38_h20_local_stability as t38
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    CLOCKS,
    EMBARGO_DAYS,
    SCENARIOS,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc


OUT = Path("results/research/temperature/t39_pre2025_currency_shrink")
REGISTERED = Path(
    "research/temperature_t39_pre2025_currency_shrink_registered.md")
T34_OUT = t34.OUT
T37_OUT = t37.OUT
T38_OUT = t38.OUT
CANDIDATE = "pre2025_currency_history_shrink_shadow"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_ALPHA = 0.5
MIN_ROWS = 200
BOOTSTRAP_DRAWS = 500
EPSILON = 1e-6


def _logit_blend(identity, expert, alpha):
    identity = np.clip(np.asarray(identity, dtype=float), EPSILON, 1 - EPSILON)
    expert = np.clip(np.asarray(expert, dtype=float), EPSILON, 1 - EPSILON)
    identity_logit = np.log(identity / (1.0 - identity))
    expert_logit = np.log(expert / (1.0 - expert))
    blended = (1.0 - alpha) * identity_logit + alpha * expert_logit
    return 1.0 / (1.0 + np.exp(-blended))


def _development_split():
    frame = pd.read_csv(T34_OUT / "development_predictions.csv.gz")
    frame["query_date"] = pd.to_datetime(frame.query_date).dt.date
    if pd.to_datetime(frame.query_date).dt.year.max() >= 2025:
        raise AssertionError("open-period row entered pre-2025 development")
    dates = frame.query_date.to_numpy()
    screen_cutoff = (
        dt.date(2024, 1, 1) - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (
        dt.date(2025, 1, 1) - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        (dates >= dt.date(2023, 1, 1))
        & (dates < dt.date(2024, 1, 1))
        & (frame.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (dates >= dt.date(2024, 1, 1))
        & (dates < dt.date(2025, 1, 1))
        & (frame.maturity_ord.to_numpy() < validation_cutoff))
    if (screen & validation).any():
        raise AssertionError("screen and validation overlap")
    return frame, screen, validation, screen_cutoff, validation_cutoff


def _metric_row(part, prediction, baseline):
    result = _probability_metrics(part.target, prediction, baseline)
    return {
        **result,
        "auc_delta": float(
            _fast_auc(part.target, prediction)
            - _fast_auc(part.target, baseline)),
        "ece_delta": float(result["ece"] - result["baseline_ece"]),
    }


def _freeze_map(frame, screen, validation):
    grid_rows = []
    gate_rows = []
    for currency in sorted(frame.currency.unique()):
        screen_part = frame.loc[screen & frame.currency.eq(currency)].copy()
        validation_part = frame.loc[
            validation & frame.currency.eq(currency)].copy()
        screen_predictions = {}
        for alpha in ALPHAS:
            prediction = _logit_blend(
                screen_part.identity_early,
                screen_part.t34_candidate,
                alpha,
            )
            screen_predictions[alpha] = prediction
            row = _metric_row(
                screen_part, prediction,
                _logit_blend(
                    screen_part.identity_early,
                    screen_part.t34_candidate,
                    DEFAULT_ALPHA,
                ),
            )
            grid_rows.append({
                "currency": currency,
                "alpha": alpha,
                "n": int(len(screen_part)),
                **row,
            })
        selected_alpha = min(
            ALPHAS,
            key=lambda alpha: (
                float(np.mean(
                    (screen_predictions[alpha]
                     - screen_part.target.to_numpy(dtype=float)) ** 2)),
                abs(alpha - DEFAULT_ALPHA),
                alpha,
            ),
        )
        base = _logit_blend(
            validation_part.identity_early,
            validation_part.t34_candidate,
            DEFAULT_ALPHA,
        )
        candidate = _logit_blend(
            validation_part.identity_early,
            validation_part.t34_candidate,
            selected_alpha,
        )
        result = _metric_row(validation_part, candidate, base)
        enough_support = bool(
            len(screen_part) >= MIN_ROWS and len(validation_part) >= MIN_ROWS)
        validation_pass = bool(
            enough_support
            and result["brier_delta"] <= 0.001
            and result["logloss_delta"] <= 0.003
            and result["ece_delta"] <= 0.01
            and result["auc_delta"] >= -0.005)
        final_alpha = selected_alpha if validation_pass else DEFAULT_ALPHA
        gate_rows.append({
            "currency": currency,
            "screen_rows": int(len(screen_part)),
            "validation_rows": int(len(validation_part)),
            "selected_alpha": selected_alpha,
            "validation_pass": validation_pass,
            "final_alpha": final_alpha,
            "changed_from_t37": bool(final_alpha != DEFAULT_ALPHA),
            **result,
        })
    return pd.DataFrame(grid_rows), pd.DataFrame(gate_rows)


def _open_predictions(frozen_map, source_frame=None):
    frame = (pd.read_csv(T37_OUT / "predictions.csv.gz")
             if source_frame is None else source_frame.copy())
    frame["query_date"] = pd.to_datetime(frame.query_date).dt.date
    frame["year"] = pd.to_datetime(frame.query_date).dt.year.astype(str)
    frame["currency_year"] = frame.currency.astype(str) + ":" + frame.year
    alpha_map = frozen_map.set_index("currency").final_alpha.to_dict()
    if set(alpha_map) != set(frame.currency.unique()):
        raise AssertionError("frozen currency map incomplete")
    frame[CANDIDATE] = frame[t37.CANDIDATE]
    history = frame.snapshot_source_kind.eq("cbr_history")
    for currency, alpha in alpha_map.items():
        mask = history & frame.currency.eq(currency)
        frame.loc[mask, CANDIDATE] = _logit_blend(
            frame.loc[mask, "identity_h20"],
            frame.loc[mask, "t34_probability"],
            float(alpha),
        )
    frame["modified_from_t37"] = (
        frame[CANDIDATE].to_numpy() != frame[t37.CANDIDATE].to_numpy())
    frame["t39_route_source"] = frame.route_source
    frame.loc[history, "t39_route_source"] = "pre2025_currency_history_shrink"
    return frame


def _rename_candidate(frame):
    output = frame.copy()
    output.columns = [
        str(column).replace(t37.CANDIDATE, CANDIDATE)
        for column in output.columns]
    if "model" in output:
        output["model"] = output.model.replace({t37.CANDIDATE: CANDIDATE})
    return output


def _identity_outputs(predictions):
    aliased = predictions.copy()
    aliased[t37.CANDIDATE] = aliased[CANDIDATE]
    evaluated = t37._evaluate(aliased)
    return {key: _rename_candidate(value)
            if isinstance(value, pd.DataFrame) else value
            for key, value in evaluated.items()}


def _pairwise_metrics(predictions, grouping):
    rows = []
    for key, part in predictions.groupby(grouping, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        result = _metric_row(
            part, part[CANDIDATE], part[t37.CANDIDATE])
        row = dict(zip(grouping, key))
        row.update({"n": int(len(part)), **result})
        row["noninferior"] = bool(
            result["brier_delta"] <= 0.001
            and result["logloss_delta"] <= 0.003
            and result["ece_delta"] <= 0.01
            and result["auc_delta"] >= -0.005)
        rows.append(row)
    return pd.DataFrame(rows)


def _clock_local_metrics(metrics):
    output = metrics[
        metrics.model.eq(CANDIDATE)
        & metrics.slice.isin(t38.LOCAL_SLICES)
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
        for slice_name in t38.LOCAL_SLICES:
            for group, part in scenario_part.groupby(slice_name, sort=True):
                result = _probability_metrics(
                    part.target, part[CANDIDATE], part.identity_h20)
                rows.append({
                    "scenario": scenario,
                    "slice": slice_name,
                    "group": str(group),
                    **result,
                    "auc_delta_identity": float(
                        result["auc"]
                        - _fast_auc(part.target, part.identity_h20)),
                    "ece_delta_identity": float(
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
        ordered.identity_h20.to_numpy(dtype=float), EPSILON, 1 - EPSILON
    ).reshape(len(dates), rows_per_date)
    candidate = np.clip(
        ordered[CANDIDATE].to_numpy(dtype=float), EPSILON, 1 - EPSILON
    ).reshape(len(dates), rows_per_date)
    daily_brier = np.mean((candidate - y) ** 2 - (base - y) ** 2, axis=1)
    daily_logloss = np.mean(
        -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
        + y * np.log(base) + (1 - y) * np.log(1 - base), axis=1)
    rows = []
    # Keep T38's exact resamples so pass-count changes reflect predictions,
    # not a different Monte Carlo draw.
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
        for slice_name in t38.LOCAL_SLICES:
            for group, part in scenario_part.groupby(slice_name, sort=True):
                rows.extend(_bootstrap_one(
                    part, scenario, slice_name, str(group)))
    return pd.DataFrame(rows)


def _pooled_local_gates(metrics, intervals):
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
    for slice_name in t38.LOCAL_SLICES:
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
    development, screen, validation, screen_cutoff, validation_cutoff = (
        _development_split())
    grid, validation_gates = _freeze_map(development, screen, validation)
    frozen_map = validation_gates[[
        "currency", "selected_alpha", "validation_pass", "final_alpha",
        "changed_from_t37",
    ]].copy()
    predictions = _open_predictions(frozen_map)
    if not (pd.to_datetime(predictions.snapshot_source_at, utc=True)
            <= pd.to_datetime(predictions.query_at, utc=True)).all():
        raise AssertionError("source timestamp after query")
    history = predictions.snapshot_source_kind.eq("cbr_history")
    non_history_exact = bool(np.array_equal(
        predictions.loc[~history, CANDIDATE].to_numpy(),
        predictions.loc[~history, t37.CANDIDATE].to_numpy()))

    identity = _identity_outputs(predictions)
    pairwise_state = _pairwise_metrics(predictions, ["scenario", "clock"])
    pairwise_pooled = _pairwise_metrics(predictions, ["scenario"])
    clock_local = _clock_local_metrics(identity["metrics"])
    pooled_local = _pooled_local_metrics(predictions)
    local_intervals = _local_bootstrap(predictions)
    pooled_local_gates = _pooled_local_gates(pooled_local, local_intervals)
    failure_summary = _failure_summary(clock_local, pooled_local_gates)

    changed_currencies = int(frozen_map.changed_from_t37.sum())
    clock_passes = int(clock_local.noninferior.sum())
    pooled_passes = int(pooled_local_gates["pass"].sum())
    year_pass = bool(pooled_local_gates.loc[
        pooled_local_gates.slice.eq("year"), "pass"].all())
    retrospective_repair_passed = bool(
        changed_currencies > 0
        and pairwise_state.noninferior.all()
        and pairwise_pooled.noninferior.all()
        and year_pass
        and clock_passes > 619
        and pooled_passes > 25
        and non_history_exact)

    source_files = [
        REGISTERED,
        Path("research/temperature_t39_pre2025_currency_shrink.py"),
        Path("research/temperature_t39_pre2025_currency_shrink_audit.py"),
        T34_OUT / "metadata.json",
        T34_OUT / "development_predictions.csv.gz",
        T37_OUT / "metadata.json",
        T37_OUT / "predictions.csv.gz",
        T38_OUT / "metadata.json",
        T38_OUT / "clock_local_metrics.csv",
        T38_OUT / "pooled_local_gates.csv",
    ]
    metadata = {
        "packet": "temperature-T39",
        "candidate": CANDIDATE,
        "screen": "mature 2023 OOS",
        "validation": "mature 2024 OOS",
        "open_diagnostic": "2025-2026 retrospective",
        "screen_rows": int(screen.sum()),
        "validation_rows": int(validation.sum()),
        "screen_cutoff_ord": screen_cutoff,
        "validation_cutoff_ord": validation_cutoff,
        "alpha_grid": list(ALPHAS),
        "default_alpha": DEFAULT_ALPHA,
        "min_rows": MIN_ROWS,
        "changed_currencies": changed_currencies,
        "clock_state_passes_vs_t37": int(pairwise_state.noninferior.sum()),
        "clock_state_count_vs_t37": int(len(pairwise_state)),
        "pooled_scenario_passes_vs_t37": int(pairwise_pooled.noninferior.sum()),
        "pooled_scenario_count_vs_t37": int(len(pairwise_pooled)),
        "clock_local_passes": clock_passes,
        "clock_local_count": int(len(clock_local)),
        "pooled_local_passes": pooled_passes,
        "pooled_local_count": int(len(pooled_local_gates)),
        "pooled_year_groups_pass": year_pass,
        "retrospective_repair_passed": retrospective_repair_passed,
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "historical_receipts_certified": False,
        "changes_push_policy": False,
        "changes_expected_future_bps": False,
        "changes_runtime_router": False,
        "checks": {
            "screen_and_validation_pre2025": True,
            "screen_validation_disjoint": True,
            "screen_mature": bool(
                development.loc[screen, "maturity_ord"].max() < screen_cutoff),
            "validation_mature": bool(
                development.loc[validation, "maturity_ord"].max()
                < validation_cutoff),
            "open_targets_not_used_for_selection": True,
            "complete_currency_map": int(len(frozen_map)) == 5,
            "source_no_later_than_query": True,
            "non_history_exact_t37": non_history_exact,
            "probability_bounded": bool(
                predictions[CANDIDATE].between(0.0, 1.0).all()),
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "screen_grid": grid,
        "validation_gates": validation_gates,
        "frozen_map": frozen_map,
        "predictions": predictions,
        **identity,
        "pairwise_state_metrics": pairwise_state,
        "pairwise_pooled_metrics": pairwise_pooled,
        "clock_local_metrics": clock_local,
        "pooled_local_metrics": pooled_local,
        "local_paired_bootstrap": local_intervals,
        "pooled_local_gates": pooled_local_gates,
        "failure_summary": failure_summary,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    frame_outputs = (
        "screen_grid", "validation_gates", "frozen_map", "predictions",
        "metrics", "reliability", "paired_bootstrap", "state_summary",
        "component_metrics", "pooled_metrics", "pooled_paired_bootstrap",
        "pooled_gate", "pairwise_state_metrics", "pairwise_pooled_metrics",
        "clock_local_metrics", "pooled_local_metrics",
        "local_paired_bootstrap", "pooled_local_gates", "failure_summary",
    )
    for key in frame_outputs:
        suffix = ".csv.gz" if key == "predictions" else ".csv"
        result[key].to_csv(
            OUT / f"{key}{suffix}", index=False,
            compression="gzip" if suffix.endswith(".gz") else None)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "frozen_map": result["frozen_map"].to_dict("records"),
        "pairwise_state_passes": (
            f"{result['metadata']['clock_state_passes_vs_t37']}/"
            f"{result['metadata']['clock_state_count_vs_t37']}"),
        "local_passes": (
            f"{result['metadata']['clock_local_passes']}/"
            f"{result['metadata']['clock_local_count']} clock; "
            f"{result['metadata']['pooled_local_passes']}/"
            f"{result['metadata']['pooled_local_count']} pooled"),
        "retrospective_repair_passed": result["metadata"][
            "retrospective_repair_passed"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
