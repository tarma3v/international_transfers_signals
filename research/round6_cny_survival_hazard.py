"""Packet-BT causal five-step survival and conditional-hazard logits."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import build_cny_basis_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_shadow_nowcast import cross_causality_check
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_survival_hazard")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
SEED = 20260905
ORDER = (
    "survival_direct_h5",
    "survival_cumulative_geometric",
    "survival_cumulative_minimum",
    "survival_hazard_product",
    "survival_hazard_stale20",
    "shadow50_hazard50",
)
TARGET_FIELDS = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "raw_vol_20", "raw_vol_60", "vol_ratio_5_60",
    "annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2",
    "dow_sin", "dow_cos", "gap_days",
)


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.03, max_iter=3000, random_state=SEED),
    )


def survival_targets(index, series):
    result = np.full((len(index), 5), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        values = series[currency].values
        if position + 5 >= len(values):
            continue
        future = values[position + 1:position + 6]
        result[row] = np.cumprod(future >= values[position]).astype(float)
    return result


def prequential_scores(matrix, survival, dates, reach, modes=("cumulative", "hazard"), verbose=True):
    scores = {}
    if "cumulative" in modes:
        scores.update({
            "direct": np.full(len(dates), np.nan),
            "geometric": np.full(len(dates), np.nan),
            "minimum": np.full(len(dates), np.nan),
        })
    if "hazard" in modes:
        scores["hazard"] = np.full(len(dates), np.nan)
    logs = []
    resolved = np.all(np.isfinite(survival), axis=1)
    for start in _quarter_starts():
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= RESET) & resolved
            & np.asarray([value < start for value in reach])
        )
        target = np.flatnonzero(
            (dates >= start) & (dates < end) & resolved
        )
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved survival path admitted")
        cumulative_parts = []
        hazard_parts = []
        for step in range(5):
            if "cumulative" in modes:
                cumulative = _model().fit(matrix[train], survival[train, step])
                cumulative_parts.append(
                    cumulative.predict_proba(matrix[target])[:, 1]
                )
            if "hazard" in modes:
                eligible = train if step == 0 else train[survival[train, step - 1] == 1]
                hazard_target = survival[eligible, step]
                if len(eligible) < 100 or len(np.unique(hazard_target)) < 2:
                    raise AssertionError(f"hazard step {step + 1} lacks training support")
                hazard = _model().fit(matrix[eligible], hazard_target)
                hazard_parts.append(hazard.predict_proba(matrix[target])[:, 1])
        if cumulative_parts:
            cumulative_parts = np.asarray(cumulative_parts)
            scores["direct"][target] = cumulative_parts[-1]
            scores["geometric"][target] = np.exp(
                np.mean(np.log(np.clip(cumulative_parts, 1e-9, 1.0)), axis=0)
            )
            scores["minimum"][target] = np.min(cumulative_parts, axis=0)
        if hazard_parts:
            scores["hazard"][target] = np.prod(np.asarray(hazard_parts), axis=0)
        logs.append({
            "quarter": str(start),
            "n_train": len(train),
            "last_resolved": str(max(reach[train])),
            "n_features": matrix.shape[1],
            "modes": "+".join(modes),
        })
        if verbose:
            print(
                f"  survival {('+'.join(modes)):<18} quarter={start} "
                f"train={len(train):5d} features={matrix.shape[1]:3d}",
                flush=True,
            )
    return scores, logs


def outcome_causality_check(matrix, survival, dates, reach):
    cutoff = np.datetime64("2025-06-30").astype(object)
    first, _ = prequential_scores(
        matrix, survival, dates, reach, verbose=False,
    )
    changed = survival.copy()
    future = np.asarray([value > cutoff for value in reach])
    changed[future] = 1.0 - changed[future]
    second, _ = prequential_scores(
        matrix, changed, dates, reach, verbose=False,
    )
    past = dates <= cutoff
    for candidate in first:
        available = past & np.isfinite(first[candidate])
        np.testing.assert_array_equal(
            first[candidate][available], second[candidate][available]
        )
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    basis, basis_names = build_cny_basis_features(index, history, references["CNY"])
    wave, wave_names = build_waveform_features(index, history)
    target_columns = np.asarray([names.index(name) for name in TARGET_FIELDS])
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ])
    local = np.column_stack([X[:, target_columns], X[:, currency_columns]])
    market = np.column_stack([basis, wave])
    aligned = np.column_stack([local, market])
    stale = np.column_stack([
        local, delayed_by_currency(market, index, rows=20),
    ])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    survival = survival_targets(index, series)
    y = build_targets(series, index)["fav_h5"]
    if not np.array_equal(
        survival[np.isfinite(y), -1], y[np.isfinite(y)]
    ):
        raise AssertionError("five-step survival target differs from fav_h5")
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    if not cross_causality_check(index, series, references["CNY"]):
        raise AssertionError("target/CNY feature causality failed")
    if not outcome_causality_check(aligned, survival, dates, reach):
        raise AssertionError("survival outcome causality failed")

    aligned_scores, aligned_logs = prequential_scores(
        aligned, survival, dates, reach,
    )
    stale_scores, stale_logs = prequential_scores(
        stale, survival, dates, reach, modes=("hazard",),
    )
    outputs = {
        "survival_direct_h5": _outputs(aligned_scores["direct"], y, dates),
        "survival_cumulative_geometric": _outputs(
            aligned_scores["geometric"], y, dates,
        ),
        "survival_cumulative_minimum": _outputs(
            aligned_scores["minimum"], y, dates,
        ),
        "survival_hazard_product": _outputs(aligned_scores["hazard"], y, dates),
        "survival_hazard_stale20": _outputs(stale_scores["hazard"], y, dates),
    }
    with SHADOW.open("rb") as handle:
        shadow = pickle.load(handle)["shadow_close_basis"]
    outputs["shadow50_hazard50"] = combine_causal(
        [shadow, outputs["survival_hazard_product"]],
        (.50, .50), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logs = pd.DataFrame(
        [{"matrix": "aligned", **row} for row in aligned_logs]
        + [{"matrix": "stale20", **row} for row in stale_logs]
    )
    logs.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                outputs[candidate], years, POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_survival_hazard_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(logs.last_resolved) < pd.to_datetime(logs.quarter)
    ))
    aligned_combined = results[
        (results.candidate == "survival_hazard_product")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale_combined = results[
        (results.candidate == "survival_hazard_stale20")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    freshness = bool(
        aligned_combined.lift > stale_combined.lift
        and aligned_combined.corridor_lift_min > stale_combined.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BT",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "model": "StandardScaler + LogisticRegression",
        "C": .03,
        "seed": SEED,
        "target_fields": TARGET_FIELDS,
        "basis_fields": basis_names,
        "waveform_fields": wave_names,
        "stale_control_rows_per_currency": 20,
        "all_training_paths_resolved": chronology_ok,
        "future_outcome_corruption_check": True,
        "aligned_hazard_beats_stale_lift_and_min_currency": freshness,
        "payload_sha256": digest,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("survival training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned hazard accepted as fresh: {freshness}")


if __name__ == "__main__":
    main()
