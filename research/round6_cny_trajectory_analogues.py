"""Packet-BD historical analogues in raw target/CNY trajectory space."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_reliability_surface import (
    CURRENCY_PENALTY,
    GLOBAL_K,
    LOCAL_K,
    SHRINKAGE,
    _hit_lcb,
)
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_trajectory_analogues")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
TARGET_NAMES = (
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "pct_range_30", "pct_range_90", "pct_range_180",
    "vol_10", "vol_30", "vol_90",
)
CNY_SUFFIXES = (
    "_ret_1", "_ret_2", "_ret_5", "_ret_10", "_ret_20",
    "_vol_5", "_vol_20", "_vol_60", "_open_close", "_intraday_range",
    "_close_wap", "_overnight_gap", "_log_trades",
)
ORDER = (
    "target_analogue_lcb",
    "cny_analogue_lcb",
    "joint_analogue_lcb",
    "primary75_joint_analogue25",
)


def build_trajectory_matrices(X, names, moex, moex_names):
    target_columns = np.asarray([names.index(name) for name in TARGET_NAMES])
    cny_columns = []
    for suffix in CNY_SUFFIXES:
        matches = [
            i for i, name in enumerate(moex_names)
            if name.startswith("moex_cnyrub_tom_") and name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise AssertionError(f"trajectory CNY field {suffix}: {matches}")
        cny_columns.append(matches[0])
    target = np.asarray(X[:, target_columns], dtype=float)
    cny = np.asarray(moex[:, cny_columns], dtype=float)
    return {
        "target": target,
        "cny": cny,
        "joint": np.column_stack([target, cny]),
    }, {
        "target": list(TARGET_NAMES),
        "cny": [moex_names[i] for i in cny_columns],
    }


def _scale(train_values, test_values):
    median = np.median(train_values, axis=0)
    q25, q75 = np.quantile(train_values, (.25, .75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return (
        np.clip((train_values - median) / scale, -10.0, 10.0),
        np.clip((test_values - median) / scale, -10.0, 10.0),
    )


def _nearest(train_values, test_value, train_currencies, currency, k, local):
    candidates = np.flatnonzero(train_currencies == currency) if local else np.arange(len(train_values))
    if not len(candidates):
        return candidates
    delta = train_values[candidates] - test_value
    distance = np.mean(delta * delta, axis=1)
    if not local:
        distance += CURRENCY_PENALTY * (train_currencies[candidates] != currency)
    count = min(k, len(candidates))
    chosen = np.argpartition(distance, count - 1)[:count]
    return candidates[chosen]


def analogue_scores(matrix, y, dates, currencies, reach, verbose=True):
    score = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        finite = np.all(np.isfinite(matrix), axis=1)
        test = (dates >= start) & (dates < end) & finite & np.isfinite(y)
        train_mask = (
            (dates >= RESET) & finite
            & np.asarray([value < start for value in reach]) & np.isfinite(y)
        )
        train, target = np.flatnonzero(train_mask), np.flatnonzero(test)
        if not len(target) or len(train) < 700:
            continue
        scaled_train, scaled_test = _scale(matrix[train], matrix[target])
        train_currencies = currencies[train]
        for local_row, row in enumerate(target):
            pooled_position = _nearest(
                scaled_train, scaled_test[local_row], train_currencies,
                currencies[row], GLOBAL_K, False,
            )
            local_position = _nearest(
                scaled_train, scaled_test[local_row], train_currencies,
                currencies[row], LOCAL_K, True,
            )
            pooled = train[pooled_position]
            local = train[local_position]
            pooled_lcb = _hit_lcb(y[pooled])
            local_lcb = _hit_lcb(y[local]) if len(local) else pooled_lcb
            local_weight = len(local) / (len(local) + SHRINKAGE)
            score[row] = local_weight * local_lcb + (1.0 - local_weight) * pooled_lcb
        logs.append({
            "quarter": str(start),
            "n_train": len(train),
            "n_test": len(target),
            "n_features": matrix.shape[1],
            "first_train": str(min(dates[train])),
            "last_resolved": str(max(reach[train])),
        })
        if verbose:
            print(
                f"  analogue features={matrix.shape[1]:2d} quarter={start} "
                f"train={len(train):5d} test={len(target):4d}",
                flush=True,
            )
    return score, logs


def future_outcome_causality_check(matrix, y, dates, currencies, reach):
    unique_dates = np.asarray(sorted(set(dates)))
    cutoff = unique_dates[int(len(unique_dates) * .72)]
    first, _ = analogue_scores(matrix, y, dates, currencies, reach, verbose=False)
    changed = y.copy()
    unresolved = np.asarray([value >= cutoff for value in reach]) & np.isfinite(y)
    changed[unresolved] = 1.0 - changed[unresolved]
    second, _ = analogue_scores(matrix, changed, dates, currencies, reach, verbose=False)
    np.testing.assert_array_equal(first[dates < cutoff], second[dates < cutoff])
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    matrices, feature_names = build_trajectory_matrices(X, names, moex, moex_names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    outputs, logs = {}, []
    for candidate, matrix_name in zip(ORDER[:3], ("target", "cny", "joint")):
        score, part = analogue_scores(
            matrices[matrix_name], y, dates, currencies, reach,
        )
        outputs[candidate] = _outputs(score, y, dates)
        for row in part:
            logs.append({"candidate": candidate, **row})
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs[ORDER[3]] = combine_causal(
        [primary, outputs[ORDER[2]]], (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)
    causality_ok = future_outcome_causality_check(
        matrices["joint"], y, dates, currencies, reach,
    )

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
        y, dates, currencies, valid, masks, "trajectory_analogues_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("trajectory analogue used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BD",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "target_feature_names": feature_names["target"],
        "cny_feature_names": feature_names["cny"],
        "scaling": "quarterly training-only median/IQR, clipped +/-10",
        "global_neighbours": GLOBAL_K,
        "local_neighbours": LOCAL_K,
        "currency_penalty": CURRENCY_PENALTY,
        "local_shrinkage_denominator": SHRINKAGE,
        "beta_prior": [0.5, 0.5],
        "primary_joint_weights": [0.75, 0.25],
        "payload_sha256": digest,
        "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
        "future_outcome_corruption_check": causality_ok,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
