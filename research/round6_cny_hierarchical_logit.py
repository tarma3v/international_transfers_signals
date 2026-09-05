"""Packet-AT one-stage partial-pooling logit with currency interactions."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    Spec,
    _bootstrap,
    _breakdown,
    _evaluate,
    prequential_scores,
)


OUT = Path("results/research/round6/cny_hierarchical_logit")
GLOBAL = Path("results/research/round6/cny_explainable/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ORDER = (
    "hierarchical_interaction_logit",
    "global50_hierarchical50",
    "primary75_hierarchical25",
)


def hierarchical_matrix(X, names, moex, moex_names):
    """Return fixed main effects plus L2-shrunk currency deviations."""
    intraday_columns = np.asarray([
        i for i, name in enumerate(moex_names)
        if "cnyrub_tom" in name and any(token in name for token in (
            "_open_close", "_intraday_range", "_close_wap",
            "_overnight_gap", "_log_trades",
        ))
    ], dtype=int)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent_names = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent_names])
    numeric = np.column_stack([
        moex[:, intraday_columns], X[:, transparent_columns],
    ])
    currency = X[:, currency_columns]
    interactions = np.column_stack([
        numeric * currency[:, [i]] for i in range(currency.shape[1])
    ])
    matrix = np.column_stack([numeric, currency, interactions])
    feature_names = (
        [moex_names[i] for i in intraday_columns]
        + list(transparent_names)
        + [names[i] for i in currency_columns]
        + [
            f"{base_name}__x__{names[currency_columns[i]]}"
            for i in range(currency.shape[1])
            for base_name in (
                [moex_names[j] for j in intraday_columns]
                + list(transparent_names)
            )
        ]
    )
    if matrix.shape[1] != len(feature_names):
        raise AssertionError("hierarchical interaction schema changed")
    return matrix, feature_names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    matrix, feature_names = hierarchical_matrix(X, names, moex, moex_names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    score, logs = prequential_scores(
        Spec(ORDER[0], "logit", "hierarchical"), matrix, y, dates, reach,
    )
    outputs = {ORDER[0]: _outputs(score, y, dates)}
    with GLOBAL.open("rb") as handle:
        global_logit = pickle.load(handle)["market_anchor_logit"]
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs[ORDER[1]] = combine_causal(
        [global_logit, outputs[ORDER[0]]], (.5, .5), dates, currencies,
    )
    outputs[ORDER[2]] = combine_causal(
        [primary, outputs[ORDER[0]]], (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

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
        raise AssertionError("training used unresolved h=5 labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AT",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "base_numeric_features": 14,
        "currency_indicators": 5,
        "interaction_features": 70,
        "matrix_features": len(feature_names),
        "logistic_penalty": {"type": "L2", "C": 0.025},
        "blend_weights": {ORDER[1]: [0.5, 0.5], ORDER[2]: [0.75, 0.25]},
        "payload_sha256": digest,
        "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
