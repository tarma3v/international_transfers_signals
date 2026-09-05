"""Packet-Z nonlinear agreement of classification and benefit experts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/nonlinear_consensus")
STACK_SOURCE = Path("results/research/round6/causal_stacking/outputs.pkl")
RANKER_SOURCE = Path("results/research/round6/direct_rankers/outputs.pkl")
OPERATORS = (
    "geometric", "harmonic", "minimum", "geom_stack25", "geom_stack75",
)


def rank(reference, values):
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def aggregate(values, operator):
    stack = np.clip(values[:, 0], 1e-4, 1.0)
    benefit = np.clip(values[:, 1], 1e-4, 1.0)
    if operator == "geometric":
        return np.sqrt(stack * benefit)
    if operator == "harmonic":
        return 2.0 / (1.0 / stack + 1.0 / benefit)
    if operator == "minimum":
        return np.minimum(stack, benefit)
    if operator == "geom_stack25":
        return np.power(stack, .25) * np.power(benefit, .75)
    if operator == "geom_stack75":
        return np.power(stack, .75) * np.power(benefit, .25)
    raise KeyError(operator)


def component_ranks(parts, dates, currencies):
    result = {}
    for year in sorted(set.intersection(*(set(part) for part in parts))):
        test = np.asarray(parts[0][year]["test_idx"], dtype=int)
        if not all(np.array_equal(test, part[year]["test_idx"]) for part in parts):
            raise AssertionError("unaligned nonlinear-consensus test rows")
        common = set(map(int, parts[0][year]["calib_idx"]))
        for part in parts[1:]:
            common &= set(map(int, part[year]["calib_idx"]))
        calibration = np.asarray(sorted(common), dtype=int)
        calibration_ranks = np.zeros((len(calibration), len(parts)), dtype=float)
        test_ranks = np.zeros((len(test), len(parts)), dtype=float)
        for column, part in enumerate(parts):
            item = part[year]
            part_calibration = np.asarray(item["calib_idx"], dtype=int)
            part_cal_score = np.asarray(item["calib_score"], dtype=float)
            score_by_row = dict(zip(part_calibration, part_cal_score))
            test_values = np.asarray(item["test_score"], dtype=float)
            for currency in np.unique(currencies[test]):
                current = np.where(currencies[test] == currency)[0]
                current = current[np.argsort(dates[test[current]])]
                reference = list(part_cal_score[currencies[part_calibration] == currency])
                if reference:
                    test_ranks[current, column] = rank(reference, test_values[current])
                else:
                    for position in current:
                        test_ranks[position, column] = (
                            rank(reference, [test_values[position]])[0]
                            if len(reference) >= 10 else .5
                        )
                        reference.append(float(test_values[position]))
            if len(calibration):
                values = np.asarray([score_by_row[row] for row in calibration])
                for currency in np.unique(currencies[calibration]):
                    mask = currencies[calibration] == currency
                    calibration_ranks[mask, column] = rank(values[mask], values[mask])
        result[year] = calibration, test, calibration_ranks, test_ranks
    return result


def nonlinear_outputs(parts, dates, currencies):
    ranks = component_ranks(parts, dates, currencies)
    outputs = {operator: {} for operator in OPERATORS}
    for year, (calibration, test, calibration_ranks, test_ranks) in ranks.items():
        for operator in OPERATORS:
            outputs[operator][year] = {
                "calib_idx": calibration, "test_idx": test,
                "calib_score": aggregate(calibration_ranks, operator),
                "test_score": aggregate(test_ranks, operator),
            }
    return outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with STACK_SOURCE.open("rb") as handle:
        stack = pickle.load(handle)["stack_resolved_extra"]
    with RANKER_SOURCE.open("rb") as handle:
        ranker = pickle.load(handle)["rank_pair_benefit_compact_quarter_anchor25"]
    outputs = nonlinear_outputs([stack, ranker], dates, currencies)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    policies = [row for row in _policy_rows() if row["policy_type"] == "rolling"]
    screen_rows = []
    for candidate, output in outputs.items():
        for policy in policies:
            item = _evaluate(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": candidate, **policy})
            screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in screen.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    later_rows = []
    for row in selected.itertuples(index=False):
        policy = _row_policy(row)
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[row.candidate], years, policy,
                             y, benefit, dates, currencies)
            item.update({"period": period, "candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            later_rows.append(item)
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_results.csv", index=False)

    boot_2025, masks_2025, valid_2025 = _bootstrap(
        selected, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        selected, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "retrospective_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown_rows = []
    for row in selected.itertuples(index=False):
        breakdown_rows.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "components": ["stack_resolved_extra", "benefit_ranker_anchor25"],
        "operators": OPERATORS, "labels_used_by_transform": False,
        "missing_2023_stack_calibration": "strictly expanding 2024 ranks",
        "policy_selected_on": 2024, "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024\n" + selected[[
        "candidate", "rate", "rolling", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].to_string(index=False))
    print("\nLATER\n" + later[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "robustness",
    ]].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).to_string(index=False))


if __name__ == "__main__":
    main()
