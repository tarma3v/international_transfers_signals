"""Packet-P fixed blends of classification and forward-benefit scores."""
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


OUT = Path("results/research/round6/multiobjective_blend")


def _rank(reference, values):
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def combine_causal(parts, weights, dates, currencies):
    """Combine parts even when one has no preceding calibration block.

    Each component uses its own prior calibration scores.  If those are absent
    (the 2024 stack), its test ranks are expanding and strictly past-only.
    """
    result = {}
    for year in sorted(set.intersection(*(set(part) for part in parts))):
        test = np.asarray(parts[0][year]["test_idx"], dtype=int)
        if not all(np.array_equal(test, part[year]["test_idx"]) for part in parts):
            raise AssertionError(f"unaligned blend test rows in {year}")
        calibration_sets = [set(map(int, part[year]["calib_idx"])) for part in parts]
        common_calibration = sorted(set.intersection(*calibration_sets))
        calibration = np.asarray(common_calibration, dtype=int)
        calibration_score = np.zeros(len(calibration), dtype=float)
        test_score = np.zeros(len(test), dtype=float)
        for part, weight in zip(parts, weights):
            item = part[year]
            part_calibration = np.asarray(item["calib_idx"], dtype=int)
            part_cal_score = np.asarray(item["calib_score"], dtype=float)
            score_by_row = dict(zip(part_calibration, part_cal_score))
            for currency in np.unique(currencies[test]):
                current = np.where(currencies[test] == currency)[0]
                current = current[np.argsort(dates[test[current]])]
                ref_mask = currencies[part_calibration] == currency
                reference = list(part_cal_score[ref_mask])
                if len(reference):
                    test_score[current] += weight * _rank(
                        reference, np.asarray(item["test_score"])[current],
                    )
                else:
                    for position in current:
                        value = float(np.asarray(item["test_score"])[position])
                        test_score[position] += weight * (
                            _rank(reference, np.asarray([value]))[0]
                            if len(reference) >= 10 else .5
                        )
                        reference.append(value)
            if len(calibration):
                values = np.asarray([score_by_row[row] for row in calibration])
                for currency in np.unique(currencies[calibration]):
                    mask = currencies[calibration] == currency
                    calibration_score[mask] += weight * _rank(values[mask], values[mask])
        result[year] = {
            "calib_idx": calibration, "test_idx": test,
            "calib_score": calibration_score, "test_score": test_score,
        }
    return result


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
    with Path("results/research/round6/causal_stacking/outputs.pkl").open("rb") as handle:
        stack = pickle.load(handle)["stack_resolved_extra"]
    with Path("results/research/round6/direct_rankers/outputs.pkl").open("rb") as handle:
        rankers = pickle.load(handle)
    ranker = rankers["rank_pair_benefit_compact_quarter_anchor25"]
    baseload = rankers["packet_e_cbr_anchor50"]
    outputs = {
        "stack_resolved_extra": stack,
        "benefit_ranker_anchor25": ranker,
        "packet_e_cbr_anchor50": baseload,
    }
    for stack_weight in (.25, .50, .75):
        outputs[f"stack{int(stack_weight*100):02d}_benefit{int((1-stack_weight)*100):02d}"] = combine_causal(
            [stack, ranker], (stack_weight, 1.0 - stack_weight), dates, currencies,
        )
    outputs["stack50_benefit25_baseload25"] = combine_causal(
        [stack, ranker, baseload], (.50, .25, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    policies = _policy_rows()
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
            item = _evaluate(
                outputs[row.candidate], years, policy, y, benefit, dates, currencies,
            )
            item.update({"period": period, "candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            later_rows.append(item)
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_results.csv", index=False)

    finalists = selected.head(7)
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        finalists, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        finalists, outputs, (2025, 2026), y, benefit, dates, currencies,
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
    for row in finalists.itertuples(index=False):
        breakdown_rows.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "components": [
            "stack_resolved_extra", "benefit_ranker_anchor25",
            "packet_e_cbr_anchor50",
        ],
        "weights": [(.25, .75), (.50, .50), (.75, .25), (.50, .25, .25)],
        "combination": "per-currency ranks against prior calibration block",
        "missing_2023_stack_calibration": (
            "2024 stack test ranks use strictly earlier 2024 stack scores"
        ),
        "weights_fitted": False,
        "policy_selected_on": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "robustness",
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
