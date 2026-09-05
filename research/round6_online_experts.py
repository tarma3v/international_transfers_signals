"""Packet-L delayed-feedback online mixtures of the new CBR experts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round3_online_mixture import HedgeSpec, _online_sequence, _rank_against
from research.round5_features import load_round5_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/online_experts")
SOURCE = Path("results/research/round6/broad_cbr/outputs.pkl")
EXPERTS = (
    "broad_full_extra",
    "broad_compact_tree_consensus",
    "broad_factor_hist_anchor25",
    "packet_e_cbr_macro",
    "anchor_multiscale_locked",
)


def ranked_parts(base: dict, currencies: np.ndarray) -> dict:
    result = {expert: {} for expert in EXPERTS}
    common_years = sorted(set.intersection(*(set(base[expert]) for expert in EXPERTS)))
    for expert in EXPERTS:
        for year in common_years:
            z = base[expert][year]
            ca = np.asarray(z["calib_idx"], dtype=int)
            te = np.asarray(z["test_idx"], dtype=int)
            ca_rank = np.full(len(ca), np.nan)
            te_rank = np.full(len(te), np.nan)
            for currency in np.unique(currencies):
                cm = currencies[ca] == currency
                tm = currencies[te] == currency
                if not cm.any() and not tm.any():
                    continue
                ca_rank[cm] = _rank_against(z["calib_score"][cm], z["calib_score"][cm])
                te_rank[tm] = _rank_against(z["calib_score"][cm], z["test_score"][tm])
            result[expert][year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": ca_rank, "test_score": te_rank,
            }
    return result


def matrix(ranked: dict, year: int, split: str):
    index_key, score_key = f"{split}_idx", f"{split}_score"
    rows = np.asarray(ranked[EXPERTS[0]][year][index_key], dtype=int)
    columns = []
    for expert in EXPERTS:
        part = ranked[expert][year]
        if not np.array_equal(rows, part[index_key]):
            raise AssertionError(f"unaligned expert rows for {year} {split}: {expert}")
        columns.append(np.asarray(part[score_key], dtype=float))
    scores = np.column_stack(columns)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite ranked expert score for {year} {split}")
    return rows, scores


def build_hedge(spec, ranked, dates, currencies, y, reach):
    output = {}
    available = sorted(set.intersection(*(set(ranked[expert]) for expert in EXPERTS)))
    for year in available:
        rows_parts, score_parts, role_parts = [], [], []
        for old_year in available:
            if old_year >= year - 1:
                continue
            old_rows, old_scores = matrix(ranked, old_year, "test")
            rows_parts.append(old_rows)
            score_parts.append(old_scores)
            role_parts.append(np.full(len(old_rows), "history", dtype=object))
        calibration_rows, calibration_scores = matrix(ranked, year, "calib")
        test_rows, test_scores = matrix(ranked, year, "test")
        rows_parts.extend([calibration_rows, test_rows])
        score_parts.extend([calibration_scores, test_scores])
        role_parts.extend([
            np.full(len(calibration_rows), "calib", dtype=object),
            np.full(len(test_rows), "test", dtype=object),
        ])
        rows = np.concatenate(rows_parts)
        scores = np.vstack(score_parts)
        roles = np.concatenate(role_parts)
        combined, restored_roles = _online_sequence(
            spec, rows, scores, roles, dates, currencies, y, reach,
        )
        output[year] = {
            "calib_idx": calibration_rows,
            "test_idx": test_rows,
            "calib_score": combined[restored_roles == "calib"],
            "test_score": combined[restored_roles == "test"],
        }
    return output


def delayed_feedback_check(spec, ranked, dates, currencies, y, reach) -> None:
    cut = np.datetime64("2025-06-30").astype(object)
    original = build_hedge(spec, ranked, dates, currencies, y, reach)
    changed_y = y.copy()
    change = np.asarray([
        np.isfinite(y[row]) and reach[row] > cut for row in range(len(y))
    ])
    changed_y[change] = 1.0 - changed_y[change]
    changed = build_hedge(spec, ranked, dates, currencies, changed_y, reach)
    rows = original[2025]["test_idx"]
    past = np.asarray([dates[row] <= cut for row in rows])
    if not np.array_equal(
        original[2025]["test_score"][past], changed[2025]["test_score"][past],
    ):
        raise AssertionError("unresolved future label changed past online score")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with SOURCE.open("rb") as handle:
        base = pickle.load(handle)
    ranked = ranked_parts(base, currencies)
    specs = [
        HedgeSpec(scope, eta, rho)
        for scope in ("global", "local", "hierarchical")
        for eta in (2.0, 5.0, 10.0, 20.0)
        for rho in (.97, .99, 1.0)
    ]
    delayed_feedback_check(specs[0], ranked, dates, currencies, y, reach)
    outputs = {}
    for spec in specs:
        outputs[spec.name] = build_hedge(
            spec, ranked, dates, currencies, y, reach,
        )
        print(f"built {spec.name}", flush=True)
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

    finalists = selected.head(8)
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
    breakdown = []
    for row in finalists.itertuples(index=False):
        breakdown.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "finalist_breakdown.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "experts": EXPERTS,
        "specs": [spec.__dict__ for spec in specs],
        "loss": "Brier loss on per-currency prior-calibration percentile ranks",
        "feedback_rule": "update only when h5 target reach date <= current date",
        "physical_unresolved_label_corruption_check": True,
        "architecture_and_policy_selected_on": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "candidate", "period", "policy_type", "frequency", "lift",
        "forward_benefit_bps", "corridor_freq_min", "corridor_lift_min",
        "quarter_frequency_min", "quarter_frequency_max", "robustness",
    ]
    print("\n2024 TOP\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "corridor_lift_min",
        "quarter_frequency_min", "robustness",
    ]].head(15).to_string(index=False))
    print("\nLATER\n" + later[columns].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).groupby("period", sort=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
