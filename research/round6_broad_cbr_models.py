"""Packet-G models on a broad, target-free, official-CBR FX panel."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_external_models import _join_external
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _anchor_outputs, _outputs
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import causality_check, load_broad_features
from research.round6_resolved_models import (
    Spec,
    _bootstrap,
    _breakdown,
    _choose,
    _evaluate,
    _policy_rows,
    _row_policy,
    prequential_scores,
)


OUT = Path("results/research/round6/broad_cbr")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
CBR_OUTPUTS = Path("results/research/round6/cbr_macro/outputs.pkl")
DATA_MANIFEST = OUT / "data_manifest.json"


def specs() -> list[Spec]:
    return [
        Spec("broad_factor_hist", "hist", "factor"),
        Spec("broad_factor_extra", "extra", "factor"),
        Spec("broad_compact_hist", "hist", "compact"),
        Spec("broad_compact_extra", "extra", "compact"),
        Spec("broad_full_hist", "hist", "full"),
        Spec("broad_full_extra", "extra", "full"),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, _paths = load_round5_features()
    broad, broad_names, references = load_broad_features(index, series)
    # This deliberately rebuilds from the physical source and corrupts every
    # observation after the cutoff.  It is slower than a symbolic audit but
    # catches accidental backward-fill and whole-sample normalization.
    causality_check(index, series, dt.date(2025, 6, 30))

    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted_columns = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    trusted = external[:, trusted_columns]
    trusted_names = [external_names[i] for i in trusted_columns]
    core = X[:, _core_columns(names)]
    trajectory_columns = np.asarray([
        i for i, name in enumerate(trajectory_names) if not name.startswith("rocket_")
    ], dtype=int)
    factor_columns = np.asarray([
        i for i, name in enumerate(broad_names)
        if name.startswith("broad_factor_")
        or name.startswith("broad_exusd_")
        or name.startswith("broad_target_minus_")
    ], dtype=int)
    matrices = {
        "factor": np.column_stack([core, trusted, broad[:, factor_columns]]),
        "compact": np.column_stack([core, trusted, broad]),
        "full": np.column_stack([
            core, trajectory[:, trajectory_columns], trusted, broad,
        ]),
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    outputs, training_log = {}, []
    for spec in specs():
        scores, rows = prequential_scores(spec, matrices[spec.matrix], y, dates, reach)
        outputs[spec.name] = _outputs(scores, y, dates)
        training_log.extend(rows)
    outputs["anchor_multiscale_locked"] = _anchor_outputs(X, names, y, dates)
    with CBR_OUTPUTS.open("rb") as handle:
        frozen_cbr = pickle.load(handle)
    outputs["packet_e_cbr_macro"] = frozen_cbr["cbr_macro_full_hist"]
    outputs["packet_e_cbr_anchor50"] = frozen_cbr["cbr_macro_full_hist_anchor50"]

    ensembles = {
        "broad_factor_hist_anchor25": (
            ("broad_factor_hist", "anchor_multiscale_locked"), (.75, .25),
        ),
        "broad_factor_hist_anchor50": (
            ("broad_factor_hist", "anchor_multiscale_locked"), (.50, .50),
        ),
        "broad_full_hist_anchor25": (
            ("broad_full_hist", "anchor_multiscale_locked"), (.75, .25),
        ),
        "broad_full_hist_anchor50": (
            ("broad_full_hist", "anchor_multiscale_locked"), (.50, .50),
        ),
        "broad_factor_tree_consensus": (
            ("broad_factor_hist", "broad_factor_extra"), (.50, .50),
        ),
        "broad_compact_tree_consensus": (
            ("broad_compact_hist", "broad_compact_extra"), (.50, .50),
        ),
        "broad_full_tree_consensus": (
            ("broad_full_hist", "broad_full_extra"), (.50, .50),
        ),
        "broad_full_hist_packet_e50": (
            ("broad_full_hist", "packet_e_cbr_macro"), (.50, .50),
        ),
    }
    for name, (members, weights) in ensembles.items():
        outputs[name] = combine_outputs(
            [outputs[member] for member in members], weights, currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    policy_rows = _policy_rows()
    grid_rows = []
    for candidate, output in outputs.items():
        for policy in policy_rows:
            item = _evaluate(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": candidate, **policy})
            grid_rows.append(item)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in grid.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    confirmation, auditing, combined = [], [], []
    for row in selected.itertuples(index=False):
        policy = _row_policy(row)
        for years, target in (
            ((2025,), confirmation), ((2026,), auditing), ((2025, 2026), combined),
        ):
            item = _evaluate(
                outputs[row.candidate], years, policy, y, benefit, dates, currencies,
            )
            item.update({"candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            if years == (2025,):
                item["clears_1p30_gate"] = bool(
                    item["lift"] >= 1.30
                    and 1.00 <= item["frequency"] <= 2.00
                    and item["corridor_freq_min"] >= .80
                    and item["quarter_frequency_min"] >= .70
                    and item["forward_benefit_bps"] > 0
                )
            target.append(item)
    confirm = pd.DataFrame(confirmation).sort_values(
        ["clears_1p30_gate", "robustness", "lift"], ascending=False,
    )
    passed = set(confirm.loc[confirm.clears_1p30_gate, "candidate"])
    audit = pd.DataFrame(auditing)
    audit["passed_2025"] = audit.candidate.isin(passed)
    audit = audit.sort_values(["passed_2025", "robustness", "lift"], ascending=False)
    together = pd.DataFrame(combined)
    together["passed_2025"] = together.candidate.isin(passed)
    together = together.sort_values(
        ["passed_2025", "macro_year_lift", "lift"], ascending=False,
    )
    confirm.to_csv(OUT / "confirm_2025.csv", index=False)
    audit.to_csv(OUT / "audit_2026.csv", index=False)
    together.to_csv(OUT / "combined_2025_2026.csv", index=False)

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
            y, dates, currencies, valid_2025, masks_2025, "confirmation_2025",
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

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("broad-CBR training chronology failed")
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    (OUT / "protocol.json").write_text(json.dumps({
        "source": "Bank of Russia official XML API",
        "source_payload_sha256": manifest["payload_sha256"],
        "reference_currencies": sorted(references),
        "targets_excluded": ["AMD", "KGS", "KZT", "TJS", "UZS"],
        "selection_used_target_labels": False,
        "asof_rule": "reference_date <= target_row_date",
        "physical_future_corruption_check": True,
        "next_rate_feature": False,
        "trusted_cbr_columns": trusted_names,
        "n_broad_features": len(broad_names),
        "n_factor_features": len(factor_columns),
        "matrix_shapes": {key: list(value.shape) for key, value in matrices.items()},
        "policy_selected_on": 2024,
        "confirmation": 2025,
        "audit": 2026,
        "n_architectures": len(outputs),
        "n_policies": len(policy_rows),
        "chronology_ok": chronology_ok,
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n2024 SELECTED\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "corridor_lift_min",
        "quarter_frequency_min", "robustness",
    ]].head(18).to_string(index=False))
    print("\n2025 CONFIRMATION\n" + confirm[[
        "candidate", "policy_type", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "clears_1p30_gate",
    ]].head(18).to_string(index=False))
    print("\n2026 AUDIT\n" + audit[[
        "candidate", "policy_type", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "passed_2025",
    ]].head(18).to_string(index=False))


if __name__ == "__main__":
    main()
