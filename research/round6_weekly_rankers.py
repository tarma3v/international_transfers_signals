"""Packet-U within-currency-week learning-to-rank models."""
from __future__ import annotations

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
from research.round6_broad_cbr_features import load_broad_features
from research.round6_direct_rankers import RankSpec, prequential_scores
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/weekly_rankers")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
STACK_SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
SEED = 20260905


def specs() -> list[RankSpec]:
    return [
        RankSpec("week_pair_fav_compact", "compact", "fav", "rank:pairwise", "week"),
        RankSpec("week_pair_ordinal_compact", "compact", "ordinal", "rank:pairwise", "week"),
        RankSpec("week_pair_benefit_compact", "compact", "benefit", "rank:pairwise", "week"),
        RankSpec("week_pair_fav_benefit_full", "full", "fav_benefit", "rank:pairwise", "week"),
        RankSpec("week_ndcg_fav_benefit_full", "full", "fav_benefit", "rank:ndcg", "week"),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, _paths = load_round5_features()
    broad, broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted_columns = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    trusted = external[:, trusted_columns]
    core = X[:, _core_columns(names)]
    trajectory_columns = np.asarray([
        i for i, name in enumerate(trajectory_names) if not name.startswith("rocket_")
    ], dtype=int)
    matrices = {
        "compact": np.column_stack([core, trusted, broad]),
        "full": np.column_stack([core, trajectory[:, trajectory_columns], trusted, broad]),
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    fav = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    ordinal = np.full(len(index), np.nan)
    benefit = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        values = series[currency].values
        if position + 5 >= len(values):
            continue
        ordinal[row] = float(np.sum(values[position + 1:position + 6] >= values[position]))
        benefit[row] = benefit_forward_only(values, position, 5)
    targets = {
        "fav": fav, "ordinal": ordinal, "benefit": benefit,
        # The prequential function converts benefit to within-training-week
        # rank and adds the binary fav term for this named objective.
        "fav_benefit": benefit,
    }

    raw_outputs, training_log = {}, []
    for spec in specs():
        score, logs = prequential_scores(
            spec, matrices[spec.matrix], targets[spec.target], fav,
            dates, currencies, reach,
        )
        raw_outputs[spec.name] = _outputs(score, fav, dates)
        training_log.extend(logs)
    anchor = _anchor_outputs(X, names, fav, dates)
    with STACK_SOURCE.open("rb") as handle:
        stack = pickle.load(handle)["stack50_benefit50"]

    outputs = dict(raw_outputs)
    for spec in specs():
        raw = raw_outputs[spec.name]
        outputs[f"{spec.name}_anchor25"] = combine_outputs(
            [raw, anchor], (.75, .25), currencies,
        )
        outputs[f"{spec.name}_anchor50"] = combine_outputs(
            [raw, anchor], (.50, .50), currencies,
        )
        outputs[f"{spec.name}_stack50"] = combine_causal(
            [raw, stack], (.50, .50), dates, currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    policies = [row for row in _policy_rows() if row["policy_type"] == "rolling"]
    screen_rows = []
    for candidate, output in outputs.items():
        for policy in policies:
            item = _evaluate(output, (2024,), policy, fav, benefit, dates, currencies)
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
                             fav, benefit, dates, currencies)
            item.update({"period": period, "candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            later_rows.append(item)
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_results.csv", index=False)

    finalists = selected.head(8)
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        finalists, outputs, (2025,), fav, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        finalists, outputs, (2025, 2026), fav, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            fav, dates, currencies, valid_2025, masks_2025, "retrospective_2025",
        ),
        _circular_shift_audit(
            fav, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown_rows = []
    for row in finalists.itertuples(index=False):
        breakdown_rows.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            fav, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("weekly ranker training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "specs": [spec.__dict__ for spec in specs()],
        "query_groups": "currency x ISO week, contiguous before fit",
        "fav_benefit_relevance": "20*binary fav + within-training-week benefit decile",
        "post_2022_only": True,
        "all_training_labels_resolved_before_refit": chronology_ok,
        "next_rate_feature": False,
        "architecture_and_policy_selected_on": 2024,
        "later_period_status": "protocol-controlled retrospective, not pristine",
        "n_broad_features": len(broad_names),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024 TOP\n" + selected[[
        "candidate", "rate", "rolling", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].head(20).to_string(index=False))
    print("\nLATER TOP\n" + later[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "robustness",
    ]].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).groupby("period", sort=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
