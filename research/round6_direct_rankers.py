"""Packet-M direct ranking objectives on the broad official-CBR panel."""
from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_external_models import _join_external
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _anchor_outputs, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/direct_rankers")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
CBR_OUTPUTS = Path("results/research/round6/cbr_macro/outputs.pkl")
SEED = 20260905


@dataclass(frozen=True)
class RankSpec:
    name: str
    matrix: str
    target: str
    objective: str
    group_period: str


def specs() -> list[RankSpec]:
    return [
        RankSpec("rank_pair_fav_compact_quarter", "compact", "fav", "rank:pairwise", "quarter"),
        RankSpec("rank_pair_fav_full_month", "full", "fav", "rank:pairwise", "month"),
        RankSpec("rank_pair_ordinal_compact_quarter", "compact", "ordinal", "rank:pairwise", "quarter"),
        RankSpec("rank_ndcg_ordinal_full_month", "full", "ordinal", "rank:ndcg", "month"),
        RankSpec("rank_pair_benefit_compact_quarter", "compact", "benefit", "rank:pairwise", "quarter"),
    ]


def _group_keys(dates, currencies, period):
    if period == "month":
        time_key = [f"{day.year:04d}-{day.month:02d}" for day in dates]
    elif period == "quarter":
        time_key = [f"{day.year:04d}Q{(day.month - 1) // 3 + 1}" for day in dates]
    elif period == "week":
        time_key = [f"{day.isocalendar().year:04d}W{day.isocalendar().week:02d}"
                    for day in dates]
    else:
        raise KeyError(period)
    return np.asarray([f"{currency}:{key}" for currency, key in zip(currencies, time_key)])


def _benefit_deciles(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    for key in np.unique(keys):
        rows = np.where(keys == key)[0]
        order = np.argsort(values[rows], kind="stable")
        ranks = np.empty(len(rows), dtype=float)
        ranks[order] = np.arange(len(rows), dtype=float)
        result[rows] = np.minimum(9.0, np.floor(ranks * 10.0 / max(1, len(rows))))
    return result


def _model(spec: RankSpec) -> XGBRanker:
    return XGBRanker(
        objective=spec.objective,
        n_estimators=420,
        max_depth=3,
        learning_rate=.025,
        min_child_weight=25,
        subsample=.80,
        colsample_bytree=.65,
        reg_lambda=20.0,
        reg_alpha=1.0,
        tree_method="hist",
        eval_metric="ndcg",
        n_jobs=4,
        random_state=SEED,
    )


def prequential_scores(spec, matrix, target, fav, dates, currencies, reach):
    scores = np.full(len(fav), np.nan)
    logs = []
    keys = _group_keys(dates, currencies, spec.group_period)
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & ~np.isnan(fav)
        train = (
            np.asarray([value < start for value in reach])
            & (dates >= RESET) & np.isfinite(target)
        )
        rows = np.where(train)[0]
        if not test.any() or len(rows) < 700:
            continue
        order = np.argsort(keys[rows], kind="stable")
        ordered = rows[order]
        _unique, counts = np.unique(keys[ordered], return_counts=True)
        relevance = target[ordered].astype(float)
        if spec.target == "benefit":
            relevance = _benefit_deciles(relevance, keys[ordered])
        elif spec.target == "fav_benefit":
            benefit_decile = _benefit_deciles(relevance, keys[ordered])
            # Integer, lexicographic relevance is accepted by both pairwise
            # and NDCG objectives; fav_h5 dominates every benefit decile.
            relevance = 20.0 * fav[ordered] + benefit_decile
        model = _model(spec)
        model.fit(matrix[ordered], relevance, group=counts.tolist(), verbose=False)
        target_rows = np.where(test)[0]
        scores[target_rows] = model.predict(matrix[target_rows])
        logs.append({
            "candidate": spec.name,
            "quarter": str(start),
            "n_train": len(ordered),
            "n_groups": len(counts),
            "last_resolved": str(max(reach[ordered])),
            "n_features": matrix.shape[1],
        })
        print(f"  {spec.name:<40} quarter={start} train={len(ordered):5d} "
              f"groups={len(counts):3d}", flush=True)
    return scores, logs


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
    targets = {"fav": fav, "ordinal": ordinal, "benefit": benefit}

    outputs, training_log = {}, []
    for spec in specs():
        score, log = prequential_scores(
            spec, matrices[spec.matrix], targets[spec.target], fav,
            dates, currencies, reach,
        )
        outputs[spec.name] = _outputs(score, fav, dates)
        training_log.extend(log)
    outputs["anchor_multiscale_locked"] = _anchor_outputs(X, names, fav, dates)
    with CBR_OUTPUTS.open("rb") as handle:
        packet_e = pickle.load(handle)
    outputs["packet_e_cbr_anchor50"] = packet_e["cbr_macro_full_hist_anchor50"]
    for spec in specs():
        outputs[f"{spec.name}_anchor25"] = combine_outputs(
            [outputs[spec.name], outputs["anchor_multiscale_locked"]],
            (.75, .25), currencies,
        )
        outputs[f"{spec.name}_baseload25"] = combine_outputs(
            [outputs[spec.name], outputs["packet_e_cbr_anchor50"]],
            (.75, .25), currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    policies = _policy_rows()
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
            item = _evaluate(
                outputs[row.candidate], years, policy, fav, benefit, dates, currencies,
            )
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
    breakdown = []
    for row in finalists.itertuples(index=False):
        breakdown.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            fav, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "finalist_breakdown.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("ranker training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "specs": [spec.__dict__ for spec in specs()],
        "query_groups": "currency x calendar month/quarter, contiguous before fit",
        "benefit_relevance": "within-training-query decile",
        "post_2022_only": True,
        "all_training_labels_resolved_before_refit": chronology_ok,
        "next_rate_feature": False,
        "architecture_and_policy_selected_on": 2024,
        "later_period_status": "protocol-controlled retrospective, not pristine",
        "n_broad_features": len(broad_names),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024 TOP\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "corridor_lift_min",
        "quarter_frequency_min", "robustness",
    ]].head(15).to_string(index=False))
    print("\nLATER\n" + later[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "robustness",
    ]].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).groupby("period", sort=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
