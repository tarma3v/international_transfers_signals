"""Packet-V shared horizon-conditioned five-step barrier models."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_external_models import _join_external
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import (
    RESET, _anchor_outputs, _next_quarter, _outputs, _quarter_starts,
)
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/shared_horizon")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
STACK_SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
SEED = 20260905
KINDS = ("hist", "extra", "xgb")
AGGREGATES = ("minimum", "geomean", "conservative")


def model(kind: str):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.035, max_leaf_nodes=11,
            min_samples_leaf=80, l2_regularization=20.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=450, max_depth=8, min_samples_leaf=40,
            max_features=.55, n_jobs=-1, random_state=SEED,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=380, max_depth=3, learning_rate=.025,
            min_child_weight=35, subsample=.80, colsample_bytree=.60,
            reg_lambda=25.0, reg_alpha=1.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=4,
            random_state=SEED,
        )
    raise KeyError(kind)


def barrier_targets(index, series) -> np.ndarray:
    result = np.full((len(index), 5), np.nan, dtype=float)
    for row, (currency, position, _day) in enumerate(index):
        values = series[currency].values
        if position + 5 >= len(values):
            continue
        result[row] = values[position] <= values[position + 1:position + 6]
    return result


def horizon_matrix(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    repeated = np.repeat(matrix[rows], 5, axis=0)
    horizons = np.tile(np.eye(5, dtype=np.float32), (len(rows), 1))
    return np.column_stack([repeated, horizons])


def aggregate(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(probabilities, 1e-6, 1.0)
    return {
        "minimum": np.min(probabilities, axis=1),
        "geomean": np.exp(np.mean(np.log(clipped), axis=1)),
        "conservative": np.mean(probabilities, axis=1) - .5 * np.std(probabilities, axis=1),
    }


def prequential_scores(kind, matrix, barriers, fav, dates, reach):
    scores = {name: np.full(len(fav), np.nan) for name in AGGREGATES}
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & np.isfinite(fav)
        train = (
            (dates >= RESET)
            & np.asarray([value < start for value in reach])
            & np.all(np.isfinite(barriers), axis=1)
        )
        rows = np.where(train)[0]
        target = np.where(test)[0]
        if len(rows) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved five-step barrier entered training")
        learner = model(kind)
        learner.fit(horizon_matrix(matrix, rows), barriers[rows].reshape(-1))
        probability = learner.predict_proba(horizon_matrix(matrix, target))[:, 1]
        probability = probability.reshape(len(target), 5)
        for name, values in aggregate(probability).items():
            scores[name][target] = values
        logs.append({
            "kind": kind, "quarter": str(start), "n_train_rows": len(rows),
            "n_train_replicas": len(rows) * 5,
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1] + 5,
        })
        print(f"  shared_{kind:<5} quarter={start} train={len(rows):5d} "
              f"replicas={len(rows)*5:6d}", flush=True)
    return scores, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _trajectory_names, _paths = load_round5_features()
    broad, broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted_columns = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    matrix = np.column_stack([X[:, _core_columns(names)], external[:, trusted_columns], broad])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    fav = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    barriers = barrier_targets(index, series)
    reach = target_reach_dates(index, series, 5)

    raw_outputs, training_log = {}, []
    for kind in KINDS:
        kind_scores, logs = prequential_scores(
            kind, matrix, barriers, fav, dates, reach,
        )
        training_log.extend(logs)
        for aggregation, score in kind_scores.items():
            raw_outputs[f"shared_{kind}_{aggregation}"] = _outputs(score, fav, dates)
    anchor = _anchor_outputs(X, names, fav, dates)
    with STACK_SOURCE.open("rb") as handle:
        stack = pickle.load(handle)["stack50_benefit50"]
    outputs = dict(raw_outputs)
    for name, raw in raw_outputs.items():
        outputs[f"{name}_anchor25"] = combine_outputs(
            [raw, anchor], (.75, .25), currencies,
        )
        outputs[f"{name}_stack50"] = combine_causal(
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
        raise AssertionError("shared-horizon training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "kinds": KINDS, "aggregates": AGGREGATES,
        "training_replication": "one row per horizon 1..5 with one-hot horizon",
        "all_five_labels_resolved_before_refit": chronology_ok,
        "matrix_shape": list(matrix.shape),
        "n_broad_features": len(broad_names),
        "next_rate_feature": False,
        "architecture_and_policy_selected_on": 2024,
        "later_period_status": "protocol-controlled retrospective, not pristine",
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
