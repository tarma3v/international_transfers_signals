"""Packet-O purged stacking on causal, prequential expert scores."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_resolved_features import load_resolved_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/causal_stacking")
SEED = 20260905
EXPERTS = (
    "benefit_ranker_anchor25",
    "broad_full_extra",
    "broad75_baseload25",
    "cbr_baseload",
    "ordinal_ndcg_baseload25",
)


@dataclass(frozen=True)
class StackSpec:
    name: str
    kind: str
    matrix: str


def specs() -> list[StackSpec]:
    return [
        StackSpec("stack_experts_logit", "logit", "experts"),
        StackSpec("stack_experts_hist", "hist", "experts"),
        StackSpec("stack_core_hist", "hist", "core"),
        StackSpec("stack_resolved_hist", "hist", "resolved"),
        StackSpec("stack_resolved_extra", "extra", "resolved"),
    ]


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.03, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.03, max_leaf_nodes=7,
            min_samples_leaf=50, l2_regularization=20.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=24,
            max_features=.70, n_jobs=-1, random_state=SEED,
        )
    raise KeyError(kind)


def load_outputs():
    with Path("results/research/round6/direct_rankers/outputs.pkl").open("rb") as handle:
        rankers = pickle.load(handle)
    with Path("results/research/round6/broad_cbr/outputs.pkl").open("rb") as handle:
        broad = pickle.load(handle)
    with Path("results/research/round6/broad_cbr_hybrid/outputs.pkl").open("rb") as handle:
        hybrid = pickle.load(handle)
    return {
        EXPERTS[0]: rankers["rank_pair_benefit_compact_quarter_anchor25"],
        EXPERTS[1]: broad["broad_full_extra"],
        EXPERTS[2]: hybrid["broad75_baseload25"],
        EXPERTS[3]: rankers["packet_e_cbr_anchor50"],
        EXPERTS[4]: rankers["rank_ndcg_ordinal_full_month_baseload25"],
    }


def _row_scores(output, n_rows):
    score = np.full(n_rows, np.nan)
    # The 2024 calibration block contains the prequential 2023 scores.
    score[output[2024]["calib_idx"]] = output[2024]["calib_score"]
    for year in (2024, 2025, 2026):
        score[output[year]["test_idx"]] = output[year]["test_score"]
    return score


def build_score_features(outputs, dates, currencies):
    raw = np.column_stack([
        _row_scores(outputs[name], len(dates)) for name in EXPERTS
    ])
    ranked = np.full_like(raw, np.nan, dtype=float)
    for currency in np.unique(currencies):
        rows = np.where(currencies == currency)[0]
        rows = rows[np.argsort(dates[rows])]
        for column in range(raw.shape[1]):
            history = []
            for row in rows:
                value = raw[row, column]
                if not np.isfinite(value):
                    continue
                reference = np.sort(np.asarray(history[-250:], dtype=float))
                ranked[row, column] = (
                    np.searchsorted(reference, value, side="right") / len(reference)
                    if len(reference) >= 10 else .5
                )
                history.append(float(value))
    rows = []
    names = [f"expert_rank_{name}" for name in EXPERTS]
    for values in ranked:
        if not np.all(np.isfinite(values)):
            rows.append([np.nan] * (len(EXPERTS) + 5 + len(EXPERTS) * 3 + 10))
            continue
        features = list(values)
        features.extend([
            float(np.mean(values)), float(np.std(values)), float(np.min(values)),
            float(np.max(values)), float(np.max(values) - np.min(values)),
        ])
        features.extend([float(value >= threshold) for value in values
                         for threshold in (.70, .80, .90)])
        features.extend([
            float(values[left] - values[right])
            for left in range(len(values)) for right in range(left + 1, len(values))
        ])
        rows.append(features)
    names.extend(["expert_mean", "expert_std", "expert_min", "expert_max", "expert_range"])
    names.extend([
        f"expert_tail_{name}_{int(threshold * 100)}"
        for name in EXPERTS for threshold in (.70, .80, .90)
    ])
    names.extend([
        f"expert_diff_{EXPERTS[left]}__{EXPERTS[right]}"
        for left in range(len(EXPERTS)) for right in range(left + 1, len(EXPERTS))
    ])
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape[1] != len(names):
        raise AssertionError(f"stack feature schema mismatch {matrix.shape[1]} != {len(names)}")
    return matrix, names, raw


def score_feature_causality_check(outputs, dates, currencies):
    full, names, _raw = build_score_features(outputs, dates, currencies)
    cut = dt.date(2025, 6, 30)
    changed = {}
    for name, output in outputs.items():
        clone = {year: {key: np.asarray(value).copy() for key, value in part.items()}
                 for year, part in output.items()}
        rows = clone[2025]["test_idx"]
        future = np.asarray([dates[row] > cut for row in rows])
        clone[2025]["test_score"][future] = np.linspace(-1000, 1000, int(future.sum()))
        changed[name] = clone
    changed_matrix, changed_names, _changed_raw = build_score_features(
        changed, dates, currencies,
    )
    past = np.asarray([day <= cut for day in dates])
    if names != changed_names or not np.array_equal(
        np.nan_to_num(full[past], nan=-999),
        np.nan_to_num(changed_matrix[past], nan=-999),
    ):
        raise AssertionError("future expert score changed a past stack feature")


def prequential_scores(spec, matrix, y, dates, reach):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        finite = np.all(np.isfinite(matrix), axis=1)
        train = (
            (dates >= dt.date(2023, 1, 1))
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & finite
        )
        test = (dates >= start) & (dates < end) & np.isfinite(y) & finite
        rows = np.where(train)[0]
        target = np.where(test)[0]
        if len(rows) < 700 or not len(target):
            continue
        model = _model(spec.kind)
        model.fit(matrix[rows], y[rows])
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": spec.name, "quarter": str(start),
            "n_train": len(rows), "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        print(f"  {spec.name:<28} quarter={start} train={len(rows):5d} ",
              f"features={matrix.shape[1]}", flush=True)
    return scores, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    _X2, _names2, index2, _series2, resolved, resolved_names = load_resolved_features()
    if index != index2:
        raise AssertionError("stack/resolved row index mismatch")
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    base_outputs = load_outputs()
    expert_features, expert_names, _raw = build_score_features(
        base_outputs, dates, currencies,
    )
    score_feature_causality_check(base_outputs, dates, currencies)
    core = X[:, _core_columns(names)]
    matrices = {
        "experts": expert_features,
        "core": np.column_stack([expert_features, core]),
        "resolved": np.column_stack([expert_features, core, resolved]),
    }

    outputs, training_log = {}, []
    for spec in specs():
        score, logs = prequential_scores(spec, matrices[spec.matrix], y, dates, reach)
        outputs[spec.name] = _outputs(score, y, dates)
        training_log.extend(logs)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

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

    finalists = selected.head(5)
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

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("stack training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "experts": EXPERTS,
        "expert_feature_names": expert_names,
        "resolved_feature_count": len(resolved_names),
        "specs": [asdict(spec) for spec in specs()],
        "score_rank_reference": "at most 250 strictly earlier scores per currency",
        "base_scores_prequential": True,
        "physical_future_score_corruption_check": True,
        "all_training_labels_resolved_before_refit": chronology_ok,
        "policy_selected_on": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "corridor_lift_min",
        "quarter_frequency_min", "robustness",
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
