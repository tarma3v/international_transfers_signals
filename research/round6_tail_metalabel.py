"""Packet-AA causal meta-labeling on a prequential candidate tail."""
from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_causal_stacking import build_score_features, load_outputs
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_features import load_resolved_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/tail_metalabel")
PRIMARY_SOURCE = Path("results/research/round6/direct_rankers/outputs.pkl")
PRIMARY = "rank_pair_benefit_compact_quarter_anchor25"
SEED = 20260905


@dataclass(frozen=True)
class Spec:
    kind: str
    tail: float

    @property
    def name(self):
        return f"tail{int(self.tail*100):02d}_{self.kind}"


def specs():
    return [Spec(kind, tail) for tail in (.50, .65) for kind in ("hist", "extra")]


def model(kind):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.03, max_leaf_nodes=7,
            min_samples_leaf=40, l2_regularization=25.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=20,
            max_features=.65, n_jobs=-1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(spec, matrix, primary_rank, y, dates, reach):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        finite = np.all(np.isfinite(matrix), axis=1) & np.isfinite(primary_rank)
        train = (
            (dates >= pd.Timestamp("2023-01-01").date())
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & finite & (primary_rank >= spec.tail)
        )
        test = (dates >= start) & (dates < end) & np.isfinite(y) & finite
        rows, target = np.where(train)[0], np.where(test)[0]
        if len(rows) < 300 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved meta-label admitted")
        learner = model(spec.kind)
        learner.fit(matrix[rows], y[rows])
        scores[target] = learner.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": spec.name, "quarter": str(start),
            "tail": spec.tail, "n_train": len(rows),
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        print(f"  {spec.name:<16} quarter={start} train={len(rows):4d}", flush=True)
    return scores, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    _X2, _names2, index2, _series2, resolved, _resolved_names = load_resolved_features()
    if index != index2:
        raise AssertionError("tail meta-label row mismatch")
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    expert_features, expert_names, _raw = build_score_features(
        load_outputs(), dates, currencies,
    )
    primary_rank = expert_features[:, 0]
    matrix = np.column_stack([expert_features, X[:, _core_columns(names)], resolved])
    with PRIMARY_SOURCE.open("rb") as handle:
        primary = pickle.load(handle)[PRIMARY]

    raw_outputs, training_log = {}, []
    for spec in specs():
        score, logs = prequential_scores(
            spec, matrix, primary_rank, y, dates, reach,
        )
        raw_outputs[spec.name] = _outputs(score, y, dates)
        training_log.extend(logs)
    outputs = dict(raw_outputs)
    for name, raw in raw_outputs.items():
        outputs[f"{name}_primary25"] = combine_causal(
            [raw, primary], (.75, .25), dates, currencies,
        )
        outputs[f"{name}_primary50"] = combine_causal(
            [raw, primary], (.50, .50), dates, currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

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
        raise AssertionError("tail meta-label chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "primary": PRIMARY, "tail_thresholds": [.50, .65],
        "meta_learners": ["hist", "extra"],
        "expert_feature_names": expert_names,
        "primary_rank": "at most 250 strictly earlier scores per currency",
        "all_training_labels_resolved_before_refit": chronology_ok,
        "next_rate_feature": False, "policy_selected_on": 2024,
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
