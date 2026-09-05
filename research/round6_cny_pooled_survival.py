"""Packet-BX pooled discrete-time survival models."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import build_cny_basis_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_survival_hazard import TARGET_FIELDS, survival_targets
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_pooled_survival")
SEPARATE = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
SEED = 20260905
ORDER = (
    "pooled_hazard_logit",
    "pooled_hazard_interaction_logit",
    "pooled_hazard_hist",
    "pooled_hazard_stale20_logit",
    "separate_hazard_product",
    "cumulative50_pooled50",
)


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.03, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=.035,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=20.0,
            random_state=SEED,
        )
    raise KeyError(kind)


def _augment(matrix, rows, step, interactions):
    onehot = np.zeros((len(rows), 5), dtype=float)
    onehot[:, step] = 1.0
    parts = [matrix[rows], onehot]
    if len(interactions):
        centered_step = (step - 2.0) / 2.0
        parts.append(matrix[rows][:, interactions] * centered_step)
    return np.column_stack(parts)


def _risk_panel(matrix, train, survival, interactions):
    matrices, labels = [], []
    for step in range(5):
        eligible = train if step == 0 else train[survival[train, step - 1] == 1]
        matrices.append(_augment(matrix, eligible, step, interactions))
        labels.append(survival[eligible, step])
    return np.vstack(matrices), np.concatenate(labels)


def pooled_scores(matrix, survival, dates, reach, kind, interactions=(), verbose=True):
    score = np.full(len(dates), np.nan)
    logs = []
    resolved = np.all(np.isfinite(survival), axis=1)
    interactions = np.asarray(interactions, dtype=int)
    for start in _quarter_starts():
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= RESET) & resolved
            & np.asarray([value < start for value in reach])
        )
        target = np.flatnonzero(
            (dates >= start) & (dates < end) & resolved
        )
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved path admitted to pooled panel")
        panel, labels = _risk_panel(matrix, train, survival, interactions)
        model = _model(kind).fit(panel, labels)
        probabilities = []
        for step in range(5):
            probabilities.append(
                model.predict_proba(
                    _augment(matrix, target, step, interactions)
                )[:, 1]
            )
        score[target] = np.prod(np.asarray(probabilities), axis=0)
        logs.append({
            "candidate": kind + ("_interaction" if len(interactions) else ""),
            "quarter": str(start),
            "n_episodes": len(train),
            "n_risk_rows": len(panel),
            "last_resolved": str(max(reach[train])),
            "n_features": panel.shape[1],
        })
        if verbose:
            print(
                f"  pooled {kind:<5} interactions={len(interactions):2d} "
                f"quarter={start} episodes={len(train):5d} risk={len(panel):5d}",
                flush=True,
            )
    return score, logs


def outcome_causality_check(matrix, survival, dates, reach):
    cutoff = np.datetime64("2025-06-30").astype(object)
    first, _ = pooled_scores(
        matrix, survival, dates, reach, "logit", verbose=False,
    )
    changed = survival.copy()
    future = np.asarray([value > cutoff for value in reach])
    changed[future] = 1.0 - changed[future]
    second, _ = pooled_scores(
        matrix, changed, dates, reach, "logit", verbose=False,
    )
    past = (dates <= cutoff) & np.isfinite(first)
    np.testing.assert_array_equal(first[past], second[past])
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    basis, basis_names = build_cny_basis_features(index, history, references["CNY"])
    wave, wave_names = build_waveform_features(index, history)
    target_columns = np.asarray([names.index(name) for name in TARGET_FIELDS])
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ])
    local = np.column_stack([X[:, target_columns], X[:, currency_columns]])
    market = np.column_stack([basis, wave])
    aligned = np.column_stack([local, market])
    stale = np.column_stack([local, delayed_by_currency(market, index, rows=20)])
    interaction_names = (
        "pct_range_30", "pct_range_90", "pct_range_180", "ret_1", "ret_5",
    )
    interactions = [TARGET_FIELDS.index(name) for name in interaction_names]
    interactions.extend([
        local.shape[1] + basis_names.index("cny_basis_close_bps"),
        local.shape[1] + len(basis_names) + wave_names.index("cny_wave_ret_lag_1"),
    ])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    survival = survival_targets(index, series)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    if not outcome_causality_check(aligned, survival, dates, reach):
        raise AssertionError("pooled outcome causality failed")
    definitions = (
        ("pooled_hazard_logit", aligned, "logit", ()),
        ("pooled_hazard_interaction_logit", aligned, "logit", interactions),
        ("pooled_hazard_hist", aligned, "hist", ()),
        ("pooled_hazard_stale20_logit", stale, "logit", ()),
    )
    outputs, logs = {}, []
    for candidate, matrix, kind, interaction in definitions:
        score, part = pooled_scores(
            matrix, survival, dates, reach, kind, interaction,
        )
        outputs[candidate] = _outputs(score, y, dates)
        logs.extend({"candidate": candidate, **row} for row in part)
    with SEPARATE.open("rb") as handle:
        separate = pickle.load(handle)
    outputs["separate_hazard_product"] = separate["survival_hazard_product"]
    outputs["cumulative50_pooled50"] = combine_causal(
        [separate["survival_cumulative_geometric"], outputs["pooled_hazard_logit"]],
        (.50, .50), dates, currencies,
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
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_pooled_survival_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
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
    aligned_row = results[
        (results.candidate == "pooled_hazard_logit")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale_row = results[
        (results.candidate == "pooled_hazard_stale20_logit")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh = bool(
        aligned_row.lift > stale_row.lift
        and aligned_row.corridor_lift_min > stale_row.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BX",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "logit_C": .03,
        "hist_parameters": {
            "max_iter": 200, "learning_rate": .035,
            "max_leaf_nodes": 7, "min_samples_leaf": 100,
            "l2_regularization": 20.0,
        },
        "interaction_names": interaction_names + (
            "cny_basis_close_bps", "cny_wave_ret_lag_1",
        ),
        "target_fields": TARGET_FIELDS,
        "basis_fields": basis_names,
        "waveform_fields": wave_names,
        "stale_control_rows_per_currency": 20,
        "all_training_paths_resolved": chronology_ok,
        "future_outcome_corruption_check": True,
        "aligned_pooled_beats_stale_lift_and_min_currency": fresh,
        "payload_sha256": digest,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("pooled training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned pooled logit accepted as fresh: {fresh}")


if __name__ == "__main__":
    main()
