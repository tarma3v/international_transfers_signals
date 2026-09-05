"""Packet-BO causal unsupervised CNY states and Bayesian outcome tables."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_unsupervised_states")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
SEED = 20260905
N_STATES = 8
LOCAL_SHRINK = 100.0
ORDER = (
    "cluster_pooled_hit_lcb",
    "cluster_shrunk_hit_lcb",
    "cluster_transition_hit_lcb",
    "cluster_shrunk_benefit_lcb",
    "cluster_state_benefit50",
    "cluster_stale20_hit_lcb",
    "primary75_cluster25",
)


def _hit_lcb(values):
    values = np.asarray(values, dtype=float)
    alpha = .5 + float(np.sum(values))
    beta = .5 + len(values) - float(np.sum(values))
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return float(mean - np.sqrt(variance))


def _benefit_lcb(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    error = float(np.std(values) / np.sqrt(len(values)))
    return float(np.mean(values) - error)


def _previous_states(states, dates, currencies):
    previous = np.full(len(states), -1, dtype=int)
    for currency in np.unique(currencies):
        rows = np.flatnonzero(currencies == currency)
        rows = rows[np.argsort(dates[rows])]
        previous[rows[1:]] = states[rows[:-1]]
    return previous


def _state_scores(states, dates, currencies, train, target, y, benefit):
    result = {
        "pooled_hit": np.full(len(target), np.nan),
        "shrunk_hit": np.full(len(target), np.nan),
        "transition_hit": np.full(len(target), np.nan),
        "shrunk_benefit": np.full(len(target), np.nan),
    }
    previous = _previous_states(states, dates, currencies)
    global_hit = _hit_lcb(y[train])
    global_benefit = _benefit_lcb(benefit[train])
    for position, row in enumerate(target):
        state = states[row]
        pooled = train[states[train] == state]
        pooled_hit = _hit_lcb(y[pooled]) if len(pooled) else global_hit
        pooled_benefit = (
            _benefit_lcb(benefit[pooled]) if len(pooled) else global_benefit
        )
        local = pooled[currencies[pooled] == currencies[row]]
        local_weight = len(local) / (len(local) + LOCAL_SHRINK)
        local_hit = _hit_lcb(y[local]) if len(local) else pooled_hit
        local_benefit = (
            _benefit_lcb(benefit[local]) if len(local) else pooled_benefit
        )
        result["pooled_hit"][position] = pooled_hit
        result["shrunk_hit"][position] = (
            local_weight * local_hit + (1.0 - local_weight) * pooled_hit
        )
        result["shrunk_benefit"][position] = (
            local_weight * local_benefit + (1.0 - local_weight) * pooled_benefit
        )

        transition = train[
            (previous[train] == previous[row]) & (states[train] == state)
        ]
        transition_pool = (
            _hit_lcb(y[transition]) if len(transition) else pooled_hit
        )
        transition_local = transition[
            currencies[transition] == currencies[row]
        ]
        transition_weight = len(transition_local) / (
            len(transition_local) + LOCAL_SHRINK
        )
        transition_local_hit = (
            _hit_lcb(y[transition_local])
            if len(transition_local) else transition_pool
        )
        result["transition_hit"][position] = (
            transition_weight * transition_local_hit
            + (1.0 - transition_weight) * transition_pool
        )
    return result


def _cluster(features, train, dates):
    unique_rows = []
    seen = set()
    for row in train:
        if dates[row] not in seen:
            seen.add(dates[row])
            unique_rows.append(row)
    unique_rows = np.asarray(unique_rows, dtype=int)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaled_train = scaler.fit_transform(features[unique_rows])
    model = KMeans(
        n_clusters=N_STATES, n_init=20, random_state=SEED,
    )
    model.fit(scaled_train)
    return model.predict(scaler.transform(features)), len(unique_rows)


def prequential_scores(wave, stale_wave, y, benefit, dates, currencies, reach):
    scores = {
        "cluster_pooled_hit_lcb": np.full(len(y), np.nan),
        "cluster_shrunk_hit_lcb": np.full(len(y), np.nan),
        "cluster_transition_hit_lcb": np.full(len(y), np.nan),
        "cluster_shrunk_benefit_lcb": np.full(len(y), np.nan),
        "cluster_stale20_hit_lcb": np.full(len(y), np.nan),
    }
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= RESET)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        target = np.flatnonzero(
            (dates >= start) & (dates < end) & np.isfinite(y)
        )
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved outcome admitted to state table")
        states, unique_dates = _cluster(wave, train, dates)
        aligned = _state_scores(
            states, dates, currencies, train, target, y, benefit,
        )
        scores["cluster_pooled_hit_lcb"][target] = aligned["pooled_hit"]
        scores["cluster_shrunk_hit_lcb"][target] = aligned["shrunk_hit"]
        scores["cluster_transition_hit_lcb"][target] = aligned["transition_hit"]
        scores["cluster_shrunk_benefit_lcb"][target] = aligned["shrunk_benefit"]
        stale_states, stale_unique_dates = _cluster(stale_wave, train, dates)
        stale = _state_scores(
            stale_states, dates, currencies, train, target, y, benefit,
        )
        scores["cluster_stale20_hit_lcb"][target] = stale["shrunk_hit"]
        logs.append({
            "quarter": str(start),
            "n_train": len(train),
            "unique_train_dates": unique_dates,
            "stale_unique_train_dates": stale_unique_dates,
            "last_resolved": str(max(reach[train])),
            "n_features": wave.shape[1],
            "n_states": N_STATES,
        })
        print(
            f"  unsupervised states          quarter={start} "
            f"train={len(train):5d} dates={unique_dates:4d}",
            flush=True,
        )
    return scores, logs


def outcome_causality_check(wave, stale_wave, y, benefit, dates, currencies, reach):
    cutoff = np.datetime64("2025-06-30").astype(object)
    original, _ = prequential_scores(
        wave, stale_wave, y, benefit, dates, currencies, reach,
    )
    changed_y = y.copy()
    changed_benefit = benefit.copy()
    unresolved = np.asarray([
        np.isfinite(value) and reach[row] > cutoff
        for row, value in enumerate(y)
    ])
    changed_y[unresolved] = 1.0 - changed_y[unresolved]
    changed_benefit[unresolved] = changed_benefit[unresolved] * -10.0 + 999.0
    changed, _ = prequential_scores(
        wave, stale_wave, changed_y, changed_benefit, dates, currencies, reach,
    )
    past = dates <= cutoff
    for candidate in original:
        available = past & np.isfinite(original[candidate])
        if not np.array_equal(
            original[candidate][available], changed[candidate][available],
        ):
            raise AssertionError(
                f"unresolved future outcome changed past {candidate} state score"
            )
    return True


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
    history, digest = load_moex_history()
    wave, wave_names = build_waveform_features(index, history)
    stale_wave = delayed_by_currency(wave, index, rows=20)
    if not outcome_causality_check(
        wave, stale_wave, y, benefit, dates, currencies, reach,
    ):
        raise AssertionError("state-table outcome causality failed")
    raw_scores, logs = prequential_scores(
        wave, stale_wave, y, benefit, dates, currencies, reach,
    )
    outputs = {
        name: _outputs(score, y, dates) for name, score in raw_scores.items()
    }
    outputs["cluster_state_benefit50"] = combine_causal(
        [outputs["cluster_shrunk_hit_lcb"],
         outputs["cluster_shrunk_benefit_lcb"]],
        (.5, .5), dates, currencies,
    )
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs["primary75_cluster25"] = combine_causal(
        [primary, outputs["cluster_shrunk_hit_lcb"]],
        (.75, .25), dates, currencies,
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
        y, dates, currencies, valid, masks, "cny_unsupervised_states_2025_2026",
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
    aligned = results[
        (results.candidate == "cluster_shrunk_hit_lcb")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale = results[
        (results.candidate == "cluster_stale20_hit_lcb")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh = bool(
        aligned.lift > stale.lift
        and aligned.corridor_lift_min > stale.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BO",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "n_states": N_STATES,
        "kmeans_n_init": 20,
        "seed": SEED,
        "scaler": "RobustScaler IQR 25--75 fit on resolved train only",
        "unique_date_clustering": True,
        "local_shrink_denominator": LOCAL_SHRINK,
        "hit_score": "Jeffreys posterior mean minus one standard deviation",
        "benefit_score": "mean minus one standard error",
        "stale_control_rows_per_currency": 20,
        "waveform_features": wave_names,
        "payload_sha256": digest,
        "future_outcome_corruption_check": True,
        "all_training_labels_resolved": chronology_ok,
        "aligned_shrunk_state_beats_stale_lift_and_min_currency": fresh,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("state-table training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned unsupervised state accepted as fresh: {fresh}")


if __name__ == "__main__":
    main()
