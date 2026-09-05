"""Packet-BB causal nearest-neighbour reliability and benefit lower bounds."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_reliability_surface")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
GAM = Path("results/research/round6/cny_gam/outputs.pkl")
RANK_WINDOW = 250
RANK_MIN = 20
GLOBAL_K = 250
LOCAL_K = 80
CURRENCY_PENALTY = .05
SHRINKAGE = 100.0
ORDER = (
    "pooled_hit_lcb",
    "shrunk_hit_lcb",
    "shrunk_benefit_lcb",
    "reliability_benefit_equal",
    "primary75_reliability25",
)


def extract_scores(output, length):
    """Recover the score known at each row, preferring its original test score."""
    result = np.full(length, np.nan)
    for year in sorted(output):
        item = output[year]
        calibration = np.asarray(item["calib_idx"], dtype=int)
        calibration_scores = np.asarray(item["calib_score"], dtype=float)
        missing = ~np.isfinite(result[calibration])
        result[calibration[missing]] = calibration_scores[missing]
    for year in sorted(output):
        item = output[year]
        result[np.asarray(item["test_idx"], dtype=int)] = np.asarray(
            item["test_score"], dtype=float,
        )
    return result


def causal_percentiles(scores, dates, currencies, window=RANK_WINDOW, minimum=RANK_MIN):
    result = np.full(len(scores), np.nan)
    for currency in np.unique(currencies):
        rows = np.flatnonzero((currencies == currency) & np.isfinite(scores))
        rows = rows[np.argsort(dates[rows])]
        history = []
        for row in rows:
            reference = np.asarray(history[-window:], dtype=float)
            if len(reference) >= minimum:
                result[row] = float(
                    np.searchsorted(np.sort(reference), scores[row], side="right")
                    / len(reference)
                )
            else:
                result[row] = .5
            history.append(float(scores[row]))
    return result


def meta_features(primary_score, gam_score, dates, currencies):
    primary_rank = causal_percentiles(primary_score, dates, currencies)
    gam_rank = causal_percentiles(gam_score, dates, currencies)
    return np.column_stack([
        primary_rank,
        gam_rank,
        np.abs(primary_rank - gam_rank),
    ])


def _neighbours(features, currencies, train, row, k, local):
    candidates = train[currencies[train] == currencies[row]] if local else train
    if not len(candidates):
        return candidates
    delta = features[candidates] - features[row]
    distance = delta[:, 0] ** 2 + delta[:, 1] ** 2 + .5 * delta[:, 2] ** 2
    if not local:
        distance = distance + CURRENCY_PENALTY * (
            currencies[candidates] != currencies[row]
        )
    count = min(k, len(candidates))
    chosen = np.argpartition(distance, count - 1)[:count]
    return candidates[chosen]


def _hit_lcb(values):
    successes = float(np.sum(values))
    alpha = .5 + successes
    beta = .5 + len(values) - successes
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return float(mean - np.sqrt(variance))


def _benefit_lcb(values):
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean
    return float(mean - np.std(values, ddof=1) / np.sqrt(len(values)))


def reliability_scores(features, y, benefit, dates, currencies, reach, verbose=True):
    scores = {name: np.full(len(y), np.nan) for name in ORDER[:3]}
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        finite_features = np.all(np.isfinite(features), axis=1)
        test = (
            (dates >= start) & (dates < end)
            & finite_features & np.isfinite(y) & np.isfinite(benefit)
        )
        train_mask = (
            (dates >= RESET) & finite_features
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & np.isfinite(benefit)
        )
        train = np.flatnonzero(train_mask)
        if not test.any() or len(train) < 700:
            continue
        for row in np.flatnonzero(test):
            pooled = _neighbours(features, currencies, train, row, GLOBAL_K, False)
            local = _neighbours(features, currencies, train, row, LOCAL_K, True)
            pooled_hit = _hit_lcb(y[pooled])
            local_hit = _hit_lcb(y[local]) if len(local) else pooled_hit
            pooled_benefit = _benefit_lcb(benefit[pooled])
            local_benefit = (
                _benefit_lcb(benefit[local]) if len(local) else pooled_benefit
            )
            local_weight = len(local) / (len(local) + SHRINKAGE)
            scores["pooled_hit_lcb"][row] = pooled_hit
            scores["shrunk_hit_lcb"][row] = (
                local_weight * local_hit + (1.0 - local_weight) * pooled_hit
            )
            scores["shrunk_benefit_lcb"][row] = (
                local_weight * local_benefit
                + (1.0 - local_weight) * pooled_benefit
            )
        logs.append({
            "quarter": str(start),
            "n_train": len(train),
            "first_train": str(min(dates[train])),
            "last_resolved": str(max(reach[train])),
            "n_test": int(test.sum()),
        })
        if verbose:
            print(
                f"  reliability surface quarter={start} train={len(train):5d} "
                f"test={test.sum():4d}",
                flush=True,
            )
    return scores, logs


def future_outcome_causality_check(features, y, benefit, dates, currencies, reach):
    cutoff = np.asarray(sorted(set(dates)))[int(len(set(dates)) * .72)]
    first, _ = reliability_scores(
        features, y, benefit, dates, currencies, reach, verbose=False,
    )
    changed_y = y.copy()
    changed_benefit = benefit.copy()
    unresolved = np.asarray([value >= cutoff for value in reach])
    changed_y[unresolved & np.isfinite(changed_y)] = (
        1.0 - changed_y[unresolved & np.isfinite(changed_y)]
    )
    changed_benefit[unresolved & np.isfinite(changed_benefit)] *= -10.0
    second, _ = reliability_scores(
        features, changed_y, changed_benefit, dates, currencies, reach,
        verbose=False,
    )
    past = dates < cutoff
    for name in first:
        np.testing.assert_array_equal(first[name][past], second[name][past])
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
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    with GAM.open("rb") as handle:
        gam = pickle.load(handle)["all_spline_gam"]
    primary_score = extract_scores(primary, len(y))
    gam_score = extract_scores(gam, len(y))
    features = meta_features(primary_score, gam_score, dates, currencies)
    raw_scores, logs = reliability_scores(
        features, y, benefit, dates, currencies, reach,
    )
    outputs = {name: _outputs(score, y, dates) for name, score in raw_scores.items()}
    outputs["reliability_benefit_equal"] = combine_causal(
        [outputs["shrunk_hit_lcb"], outputs["shrunk_benefit_lcb"]],
        (.5, .5), dates, currencies,
    )
    outputs["primary75_reliability25"] = combine_causal(
        [primary, outputs["reliability_benefit_equal"]],
        (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)
    causality_ok = future_outcome_causality_check(
        features, y, benefit, dates, currencies, reach,
    )

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
        y, dates, currencies, valid, masks, "reliability_surface_2025_2026",
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
    if not chronology_ok:
        raise AssertionError("reliability surface used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BB",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MIN,
        "global_neighbours": GLOBAL_K,
        "local_neighbours": LOCAL_K,
        "currency_mismatch_distance_penalty": CURRENCY_PENALTY,
        "beta_prior": [0.5, 0.5],
        "lower_bound_standard_deviations": 1.0,
        "local_shrinkage_denominator": SHRINKAGE,
        "blend_weights": {"hit_benefit": [0.5, 0.5], "primary_meta": [0.75, 0.25]},
        "all_training_labels_resolved": chronology_ok,
        "future_outcome_corruption_check": causality_ok,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
