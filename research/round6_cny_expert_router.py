"""Packet-BZ causal resolved-error router over shadow and survival experts."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import build_cny_basis_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_cny_survival_hazard import TARGET_FIELDS
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_expert_router")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
SURVIVAL = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
SEED = 20260905
RANK_WINDOW = 250
RANK_MINIMUM = 20
ORDER = (
    "router_logit_soft",
    "router_logit_hard",
    "router_tree_soft",
    "router_tree_hard",
    "router_logit_soft_stale20",
    "shadow50_survival50",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def router_matrix(shadow_rank, survival_rank, X, names, basis, basis_names, wave, wave_names):
    pair = np.column_stack([shadow_rank, survival_rank])
    target_names = (
        "pct_range_30", "pct_range_90", "pct_range_180", "ret_1", "ret_5",
        "annual_sin_1", "annual_cos_1", "dow_sin", "dow_cos",
    )
    target = X[:, [names.index(name) for name in target_names]]
    currency = X[:, [
        i for i, name in enumerate(names) if name.startswith("currency_")
    ]]
    market = np.column_stack([
        basis[:, basis_names.index("cny_basis_close_bps")],
        wave[:, wave_names.index("cny_wave_vol_5")],
        wave[:, wave_names.index("cny_wave_vol_20")],
        wave[:, wave_names.index("cny_wave_last_z_20")],
        wave[:, wave_names.index("cny_wave_acceleration_5_5")],
    ])
    return np.column_stack([
        pair,
        survival_rank - shadow_rank,
        np.abs(survival_rank - shadow_rank),
        np.min(pair, axis=1),
        np.max(pair, axis=1),
        np.mean(pair, axis=1),
        currency,
        target,
        market,
    ])


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.02, max_iter=3000, random_state=SEED),
        )
    if kind == "tree":
        return DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=150, random_state=SEED,
        )
    raise KeyError(kind)


def routed_scores(name, kind, gate, shadow_rank, survival_rank, router_y, y, dates, reach, verbose=True):
    soft = np.full(len(y), np.nan)
    hard = np.full(len(y), np.nan)
    weights = np.full(len(y), np.nan)
    logs = []
    finite = np.all(np.isfinite(gate), axis=1)
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = np.flatnonzero(
            (dates >= dt.date(2023, 1, 1)) & finite & np.isfinite(y)
            & np.asarray([value < start for value in reach])
        )
        target = np.flatnonzero(
            (dates >= start) & (dates < end) & finite & np.isfinite(y)
        )
        if len(train) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in train):
            raise AssertionError("unresolved router label admitted")
        model = _model(kind).fit(gate[train], router_y[train])
        weight = model.predict_proba(gate[target])[:, 1]
        weights[target] = weight
        soft[target] = (
            weight * survival_rank[target]
            + (1.0 - weight) * shadow_rank[target]
        )
        hard[target] = np.where(
            weight >= .5, survival_rank[target], shadow_rank[target]
        )
        logs.append({
            "candidate": name,
            "quarter": str(start),
            "n_train": len(train),
            "last_resolved": str(max(reach[train])),
            "n_features": gate.shape[1],
            "survival_weight_mean": float(np.mean(weight)),
        })
        if verbose:
            print(
                f"  router {name:<8} quarter={start} train={len(train):5d} "
                f"mean_weight={np.mean(weight):.3f}",
                flush=True,
            )
    return soft, hard, weights, logs


def outcome_causality_check(gate, shadow_rank, survival_rank, router_y, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    first, _hard, _weights, _ = routed_scores(
        "causal", "logit", gate, shadow_rank, survival_rank,
        router_y, y, dates, reach, verbose=False,
    )
    changed = router_y.copy()
    future = np.asarray([value > cutoff for value in reach])
    changed[future] = 1.0 - changed[future]
    second, _hard, _weights, _ = routed_scores(
        "changed", "logit", gate, shadow_rank, survival_rank,
        changed, y, dates, reach, verbose=False,
    )
    past = (dates <= cutoff) & np.isfinite(first)
    np.testing.assert_array_equal(first[past], second[past])
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    shadow_output = _load(SHADOW)["shadow_close_basis"]
    survival_output = _load(SURVIVAL)["survival_cumulative_geometric"]
    shadow_score = row_scores(shadow_output, len(y))
    survival_score = row_scores(survival_output, len(y))
    shadow_rank = causal_percentiles(
        shadow_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    survival_rank = causal_percentiles(
        survival_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    router_y = (
        np.abs(survival_rank - y) < np.abs(shadow_rank - y)
    ).astype(float)
    history, digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    basis, basis_names = build_cny_basis_features(index, history, references["CNY"])
    wave, wave_names = build_waveform_features(index, history)
    gate = router_matrix(
        shadow_rank, survival_rank, X, names, basis, basis_names, wave, wave_names,
    )
    stale_gate = delayed_by_currency(gate, index, rows=20)
    if not outcome_causality_check(
        gate, shadow_rank, survival_rank, router_y, y, dates, reach,
    ):
        raise AssertionError("router outcome causality failed")

    outputs, logs, weight_columns = {}, [], {}
    for name, kind, matrix in (
        ("router_logit", "logit", gate),
        ("router_tree", "tree", gate),
        ("router_logit_stale20", "logit", stale_gate),
    ):
        soft, hard, weights, part = routed_scores(
            name, kind, matrix, shadow_rank, survival_rank,
            router_y, y, dates, reach,
        )
        output_name = name + "_soft" if name != "router_logit_stale20" else "router_logit_soft_stale20"
        outputs[output_name] = _outputs(soft, y, dates)
        if name != "router_logit_stale20":
            outputs[name + "_hard"] = _outputs(hard, y, dates)
        weight_columns[name] = weights
        logs.extend(part)
    outputs["shadow50_survival50"] = combine_causal(
        [shadow_output, survival_output], (.50, .50), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    pd.DataFrame({"date": dates, "currency": currencies, **weight_columns}).to_csv(
        OUT / "router_weights.csv", index=False,
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
        y, dates, currencies, valid, masks, "cny_expert_router_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    training = pd.DataFrame(logs)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    aligned_row = results[
        (results.candidate == "router_logit_soft")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale_row = results[
        (results.candidate == "router_logit_soft_stale20")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh = bool(
        aligned_row.lift > stale_row.lift
        and aligned_row.corridor_lift_min > stale_row.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BZ",
        "variants": ORDER,
        "experts": ["shadow_close_basis", "survival_cumulative_geometric"],
        "fixed_policy": POLICY,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "router_label": "expert with lower resolved Brier loss; ties shadow",
        "logit_C": .02,
        "tree_max_depth": 2,
        "tree_min_samples_leaf": 150,
        "stale_control_rows_per_currency": 20,
        "all_training_labels_resolved": chronology_ok,
        "future_router_label_corruption_check": True,
        "aligned_soft_logit_beats_stale_lift_and_min_currency": fresh,
        "payload_sha256": digest,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("router training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned router accepted as fresh: {fresh}")


if __name__ == "__main__":
    main()
