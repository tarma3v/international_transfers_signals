"""Packet-EM: causal regime normalization of the 15:30 fixing basis."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_availability_router import availability_route
from research.round6_fixing_proxies import proxy_causality_check
from research.round6_moex_spot_1530_features import (
    build_spot_1530_features,
    load_spot_1530_history,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_state_agreement_geometry import future_rank_check
from research.round6_uzbek_central_bank_models import (
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_regime_normalization")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
FIXING_PATH = Path("results/research/round6/fixing_proxies/outputs.pkl")
ROUTER_PATH = Path(
    "results/research/round6/fixing_availability_router/outputs.pkl"
)
RANK_WINDOW = 250
RANK_MINIMUM = 20
STALE_ROWS = 20
ROBUST_SCALE = 1.4826
ROBUST_FLOOR = 1.0
FEATURES = (
    "innovation_1",
    "median_excess_20",
    "median_excess_60",
    "robust_z_20",
    "robust_z_60",
    "persistent_mean_3",
    "persistent_min_2",
)
BLENDS = (
    "innovation_1",
    "robust_z_20",
    "robust_z_60",
    "persistent_mean_3",
    "persistent_min_2",
)


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def _robust_z(value: float, reference: np.ndarray) -> float:
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(ROBUST_SCALE * mad, ROBUST_FLOOR)
    return (value - median) / scale


def regime_features(raw, available, dates, currencies):
    """Build features from current and strictly earlier available sessions."""
    raw = np.asarray(raw, dtype=float)
    available = np.asarray(available, dtype=bool)
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies, dtype=object)
    if not (raw.shape == available.shape == dates.shape == currencies.shape):
        raise ValueError("unaligned fixing-normalization inputs")
    result = {name: np.full(len(raw), np.nan) for name in FEATURES}
    for currency in np.unique(currencies):
        rows = np.flatnonzero(currencies == currency)
        rows = rows[np.argsort(dates[rows])]
        history = []
        for row in rows:
            if not available[row] or not np.isfinite(raw[row]):
                continue
            value = float(raw[row])
            if len(history) >= 1:
                result["innovation_1"][row] = value - history[-1]
                result["persistent_min_2"][row] = min(value, history[-1])
            if len(history) >= 2:
                result["persistent_mean_3"][row] = float(
                    np.mean([history[-2], history[-1], value])
                )
            if len(history) >= 20:
                reference = np.asarray(history[-20:], dtype=float)
                result["median_excess_20"][row] = (
                    value - float(np.median(reference))
                )
                result["robust_z_20"][row] = _robust_z(value, reference)
            if len(history) >= 60:
                reference = np.asarray(history[-60:], dtype=float)
                result["median_excess_60"][row] = (
                    value - float(np.median(reference))
                )
                result["robust_z_60"][row] = _robust_z(value, reference)
            history.append(value)
    return result


def feature_causality_check(raw, available, dates, currencies):
    cutoff = dt.date(2025, 6, 30)
    original = regime_features(raw, available, dates, currencies)
    changed_raw = np.asarray(raw, dtype=float).copy()
    future = (dates > cutoff) & available
    changed_raw[future] = changed_raw[future] * 100.0 + 10000.0
    changed = regime_features(changed_raw, available, dates, currencies)
    past = dates <= cutoff
    future_changed = False
    for name in FEATURES:
        np.testing.assert_array_equal(original[name][past], changed[name][past])
        valid = future & np.isfinite(original[name]) & np.isfinite(changed[name])
        future_changed |= bool(np.any(original[name][valid] != changed[name][valid]))
    if not future_changed:
        raise AssertionError("future fixing corruption did not affect future features")
    return True


def _selected_feature(candidate: str):
    if candidate.startswith("route_"):
        return candidate.removeprefix("route_"), False
    if candidate.startswith("router75_") and candidate.endswith("_25"):
        return candidate[len("router75_"):-len("_25")], True
    return None, False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    proxy_causality_check(index, history, references)
    market, market_names = build_spot_1530_features(
        index, history, references,
    )
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    missing = market[:, market_names.index(
        "moex_1530_cnyrub_tom_missing"
    )].astype(bool)
    fixing = _load(FIXING_PATH, "selected")
    raw = row_scores(fixing, len(index))
    available = ~missing & np.isfinite(raw)
    features = regime_features(raw, available, dates, currencies)
    feature_causality_check(raw, available, dates, currencies)

    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    noon = _load(NOON_PATH, "selected")
    router = _load(ROUTER_PATH, "availability_route")
    noon_score = row_scores(noon, len(index))
    future_rank_check(noon_score, dates, currencies)
    noon_rank = causal_percentiles(
        noon_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    raw_rank = causal_percentiles(
        raw, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    router_score = availability_route(noon_rank, raw_rank, missing)
    reconstructed_router = _outputs(router_score, y5, dates)
    for year in sorted(set(router) & set(reconstructed_router)):
        np.testing.assert_array_equal(
            router[year]["test_idx"], reconstructed_router[year]["test_idx"],
        )
        np.testing.assert_allclose(
            router[year]["test_score"],
            reconstructed_router[year]["test_score"],
            rtol=0.0,
            atol=0.0,
        )

    feature_ranks = {}
    route_scores = {}
    candidates = {"availability_router": router}
    candidate_feature = {}
    for name in FEATURES:
        future_rank_check(features[name], dates, currencies)
        rank = causal_percentiles(
            features[name], dates, currencies, RANK_WINDOW, RANK_MINIMUM,
        )
        feature_ranks[name] = rank
        route = availability_route(noon_rank, rank, missing)
        route_scores[name] = route
        candidate = f"route_{name}"
        candidates[candidate] = _outputs(route, y5, dates)
        candidate_feature[candidate] = name
    for name in BLENDS:
        candidate = f"router75_{name}_25"
        score = .75 * router_score + .25 * route_scores[name]
        candidates[candidate] = _outputs(score, y5, dates)
        candidate_feature[candidate] = name

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    screen_h5 = []
    for name, output in candidates.items():
        item = _evaluate(
            output, (2024,), POLICY, y5, forwards[5], dates, currencies,
        )
        screen_h5.append({"candidate": name, "frequency": item["frequency"]})
    screen_summary = screen_summary.merge(pd.DataFrame(screen_h5), on="candidate")
    feasible = screen_summary[
        screen_summary.frequency.between(1.0, 2.0)
        & screen_summary.symmetric_benefit_min.gt(0)
        & screen_summary.future_benefit_min.gt(0)
    ]
    pool = feasible if len(feasible) else screen_summary
    selected = str(pool.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0].candidate)

    comparison = {
        "availability_router": router,
        "selected": candidates[selected],
    }
    feature_name, blended = _selected_feature(selected)
    if feature_name is None:
        comparison["matched_stale20"] = _load(
            ROUTER_PATH, "matched_stale20",
        )
    else:
        delayed = delayed_by_currency(
            features[feature_name][:, None], index, rows=STALE_ROWS,
        )[:, 0]
        delayed[missing] = np.nan
        delayed_rank = causal_percentiles(
            delayed, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
        )
        stale_route = availability_route(noon_rank, delayed_rank, missing)
        stale_score = (
            .75 * router_score + .25 * stale_route if blended else stale_route
        )
        comparison["matched_stale20"] = _outputs(stale_score, y5, dates)

    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)

    h5_rows = []
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], comparison, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks,
        "fixing_regime_normalization_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EM",
        "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "raw_anchor": "positive CNY session-mean basis to current CBR",
        "features": FEATURES,
        "blend_features": BLENDS,
        "blend_weights": {"availability_router": .75, "alternative": .25},
        "history": "strictly earlier available sessions of same currency",
        "robust_scale": ROBUST_SCALE,
        "robust_scale_floor_bps": ROBUST_FLOOR,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "selection_period": 2024,
        "selected": selected,
        "matched_control": "selected feature delayed 20 target rows",
        "feature_future_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": (
            "protocol-controlled retrospective opened after 2024 selection"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
