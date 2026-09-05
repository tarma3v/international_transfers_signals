"""Packet-EO: causal target-to-CNY transmission projection."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import Series
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
    _reference_last,
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


OUT = Path("results/research/round6/fixing_crossrate_projection")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
FIXING_PATH = Path("results/research/round6/fixing_proxies/outputs.pkl")
ROUTER_PATH = Path(
    "results/research/round6/fixing_availability_router/outputs.pkl"
)
RANK_WINDOW = 250
RANK_MINIMUM = 20
STALE_ROWS = 20
BETA_WINDOWS = (60, 120)
BETA_MINIMUM = 20
BETA_BOUNDS = (0.0, 2.0)
REVERSION_WEIGHT = .25
SCORES = (
    "beta_60_basis",
    "beta_120_basis",
    "cross_reversion_60",
    "cross_reversion_120",
    "projected_60",
    "projected_120",
)
BLENDS = ("cross_reversion_60", "projected_60")


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def _beta(cny_returns, target_returns, window):
    x = np.asarray(cny_returns[-window:], dtype=float)
    y = np.asarray(target_returns[-window:], dtype=float)
    if len(x) < BETA_MINIMUM:
        return np.nan
    centered = x - float(np.mean(x))
    variance = float(np.dot(centered, centered))
    if variance <= 1e-12:
        return 0.0
    covariance = float(np.dot(centered, y - float(np.mean(y))))
    return float(np.clip(covariance / variance, *BETA_BOUNDS))


def transmission_features(index, series, cny_reference):
    """Estimate currency transmission using only observations before each row."""
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    target_level = np.asarray([
        series[currency].values[position]
        for currency, position, _day in index
    ], dtype=float)
    cny_level = np.asarray([
        _reference_last(cny_reference, day) for day in dates
    ], dtype=float)
    result = {
        f"beta_{window}": np.full(len(index), np.nan)
        for window in BETA_WINDOWS
    }
    result.update({
        f"cross_reversion_{window}": np.full(len(index), np.nan)
        for window in BETA_WINDOWS
    })
    for currency in np.unique(currencies):
        rows = np.flatnonzero(currencies == currency)
        rows = rows[np.argsort(dates[rows])]
        target_returns, cny_returns, cross_history = [], [], []
        previous_target = previous_cny = None
        for row in rows:
            target = float(target_level[row])
            cny = float(cny_level[row])
            if not (target > 0.0 and cny > 0.0):
                continue
            current_cross = float(np.log(target / cny))
            for window in BETA_WINDOWS:
                result[f"beta_{window}"][row] = _beta(
                    cny_returns, target_returns, window,
                )
                reference = np.asarray(cross_history[-window:], dtype=float)
                if len(reference) >= BETA_MINIMUM:
                    result[f"cross_reversion_{window}"][row] = (
                        float(np.median(reference)) - current_cross
                    ) * 10000.0
            if previous_target is not None and previous_cny is not None:
                target_returns.append(float(np.log(target / previous_target)))
                cny_returns.append(float(np.log(cny / previous_cny)))
            cross_history.append(current_cross)
            previous_target, previous_cny = target, cny
    return result


def transmission_causality_check(index, series, cny_reference):
    cutoff = dt.date(2025, 6, 30)
    original = transmission_features(index, series, cny_reference)
    altered_series = {}
    for currency, item in series.items():
        values = item.values.copy()
        future = np.asarray([day > cutoff for day in item.dates])
        values[future] *= np.exp(.01 * np.arange(1, future.sum() + 1))
        altered_series[currency] = Series(
            item.code, item.dates.copy(), values,
        )
    cny_values = cny_reference.values.copy()
    cny_future = np.asarray([day > cutoff for day in cny_reference.dates])
    cny_values[cny_future] *= np.exp(
        -.008 * np.arange(1, cny_future.sum() + 1)
    )
    altered_cny = Series(
        cny_reference.code, cny_reference.dates.copy(), cny_values,
    )
    altered = transmission_features(index, altered_series, altered_cny)
    dates = np.asarray([row[2] for row in index], dtype=object)
    past, future = dates <= cutoff, dates > cutoff
    changed_future = False
    for name in original:
        np.testing.assert_array_equal(original[name][past], altered[name][past])
        finite = future & np.isfinite(original[name]) & np.isfinite(altered[name])
        changed_future |= bool(np.any(original[name][finite] != altered[name][finite]))
    if not changed_future:
        raise AssertionError("future CBR corruption did not change transmission")
    return True


def _selected_score(candidate: str):
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
    transmission_causality_check(index, series, references["CNY"])
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
    transmission = transmission_features(index, series, references["CNY"])
    raw_scores = {
        "beta_60_basis": raw * transmission["beta_60"],
        "beta_120_basis": raw * transmission["beta_120"],
        "cross_reversion_60": transmission["cross_reversion_60"],
        "cross_reversion_120": transmission["cross_reversion_120"],
        "projected_60": (
            raw * transmission["beta_60"]
            + REVERSION_WEIGHT * transmission["cross_reversion_60"]
        ),
        "projected_120": (
            raw * transmission["beta_120"]
            + REVERSION_WEIGHT * transmission["cross_reversion_120"]
        ),
    }
    for score in raw_scores.values():
        score[missing] = np.nan

    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    noon = _load(NOON_PATH, "selected")
    router = _load(ROUTER_PATH, "availability_route")
    noon_score = row_scores(noon, len(index))
    noon_rank = causal_percentiles(
        noon_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    raw_rank = causal_percentiles(
        raw, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    router_score = availability_route(noon_rank, raw_rank, missing)
    reconstructed = _outputs(router_score, y5, dates)
    for year in sorted(set(router) & set(reconstructed)):
        np.testing.assert_array_equal(
            router[year]["test_idx"], reconstructed[year]["test_idx"],
        )
        np.testing.assert_array_equal(
            router[year]["test_score"], reconstructed[year]["test_score"],
        )

    route_scores = {}
    candidates = {"availability_router": router}
    for name in SCORES:
        future_rank_check(raw_scores[name], dates, currencies)
        rank = causal_percentiles(
            raw_scores[name], dates, currencies, RANK_WINDOW, RANK_MINIMUM,
        )
        route = availability_route(noon_rank, rank, missing)
        route_scores[name] = route
        candidates[f"route_{name}"] = _outputs(route, y5, dates)
    for name in BLENDS:
        candidates[f"router75_{name}_25"] = _outputs(
            .75 * router_score + .25 * route_scores[name], y5, dates,
        )

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
    score_name, blended = _selected_score(selected)
    if score_name is None:
        comparison["matched_stale20"] = _load(
            ROUTER_PATH, "matched_stale20",
        )
    else:
        delayed = delayed_by_currency(
            raw_scores[score_name][:, None], index, rows=STALE_ROWS,
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
        "fixing_crossrate_projection_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EO",
        "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "scores": SCORES,
        "beta_windows": BETA_WINDOWS,
        "beta_minimum": BETA_MINIMUM,
        "beta_bounds": BETA_BOUNDS,
        "reversion_weight": REVERSION_WEIGHT,
        "blend_scores": BLENDS,
        "blend_weights": {"availability_router": .75, "projection": .25},
        "strict_returns": "only returns ending before the signal row",
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "selection_period": 2024,
        "selected": selected,
        "matched_control": "selected raw score delayed 20 target rows",
        "physical_future_corruption_check": True,
        "outcome_labels_used": False,
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
