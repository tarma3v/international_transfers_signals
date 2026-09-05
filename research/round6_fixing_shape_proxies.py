"""Packet-EN: reference-invariant intraday persistence proxies for fixing."""
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
    DECISION_TIME,
    SESSION_START,
    _arrays,
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


OUT = Path("results/research/round6/fixing_shape_proxies")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
FIXING_PATH = Path("results/research/round6/fixing_proxies/outputs.pkl")
ROUTER_PATH = Path(
    "results/research/round6/fixing_availability_router/outputs.pkl"
)
RANK_WINDOW = 250
RANK_MINIMUM = 20
STALE_ROWS = 20
FEATURES = (
    "median_close",
    "q25_close",
    "lower_half_mean",
    "block_min_mean",
    "lcb_half_std",
    "lcb_quarter_range",
    "trimmed_mean",
)
BLENDS = (
    "q25_close",
    "lower_half_mean",
    "block_min_mean",
    "lcb_half_std",
    "lcb_quarter_range",
)
BLOCK_BOUNDARIES = (dt.time(12, 0), dt.time(14, 0))


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def _log_adjustment(alternative: float, mean: float) -> float:
    if alternative <= 0.0 or mean <= 0.0:
        return np.nan
    return float(np.log(alternative / mean) * 10000.0)


def session_adjustments(closes, blocks):
    """Return scale/reference-invariant corrections for one complete session."""
    closes = np.asarray(closes, dtype=float)
    blocks = np.asarray(blocks, dtype=int)
    if closes.ndim != 1 or closes.shape != blocks.shape or not len(closes):
        raise ValueError("invalid fixing-session arrays")
    if not np.all(np.isfinite(closes)) or np.any(closes <= 0.0):
        raise ValueError("non-positive fixing-session close")
    mean = float(np.mean(closes))
    ordered = np.sort(closes)
    lower = float(np.mean(ordered[: (len(ordered) + 1) // 2]))
    block_means = [float(np.mean(closes[blocks == block])) for block in range(3)]
    block_min = min(block_means) if all(
        np.any(blocks == block) for block in range(3)
    ) else np.nan
    trim = max(1, int(np.floor(.10 * len(ordered))))
    trimmed = (
        float(np.mean(ordered[trim:-trim]))
        if 2 * trim < len(ordered) else mean
    )
    log_closes = np.log(closes)
    return {
        "median_close": _log_adjustment(float(np.median(closes)), mean),
        "q25_close": _log_adjustment(float(np.quantile(closes, .25)), mean),
        "lower_half_mean": _log_adjustment(lower, mean),
        "block_min_mean": _log_adjustment(block_min, mean),
        "lcb_half_std": -0.5 * float(np.std(log_closes)) * 10000.0,
        "lcb_quarter_range": (
            -0.25 * float(np.max(log_closes) - np.min(log_closes)) * 10000.0
        ),
        "trimmed_mean": _log_adjustment(trimmed, mean),
    }


def shape_adjustments(index, history):
    item = _arrays(history)["CNYRUB_TOM"]
    result = {name: np.full(len(index), np.nan) for name in FEATURES}
    for row_number, (_currency, _position, day) in enumerate(index):
        start_time = dt.datetime.combine(day, SESSION_START)
        cutoff = dt.datetime.combine(day, DECISION_TIME)
        start = int(np.searchsorted(item["begin"], start_time, side="left"))
        stop = int(np.searchsorted(item["end"], cutoff, side="left"))
        rows = np.arange(start, stop, dtype=int)
        rows = rows[item["begin"][rows] >= start_time]
        if not len(rows):
            continue
        blocks = np.asarray([
            0 if item["begin"][row].time() < BLOCK_BOUNDARIES[0]
            else 1 if item["begin"][row].time() < BLOCK_BOUNDARIES[1]
            else 2
            for row in rows
        ], dtype=int)
        current = session_adjustments(item["close"][rows], blocks)
        for name, value in current.items():
            result[name][row_number] = value
    return result


def shape_causality_check(index, history):
    cutoff = dt.date(2025, 6, 30)
    original = shape_adjustments(index, history)
    changed = {}
    cutoff_time = dt.datetime.combine(cutoff, DECISION_TIME)
    for ticker, rows in history.items():
        changed[ticker] = []
        for source in rows:
            clone = dict(source)
            if ticker == "CNYRUB_TOM" and source["end"] >= cutoff_time:
                phase = (source["end"].hour * 6 + source["end"].minute // 10) % 7
                clone["close"] *= 1.0 + .01 * (phase + 1)
            changed[ticker].append(clone)
    altered = shape_adjustments(index, changed)
    dates = np.asarray([row[2] for row in index], dtype=object)
    past = dates <= cutoff
    future = dates > cutoff
    changed_future = False
    for name in FEATURES:
        np.testing.assert_array_equal(original[name][past], altered[name][past])
        finite = future & np.isfinite(original[name]) & np.isfinite(altered[name])
        changed_future |= bool(np.any(original[name][finite] != altered[name][finite]))
    if not changed_future:
        raise AssertionError("future candle corruption did not change shape proxies")
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
    shape_causality_check(index, history)
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
    adjustments = shape_adjustments(index, history)

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

    corrected, route_scores = {}, {}
    candidates = {"availability_router": router}
    for name in FEATURES:
        corrected[name] = raw + adjustments[name]
        corrected[name][missing] = np.nan
        future_rank_check(corrected[name], dates, currencies)
        rank = causal_percentiles(
            corrected[name], dates, currencies, RANK_WINDOW, RANK_MINIMUM,
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
    feature_name, blended = _selected_feature(selected)
    if feature_name is None:
        comparison["matched_stale20"] = _load(
            ROUTER_PATH, "matched_stale20",
        )
    else:
        delayed = delayed_by_currency(
            corrected[feature_name][:, None], index, rows=STALE_ROWS,
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
        y5, dates, currencies, valid, masks, "fixing_shape_proxies_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EN",
        "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "methodology_source": (
            "https://cbr.ru/Content/Document/File/162004/metod_6290-u.pdf"
        ),
        "methodology": "10:00-15:30 trade-volume-weighted CNYRUB_TOM",
        "true_vwap_claimed": False,
        "features": FEATURES,
        "blend_features": BLENDS,
        "blend_weights": {"availability_router": .75, "shape_route": .25},
        "shape_term": "10000*log(alternative/session_mean)",
        "shape_reference_invariant": True,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "selection_period": 2024,
        "selected": selected,
        "matched_control": "selected corrected score delayed 20 target rows",
        "physical_future_corruption_check": True,
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
