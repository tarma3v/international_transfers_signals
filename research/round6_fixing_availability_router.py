"""Packet-ED: causal noon fallback when the 15:30 CNY market is unavailable."""
from __future__ import annotations

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
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_moex_spot_1530_features import (
    build_spot_1530_features,
    load_spot_1530_history,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_state_agreement_geometry import future_rank_check
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_availability_router")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
FIXING_PATH = Path("results/research/round6/fixing_proxies/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def availability_route(noon_rank, fixing_rank, missing):
    noon_rank = np.asarray(noon_rank, dtype=float)
    fixing_rank = np.asarray(fixing_rank, dtype=float)
    missing = np.asarray(missing, dtype=bool)
    if not (noon_rank.shape == fixing_rank.shape == missing.shape):
        raise ValueError("unaligned availability route inputs")
    return np.where(missing, noon_rank, fixing_rank)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    market, market_names = build_spot_1530_features(index, history, references)
    missing = market[:, market_names.index(
        "moex_1530_cnyrub_tom_missing"
    )].astype(bool)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}

    noon = _load(NOON_PATH, "selected")
    fixing = _load(FIXING_PATH, "selected")
    fixing_stale = _load(FIXING_PATH, "matched_stale20")
    noon_score = row_scores(noon, len(index))
    fixing_score = row_scores(fixing, len(index))
    fixing_stale_score = row_scores(fixing_stale, len(index))
    for score in (noon_score, fixing_score, fixing_stale_score):
        future_rank_check(score, dates, currencies)
    noon_rank = causal_percentiles(
        noon_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    fixing_rank = causal_percentiles(
        fixing_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    fixing_stale_rank = causal_percentiles(
        fixing_stale_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    route_score = availability_route(noon_rank, fixing_rank, missing)
    stale_route_score = availability_route(
        noon_rank, fixing_stale_rank, missing,
    )
    route = _outputs(route_score, y5, dates)
    stale_route = _outputs(stale_route_score, y5, dates)
    candidates = {
        "noon_consensus": noon,
        "fixing_basis": fixing,
        "availability_route": route,
    }
    stale_outputs = {
        "fixing_basis": fixing_stale,
        "availability_route": stale_route,
    }
    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "noon_consensus": noon,
        "fixing_basis": fixing,
        "availability_route": route,
        "selected": candidates[selected],
    }
    if selected in stale_outputs:
        comparison["matched_stale20"] = stale_outputs[selected]
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
        y5, dates, currencies, valid, masks, "fixing_availability_router_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "ED", "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "route": "fixing rank when CNY session exists, noon rank otherwise",
        "availability_labels_used": False,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "missing_rows": int(missing.sum()),
        "total_rows": int(len(missing)),
        "selection_period": 2024,
        "selected": selected,
        "matched_control": "fixing input delayed 20 target rows only",
        "next_cbr_rate_used": False,
        "later_period_status": (
            "protocol-controlled retrospective opened after 2024 selection"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("Missing rows:", int(missing.sum()), "/", len(missing))
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
