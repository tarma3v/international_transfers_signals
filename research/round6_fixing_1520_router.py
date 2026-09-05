"""Packet-EJ: causal availability router at the selected 15:20 cutoff."""
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
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_availability_router import availability_route
from research.round6_moex_spot_1530_features import (
    SESSION_START,
    _arrays,
    load_spot_1530_history,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_state_agreement_geometry import future_rank_check
from research.round6_uzbek_central_bank_models import (
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_1520_router")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
CUTOFF_PATH = Path("results/research/round6/fixing_cutoff_frontier/outputs.pkl")
ROUTE_1530_PATH = Path(
    "results/research/round6/fixing_availability_router/outputs.pkl"
)
CUTOFF = dt.time(15, 20)
RANK_WINDOW = 250
RANK_MINIMUM = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def market_available(index, history, cutoff=CUTOFF):
    item = _arrays(history)["CNYRUB_TOM"]
    available = np.zeros(len(index), dtype=bool)
    for row_number, (_currency, _position, day) in enumerate(index):
        start_time = dt.datetime.combine(day, SESSION_START)
        cutoff_time = dt.datetime.combine(day, cutoff)
        start = int(np.searchsorted(item["begin"], start_time, side="left"))
        stop = int(np.searchsorted(item["end"], cutoff_time, side="left"))
        rows = np.arange(start, stop, dtype=int)
        available[row_number] = bool(np.any(item["begin"][rows] >= start_time))
    return available


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    history, digest = load_spot_1530_history()
    available = market_available(index, history)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}

    noon = _load(NOON_PATH, "selected")
    raw_1520 = _load(CUTOFF_PATH, "selected")
    stale_1520 = _load(CUTOFF_PATH, "matched_stale20")
    route_1530 = _load(ROUTE_1530_PATH, "availability_route")
    scores = {
        "noon": row_scores(noon, len(index)),
        "raw_1520": row_scores(raw_1520, len(index)),
        "stale_1520": row_scores(stale_1520, len(index)),
    }
    for score in scores.values():
        future_rank_check(score, dates, currencies)
    ranks = {
        name: causal_percentiles(
            score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
        )
        for name, score in scores.items()
    }
    route_score = availability_route(
        ranks["noon"], ranks["raw_1520"], ~available,
    )
    stale_route_score = availability_route(
        ranks["noon"], ranks["stale_1520"], ~available,
    )
    route_1520 = _outputs(route_score, y5, dates)
    matched_stale = _outputs(stale_route_score, y5, dates)
    comparison = {
        "noon_consensus": noon,
        "raw_1520": raw_1520,
        "route_1530": route_1530,
        "route_1520": route_1520,
        "matched_stale20": matched_stale,
    }

    screen = horizon_rows(
        comparison, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
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
        y5, dates, currencies, valid, masks, "fixing_1520_router_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    route_combined = h5[
        (h5.candidate == "route_1520")
        & (h5.period == "combined_2025_2026")
    ].iloc[0]
    route_years = h5[
        (h5.candidate == "route_1520")
        & h5.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    route_horizons = later[
        (later.candidate == "route_1520")
        & (later.period == "combined_2025_2026")
    ]
    point_gates = {
        "all_five_lifts_at_least_1p30": bool(
            route_horizons.case_lift.ge(1.30).all()
        ),
        "annual_rate_between_1_and_2": bool(
            route_years.frequency.between(1.0, 2.0).all()
        ),
        "minimum_currency_h5_lift_at_least_1p30": bool(
            route_combined.corridor_lift_min >= 1.30
        ),
        "minimum_quarter_rate_at_least_0p95": bool(
            route_combined.quarter_frequency_min >= .95
        ),
        "all_symmetric_benefits_positive": bool(
            route_horizons.symmetric_benefit_bps.gt(0).all()
        ),
        "all_future_benefits_positive": bool(
            route_horizons.future_only_benefit_bps.gt(0).all()
        ),
    }
    point_gates["point_operational_gates_pass"] = bool(all(point_gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EJ",
        "cutoff": CUTOFF.isoformat(),
        "source_cutoff_selection": "packet-EI selected cutoff_1520 on 2024",
        "route": "15:20 fixing rank when session exists, noon rank otherwise",
        "availability_labels_used": False,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MINIMUM,
        "available_rows": int(available.sum()),
        "total_rows": int(len(available)),
        "payload_sha256": digest,
        "fixed_policy": POLICY,
        "matched_control": "15:20 fixing input delayed 20 target rows only",
        "point_gates": point_gates,
        "next_cbr_rate_used": False,
        "later_period_status": "fixed retrospective timing product",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSCREEN\n" + screen_summary.to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))
    print("\nGATES\n" + json.dumps(point_gates, indent=2))


if __name__ == "__main__":
    main()
