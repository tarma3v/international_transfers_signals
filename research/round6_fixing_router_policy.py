"""Packet-EF: cadence-first causal policy plateau for the fixing router."""
from __future__ import annotations

from dataclasses import asdict
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_policy_screen import (
    PolicySpec,
    evaluate,
    fire,
    future_score_check,
)
from research.round6_uzbek_central_bank_models import _forward
from ml.targets import HORIZONS


OUT = Path("results/research/round6/fixing_router_policy")
SOURCE = Path("results/research/round6/fixing_availability_router/outputs.pkl")
CANDIDATE = "availability_route"
RATES = (.20, .22, .24, .26)
ROLLING_MEMORIES = (15, 20, 30, 40, 60)
EXPONENTIAL_HALF_LIVES = (10, 20, 40)


def policy_grid():
    rolling = [
        PolicySpec("rolling", rate, memory)
        for rate in RATES
        for memory in ROLLING_MEMORIES
    ]
    exponential = [
        PolicySpec("exponential", rate, half_life)
        for rate in RATES
        for half_life in EXPONENTIAL_HALF_LIVES
    ]
    return rolling + exponential


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    with SOURCE.open("rb") as handle:
        output = pickle.load(handle)[CANDIDATE]
    future_score_check(output, dates, currencies, targets["fav_h5"])

    screen_rows, screen_detail = [], []
    for spec in policy_grid():
        detail, summary = evaluate(
            output, (2024,), spec, targets, forwards, dates, currencies,
        )
        name = f"{spec.kind}_r{int(spec.rate*100):02d}_m{spec.memory:03d}"
        screen_rows.append({"policy": name, **asdict(spec), **summary})
        screen_detail.extend({
            "policy": name, **asdict(spec), **row,
        } for row in detail)
    screen = pd.DataFrame(screen_rows)
    detail = pd.DataFrame(screen_detail)
    screen.to_csv(OUT / "screen_2024_summary.csv", index=False)
    detail.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    feasible = screen[
        screen.frequency.between(1.0, 2.0)
        & screen.corridor_frequency_min.ge(1.0)
        & screen.quarter_frequency_min.ge(1.0)
        & screen.symmetric_benefit_min.gt(0)
        & screen.future_benefit_min.gt(0)
    ]
    if feasible.empty:
        raise RuntimeError("no cadence-feasible packet-EF policy on 2024")
    selected_row = feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "quarter_frequency_min"],
        ascending=False,
    ).iloc[0]
    selected = PolicySpec(
        str(selected_row.kind),
        float(selected_row.rate),
        int(selected_row.memory),
    )
    default = PolicySpec(
        "rolling", float(POLICY["rate"]), int(POLICY["rolling"]),
    )

    later_rows, later_summary = [], []
    for label, spec in (("default", default), ("selected", selected)):
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            rows, summary = evaluate(
                output, years, spec, targets, forwards, dates, currencies,
            )
            later_rows.extend({
                "candidate": label,
                "period": period,
                **asdict(spec),
                **row,
            } for row in rows)
            later_summary.append({
                "candidate": label,
                "period": period,
                **asdict(spec),
                **summary,
            })
    later = pd.DataFrame(later_rows)
    summaries = pd.DataFrame(later_summary)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summaries.to_csv(OUT / "later_summary.csv", index=False)

    selected_later = summaries[summaries.candidate == "selected"]
    selected_combined = selected_later[
        selected_later.period == "combined_2025_2026"
    ].iloc[0]
    selected_years = selected_later[selected_later.period != "combined_2025_2026"]
    valid_h5, fired_h5 = fire(
        output, (2025, 2026), dates, currencies, targets["fav_h5"], selected,
    )
    corridor_lift_min = min(
        float(np.mean(targets["fav_h5"][fired_h5 & valid_h5 & (currencies == currency)])
              / np.mean(targets["fav_h5"][valid_h5 & (currencies == currency)]))
        for currency in sorted(set(currencies))
    )
    gates = {
        "all_five_lifts_at_least_1p30": bool(
            later[
                (later.candidate == "selected")
                & (later.period == "combined_2025_2026")
            ].case_lift.ge(1.30).all()
        ),
        "annual_rate_between_1_and_2": bool(
            selected_years.frequency.between(1.0, 2.0).all()
        ),
        "minimum_currency_lift_at_least_1p30": bool(
            corridor_lift_min >= 1.30
        ),
        "minimum_quarter_rate_at_least_1": bool(
            selected_combined.quarter_frequency_min >= 1.0
        ),
        "all_symmetric_benefits_positive": bool(
            selected_combined.symmetric_benefit_min > 0.0
        ),
        "all_future_benefits_positive": bool(
            selected_combined.future_benefit_min > 0.0
        ),
    }
    gates["point_operational_gates_pass"] = bool(all(gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EF",
        "source_candidate": "packet-ED availability_route",
        "selection_period": 2024,
        "selected_policy": asdict(selected),
        "default_policy": asdict(default),
        "rates": RATES,
        "rolling_memories": ROLLING_MEMORIES,
        "exponential_half_lives": EXPONENTIAL_HALF_LIVES,
        "candidate_count": len(screen),
        "selection_objective": (
            "maximum worst official lift, then mean lift, then quarter cadence"
        ),
        "selection_constraints": (
            "h5 rate 1-2, minimum currency and quarter rate >=1, "
            "all benefits positive"
        ),
        "threshold_history": "strictly earlier same-currency scores",
        "physical_future_score_corruption_check": True,
        "next_cbr_rate_used": False,
        "combined_h5_minimum_currency_lift": corridor_lift_min,
        "point_gates": gates,
        "later_period_status": (
            "protocol-controlled retrospective opened after 2024 selection"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected:", selected)
    print("\nSCREEN TOP\n" + feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + summaries.to_string(index=False))
    print("\nCOMBINED BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))
    print("\nGATES\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
