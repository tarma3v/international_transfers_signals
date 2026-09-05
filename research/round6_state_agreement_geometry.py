"""Packet-DF: label-free geometry of incumbent and target-state ranks."""
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
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/state_agreement_geometry")
STATE_PATH = Path("results/research/round6/target_state_space/outputs.pkl")
STATE = "state_selected"
RANK_WINDOW = 250
RANK_MINIMUM = 20
FORMULAS = (
    "minimum",
    "geometric",
    "harmonic",
    "min75_max25",
    "min90_max10",
    "low_penalty10",
    "low_penalty20",
    "high_bonus10",
    "high_bonus20",
    "disagreement_penalty20",
)


def geometries(incumbent, state):
    lower = np.minimum(incumbent, state)
    upper = np.maximum(incumbent, state)
    return {
        "minimum": lower,
        "geometric": np.sqrt(np.clip(incumbent * state, 0.0, 1.0)),
        "harmonic": 2.0 * incumbent * state / np.clip(
            incumbent + state, 1e-9, None,
        ),
        "min75_max25": .75 * lower + .25 * upper,
        "min90_max10": .90 * lower + .10 * upper,
        "low_penalty10": incumbent - .10 * np.maximum(.5 - state, 0.0),
        "low_penalty20": incumbent - .20 * np.maximum(.5 - state, 0.0),
        "high_bonus10": incumbent + .10 * np.maximum(state - .5, 0.0),
        "high_bonus20": incumbent + .20 * np.maximum(state - .5, 0.0),
        "disagreement_penalty20": incumbent - .20 * np.abs(incumbent - state),
    }


def future_rank_check(scores, dates, currencies):
    original = causal_percentiles(
        scores, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    cutoff = np.datetime64("2025-06-30")
    day64 = np.asarray(dates, dtype="datetime64[D]")
    changed = np.asarray(scores, dtype=float).copy()
    future = day64 > cutoff
    changed[future] = np.linspace(-1e6, 1e6, int(future.sum()))
    altered = causal_percentiles(
        changed, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    np.testing.assert_array_equal(original[~future], altered[~future])
    return True


def _load(path, candidate):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    incumbent_output = _load(INCUMBENT_PATH, INCUMBENT)
    state_output = _load(STATE_PATH, STATE)
    incumbent_score = row_scores(incumbent_output, len(index))
    state_score = row_scores(state_output, len(index))
    future_rank_check(state_score, dates, currencies)
    incumbent_rank = causal_percentiles(
        incumbent_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    state_rank = causal_percentiles(
        state_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    stale_state_rank = delayed_by_currency(
        state_rank[:, None], index, rows=20,
    )[:, 0]
    aligned_scores = geometries(incumbent_rank, state_rank)
    stale_scores = geometries(incumbent_rank, stale_state_rank)
    outputs = {"incumbent": incumbent_output}
    for name in FORMULAS:
        outputs[name] = _outputs(aligned_scores[name], y5, dates)
        outputs[f"{name}_stale20"] = _outputs(stale_scores[name], y5, dates)

    screen_candidates = {"incumbent": outputs["incumbent"]}
    screen_candidates.update({name: outputs[name] for name in FORMULAS})
    screen = horizon_rows(
        screen_candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    if selected == "incumbent":
        comparison = {
            "incumbent": outputs["incumbent"],
            "selected": outputs["incumbent"],
        }
    else:
        comparison = {
            "incumbent": outputs["incumbent"],
            "selected": outputs[selected],
            "matched_stale20": outputs[f"{selected}_stale20"],
        }
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(comparison, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summarize(later[later.period == "combined_2025_2026"]).to_csv(
        OUT / "later_summary.csv", index=False,
    )

    h5_rows = []
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, POLICY, y5, forwards[5], dates, currencies)
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
        y5, dates, currencies, valid, masks, "state_agreement_geometry_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DF", "fixed_policy": POLICY,
        "incumbent": INCUMBENT, "state_expert": STATE,
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "formulas": FORMULAS, "matched_state_stale_rows": 20,
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "geometry_is_label_free": True,
        "current_score_added_after_rank": True,
        "physical_future_score_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + summarize(
        later[later.period == "combined_2025_2026"]
    ).to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
