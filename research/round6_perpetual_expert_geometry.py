"""Packet-DI: label-free incumbent/perpetual-futures expert geometry."""
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
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/perpetual_expert_geometry")
FUTURES_PATH = Path("results/research/round6/moex_perpetual/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20
FORMULAS = (
    "minimum", "geometric", "harmonic", "arithmetic",
    "min75_max25", "min90_max10", "maximum",
    "highgate65", "highgate75", "disagreement_penalty25",
)


def geometries(incumbent, futures):
    lower = np.minimum(incumbent, futures)
    upper = np.maximum(incumbent, futures)
    mean = .5 * (incumbent + futures)
    return {
        "minimum": lower,
        "geometric": np.sqrt(np.clip(incumbent * futures, 0.0, 1.0)),
        "harmonic": 2.0 * incumbent * futures / np.clip(
            incumbent + futures, 1e-9, None,
        ),
        "arithmetic": mean,
        "min75_max25": .75 * lower + .25 * upper,
        "min90_max10": .90 * lower + .10 * upper,
        "maximum": upper,
        "highgate65": np.where(mean >= .65, upper, lower),
        "highgate75": np.where(mean >= .75, upper, lower),
        "disagreement_penalty25": mean - .25 * np.abs(incumbent - futures),
    }


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
    futures_output = _load(FUTURES_PATH, "selected")
    stale_output = _load(FUTURES_PATH, "matched_stale20")
    incumbent_rank = causal_percentiles(
        row_scores(incumbent_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    futures_rank = causal_percentiles(
        row_scores(futures_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    stale_rank = causal_percentiles(
        row_scores(stale_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    aligned_scores = geometries(incumbent_rank, futures_rank)
    stale_scores = geometries(incumbent_rank, stale_rank)
    aligned_outputs = {"incumbent": incumbent_output}
    stale_outputs = {"incumbent": incumbent_output}
    for name in FORMULAS:
        aligned_outputs[name] = _outputs(aligned_scores[name], y5, dates)
        stale_outputs[name] = _outputs(stale_scores[name], y5, dates)

    screen = horizon_rows(
        aligned_outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "incumbent": incumbent_output,
        "selected": aligned_outputs[selected],
    }
    if selected != "incumbent":
        comparison["matched_stale20"] = stale_outputs[selected]
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
        y5, dates, currencies, valid, masks, "perpetual_geometry_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DI", "fixed_policy": POLICY,
        "incumbent": INCUMBENT,
        "futures_expert": "packet-DH selected futures_extra",
        "matched_stale_expert": "packet-DH matched_stale20",
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "formulas": FORMULAS,
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "geometry_is_label_free": True,
        "current_score_added_after_rank": True,
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
