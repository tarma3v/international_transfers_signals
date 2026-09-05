"""Packet-DO: label-free incumbent/daily/noon futures consensus."""
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


OUT = Path("results/research/round6/three_view_futures_consensus")
DAILY_PATH = Path("results/research/round6/moex_perpetual/outputs.pkl")
NOON_PATH = Path("results/research/round6/moex_perpetual_hourly_models/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20
PAIR_FORMULAS = (
    "pair_minimum", "pair_geometric", "pair_harmonic",
    "pair_arithmetic", "pair_min75_max25",
)
TRI_FORMULAS = (
    "tri_minimum", "tri_geometric", "tri_harmonic", "tri_arithmetic",
    "tri_median", "tri_lower_quartile", "tri_min75_median25",
)


def formulas(incumbent, daily, noon):
    pair = np.column_stack((incumbent, noon))
    pair_min = np.min(pair, axis=1)
    pair_max = np.max(pair, axis=1)
    tri = np.column_stack((incumbent, daily, noon))
    tri_min = np.min(tri, axis=1)
    tri_median = np.median(tri, axis=1)
    return {
        "pair_minimum": pair_min,
        "pair_geometric": np.exp(np.mean(np.log(np.clip(pair, 1e-9, 1.0)), axis=1)),
        "pair_harmonic": 2.0 / np.sum(1.0 / np.clip(pair, 1e-9, 1.0), axis=1),
        "pair_arithmetic": np.mean(pair, axis=1),
        "pair_min75_max25": .75 * pair_min + .25 * pair_max,
        "tri_minimum": tri_min,
        "tri_geometric": np.exp(np.mean(np.log(np.clip(tri, 1e-9, 1.0)), axis=1)),
        "tri_harmonic": 3.0 / np.sum(1.0 / np.clip(tri, 1e-9, 1.0), axis=1),
        "tri_arithmetic": np.mean(tri, axis=1),
        "tri_median": tri_median,
        "tri_lower_quartile": np.quantile(tri, .25, axis=1),
        "tri_min75_median25": .75 * tri_min + .25 * tri_median,
    }


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}

    incumbent_output = _load(INCUMBENT_PATH, INCUMBENT)
    daily_output = _load(DAILY_PATH, "selected")
    daily_stale_output = _load(DAILY_PATH, "matched_stale20")
    noon_output = _load(NOON_PATH, "selected")
    noon_stale_output = _load(NOON_PATH, "matched_stale20")
    ranks = {}
    for name, output in (
        ("incumbent", incumbent_output),
        ("daily", daily_output),
        ("daily_stale", daily_stale_output),
        ("noon", noon_output),
        ("noon_stale", noon_stale_output),
    ):
        ranks[name] = causal_percentiles(
            row_scores(output, len(index)), dates, currencies,
            RANK_WINDOW, RANK_MINIMUM,
        )
    aligned = formulas(ranks["incumbent"], ranks["daily"], ranks["noon"])
    stale = formulas(
        ranks["incumbent"], ranks["daily_stale"], ranks["noon_stale"],
    )
    formula_names = PAIR_FORMULAS + TRI_FORMULAS
    aligned_outputs = {"incumbent": incumbent_output}
    stale_outputs = {"incumbent": incumbent_output}
    for name in formula_names:
        aligned_outputs[name] = _outputs(aligned[name], y5, dates)
        stale_outputs[name] = _outputs(stale[name], y5, dates)

    screen = horizon_rows(
        aligned_outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {"incumbent": incumbent_output, "selected": aligned_outputs[selected]}
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
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
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
        y5, dates, currencies, valid, masks, "three_view_futures_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DO", "fixed_policy": POLICY,
        "incumbent": INCUMBENT,
        "daily_expert": "packet-DH futures_extra",
        "noon_expert": "packet-DN noon_hist at 12:00 Europe/Moscow",
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "pair_formulas": PAIR_FORMULAS, "three_view_formulas": TRI_FORMULAS,
        "matched_stale": "both daily and noon market experts delayed 20 target rows",
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "formula_is_label_free": True,
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
