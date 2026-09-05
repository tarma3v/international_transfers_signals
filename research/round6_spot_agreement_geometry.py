"""Packet-DX: label-free agreement geometry for noon and signed spot."""
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


OUT = Path("results/research/round6/spot_agreement_geometry")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
SPOT_PATH = Path("results/research/round6/spot_signed_nowcast/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20
FORMULAS = (
    "minimum", "geometric", "harmonic", "arithmetic", "min75_max25",
)


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def formulas(first, second):
    pair = np.column_stack((first, second))
    minimum = np.min(pair, axis=1)
    maximum = np.max(pair, axis=1)
    clipped = np.clip(pair, 1e-9, 1.0)
    return {
        "minimum": minimum,
        "geometric": np.exp(np.mean(np.log(clipped), axis=1)),
        "harmonic": 2.0 / np.sum(1.0 / clipped, axis=1),
        "arithmetic": np.mean(pair, axis=1),
        "min75_max25": .75 * minimum + .25 * maximum,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}

    noon_output = _load(NOON_PATH, "selected")
    spot_output = _load(SPOT_PATH, "selected")
    stale_spot_output = _load(SPOT_PATH, "matched_stale20")
    noon_rank = causal_percentiles(
        row_scores(noon_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    spot_rank = causal_percentiles(
        row_scores(spot_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    stale_spot_rank = causal_percentiles(
        row_scores(stale_spot_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    fresh_formula = formulas(noon_rank, spot_rank)
    stale_formula = formulas(noon_rank, stale_spot_rank)
    candidates = {
        "noon_consensus": noon_output,
        "signed_spot": spot_output,
        **{
            name: _outputs(score, y5, dates)
            for name, score in fresh_formula.items()
        },
    }
    stale_outputs = {
        name: _outputs(score, y5, dates)
        for name, score in stale_formula.items()
    }
    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "noon_consensus": noon_output,
        "signed_spot": spot_output,
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
            ("screen_2024", (2024,)), ("retrospective_2025", (2025,)),
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
        y5, dates, currencies, valid, masks, "spot_agreement_geometry_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DX", "fixed_policy": POLICY,
        "experts": ("packet-DO noon consensus", "packet-DU signed spot"),
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "formulas": FORMULAS, "formula_labels_used": False,
        "selection_period": 2024, "selected": selected,
        "matched_control": "signed spot delayed 20 target rows only",
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
