"""Packet-DQ: fixed causal balance of noon consensus and target state."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/noon_state_balance")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
STATE_PATH = Path("results/research/round6/target_state_space/outputs.pkl")
STATE_WEIGHTS = (.10, .25, .40)


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def candidates(noon, state, incumbent, dates, currencies):
    result = {
        "incumbent": incumbent,
        "noon_consensus": noon,
        "state": state,
    }
    for weight in STATE_WEIGHTS:
        name = f"noon{int((1.0-weight)*100)}_state{int(weight*100)}"
        result[name] = combine_causal(
            (noon, state), (1.0 - weight, weight), dates, currencies,
        )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = pd.Series([row[2] for row in index]).to_numpy(dtype=object)
    currencies = pd.Series([row[0] for row in index]).to_numpy(dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]

    noon = _load(NOON_PATH, "selected")
    state = _load(STATE_PATH, "state_selected")
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    outputs = candidates(noon, state, incumbent, dates, currencies)

    screen = horizon_rows(outputs, (2024,), targets, forwards, dates, currencies)
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "incumbent": incumbent,
        "noon_consensus": noon,
        "state": state,
        "selected": outputs[selected],
    }
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
        y5, dates, currencies, valid, masks, "noon_state_balance_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DQ", "fixed_policy": POLICY,
        "short_medium_expert": "packet-DO selected noon/incumbent consensus",
        "long_expert": "packet-DE raw selected target state-space score",
        "state_weights": STATE_WEIGHTS,
        "combination": "causal same-currency ranks against prior calibration scores",
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "next_cbr_rate_used": False,
        "status": "retrospective causal challenger; coarse family frozen before metrics",
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
