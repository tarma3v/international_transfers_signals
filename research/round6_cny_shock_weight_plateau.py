"""Packet-AX sensitivity plateau for fixed CNY/anchor shock-bridge weights."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS, _stitch
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _breakdown, _evaluate


OUT = Path("results/research/round6/cny_shock_weight_plateau")
PRE = Path("results/research/round6/cny_pre2022/outputs.pkl")
BRIDGE = Path("results/research/round6/cny_shock_bridge/outputs.pkl")
LATER = Path("results/research/round6/cny_history_weighting/outputs.pkl")
ANCHOR = Path("results/research/candidate_outputs_h5_v2.pkl")
WEIGHTS = (.50, .60, .70, .75, .80, .90, 1.00)
SHOCK_YEARS = (2022, 2023)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    pre = _load(PRE)["logit50_extra50"]
    cny = _load(BRIDGE)["expanding_consensus"]
    later = _load(LATER)["hard_reset"]
    anchor = _load(ANCHOR)["anchor_multiscale"]
    shock_outputs, lifecycle_outputs = {}, {}
    for weight in WEIGHTS:
        name = f"cny{int(round(weight * 100)):03d}_anchor{int(round((1-weight)*100)):03d}"
        if weight == 1.0:
            shock = cny
        else:
            shock = combine_causal(
                [cny, anchor], (weight, 1.0 - weight), dates, currencies,
            )
        shock_outputs[name] = shock
        lifecycle_outputs[name] = _stitch(pre, shock, later)
    with (OUT / "shock_outputs.pkl").open("wb") as handle:
        pickle.dump(shock_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT / "lifecycle_outputs.pkl").open("wb") as handle:
        pickle.dump(lifecycle_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows, quarter_rows = [], []
    for weight, name in zip(WEIGHTS, shock_outputs):
        detail = _breakdown(
            name, shock_outputs[name], SHOCK_YEARS, POLICY,
            y, benefit, dates, currencies,
        )
        quarter_rows.extend({**row, "cny_weight": weight} for row in detail)
        quarters = [row for row in detail if row["breakdown"] == "quarter"]
        shock_metrics = {}
        for period, years in (
            ("2022", (2022,)), ("2023", (2023,)),
            ("2022_2023", SHOCK_YEARS),
        ):
            shock_metrics[period] = _evaluate(
                shock_outputs[name], years, POLICY,
                y, benefit, dates, currencies,
            )
        life = _evaluate(
            lifecycle_outputs[name], YEARS, POLICY,
            y, benefit, dates, currencies,
        )
        rows.append({
            "candidate": name,
            "cny_weight": weight,
            "anchor_weight": 1.0 - weight,
            "lift_2022": shock_metrics["2022"]["lift"],
            "rate_2022": shock_metrics["2022"]["frequency"],
            "benefit_2022": shock_metrics["2022"]["forward_benefit_bps"],
            "lift_2023": shock_metrics["2023"]["lift"],
            "rate_2023": shock_metrics["2023"]["frequency"],
            "benefit_2023": shock_metrics["2023"]["forward_benefit_bps"],
            "shock_lift": shock_metrics["2022_2023"]["lift"],
            "shock_benefit": shock_metrics["2022_2023"]["forward_benefit_bps"],
            "shock_quarter_lift_min": min(row["lift"] for row in quarters),
            "shock_quarter_rate_min": min(row["frequency"] for row in quarters),
            "lifecycle_lift": life["lift"],
            "lifecycle_rate": life["frequency"],
            "lifecycle_benefit": life["forward_benefit_bps"],
            "lifecycle_year_lift_min": life["year_lift_min"],
            "lifecycle_year_rate_min": life["year_frequency_min"],
            "lifecycle_currency_lift_min": life["corridor_lift_min"],
        })
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "weight_plateau.csv", index=False)
    pd.DataFrame(quarter_rows).to_csv(OUT / "shock_breakdown.csv", index=False)
    control = grid[grid.cny_weight == 1.0].iloc[0]
    neighbouring = grid[grid.cny_weight.isin((.60, .70, .75, .80, .90))]
    robust_cells = neighbouring[
        neighbouring.lift_2022.gt(control.lift_2022)
        & neighbouring.lift_2023.gt(control.lift_2023)
        & neighbouring.lifecycle_year_lift_min.gt(control.lifecycle_year_lift_min)
    ]
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AX",
        "cny_weights": WEIGHTS,
        "fixed_policy": POLICY,
        "original_challenger_weight": 0.75,
        "grid_used_for_selection": False,
        "neighbouring_cells_improving_both_years_and_min_year": int(len(robust_cells)),
        "model_refit": False,
        "later_period_status": "post-diagnostic retrospective sensitivity",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(grid.to_string(index=False))
    print(f"\nRobust neighbouring cells: {len(robust_cells)}/{len(neighbouring)}")


if __name__ == "__main__":
    main()
