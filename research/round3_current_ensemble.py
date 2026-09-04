"""Chronological post-break ensembles for the current operating regime.

The ensemble combines three structurally different scores: a model trained only
after the 2022 break, a delayed-feedback online expert mix, and the transparent
multiscale price-position anchor.  Weights and policies are selected on 2024,
gated on 2025, and read on 2026.  The entire exercise remains retrospective.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from research.extended_features import load_or_build
from research.model_study import combine_outputs, evaluate
from research.round2_diverse_models import _metric_grid, _select
from research.round3_postshock_reset import _benefit

ROOT = Path("results/research")
OUT = ROOT / "round3"


def _load() -> dict:
    with (OUT / "postshock_reset_outputs.pkl").open("rb") as fh:
        reset = pickle.load(fh)["reset_xgb"]
    with (OUT / "online_mixture_outputs.pkl").open("rb") as fh:
        online_cache = pickle.load(fh)
        online = online_cache["hedge_local_eta2_rho0p99"]
        consensus = online_cache["consensus_geometric"]
    with (ROOT / "candidate_outputs_h5_v2.pkl").open("rb") as fh:
        anchor = pickle.load(fh)["anchor_multiscale"]
    return {"reset_xgb": reset, "online_hedge": online,
            "consensus_geometric": consensus, "multiscale_anchor": anchor}


def _specs() -> list[tuple[str, tuple[str, ...], tuple[float, ...]]]:
    result = []
    for left, right in (
        ("reset_xgb", "online_hedge"),
        ("reset_xgb", "multiscale_anchor"),
        ("online_hedge", "multiscale_anchor"),
        ("reset_xgb", "consensus_geometric"),
        ("consensus_geometric", "multiscale_anchor"),
    ):
        for weight in (.25, .50, .75):
            result.append((
                f"{weight:.2f}*{left}+{1-weight:.2f}*{right}",
                (left, right), (weight, 1.0 - weight),
            ))
    result.extend([
        ("equal_three", ("reset_xgb", "online_hedge", "multiscale_anchor"),
         (1 / 3, 1 / 3, 1 / 3)),
        ("reset_half_online_anchor_quarters",
         ("reset_xgb", "online_hedge", "multiscale_anchor"), (.5, .25, .25)),
        ("online_half_reset_anchor_quarters",
         ("reset_xgb", "online_hedge", "multiscale_anchor"), (.25, .5, .25)),
        ("equal_reset_consensus_anchor",
         ("reset_xgb", "consensus_geometric", "multiscale_anchor"),
         (1 / 3, 1 / 3, 1 / 3)),
        ("reset_half_consensus_anchor_quarters",
         ("reset_xgb", "consensus_geometric", "multiscale_anchor"), (.5, .25, .25)),
        ("consensus_half_reset_anchor_quarters",
         ("reset_xgb", "consensus_geometric", "multiscale_anchor"), (.25, .5, .25)),
    ])
    return result


def main() -> None:
    outputs = _load()
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = _benefit(series, index)

    ensembles = {}
    registry = []
    for name, members, weights in _specs():
        ensembles[name] = combine_outputs([outputs[m] for m in members], weights, currencies)
        registry.append({"candidate": name, "members": members, "weights": weights})
    with (OUT / "current_ensemble_outputs.pkl").open("wb") as fh:
        pickle.dump(ensembles, fh, protocol=pickle.HIGHEST_PROTOCOL)

    grid = []
    for name, output in ensembles.items():
        grid.extend(_metric_grid(output, y, dates, currencies, benefit, (2024,), name))
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "current_ensemble_2024_grid.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _n, z in grid.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "current_ensemble_stage1_2024.csv", index=False)

    gate = []
    for row in stage1.itertuples(index=False):
        result = evaluate(
            ensembles[row.candidate], y, dates, currencies, benefit, (2025,),
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate, "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        gate.append(result)
    gate = pd.DataFrame(gate)
    gate["robustness"] = gate[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    gate = gate.sort_values(["robustness", "lift"], ascending=False)
    gate.to_csv(OUT / "current_ensemble_stage2_2025.csv", index=False)

    final = []
    for row in gate.head(4).itertuples(index=False):
        result = evaluate(
            ensembles[row.candidate], y, dates, currencies, benefit, (2026,),
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective; 2026 inspected in earlier rounds"})
        final.append(result)
    final = pd.DataFrame(final).sort_values("lift", ascending=False)
    final.to_csv(OUT / "current_ensemble_2026_retrospective.csv", index=False)
    (OUT / "current_ensemble_protocol.json").write_text(json.dumps({
        "registry": registry, "selection_year": 2024, "gate_year": 2025,
        "reported_last_year": 2026,
        "limitation": "no pristine post-break holdout remains",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "corridor_freq_min", "corridor_lift_min", "robustness"]
    print("\n2024", stage1[columns].to_string(index=False), sep="\n")
    print("\n2025", gate[columns].to_string(index=False), sep="\n")
    print("\n2026", final[[c for c in columns if c in final]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
