"""Packet-CD: label-free geometry of the CNY shadow and survival experts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_expert_geometry")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
SURVIVAL = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20
SCREEN_ORDER = (
    "geometry_minimum",
    "geometry_geometric",
    "geometry_harmonic",
    "geometry_min75_max25",
    "geometry_min90_max10",
    "geometry_highgate65",
    "geometry_highgate70",
    "geometry_highgate75",
    "geometry_stale20_minimum",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _choose(results):
    part = results[results.period == "screen_2024"].copy()
    feasible = part[
        part.frequency.between(1.0, 2.0)
        & part.corridor_freq_min.ge(.80)
        & part.quarter_frequency_min.ge(.70)
        & part.forward_benefit_bps.gt(0.0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return str(pool.sort_values(
        ["robustness", "lift", "forward_benefit_bps"], ascending=False,
    ).iloc[0].candidate)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    shadow_output = _load(SHADOW)["shadow_close_basis"]
    survival_output = _load(SURVIVAL)["survival_cumulative_geometric"]
    shadow = causal_percentiles(
        row_scores(shadow_output, len(y)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    survival = causal_percentiles(
        row_scores(survival_output, len(y)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    lower = np.minimum(shadow, survival)
    upper = np.maximum(shadow, survival)
    mean = .50 * shadow + .50 * survival
    raw = {
        "geometry_minimum": lower,
        "geometry_geometric": np.sqrt(np.clip(shadow * survival, 0.0, 1.0)),
        "geometry_harmonic": 2.0 * shadow * survival / np.clip(
            shadow + survival, 1e-9, None,
        ),
        "geometry_min75_max25": .75 * lower + .25 * upper,
        "geometry_min90_max10": .90 * lower + .10 * upper,
        "geometry_highgate65": np.where(mean > .65, survival, lower),
        "geometry_highgate70": np.where(mean > .70, survival, lower),
        "geometry_highgate75": np.where(mean > .75, survival, lower),
        "geometry_stale20_minimum": delayed_by_currency(
            np.column_stack([shadow, survival]), index, rows=20,
        ).min(axis=1),
    }
    outputs = {name: _outputs(score, y, dates) for name, score in raw.items()}
    screen_rows = []
    for candidate in SCREEN_ORDER:
        item = _evaluate(outputs[candidate], (2024,), POLICY, y, benefit, dates, currencies)
        item.update({"candidate": candidate, "period": "screen_2024", **POLICY})
        screen_rows.append(item)
    chosen = _choose(pd.DataFrame(screen_rows))
    primary = _load(PRIMARY)["logit50_extra50"]
    for weight in (.10, .25, .40):
        name = f"primary{int((1-weight)*100)}_{chosen}{int(weight*100)}"
        outputs[name] = combine_causal(
            [primary, outputs[chosen]], (1.0 - weight, weight), dates, currencies,
        )
    order = SCREEN_ORDER + tuple(
        f"primary{int((1-weight)*100)}_{chosen}{int(weight*100)}"
        for weight in (.10, .25, .40)
    )
    rows = []
    for candidate in order:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[candidate], years, POLICY, y, benefit, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(order)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_expert_geometry_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in order:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CD", "variants": order, "fixed_policy": POLICY,
        "selection_period": 2024, "selected_geometry": chosen,
        "experts": ["shadow_close_basis", "survival_cumulative_geometric"],
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "geometry_is_label_free": True,
        "stale_control_rows_per_currency": 20,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    display = results.sort_values(["period", "predeclared_order"])
    print(f"Selected on 2024: {chosen}\n")
    print(display[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "robustness",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
