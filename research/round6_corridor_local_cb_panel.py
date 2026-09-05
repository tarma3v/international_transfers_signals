"""Packet-CT: one separately screened local-central-bank expert per corridor."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_armenian_central_bank_features import build_cba_features, load_cba
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_kazakh_central_bank_features import (
    build_kazakh_nbk_features, load_kazakh_nbk,
)
from research.round6_kyrgyz_central_bank_features import (
    build_kyrgyz_nbkr_features, load_kyrgyz_nbkr,
)
from research.round6_local_central_bank_features import build_nbt_features, load_nbt
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _fire
from research.round6_uzbek_central_bank_features import (
    build_uzbek_cbu_features, load_uzbek_cbu,
)
from research.round6_uzbek_central_bank_models import _forward, horizon_rows, summarize


OUT = Path("results/research/round6/corridor_local_cb_panel")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
SPECS = {
    "TJS": ("nbt", load_nbt, build_nbt_features),
    "UZS": ("uzbek_cbu", load_uzbek_cbu, build_uzbek_cbu_features),
    "KGS": ("kyrgyz_nbkr", load_kyrgyz_nbkr, build_kyrgyz_nbkr_features),
    "AMD": ("cba", load_cba, build_cba_features),
    "KZT": ("kazakh_nbk", load_kazakh_nbk, build_kazakh_nbk_features),
}


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def formula_family(matrix, names, prefix):
    col = {name: matrix[:, names.index(name)].astype(float) for name in names}
    basis = {
        "direct_basis": col[f"{prefix}_direct_basis_bps"],
        "usd_basis": col[f"{prefix}_usd_basis_bps"],
        "cny_basis": col[f"{prefix}_cny_basis_bps"],
        "consensus_basis": col[f"{prefix}_consensus_basis_bps"],
        "direct_minus_usd": col[f"{prefix}_direct_minus_usd_bps"],
        "direct_minus_cny": col[f"{prefix}_direct_minus_cny_bps"],
        "usd_minus_cny": col[f"{prefix}_usd_minus_cny_bps"],
    }
    result = {}
    for name, values in basis.items():
        result[name] = values
        result[f"negative_{name}"] = -values
    for lag in (1, 2, 5, 10, 20):
        values = col[f"{prefix}_rub_quote_ret_{lag}"]
        result[f"rub_momentum_{lag}"] = values
        result[f"rub_reversal_{lag}"] = -values
    return result


def _corridor_screen(output, currency, targets, forwards, dates, currencies):
    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        valid, fired = _fire(output, (2024,), POLICY, y, dates, currencies)
        scope = valid & (currencies == currency)
        active = scope & fired
        rows.append({
            "horizon": h,
            "lift": float(y[active].mean() / y[scope].mean()),
            "symmetric_benefit_bps": float(np.nanmean(
                targets[f"benefit_h{h}"][active]
            )),
            "future_benefit_bps": float(np.nanmean(forwards[h][active])),
            "n_signals": int(active.sum()),
        })
    return rows


def _select(part: pd.DataFrame) -> str:
    summary = part.groupby("formula", as_index=False).agg(
        horizon_lift_min=("lift", "min"),
        horizon_lift_mean=("lift", "mean"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_benefit_bps", "min"),
    )
    feasible = summary[
        summary.symmetric_benefit_min.gt(0)
        & summary.future_benefit_min.gt(0)
    ]
    pool = feasible if len(feasible) else summary
    return str(pool.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0].formula)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    source_scores = {}
    source_outputs = {}
    digests = {}
    screen_rows = []
    selected = {}
    for currency, (prefix, loader, builder) in SPECS.items():
        source, digest = loader()
        digests[currency] = digest
        matrix, names = builder(index, series, references, source)
        formulas = formula_family(matrix, names, prefix)
        source_scores[currency] = formulas
        source_outputs[currency] = {
            name: _outputs(score, y5, dates) for name, score in formulas.items()
        }
        rows = []
        for formula, output in source_outputs[currency].items():
            for row in _corridor_screen(
                output, currency, targets, forwards, dates, currencies,
            ):
                rows.append({"currency": currency, "formula": formula, **row})
        selected[currency] = _select(pd.DataFrame(rows))
        screen_rows.extend(rows)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024_by_corridor_horizon.csv", index=False)
    pd.DataFrame([
        {"currency": currency, "selected_formula": formula}
        for currency, formula in selected.items()
    ]).to_csv(OUT / "selected_formulas.csv", index=False)

    panel_score = np.full(len(index), np.nan)
    for currency, formula in selected.items():
        mask = currencies == currency
        panel_score[mask] = source_scores[currency][formula][mask]
    panel = _outputs(panel_score, y5, dates)
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    outputs = {"incumbent": incumbent, "local_cb_panel": panel}
    for weight in (.05, .10, .20, .30, .40):
        outputs[f"incumbent{int((1-weight)*100)}_local{int(weight*100)}"] = combine_causal(
            [incumbent, panel], (1.0 - weight, weight), dates, currencies,
        )
    finalist_screen = horizon_rows(
        outputs, (2024,), targets, forwards, dates, currencies,
    )
    finalist_summary = summarize(finalist_screen)
    finalist_screen.to_csv(OUT / "finalist_screen_2024_by_horizon.csv", index=False)
    finalist_summary.to_csv(OUT / "finalist_screen_2024_summary.csv", index=False)
    feasible = finalist_summary[
        finalist_summary.symmetric_benefit_min.gt(0)
        & finalist_summary.future_benefit_min.gt(0)
    ]
    selected_finalist = str(feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).iloc[0].candidate)

    comparison = {"incumbent": incumbent, "selected": outputs[selected_finalist]}
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
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CT", "fixed_policy": POLICY,
        "selection_period": 2024, "selected_by_corridor": selected,
        "selected_finalist": selected_finalist,
        "formula_count_per_corridor": len(next(iter(source_scores.values()))),
        "selection_objective": "per-corridor maximum worst official lift over all five horizons with positive benefits",
        "asof_rule": "every local-CB effective date strictly before signal date; CBR date <= signal date",
        "payload_sha256": digests, "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after all 2024 choices",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected formulas:", json.dumps(selected, ensure_ascii=False))
    print("Selected finalist:", selected_finalist)
    print("\nSCREEN\n" + finalist_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
