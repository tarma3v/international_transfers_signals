"""Packet-CZ: use cross-bank dispersion only as a nonlinear uncertainty veto."""
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
from research.round6_armenian_central_bank_features import load_cba
from research.round6_belarus_nbrb_features import load_nbrb
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_crossbank_consensus import build_crossbank_features
from research.round6_georgia_nbg_features import load_nbg
from research.round6_kazakh_central_bank_features import load_kazakh_nbk
from research.round6_kyrgyz_central_bank_features import load_kyrgyz_nbkr
from research.round6_local_central_bank_features import load_nbt
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_features import load_uzbek_cbu
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/crossbank_uncertainty")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
RANK_WINDOW = 250
RANK_MINIMUM = 20
SOURCES = {
    "armenia_cba": load_cba,
    "tajikistan_nbt": load_nbt,
    "uzbekistan_cbu": load_uzbek_cbu,
    "kazakhstan_nbk": load_kazakh_nbk,
    "kyrgyzstan_nbkr": load_kyrgyz_nbkr,
    "georgia_nbg": load_nbg,
    "belarus_nbrb": load_nbrb,
}
QUANTILES = (.70, .80, .90)
PENALTIES = (.15, .30, .50)
BONUSES = (.10, .20)


def adjusted_scores(incumbent_score, dispersion, dates, currencies):
    """Create fixed nonlinear corrections using strictly preceding ranks."""
    incumbent_rank = causal_percentiles(
        incumbent_score, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    dispersion_rank = causal_percentiles(
        dispersion, dates, currencies, RANK_WINDOW, RANK_MINIMUM,
    )
    result = {}
    for quantile in QUANTILES:
        excess = np.maximum(dispersion_rank - quantile, 0.0)
        low = np.maximum((1.0 - quantile) - dispersion_rank, 0.0)
        for penalty in PENALTIES:
            label = f"veto_q{int(quantile*100)}_p{int(penalty*100)}"
            result[label] = incumbent_rank - penalty * excess
        for bonus in BONUSES:
            label = f"confirm_q{int((1-quantile)*100)}_b{int(bonus*100)}"
            result[label] = incumbent_rank + bonus * low
    for quantile in (.85, .95):
        result[f"hard_veto_q{int(quantile*100)}"] = np.where(
            dispersion_rank > quantile, incumbent_rank - 1.0, incumbent_rank,
        )
    return result


def causality_check(incumbent_score, dispersion, dates, currencies):
    unique = np.asarray(sorted(set(dates)))
    cutoff = unique[int(len(unique) * .72)]
    first = adjusted_scores(incumbent_score, dispersion, dates, currencies)
    changed = dispersion.copy()
    future = dates >= cutoff
    changed[future] = np.linspace(-1000.0, 1000.0, int(future.sum()))
    second = adjusted_scores(incumbent_score, changed, dates, currencies)
    for name in first:
        np.testing.assert_array_equal(first[name][dates < cutoff], second[name][dates < cutoff])
    return True


def _load_output():
    with INCUMBENT_PATH.open("rb") as handle:
        return pickle.load(handle)[INCUMBENT]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    sources = {name: loader()[0] for name, loader in SOURCES.items()}
    matrix, names, _availability = build_crossbank_features(index, references, sources)
    dispersion = -matrix[:, names.index("negative_dispersion")].astype(float)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    incumbent = _load_output()
    incumbent_score = row_scores(incumbent, len(index))
    raw = adjusted_scores(incumbent_score, dispersion, dates, currencies)
    causality_check(incumbent_score, dispersion, dates, currencies)
    outputs = {"incumbent": incumbent}
    outputs.update({name: _outputs(score, y5, dates) for name, score in raw.items()})

    screen = horizon_rows(outputs, (2024,), targets, forwards, dates, currencies)
    summary = summarize(screen)
    selected = _choose(summary)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

    comparison = {"incumbent": incumbent, "selected": outputs[selected]}
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
    for candidate, output in outputs.items():
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
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], outputs, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "crossbank_uncertainty_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in outputs.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CZ", "fixed_policy": POLICY,
        "incumbent": INCUMBENT,
        "uncertainty": "cross-source MAD of strictly lagged USD/RUB and CNY/RUB basis consensus",
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "quantiles": QUANTILES, "penalties": PENALTIES, "bonuses": BONUSES,
        "hard_veto_quantiles": [.85, .95],
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "current_value_enters_rank_history_after_decision": True,
        "physical_future_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + summary.sort_values(
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
