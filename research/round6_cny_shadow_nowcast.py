"""Packet-BQ transparent market shadow-rate nowcast."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import Series
from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import (
    build_cny_basis_features,
    causality_check as basis_causality_check,
)
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_shadow_nowcast")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ORDER = (
    "shadow_close_basis",
    "shadow_close_cross5",
    "primary75_shadow25",
    "primary50_shadow50",
    "primary75_shadow_cross25",
)
SCREENED_2024 = (
    "close basis",
    "WAP basis",
    "minimum close/open/WAP basis",
    "close basis - one-day target/CNY cross return",
    "close basis + one-day target/CNY cross return",
    "close basis - 0.20 * five-day target/CNY cross return",
    "negative 20-day cross z-score",
    "negative 60-day cross z-score",
    "close basis - 0.25 * 20-day cross z-score",
    "close basis - 0.25 * 60-day cross z-score",
)


def build_cross_return_5(index, series, cbr_cny):
    values = np.zeros(len(index), dtype=float)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    dates = np.asarray([row[2] for row in index], dtype=object)
    cross = np.zeros(len(index), dtype=float)
    for row, (currency, position, day) in enumerate(index):
        stop = int(np.searchsorted(cbr_cny.dates, day, side="right"))
        if not stop:
            continue
        cross[row] = (
            np.log(series[currency].values[position] / cbr_cny.values[stop - 1])
            * 10000.0
        )
    for currency in np.unique(currencies):
        rows = np.flatnonzero(currencies == currency)
        rows = rows[np.argsort(dates[rows])]
        if len(rows) > 5:
            values[rows[5:]] = cross[rows[5:]] - cross[rows[:-5]]
    return values


def cross_causality_check(index, series, cbr_cny, cutoff=dt.date(2025, 6, 30)):
    first = build_cross_return_5(index, series, cbr_cny)
    changed_series = {}
    for currency, item in series.items():
        values = item.values.copy()
        future = item.dates > cutoff
        values[future] *= np.linspace(2.0, 20.0, int(future.sum()))
        changed_series[currency] = Series(currency, item.dates.copy(), values)
    cny_values = cbr_cny.values.copy()
    future = cbr_cny.dates > cutoff
    cny_values[future] *= np.linspace(3.0, 30.0, int(future.sum()))
    changed_cny = Series(cbr_cny.code, cbr_cny.dates.copy(), cny_values)
    second = build_cross_return_5(index, changed_series, changed_cny)
    past = np.asarray([row[2] <= cutoff for row in index])
    if not np.array_equal(first[past], second[past]):
        raise AssertionError("future official rate changed a past cross return")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    cbr_cny = references["CNY"]
    basis, basis_names = build_cny_basis_features(index, history, cbr_cny)
    basis_causality_check(index, history, cbr_cny)
    if not cross_causality_check(index, series, cbr_cny):
        raise AssertionError("cross-return causality check failed")
    close = np.asarray(basis[:, basis_names.index("cny_basis_close_bps")])
    cross5 = build_cross_return_5(index, series, cbr_cny)
    raw = {
        "shadow_close_basis": close,
        "shadow_close_cross5": close - 0.20 * cross5,
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    outputs = {name: _outputs(score, y, dates) for name, score in raw.items()}
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs["primary75_shadow25"] = combine_causal(
        [primary, outputs["shadow_close_basis"]], (.75, .25), dates, currencies,
    )
    outputs["primary50_shadow50"] = combine_causal(
        [primary, outputs["shadow_close_basis"]], (.50, .50), dates, currencies,
    )
    outputs["primary75_shadow_cross25"] = combine_causal(
        [primary, outputs["shadow_close_cross5"]], (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                outputs[candidate], years, POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_shadow_nowcast_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BQ",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "architecture_screen_year": 2024,
        "screened_formulas": SCREENED_2024,
        "advanced_formulas": {
            "shadow_close_basis": "CNY MOEX close / official CBR CNY log basis",
            "shadow_close_cross5": "close basis - 0.20 * target/CNY cross return 5",
        },
        "blend_weights": [[.75, .25], [.50, .50]],
        "basis_feature": "cny_basis_close_bps",
        "asof_rule": "MOEX TRADEDATE < signal; official CBR date <= signal",
        "same_day_market_close_allowed": False,
        "physical_future_corruption_check": True,
        "payload_sha256": digest,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
