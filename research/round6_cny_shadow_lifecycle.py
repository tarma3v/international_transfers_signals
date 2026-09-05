"""Packet-BS full-history transport of the direct CNY shadow nowcast."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_basis_features import build_cny_basis_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import load_moex_history
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_shadow_lifecycle")
YEARS = tuple(range(2017, 2027))
ORDER = ("shadow_raw_lifecycle", "shadow_available_lifecycle")


def _full_outputs(score, y, dates):
    result = {}
    for year in YEARS:
        calibration = np.asarray([day.year == year - 1 for day in dates])
        test = np.asarray([day.year == year for day in dates])
        calibration &= np.isfinite(y) & np.isfinite(score)
        test &= np.isfinite(y) & np.isfinite(score)
        ca, te = np.flatnonzero(calibration), np.flatnonzero(test)
        result[year] = {
            "calib_idx": ca,
            "test_idx": te,
            "calib_score": score[ca],
            "test_score": score[te],
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    _broad, _broad_names, references = load_broad_features(index, series)
    basis, names = build_cny_basis_features(index, history, references["CNY"])
    close = np.asarray(basis[:, names.index("cny_basis_close_bps")], dtype=float)
    missing = basis[:, names.index("cny_basis_missing")].astype(bool)
    available = close.copy()
    available[missing] = -1e9
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    outputs = {
        "shadow_raw_lifecycle": _full_outputs(close, y, dates),
        "shadow_available_lifecycle": _full_outputs(available, y, dates),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    periods = (
        ("pre_svo_2017_2021", tuple(range(2017, 2022))),
        ("transition_2022_2023", (2022, 2023)),
        ("post_2024_2026", (2024, 2025, 2026)),
        ("full_2017_2026", YEARS),
    )
    rows = []
    for candidate in ORDER:
        for period, years in periods:
            item = _evaluate(
                outputs[candidate], years, POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            rows.append(item)
        for year in YEARS:
            item = _evaluate(
                outputs[candidate], (year,), POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": f"year_{year}", **POLICY})
            rows.append(item)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "matched_results.csv", index=False)

    coverage = []
    for year in YEARS:
        scope = np.asarray([day.year == year for day in dates]) & np.isfinite(y)
        coverage.append({
            "year": year,
            "n": int(scope.sum()),
            "available": int((scope & ~missing).sum()),
            "coverage": float((~missing[scope]).mean()),
        })
    pd.DataFrame(coverage).to_csv(OUT / "market_coverage.csv", index=False)

    selected = results[results.period == "full_2017_2026"].copy()
    bootstrap, masks, valid = _bootstrap(
        selected, outputs, YEARS, y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2017_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_shadow_lifecycle_2017_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
    breakdown = pd.DataFrame(breakdown)
    breakdown.to_csv(OUT / "breakdown_2017_2026.csv", index=False)

    summaries = []
    for candidate in ORDER:
        annual = results[
            (results.candidate == candidate) & results.period.str.startswith("year_")
        ]
        combined = results[
            (results.candidate == candidate) & (results.period == "full_2017_2026")
        ].iloc[0]
        gates = {
            "annual_lift_at_least_1_30": bool(annual.lift.ge(1.30).all()),
            "annual_rate_between_1_and_2": bool(annual.frequency.between(1, 2).all()),
            "minimum_currency_lift_at_least_1_30": bool(
                combined.corridor_lift_min >= 1.30
            ),
            "minimum_quarter_rate_at_least_0_75": bool(
                combined.quarter_frequency_min >= .75
            ),
        }
        summaries.append({
            "candidate": candidate,
            "combined_lift": float(combined.lift),
            "combined_frequency": float(combined.frequency),
            "combined_benefit_bps": float(combined.forward_benefit_bps),
            "minimum_annual_lift": float(annual.lift.min()),
            "minimum_annual_frequency": float(annual.frequency.min()),
            "maximum_annual_frequency": float(annual.frequency.max()),
            "minimum_currency_lift": float(combined.corridor_lift_min),
            "minimum_quarter_frequency": float(combined.quarter_frequency_min),
            **gates,
            "all_transport_gates": bool(all(gates.values())),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "lifecycle_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BS",
        "variants": ORDER,
        "years": YEARS,
        "periods": {name: years for name, years in periods},
        "fixed_policy": POLICY,
        "availability_gate": "basis_missing -> score -1e9",
        "asof_rule": "MOEX TRADEDATE < signal date",
        "payload_sha256": digest,
        "labels_fitted": False,
        "next_cbr_rate_used": False,
        "older_period_status": "frozen transport audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min",
    ]].to_string(index=False))
    print("\nSUMMARY\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
