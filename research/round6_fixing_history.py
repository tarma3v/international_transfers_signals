"""Packet-EG: fixed 2022--2026 lifecycle audit of the 15:30 fixing proxy."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_fixing_proxies import proxy_causality_check, proxy_scores
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_history")
YEARS = tuple(range(2022, 2027))
FIRST_AVAILABLE = dt.date(2022, 1, 3)


def calendar_outputs(scores, y, dates, years=YEARS):
    scores = np.asarray(scores, dtype=float)
    result = {}
    for year in years:
        calibration = (
            (dates >= dt.date(year - 1, 1, 1))
            & (dates < dt.date(year, 1, 1))
            & np.isfinite(scores)
            & np.isfinite(y)
        )
        test = (
            np.asarray([day.year == year for day in dates])
            & np.isfinite(scores)
            & np.isfinite(y)
        )
        ca, te = np.where(calibration)[0], np.where(test)[0]
        result[year] = {
            "calib_idx": ca,
            "test_idx": te,
            "calib_score": scores[ca],
            "test_score": scores[te],
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    matrix = proxy_scores(index, history, references)
    proxy_causality_check(index, history, references)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    score = matrix[:, 0].astype(float)
    score[dates < FIRST_AVAILABLE] = np.nan
    output = calendar_outputs(score, y5, dates)
    outputs = {"fixing_basis": output}

    horizon_parts = []
    periods = [(str(year), (year,)) for year in YEARS]
    periods.append(("combined_2022_2026", YEARS))
    for period, years in periods:
        part = horizon_rows(
            outputs, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        horizon_parts.append(part)
    by_horizon = pd.concat(horizon_parts, ignore_index=True)
    by_horizon.to_csv(OUT / "by_year_horizon.csv", index=False)
    summaries = pd.concat([
        summarize(by_horizon[by_horizon.period == period]).assign(period=period)
        for period, _years in periods
    ], ignore_index=True)
    summaries.to_csv(OUT / "by_year_summary.csv", index=False)

    h5_rows = []
    for period, years in periods:
        item = _evaluate(
            output, years, POLICY, y5, forwards[5], dates, currencies,
        )
        item.update({"candidate": "fixing_basis", "period": period, **POLICY})
        h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    combined = h5[h5.period == "combined_2022_2026"]
    bootstrap, masks, valid = _bootstrap(
        combined, outputs, YEARS, y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "fixing_history_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = _breakdown(
        "fixing_basis", output, YEARS, POLICY,
        y5, forwards[5], dates, currencies,
    )
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EG",
        "years": YEARS,
        "first_available": str(FIRST_AVAILABLE),
        "candidate": "packet-EB arithmetic mean-close CNY fixing basis",
        "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "strict_asof": "10-minute candle end < signal date 15:30",
        "preavailability_rows_are_invalid": True,
        "postavailability_market_closed_rows_are_neutral_zero": True,
        "selector_used": False,
        "next_cbr_rate_used": False,
        "later_period_status": "fixed retrospective lifecycle audit",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nYEAR SUMMARY\n" + summaries.to_string(index=False))
    print("\nH5\n" + h5[[
        "period", "frequency", "lift", "forward_benefit_bps",
        "year_lift_min", "corridor_lift_min", "quarter_frequency_min",
    ]].to_string(index=False))
    print("\nBY HORIZON\n" + by_horizon.to_string(index=False))


if __name__ == "__main__":
    main()
