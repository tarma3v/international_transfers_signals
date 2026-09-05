"""Packet-AK expanding pre-2022 transport test for the lagged CNY signal."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_external_models import _join_external
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _evaluate, _model,
)


OUT = Path("results/research/round6/cny_pre2022")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
CNY_START = dt.date(2013, 4, 15)
EVALUATION_YEARS = tuple(range(2017, 2022))


def quarter_starts():
    return [
        dt.date(year, month, 1)
        for year in range(2016, 2022) for month in (1, 4, 7, 10)
    ]


def next_quarter(day):
    return dt.date(day.year + (day.month == 10), 1 if day.month == 10 else day.month + 3, 1)


def expanding_scores(name, kind, matrix, y, dates, reach):
    score = np.full(len(y), np.nan)
    logs = []
    for start in quarter_starts():
        end = next_quarter(start)
        test = (dates >= start) & (dates < end) & np.isfinite(y)
        train = (
            (dates >= CNY_START)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        rows = np.flatnonzero(train)
        if not test.any() or len(rows) < 700:
            continue
        model = _model(kind)
        model.fit(matrix[rows], y[rows])
        target = np.flatnonzero(test)
        score[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": name, "quarter": str(start), "n_train": len(rows),
            "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])), "n_features": matrix.shape[1],
        })
        print(f"  {name:<27} quarter={start} train={len(rows):5d} "
              f"features={matrix.shape[1]:3d}", flush=True)
    return score, logs


def yearly_outputs(score, y, dates):
    result = {}
    for year in EVALUATION_YEARS:
        calibration = (
            (dates >= dt.date(year - 1, 1, 1))
            & (dates < dt.date(year, 1, 1))
            & np.isfinite(score) & np.isfinite(y)
        )
        test = (
            np.asarray([day.year == year for day in dates])
            & np.isfinite(score) & np.isfinite(y)
        )
        ca, te = np.flatnonzero(calibration), np.flatnonzero(test)
        result[year] = {
            "calib_idx": ca, "test_idx": te,
            "calib_score": score[ca], "test_score": score[te],
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    broad, _broad_names, _references = load_broad_features(index, series)
    base = np.column_stack([X[:, _core_columns(names)], external[:, trusted], broad])
    intraday_columns = np.asarray([
        i for i, name in enumerate(moex_names)
        if "cnyrub_tom" in name and any(token in name for token in (
            "_open_close", "_intraday_range", "_close_wap",
            "_overnight_gap", "_log_trades",
        ))
    ], dtype=int)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent = ("pct_range_30", "pct_range_90", "pct_range_180",
                   "ret_1", "ret_5", "ret_20")
    transparent_columns = np.asarray([names.index(name) for name in transparent])
    intraday = moex[:, intraday_columns]
    logit_matrix = np.column_stack([
        intraday, X[:, currency_columns], X[:, transparent_columns],
    ])
    extra_matrix = np.column_stack([base, intraday])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    logit_score, logit_logs = expanding_scores(
        "pre2022_market_anchor_logit", "logit", logit_matrix, y, dates, reach,
    )
    extra_score, extra_logs = expanding_scores(
        "pre2022_cny_intraday_extra", "extra", extra_matrix, y, dates, reach,
    )
    logit = yearly_outputs(logit_score, y, dates)
    extra = yearly_outputs(extra_score, y, dates)
    outputs = {
        "market_anchor_logit": logit,
        "cny_intraday_extra": extra,
        "logit50_extra50": combine_causal(
            [logit, extra], (.50, .50), dates, currencies,
        ),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logit_logs + extra_logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in outputs:
        for year in EVALUATION_YEARS:
            item = _evaluate(outputs[candidate], (year,), POLICY,
                             y, benefit, dates, currencies)
            item.update({"candidate": candidate, "period": str(year), **POLICY})
            rows.append(item)
        item = _evaluate(outputs[candidate], EVALUATION_YEARS, POLICY,
                         y, benefit, dates, currencies)
        item.update({"candidate": candidate, "period": "2017_2021", **POLICY})
        rows.append(item)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "matched_results.csv", index=False)
    selected = results[results.period == "2017_2021"].copy()
    bootstrap, masks, valid = _bootstrap(
        selected, outputs, EVALUATION_YEARS, y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2017_2021.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "pre2022_2017_2021",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in outputs:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], EVALUATION_YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2017_2021.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("training used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AK", "evaluation_years": EVALUATION_YEARS,
        "training_start": str(CNY_START), "refit": "calendar quarter expanding",
        "fixed_policy": POLICY, "fixed_weights": [0.5, 0.5],
        "payload_sha256": digest, "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
        "later_period_status": "protocol-controlled pre-2022 transport audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min",
    ]].sort_values(["candidate", "period"]).to_string(index=False))


if __name__ == "__main__":
    main()
