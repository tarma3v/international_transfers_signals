"""Packet-AL causal bridge across the 2022 market-regime shock."""
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
from research.round5_adaptation import RESET
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_pre2022 import CNY_START, next_quarter
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _evaluate, _model,
)


OUT = Path("results/research/round6/cny_shock_bridge")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
YEARS = (2022, 2023)
STARTS = tuple(
    dt.date(year, month, 1)
    for year in (2021, 2022, 2023) for month in (1, 4, 7, 10)
)


def scores(name, kind, matrix, y, dates, reach, lower):
    result = np.full(len(y), np.nan)
    logs = []
    for start in STARTS:
        end = next_quarter(start)
        test = (dates >= start) & (dates < end) & np.isfinite(y)
        train = (
            (dates >= lower)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        rows = np.flatnonzero(train)
        if not test.any() or len(rows) < 700:
            continue
        model = _model(kind)
        model.fit(matrix[rows], y[rows])
        target = np.flatnonzero(test)
        result[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": name, "quarter": str(start), "n_train": len(rows),
            "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])), "n_features": matrix.shape[1],
        })
        print(f"  {name:<27} quarter={start} train={len(rows):5d}", flush=True)
    return result, logs


def outputs(score, y, dates):
    result = {}
    for year in YEARS:
        calibration = (
            np.asarray([day.year == year - 1 for day in dates])
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

    expanding_logit, log1 = scores(
        "expanding_logit", "logit", logit_matrix, y, dates, reach, CNY_START,
    )
    expanding_extra, log2 = scores(
        "expanding_extra", "extra", extra_matrix, y, dates, reach, CNY_START,
    )
    reset_logit, log3 = scores(
        "reset_logit", "logit", logit_matrix, y, dates, reach, RESET,
    )
    reset_extra, log4 = scores(
        "reset_extra", "extra", extra_matrix, y, dates, reach, RESET,
    )
    reset_ready = np.isfinite(reset_logit) & np.isfinite(reset_extra)
    hybrid_logit = np.where(reset_ready, reset_logit, expanding_logit)
    hybrid_extra = np.where(reset_ready, reset_extra, expanding_extra)

    component_outputs = {
        "expanding_logit": outputs(expanding_logit, y, dates),
        "expanding_extra": outputs(expanding_extra, y, dates),
        "hybrid_logit": outputs(hybrid_logit, y, dates),
        "hybrid_extra": outputs(hybrid_extra, y, dates),
    }
    model_outputs = {
        "expanding_consensus": combine_causal(
            [component_outputs["expanding_logit"], component_outputs["expanding_extra"]],
            (.5, .5), dates, currencies,
        ),
        "mechanical_reset_hybrid": combine_causal(
            [component_outputs["hybrid_logit"], component_outputs["hybrid_extra"]],
            (.5, .5), dates, currencies,
        ),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(model_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(log1 + log2 + log3 + log4)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in model_outputs:
        for period, years in (("2022", (2022,)), ("2023", (2023,)),
                              ("2022_2023", YEARS)):
            item = _evaluate(model_outputs[candidate], years, POLICY,
                             y, benefit, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            rows.append(item)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "matched_results.csv", index=False)
    selected = results[results.period == "2022_2023"].copy()
    bootstrap, masks, valid = _bootstrap(
        selected, model_outputs, YEARS, y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2022_2023.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "shock_bridge_2022_2023",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in model_outputs:
        breakdown.extend(_breakdown(
            candidate, model_outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2022_2023.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("training used unresolved labels")
    first_reset = min(
        dates[np.isfinite(reset_logit) & np.isfinite(reset_extra)], default=None,
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AL", "years": YEARS, "reset_date": str(RESET),
        "minimum_resolved_reset_rows": 700, "first_reset_score": str(first_reset),
        "hybrid_rule": "expanding until both reset components exist, reset thereafter",
        "fixed_policy": POLICY, "fixed_weights": [0.5, 0.5],
        "payload_sha256": digest, "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min",
    ]].sort_values(["period", "candidate"]).to_string(index=False))
    print("first reset score:", first_reset)


if __name__ == "__main__":
    main()
