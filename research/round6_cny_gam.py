"""Packet-AZ regularized additive spline models for lagged CNY state."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_gam")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
N_MARKET = 8
N_NUMERIC = 14
N_CURRENCY = 5
N_KNOTS = 5
MIN_GLOBAL_ROWS = 700
MIN_LOCAL_ROWS = 140
ORDER = (
    "market_spline_gam",
    "all_spline_gam",
    "local_all_spline_gam",
    "global75_local_gam25",
    "primary75_all_gam25",
    "primary75_local_gam25",
)


def raw_matrix(X, names, moex, moex_names):
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
    transparent_names = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent_names])
    matrix = np.column_stack([
        moex[:, intraday_columns], X[:, transparent_columns], X[:, currency_columns],
    ])
    feature_names = (
        [moex_names[i] for i in intraday_columns]
        + list(transparent_names)
        + [names[i] for i in currency_columns]
    )
    if matrix.shape[1] != N_NUMERIC + N_CURRENCY or len(intraday_columns) != N_MARKET:
        raise AssertionError(
            f"GAM schema changed: market={len(intraday_columns)}, total={matrix.shape[1]}"
        )
    return matrix, feature_names


def gam_model(kind):
    spline = SplineTransformer(
        n_knots=N_KNOTS,
        degree=2,
        knots="quantile",
        include_bias=False,
        extrapolation="linear",
    )
    if kind == "market":
        transformer = ColumnTransformer([
            ("market_spline", spline, list(range(N_MARKET))),
            ("linear", "passthrough", list(range(N_MARKET, N_NUMERIC + N_CURRENCY))),
        ])
    elif kind == "all":
        transformer = ColumnTransformer([
            ("numeric_spline", spline, list(range(N_NUMERIC))),
            ("currency", "passthrough", list(range(N_NUMERIC, N_NUMERIC + N_CURRENCY))),
        ])
    else:
        raise KeyError(kind)
    return make_pipeline(
        transformer,
        StandardScaler(),
        LogisticRegression(C=.025, max_iter=4000, random_state=20260905),
    )


def global_scores(name, kind, matrix, y, dates, reach):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & np.isfinite(y)
        train = (
            (dates >= RESET)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        rows = np.flatnonzero(train)
        if not test.any() or len(rows) < MIN_GLOBAL_ROWS:
            continue
        model = gam_model(kind)
        model.fit(matrix[rows], y[rows])
        target = np.flatnonzero(test)
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": name,
            "currency": "ALL",
            "quarter": str(start),
            "n_train": len(rows),
            "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])),
            "n_raw_features": matrix.shape[1],
        })
        print(f"  {name:<27} quarter={start} train={len(rows):5d}", flush=True)
    return scores, logs


def local_scores(name, matrix, y, dates, currencies, reach):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        for currency in np.unique(currencies):
            test = (
                (dates >= start) & (dates < end)
                & (currencies == currency) & np.isfinite(y)
            )
            train = (
                (dates >= RESET) & (currencies == currency)
                & np.asarray([value < start for value in reach])
                & np.isfinite(y)
            )
            rows = np.flatnonzero(train)
            if not test.any() or len(rows) < MIN_LOCAL_ROWS:
                continue
            model = gam_model("all")
            model.fit(matrix[rows], y[rows])
            target = np.flatnonzero(test)
            scores[target] = model.predict_proba(matrix[target])[:, 1]
            logs.append({
                "candidate": name,
                "currency": currency,
                "quarter": str(start),
                "n_train": len(rows),
                "first_train": str(min(dates[rows])),
                "last_resolved": str(max(reach[rows])),
                "n_raw_features": matrix.shape[1],
            })
        print(f"  {name:<27} quarter={start}", flush=True)
    return scores, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    matrix, feature_names = raw_matrix(X, names, moex, moex_names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    outputs, logs = {}, []
    for name, kind in ((ORDER[0], "market"), (ORDER[1], "all")):
        score, part = global_scores(name, kind, matrix, y, dates, reach)
        outputs[name] = _outputs(score, y, dates)
        logs.extend(part)
    score, part = local_scores(ORDER[2], matrix, y, dates, currencies, reach)
    outputs[ORDER[2]] = _outputs(score, y, dates)
    logs.extend(part)
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs[ORDER[3]] = combine_causal(
        [outputs[ORDER[1]], outputs[ORDER[2]]], (.75, .25), dates, currencies,
    )
    outputs[ORDER[4]] = combine_causal(
        [primary, outputs[ORDER[1]]], (.75, .25), dates, currencies,
    )
    outputs[ORDER[5]] = combine_causal(
        [primary, outputs[ORDER[2]]], (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

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
        y, dates, currencies, valid, masks, "gam_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("GAM training used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AZ",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "raw_feature_names": feature_names,
        "spline": {
            "n_knots": N_KNOTS,
            "degree": 2,
            "knots": "quantile",
            "include_bias": False,
            "extrapolation": "linear",
        },
        "logistic_penalty": {"type": "L2", "C": 0.025},
        "minimum_global_rows": MIN_GLOBAL_ROWS,
        "minimum_local_rows": MIN_LOCAL_ROWS,
        "blend_weights": [0.75, 0.25],
        "payload_sha256": digest,
        "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
