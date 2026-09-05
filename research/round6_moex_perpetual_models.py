"""Packet-DH: leakage-safe learners on lagged MOEX perpetual FX futures."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_moex_perpetual_features import (
    build_perpetual_features,
    causality_check as feature_causality_check,
    load_perpetual_history,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/moex_perpetual")
SEED = 20260905
TRAIN_START = dt.date(2022, 5, 1)
STALE_ROWS = 20
BLEND_WEIGHTS = (.10, .25)
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)
KINDS = ("logit", "hist", "extra")


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.025, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=100, l2_regularization=30.0,
            random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=45,
            max_features=.65, n_jobs=1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(kind, matrix, y, dates, reach, verbose=True):
    scores = np.full(len(y), np.nan, dtype=float)
    logs = []
    finite = np.all(np.isfinite(matrix), axis=1)
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = (
            (dates >= TRAIN_START)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & finite
        )
        test = (
            (dates >= start) & (dates < end)
            & np.isfinite(y) & finite
        )
        rows = np.flatnonzero(train)
        target = np.flatnonzero(test)
        if len(rows) < 1000 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved target admitted to futures learner")
        model = _model(kind)
        model.fit(matrix[rows], y[rows])
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": kind,
            "quarter": str(start),
            "n_train": len(rows),
            "n_test": len(target),
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        if verbose:
            print(
                f"  futures {kind:<6} quarter={start} "
                f"train={len(rows):5d} test={len(target):4d}", flush=True,
            )
    return scores, logs


def outcome_causality_check(matrix, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    original, _ = prequential_scores(
        "logit", matrix, y, dates, reach, verbose=False,
    )
    changed = y.copy()
    unresolved = np.asarray([
        np.isfinite(y[row]) and reach[row] > cutoff for row in range(len(y))
    ])
    changed[unresolved] = 1.0 - changed[unresolved]
    altered, _ = prequential_scores(
        "logit", matrix, changed, dates, reach, verbose=False,
    )
    past = (dates <= cutoff) & np.isfinite(original)
    np.testing.assert_array_equal(original[past], altered[past])
    return True


def _load_incumbent():
    with INCUMBENT_PATH.open("rb") as handle:
        return pickle.load(handle)[INCUMBENT]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_perpetual_history()
    futures, futures_names = build_perpetual_features(index, history, references)
    feature_causality_check(index, history, references)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    static = np.column_stack([X[:, currency_columns], X[:, target_columns]])
    aligned = np.column_stack([static, futures])
    stale_futures = delayed_by_currency(futures, index, rows=STALE_ROWS)
    stale = np.column_stack([static, stale_futures])

    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    outcome_causality_check(aligned, y5, dates, reach)
    raw, stale_raw, logs = {}, {}, []
    for kind in KINDS:
        score, part = prequential_scores(kind, aligned, y5, dates, reach)
        stale_score, stale_part = prequential_scores(kind, stale, y5, dates, reach)
        raw[f"futures_{kind}"] = _outputs(score, y5, dates)
        stale_raw[f"futures_{kind}"] = _outputs(stale_score, y5, dates)
        logs.extend(part)
        logs.extend({**row, "candidate": f"{kind}_stale20"} for row in stale_part)

    incumbent = _load_incumbent()
    aligned_outputs = {"incumbent": incumbent, **raw}
    stale_outputs = {"incumbent": incumbent, **stale_raw}
    for kind in KINDS:
        raw_name = f"futures_{kind}"
        for weight in BLEND_WEIGHTS:
            name = f"incumbent{int((1-weight)*100)}_{raw_name}{int(weight*100)}"
            aligned_outputs[name] = combine_causal(
                [incumbent, raw[raw_name]], (1.0 - weight, weight),
                dates, currencies,
            )
            stale_outputs[name] = combine_causal(
                [incumbent, stale_raw[raw_name]], (1.0 - weight, weight),
                dates, currencies,
            )

    screen = horizon_rows(
        aligned_outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {"incumbent": incumbent, "selected": aligned_outputs[selected]}
    if selected != "incumbent":
        comparison["matched_stale20"] = stale_outputs[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)

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
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, POLICY, y5, forwards[5], dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], comparison, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "moex_perpetual_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DH", "fixed_policy": POLICY,
        "source_payload_sha256": digest,
        "futures_feature_count": len(futures_names),
        "futures_feature_names": futures_names,
        "target_features": TARGET_FEATURES,
        "models": KINDS,
        "training_start": TRAIN_START,
        "quarterly_refit": True,
        "all_h5_training_labels_resolved": True,
        "external_stale_control_rows_per_currency": STALE_ROWS,
        "blend_weights": BLEND_WEIGHTS,
        "selection_period": 2024,
        "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "strict_futures_asof": "TRADEDATE < signal_date",
        "future_outcome_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + screen_summary.sort_values(
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
