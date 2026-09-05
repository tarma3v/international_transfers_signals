"""Packet-AM long-history, post-shock weighting, and reset consensus audit."""
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
from research.round5_adaptation import RESET, _outputs, _quarter_starts, _next_quarter
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_pre2022 import CNY_START
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _evaluate, _model,
)


OUT = Path("results/research/round6/cny_history_weighting")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
RESET_OUTPUTS = Path("results/research/round6/cny_consensus/outputs.pkl")
ORDER = ("all_history", "post2022_weight3", "hard_reset", "history50_reset50")


def weighted_scores(name, kind, matrix, y, dates, reach, post_weight):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & np.isfinite(y)
        train = (
            (dates >= CNY_START)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        rows = np.flatnonzero(train)
        if not test.any() or len(rows) < 700:
            continue
        weights = np.where(dates[rows] >= RESET, post_weight, 1.0)
        model = _model(kind)
        if hasattr(model, "named_steps"):
            model.fit(matrix[rows], y[rows], logisticregression__sample_weight=weights)
        else:
            model.fit(matrix[rows], y[rows], sample_weight=weights)
        target = np.flatnonzero(test)
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": name, "quarter": str(start), "n_train": len(rows),
            "post_weight": post_weight,
            "post_rows": int(np.sum(dates[rows] >= RESET)),
            "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])), "n_features": matrix.shape[1],
        })
        print(f"  {name:<27} quarter={start} train={len(rows):5d} "
              f"post={np.sum(dates[rows] >= RESET):4d}", flush=True)
    return scores, logs


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

    components, logs = {}, []
    for label, weight in (("all", 1.0), ("post3", 3.0)):
        logit_score, part = weighted_scores(
            f"{label}_logit", "logit", logit_matrix, y, dates, reach, weight,
        )
        logs.extend(part)
        extra_score, part = weighted_scores(
            f"{label}_extra", "extra", extra_matrix, y, dates, reach, weight,
        )
        logs.extend(part)
        components[label] = (_outputs(logit_score, y, dates),
                             _outputs(extra_score, y, dates))
    outputs = {
        "all_history": combine_causal(
            components["all"], (.5, .5), dates, currencies,
        ),
        "post2022_weight3": combine_causal(
            components["post3"], (.5, .5), dates, currencies,
        ),
    }
    with RESET_OUTPUTS.open("rb") as handle:
        outputs["hard_reset"] = pickle.load(handle)["logit50_extra50"]
    outputs["history50_reset50"] = combine_causal(
        [outputs["all_history"], outputs["hard_reset"]],
        (.5, .5), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)), ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[candidate], years, POLICY,
                             y, benefit, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    selected = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        selected, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "retrospective_2025_2026",
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
        raise AssertionError("training used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AM", "variants": ORDER, "training_start": str(CNY_START),
        "post2022_weight": 3.0, "fixed_consensus_weights": [0.5, 0.5],
        "fixed_history_reset_weights": [0.5, 0.5], "fixed_policy": POLICY,
        "payload_sha256": digest, "asof_rule": "TRADEDATE < signal_date",
        "all_training_labels_resolved": chronology_ok,
        "later_period_status": "post-diagnostic retrospective exploration",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
