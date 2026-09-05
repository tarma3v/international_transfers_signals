"""Packet-AR per-currency CNY experts with fixed global shrinkage."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_external_models import _join_external
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import RESET, _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _evaluate, _model,
)


OUT = Path("results/research/round6/cny_local_experts")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
GLOBAL = Path("results/research/round6/cny_explainable/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
MIN_LOCAL_ROWS = 140
ORDER = (
    "local_logit", "global75_local_logit25", "local_extra",
    "global75_local_extra25", "local_logit50_extra50",
    "primary75_local_consensus25",
)


def local_scores(name, kind, matrix, y, dates, currencies, reach):
    scores = np.full(len(y), np.nan)
    logs = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        for currency in np.unique(currencies):
            test = ((dates >= start) & (dates < end) & (currencies == currency)
                    & np.isfinite(y))
            train = ((dates >= RESET) & (currencies == currency)
                     & np.asarray([value < start for value in reach]) & np.isfinite(y))
            rows = np.flatnonzero(train)
            if not test.any() or len(rows) < MIN_LOCAL_ROWS:
                continue
            model = _model(kind)
            model.fit(matrix[rows], y[rows])
            target = np.flatnonzero(test)
            scores[target] = model.predict_proba(matrix[target])[:, 1]
            logs.append({
                "candidate": name, "currency": currency, "quarter": str(start),
                "n_train": len(rows), "first_train": str(min(dates[rows])),
                "last_resolved": str(max(reach[rows])), "n_features": matrix.shape[1],
            })
        print(f"  {name:<27} quarter={start}", flush=True)
    return scores, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    broad, _broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    base = np.column_stack([X[:, _core_columns(names)], external[:, trusted], broad])
    intraday_columns = np.asarray([
        i for i, name in enumerate(moex_names)
        if "cnyrub_tom" in name and any(token in name for token in (
            "_open_close", "_intraday_range", "_close_wap",
            "_overnight_gap", "_log_trades",
        ))
    ], dtype=int)
    transparent = ("pct_range_30", "pct_range_90", "pct_range_180",
                   "ret_1", "ret_5", "ret_20")
    transparent_columns = np.asarray([names.index(name) for name in transparent])
    intraday = moex[:, intraday_columns]
    matrices = {
        "local_logit": np.column_stack([intraday, X[:, transparent_columns]]),
        "local_extra": np.column_stack([base, intraday]),
    }
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
    for candidate, kind in (("local_logit", "logit"), ("local_extra", "extra")):
        score, part = local_scores(
            candidate, kind, matrices[candidate], y, dates, currencies, reach,
        )
        outputs[candidate] = _outputs(score, y, dates)
        logs.extend(part)
    with GLOBAL.open("rb") as handle:
        global_outputs = pickle.load(handle)
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs["global75_local_logit25"] = combine_causal(
        [global_outputs["market_anchor_logit"], outputs["local_logit"]],
        (.75, .25), dates, currencies,
    )
    outputs["global75_local_extra25"] = combine_causal(
        [global_outputs["cny_intraday_extra"], outputs["local_extra"]],
        (.75, .25), dates, currencies,
    )
    outputs["local_logit50_extra50"] = combine_causal(
        [outputs["local_logit"], outputs["local_extra"]],
        (.5, .5), dates, currencies,
    )
    outputs["primary75_local_consensus25"] = combine_causal(
        [primary, outputs["local_logit50_extra50"]],
        (.75, .25), dates, currencies,
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
        "packet": "AR", "variants": ORDER, "fixed_policy": POLICY,
        "minimum_local_rows": MIN_LOCAL_ROWS,
        "global_local_weights": [0.75, 0.25],
        "local_consensus_weights": [0.5, 0.5],
        "asof_rule": "TRADEDATE < signal_date", "payload_sha256": digest,
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
