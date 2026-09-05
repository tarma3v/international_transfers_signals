"""Packet-AH low-dimensional, explainable models for the lagged CNY signal."""
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
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_resolved_models import (
    Spec, _bootstrap, _breakdown, _evaluate, prequential_scores,
)


OUT = Path("results/research/round6/cny_explainable")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
AG_OUTPUTS = Path("results/research/round6/cny_decomposition/outputs.pkl")
ORDER = (
    "market_currency_logit", "market_anchor_logit", "market_anchor_hist",
    "full_market_logit", "cny_intraday_extra",
)


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
    intraday = moex[:, intraday_columns]
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent_names = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent_names])
    market_currency = np.column_stack([intraday, X[:, currency_columns]])
    market_anchor = np.column_stack([
        intraday, X[:, currency_columns], X[:, transparent_columns],
    ])
    matrices = {
        "market_currency_logit": market_currency,
        "market_anchor_logit": market_anchor,
        "market_anchor_hist": market_anchor,
        "full_market_logit": np.column_stack([base, intraday]),
    }
    kinds = {
        "market_currency_logit": "logit", "market_anchor_logit": "logit",
        "market_anchor_hist": "hist", "full_market_logit": "logit",
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
    for candidate in ORDER[:-1]:
        score, rows = prequential_scores(
            Spec(candidate, kinds[candidate], candidate), matrices[candidate],
            y, dates, reach,
        )
        outputs[candidate] = _outputs(score, y, dates)
        logs.extend(rows)
    with AG_OUTPUTS.open("rb") as handle:
        outputs["cny_intraday_extra"] = pickle.load(handle)["cny_intraday"]
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    result_rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[candidate], years, POLICY,
                             y, benefit, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            result_rows.append(item)
    results = pd.DataFrame(result_rows)
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
    breakdown_rows = []
    for candidate in ORDER:
        breakdown_rows.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "breakdown_2025_2026.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("training used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AH", "variants": ORDER, "fixed_policy": POLICY,
        "market_feature_names": [moex_names[i] for i in intraday_columns],
        "transparent_target_feature_names": list(transparent_names),
        "payload_sha256": digest, "asof_rule": "TRADEDATE < signal_date",
        "same_day_close_allowed": False, "all_training_labels_resolved": chronology_ok,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
