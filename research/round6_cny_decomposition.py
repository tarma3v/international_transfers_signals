"""Packet-AG decomposition and stale negative control for the MOEX CNY signal."""
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
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_resolved_models import (
    Spec, _bootstrap, _breakdown, _evaluate, prequential_scores,
)


OUT = Path("results/research/round6/cny_decomposition")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
AF_OUTPUTS = Path("results/research/round6/moex_audit/outputs.pkl")
ORDER = ("no_moex", "cny_all", "cny_trend", "cny_intraday", "cny_stale20")
POLICY = {
    "policy_type": "rolling", "rate": .22, "rolling": 20, "cooldown": 0,
    "history": 0, "strong": 0.0, "late": 0.0, "late_weekday": 0,
    "weekly_cap": 0,
}


def delayed_by_currency(matrix, index, rows=20):
    """Delay target-row features within currency without backward filling."""
    delayed = np.zeros_like(matrix)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    dates = np.asarray([row[2] for row in index], dtype=object)
    for currency in np.unique(currencies):
        positions = np.flatnonzero(currencies == currency)
        positions = positions[np.argsort(dates[positions])]
        if len(positions) > rows:
            delayed[positions[rows:]] = matrix[positions[:-rows]]
    return delayed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    broad, broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    base = np.column_stack([
        X[:, _core_columns(names)], external[:, trusted], broad,
    ])
    cny_columns = np.asarray([
        i for i, name in enumerate(moex_names) if "cnyrub_tom" in name
    ], dtype=int)
    cny_names = [moex_names[i] for i in cny_columns]
    cny = moex[:, cny_columns]
    trend_columns = np.asarray([
        i for i, name in enumerate(cny_names)
        if "_ret_" in name or "_vol_" in name
        or name.endswith("_age_days") or name.endswith("_missing")
    ], dtype=int)
    intraday_columns = np.asarray([
        i for i, name in enumerate(cny_names)
        if any(token in name for token in (
            "_open_close", "_intraday_range", "_close_wap",
            "_overnight_gap", "_log_trades",
        ))
    ], dtype=int)
    stale = delayed_by_currency(cny, index, rows=20)
    matrices = {
        "cny_trend": np.column_stack([base, cny[:, trend_columns]]),
        "cny_intraday": np.column_stack([base, cny[:, intraday_columns]]),
        "cny_stale20": np.column_stack([base, stale]),
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

    # These two controls are byte-for-byte the already-frozen AF outputs from
    # the identical learner/matrix, avoiding a redundant deterministic refit.
    with AF_OUTPUTS.open("rb") as handle:
        af_outputs = pickle.load(handle)
    outputs = {"no_moex": af_outputs["no_moex"], "cny_all": af_outputs["cny_only"]}
    training_log = []
    for candidate in ("cny_trend", "cny_intraday", "cny_stale20"):
        score, logs = prequential_scores(
            Spec(candidate, "extra", candidate), matrices[candidate], y, dates, reach,
        )
        outputs[candidate] = _outputs(score, y, dates)
        training_log.extend(logs)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
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
        "packet": "AG", "variants": ORDER, "fixed_policy": POLICY,
        "cny_feature_count": len(cny_names), "trend_feature_names": [
            cny_names[i] for i in trend_columns
        ], "intraday_feature_names": [cny_names[i] for i in intraday_columns],
        "stale_control": "all CNY features delayed by 20 target rows per currency",
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
