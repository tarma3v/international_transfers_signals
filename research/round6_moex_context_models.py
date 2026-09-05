"""Packet-AO matched models for independent MOEX risk/liquidity context."""
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
from research.round5_adaptation import RESET, _outputs
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_moex_context_features import (
    MAX_STALENESS_DAYS, TICKERS, build_context_features, causality_check,
    load_context_history,
)
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    Spec, _bootstrap, _breakdown, _evaluate, _model, prequential_scores,
)


OUT = Path("results/research/round6/moex_context")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
COMPONENTS = Path("results/research/round6/cny_explainable/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ORDER = (
    "no_context", "imoex_only", "rgbi_only", "rusfar_only", "gold_only",
    "all_context", "stale20_context", "primary50_all_context50",
)


def coverage(index, history):
    rows = []
    for ticker in TICKERS:
        dates = np.asarray([row["date"] for row in history[ticker]], dtype=object)
        for _currency, _position, day in index:
            end = int(np.searchsorted(dates, day, side="left"))
            source = dates[end - 1] if end else None
            age = (day - source).days if source is not None else np.nan
            rows.append({
                "ticker": ticker, "year": day.year,
                "available": bool(source is not None and age <= MAX_STALENESS_DAYS),
                "age_days": age,
            })
    return (pd.DataFrame(rows).groupby(["ticker", "year"], as_index=False)
            .agg(rows=("available", "size"), available_rate=("available", "mean"),
                 median_age_days=("age_days", "median"), max_age_days=("age_days", "max")))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    cny_history, _cny_digest = load_moex_history()
    moex, moex_names = build_moex_features(index, cny_history)
    history, digest = load_context_history()
    context, context_names = build_context_features(index, history)
    causality_check(index, history)
    broad, broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    base = np.column_stack([X[:, _core_columns(names)], external[:, trusted], broad])
    base_names = ([names[i] for i in _core_columns(names)]
                  + [external_names[i] for i in trusted] + broad_names)
    intraday_columns = np.asarray([
        i for i, name in enumerate(moex_names)
        if "cnyrub_tom" in name and any(token in name for token in (
            "_open_close", "_intraday_range", "_close_wap",
            "_overnight_gap", "_log_trades",
        ))
    ], dtype=int)
    intraday = moex[:, intraday_columns]
    intraday_names = [moex_names[i] for i in intraday_columns]
    groups = {
        "imoex_only": [i for i, name in enumerate(context_names) if name.startswith("context_imoex_")],
        "rgbi_only": [i for i, name in enumerate(context_names) if name.startswith("context_rgbi_")],
        "rusfar_only": [i for i, name in enumerate(context_names) if name.startswith("context_rusfar_")],
        "gold_only": [i for i, name in enumerate(context_names) if name.startswith("context_gldrub_tom_")],
        "all_context": list(range(len(context_names))),
    }
    matrices = {
        candidate: np.column_stack([base, intraday, context[:, columns]])
        for candidate, columns in groups.items()
    }
    matrices["stale20_context"] = np.column_stack([
        base, intraday, delayed_by_currency(context, index, rows=20),
    ])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with COMPONENTS.open("rb") as handle:
        outputs = {"no_context": pickle.load(handle)["cny_intraday_extra"]}
    logs = []
    for candidate in (
        "imoex_only", "rgbi_only", "rusfar_only", "gold_only",
        "all_context", "stale20_context",
    ):
        score, part = prequential_scores(
            Spec(candidate, "extra", candidate), matrices[candidate], y, dates, reach,
        )
        outputs[candidate] = _outputs(score, y, dates)
        logs.extend(part)
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs["primary50_all_context50"] = combine_causal(
        [primary, outputs["all_context"]], (.5, .5), dates, currencies,
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
    availability = coverage(index, history)
    availability.to_csv(OUT / "availability_by_year.csv", index=False)

    # Diagnostic-only train-resolved importance, never used to choose a model.
    start = pd.Timestamp("2026-07-01").date()
    train = ((dates >= RESET) & np.asarray([value < start for value in reach]) & np.isfinite(y))
    learner = _model("extra")
    learner.fit(matrices["all_context"][train], y[train])
    all_names = base_names + intraday_names + context_names
    importance = pd.DataFrame({
        "feature": all_names, "importance": learner.feature_importances_,
        "is_context": [name.startswith("context_") for name in all_names],
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "feature_importance_2026q3_train_only.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("training used unresolved labels")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AO", "variants": ORDER, "fixed_policy": POLICY,
        "instruments": TICKERS, "feature_count": len(context_names),
        "feature_names": context_names, "payload_sha256": digest,
        "asof_rule": "TRADEDATE < signal_date", "same_day_close_allowed": False,
        "physical_future_corruption_check": True,
        "stale_control": "all context delayed by 20 target rows per currency",
        "all_training_labels_resolved": chronology_ok,
        "later_period_status": "post-diagnostic retrospective exploration",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print("\nCOVERAGE\n", availability[availability.year >= 2022].to_string(index=False))
    print("\nCONTEXT IMPORTANCE\n", importance[importance.is_context].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
