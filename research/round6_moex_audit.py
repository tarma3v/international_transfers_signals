"""Packet-AF matched MOEX ablations and strict timestamp audit."""
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
from research.round6_moex_features import (
    MAX_STALENESS_DAYS, TICKERS, build_moex_features, causality_check,
    load_moex_history,
)
from research.round6_resolved_models import (
    Spec, _bootstrap, _breakdown, _evaluate, _model, prequential_scores,
)


OUT = Path("results/research/round6/moex_audit")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
ORDER = ("no_moex", "cny_only", "usd_only", "eur_only", "all_moex")
POLICY = {
    "policy_type": "rolling", "rate": .22, "rolling": 20, "cooldown": 0,
    "history": 0, "strong": 0.0, "late": 0.0, "late_weekday": 0,
    "weekly_cap": 0,
}


def availability(index, history):
    rows = []
    for ticker in TICKERS:
        dates = np.asarray([row["date"] for row in history[ticker]], dtype=object)
        for _currency, _position, day in index:
            end = int(np.searchsorted(dates, day, side="left"))
            source = dates[end - 1] if end else None
            if source is not None and not source < day:
                raise AssertionError("MOEX source date is not strictly before signal")
            age = (day - source).days if source is not None else np.nan
            rows.append({
                "ticker": ticker, "signal_year": day.year,
                "source_date": source, "signal_date": day,
                "age_days": age,
                "available": bool(source is not None and age <= MAX_STALENESS_DAYS),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, payload_digest = load_moex_history()
    moex, moex_names = build_moex_features(index, history)
    causality_check(index, history)
    broad, broad_names, _references = load_broad_features(index, series)
    joined, joined_names = _join_external(X, names, index, EXTERNAL)
    external = joined[:, len(names):]
    external_names = joined_names[len(names):]
    trusted_columns = np.asarray([
        i for i, name in enumerate(external_names)
        if not name.startswith("brent_") and not name.startswith("broad_dollar_")
    ], dtype=int)
    base = np.column_stack([
        X[:, _core_columns(names)], external[:, trusted_columns], broad,
    ])
    base_names = (
        [names[i] for i in _core_columns(names)]
        + [external_names[i] for i in trusted_columns] + broad_names
    )
    groups = {
        "cny_only": [i for i, name in enumerate(moex_names) if "cnyrub_tom" in name],
        "usd_only": [i for i, name in enumerate(moex_names) if "usd000utstom" in name],
        "eur_only": [i for i, name in enumerate(moex_names) if "eur_rub__tom" in name],
        "all_moex": list(range(len(moex_names))),
    }
    matrices = {name: np.column_stack([base, moex[:, columns]]) for name, columns in groups.items()}
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    with Path("results/research/round6/broad_cbr/outputs.pkl").open("rb") as handle:
        broad_outputs = pickle.load(handle)
    with Path("results/research/round6/moex/outputs.pkl").open("rb") as handle:
        moex_outputs = pickle.load(handle)
    outputs = {
        "no_moex": broad_outputs["broad_compact_extra"],
        "all_moex": moex_outputs["moex_extra"],
    }
    training_log = []
    for name in ("cny_only", "usd_only", "eur_only"):
        score, logs = prequential_scores(
            Spec(name, "extra", name), matrices[name], y, dates, reach,
        )
        outputs[name] = _outputs(score, y, dates)
        training_log.extend(logs)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(training_log).to_csv(OUT / "training_log.csv", index=False)

    results = []
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
            results.append(item)
    results = pd.DataFrame(results)
    results["predeclared_order"] = results.candidate.map({name: i + 1 for i, name in enumerate(ORDER)})
    results.to_csv(OUT / "matched_results.csv", index=False)

    selected = results[results.period == "screen_2024"].copy()
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        selected, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        selected, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "retrospective_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown_rows = []
    for candidate in ORDER:
        breakdown_rows.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "breakdown_2025_2026.csv", index=False)

    available = availability(index, history)
    available.to_csv(OUT / "asof_rows.csv", index=False)
    coverage = (
        available.groupby(["ticker", "signal_year"], as_index=False)
        .agg(rows=("available", "size"), available_rate=("available", "mean"),
             median_age_days=("age_days", "median"), max_age_days=("age_days", "max"))
    )
    coverage.to_csv(OUT / "availability_by_year.csv", index=False)

    # Train-only impurity importance for the last refit; it never selects the
    # architecture or policy and is explicitly diagnostic.
    start = pd.Timestamp("2026-07-01").date()
    train = (
        (dates >= RESET) & np.asarray([value < start for value in reach])
        & np.isfinite(y)
    )
    learner = _model("extra")
    learner.fit(matrices["all_moex"][train], y[train])
    all_names = base_names + moex_names
    importance = pd.DataFrame({
        "feature": all_names, "importance": learner.feature_importances_,
        "is_moex": [name.startswith("moex_") for name in all_names],
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "feature_importance_2026q3_train_only.csv", index=False)

    chronology_ok = all(
        pd.to_datetime(pd.DataFrame(training_log).last_resolved)
        < pd.to_datetime(pd.DataFrame(training_log).quarter)
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "variants": ORDER, "fixed_policy": POLICY,
        "payload_sha256": payload_digest,
        "asof_rule": "source_trade_date < signal_date",
        "all_timestamp_rows_strict": True,
        "physical_same_date_and_future_corruption_check": True,
        "all_training_labels_resolved": bool(chronology_ok),
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMATCHED\n" + results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print("\nCOVERAGE\n" + coverage[coverage.signal_year >= 2022].to_string(index=False))
    print("\nTOP IMPORTANCE\n" + importance.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
