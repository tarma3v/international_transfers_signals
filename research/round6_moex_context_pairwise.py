"""Packet-AP per-instrument stale controls and low-dose primary blends."""
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
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_moex_context_features import (
    TICKERS, build_context_features, load_context_history,
)
from research.round6_moex_features import build_moex_features, load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    Spec, _bootstrap, _breakdown, _evaluate, prequential_scores,
)


OUT = Path("results/research/round6/moex_context_pairwise")
EXTERNAL = Path("results/research/round2/external_features_b5_d10.csv")
AO_OUTPUTS = Path("results/research/round6/moex_context/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
LABELS = {"IMOEX": "imoex", "RGBI": "rgbi", "RUSFAR": "rusfar", "GLDRUB_TOM": "gold"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    cny_history, _digest = load_moex_history()
    moex, moex_names = build_moex_features(index, cny_history)
    history, context_digest = load_context_history()
    context, context_names = build_context_features(index, history)
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
    intraday = moex[:, intraday_columns]
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with AO_OUTPUTS.open("rb") as handle:
        aligned = pickle.load(handle)
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs, logs, order = {}, [], []
    for ticker in TICKERS:
        label = LABELS[ticker]
        columns = np.asarray([
            i for i, name in enumerate(context_names)
            if name.startswith(f"context_{ticker.lower()}_")
        ], dtype=int)
        aligned_name = f"{label}_aligned"
        stale_name = f"{label}_stale20"
        blend_name = f"primary75_{label}25"
        outputs[aligned_name] = aligned[f"{label}_only"]
        stale = delayed_by_currency(context[:, columns], index, rows=20)
        matrix = np.column_stack([base, intraday, stale])
        score, part = prequential_scores(
            Spec(stale_name, "extra", stale_name), matrix, y, dates, reach,
        )
        outputs[stale_name] = _outputs(score, y, dates)
        logs.extend(part)
        outputs[blend_name] = combine_causal(
            [primary, outputs[aligned_name]], (.75, .25), dates, currencies,
        )
        order.extend([aligned_name, stale_name, blend_name])
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in order:
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
        {name: i + 1 for i, name in enumerate(order)}
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
    for candidate in order:
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
        "packet": "AP", "order": order, "fixed_policy": POLICY,
        "aligned_weight_in_blend": 0.25,
        "stale_control": "individual 22-feature group delayed 20 target rows per currency",
        "payload_sha256": context_digest, "asof_rule": "TRADEDATE < signal_date",
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
