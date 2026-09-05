"""Packet-DN: quarterly models on the fixed noon-Moscow futures state."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_moex_perpetual_features import (
    build_perpetual_features,
    load_perpetual_history,
)
from research.round6_moex_perpetual_hourly_features import (
    build_hourly_features,
    causality_check as hourly_feature_causality_check,
    load_hourly_history,
)
from research.round6_moex_perpetual_models import (
    outcome_causality_check,
    prequential_scores,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/moex_perpetual_hourly_models")
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)
VIEWS = ("noon", "full")
KINDS = ("logit", "hist", "extra")
BLEND_WEIGHTS = (.10, .25)
STALE_ROWS = 20


def _load_incumbent():
    with INCUMBENT_PATH.open("rb") as handle:
        return pickle.load(handle)[INCUMBENT]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    daily_history, daily_digest = load_perpetual_history()
    daily, daily_names = build_perpetual_features(index, daily_history, references)
    hourly_history, hourly_digest = load_hourly_history()
    hourly, hourly_names = build_hourly_features(index, hourly_history, references)
    hourly_feature_causality_check(index, hourly_history, references)

    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    static = np.column_stack([X[:, currency_columns], X[:, target_columns]])
    stale_hourly = delayed_by_currency(hourly, index, rows=STALE_ROWS)
    matrices = {
        "noon": np.column_stack([static, hourly]),
        "full": np.column_stack([static, daily, hourly]),
    }
    stale_matrices = {
        "noon": np.column_stack([static, stale_hourly]),
        "full": np.column_stack([static, daily, stale_hourly]),
    }

    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    outcome_causality_check(matrices["full"], y5, dates, reach)

    aligned_raw, stale_raw, logs = {}, {}, []
    for view in VIEWS:
        for kind in KINDS:
            name = f"{view}_{kind}"
            score, part = prequential_scores(
                kind, matrices[view], y5, dates, reach,
            )
            stale_score, stale_part = prequential_scores(
                kind, stale_matrices[view], y5, dates, reach,
            )
            aligned_raw[name] = _outputs(score, y5, dates)
            stale_raw[name] = _outputs(stale_score, y5, dates)
            logs.extend({**row, "view": view, "stale": False} for row in part)
            logs.extend({**row, "view": view, "stale": True} for row in stale_part)

    incumbent = _load_incumbent()
    aligned_outputs = {"incumbent": incumbent, **aligned_raw}
    stale_outputs = {"incumbent": incumbent, **stale_raw}
    for name in aligned_raw:
        for weight in BLEND_WEIGHTS:
            candidate = f"incumbent{int((1-weight)*100)}_{name}{int(weight*100)}"
            aligned_outputs[candidate] = combine_causal(
                [incumbent, aligned_raw[name]], (1.0 - weight, weight),
                dates, currencies,
            )
            stale_outputs[candidate] = combine_causal(
                [incumbent, stale_raw[name]], (1.0 - weight, weight),
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
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
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
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
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
        y5, dates, currencies, valid, masks, "noon_moex_perpetual_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DN", "fixed_policy": POLICY,
        "incumbent": INCUMBENT,
        "hourly_source_payload_sha256": hourly_digest,
        "daily_source_payload_sha256": daily_digest,
        "decision_time": "12:00:00 Europe/Moscow",
        "strict_hourly_asof": "candle end < signal_date 12:00:00",
        "hourly_feature_count": len(hourly_names),
        "daily_feature_count": len(daily_names),
        "target_features": TARGET_FEATURES,
        "views": VIEWS, "models": KINDS,
        "external_stale_control_rows_per_currency": STALE_ROWS,
        "blend_weights": BLEND_WEIGHTS,
        "training_start": "2022-05-01", "quarterly_refit": True,
        "all_h5_training_labels_resolved": True,
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "physical_hourly_corruption_check": True,
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
