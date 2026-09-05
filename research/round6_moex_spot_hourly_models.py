"""Packet-DT: quarterly learners on the fixed pre-noon MOEX spot state."""
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
from research.round6_moex_perpetual_hourly_features import (
    build_hourly_features,
    load_hourly_history,
)
from research.round6_moex_perpetual_models import (
    outcome_causality_check,
    prequential_scores,
)
from research.round6_moex_spot_hourly_features import (
    SPOT_ONLY_FEATURES,
    build_spot_features,
    causality_check as spot_feature_causality_check,
    load_spot_history,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/moex_spot_hourly_models")
NOON_CONSENSUS_PATH = Path(
    "results/research/round6/three_view_futures_consensus/outputs.pkl"
)
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)
VIEWS = ("spot", "spot_perpetual")
KINDS = ("logit", "hist", "extra")
BLEND_WEIGHTS = (.25, .50)
STALE_ROWS = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    perpetual_history, perpetual_digest = load_hourly_history()
    perpetual, perpetual_names = build_hourly_features(
        index, perpetual_history, references,
    )
    spot_history, spot_digest = load_spot_history()
    spot, spot_names = build_spot_features(
        index, spot_history, references, perpetual_history,
    )
    spot_feature_causality_check(
        index, spot_history, references, perpetual_history,
    )

    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    static = np.column_stack((X[:, currency_columns], X[:, target_columns]))
    spot_base = spot[:, :SPOT_ONLY_FEATURES]
    spot_derived = spot[:, SPOT_ONLY_FEATURES:]
    delayed_spot = delayed_by_currency(spot, index, rows=STALE_ROWS)
    delayed_base = delayed_spot[:, :SPOT_ONLY_FEATURES]
    delayed_derived = delayed_spot[:, SPOT_ONLY_FEATURES:]
    matrices = {
        "spot": np.column_stack((static, spot_base)),
        "spot_perpetual": np.column_stack((
            static, spot_base, perpetual, spot_derived,
        )),
    }
    stale_matrices = {
        "spot": np.column_stack((static, delayed_base)),
        "spot_perpetual": np.column_stack((
            static, delayed_base, perpetual, delayed_derived,
        )),
    }

    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    outcome_causality_check(matrices["spot_perpetual"], y5, dates, reach)

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

    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    noon_consensus = _load(NOON_CONSENSUS_PATH, "selected")
    candidates = {
        "incumbent": incumbent,
        "noon_consensus": noon_consensus,
        **aligned_raw,
    }
    matched_stale = dict(stale_raw)
    for name, raw in aligned_raw.items():
        for base_name, base in (
            ("incumbent", incumbent), ("noon_consensus", noon_consensus),
        ):
            for weight in BLEND_WEIGHTS:
                candidate = (
                    f"{base_name}{int((1-weight)*100)}_"
                    f"{name}{int(weight*100)}"
                )
                candidates[candidate] = combine_causal(
                    (base, raw), (1.0 - weight, weight), dates, currencies,
                )
                matched_stale[candidate] = combine_causal(
                    (base, stale_raw[name]),
                    (1.0 - weight, weight), dates, currencies,
                )

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "incumbent": incumbent,
        "noon_consensus": noon_consensus,
        "selected": candidates[selected],
    }
    if selected in matched_stale:
        comparison["matched_stale20"] = matched_stale[selected]
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
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)

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
        y5, dates, currencies, valid, masks, "noon_moex_spot_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DT", "fixed_policy": POLICY,
        "spot_payload_sha256": spot_digest,
        "perpetual_payload_sha256": perpetual_digest,
        "decision_time": "12:00:00 Europe/Moscow",
        "strict_spot_asof": "candle end < signal_date 12:00:00",
        "spot_feature_count": len(spot_names),
        "spot_only_feature_count": SPOT_ONLY_FEATURES,
        "perpetual_feature_count": len(perpetual_names),
        "target_features": TARGET_FEATURES,
        "views": VIEWS, "models": KINDS,
        "new_spot_stale_control_rows": STALE_ROWS,
        "blend_bases": ("incumbent", "packet-DO noon_consensus"),
        "blend_weights": BLEND_WEIGHTS,
        "training_start": "2022-05-01", "quarterly_refit": True,
        "all_h5_training_labels_resolved": True,
        "selection_period": 2024, "selected": selected,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "physical_spot_corruption_check": True,
        "future_outcome_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN TOP\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
