"""Packet-DZ: signed label-free partial-fixing nowcast at 15:30 Moscow."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_moex_spot_1530_features import (
    build_spot_1530_features,
    causality_check,
    load_spot_1530_history,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/spot_1530_nowcast")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
BLEND_WEIGHTS = (.10, .25, .40)
STALE_ROWS = 20
RAW_NAMES = (
    "cny_last_basis", "cny_mean_basis", "usd_last_basis", "usd_mean_basis",
    "last_basis_arithmetic", "mean_basis_arithmetic",
    "last_basis_maximum", "last_basis_minimum",
)


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def raw_scores(matrix, names):
    cny_last = matrix[:, names.index("moex_1530_cnyrub_tom_last_cbr_basis")].astype(float)
    cny_mean = matrix[:, names.index("moex_1530_cnyrub_tom_mean_cbr_basis")].astype(float)
    usd_last = matrix[:, names.index("moex_1530_usd000utstom_last_cbr_basis")].astype(float)
    usd_mean = matrix[:, names.index("moex_1530_usd000utstom_mean_cbr_basis")].astype(float)
    last = np.column_stack((cny_last, usd_last))
    mean = np.column_stack((cny_mean, usd_mean))
    return {
        "cny_last_basis": cny_last,
        "cny_mean_basis": cny_mean,
        "usd_last_basis": usd_last,
        "usd_mean_basis": usd_mean,
        "last_basis_arithmetic": np.mean(last, axis=1),
        "mean_basis_arithmetic": np.mean(mean, axis=1),
        "last_basis_maximum": np.max(last, axis=1),
        "last_basis_minimum": np.min(last, axis=1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    matrix, names = build_spot_1530_features(index, history, references)
    causality_check(index, history, references)
    raw = raw_scores(matrix, names)
    if tuple(raw) != RAW_NAMES:
        raise AssertionError("15:30 score order changed")

    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    aligned_raw = {name: _outputs(score, y5, dates) for name, score in raw.items()}
    stale_raw = {
        name: _outputs(
            delayed_by_currency(score[:, None], index, rows=STALE_ROWS)[:, 0],
            y5, dates,
        ) for name, score in raw.items()
    }
    noon = _load(NOON_PATH, "selected")
    candidates = {"noon_consensus": noon, **aligned_raw}
    matched_stale = dict(stale_raw)
    for name, output in aligned_raw.items():
        for weight in BLEND_WEIGHTS:
            candidate = f"noon{int((1-weight)*100)}_{name}{int(weight*100)}"
            candidates[candidate] = combine_causal(
                (noon, output), (1.0 - weight, weight), dates, currencies,
            )
            matched_stale[candidate] = combine_causal(
                (noon, stale_raw[name]),
                (1.0 - weight, weight), dates, currencies,
            )

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {"noon_consensus": noon, "selected": candidates[selected]}
    if selected in matched_stale:
        comparison["matched_stale20"] = matched_stale[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

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
            ("screen_2024", (2024,)), ("retrospective_2025", (2025,)),
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
        y5, dates, currencies, valid, masks, "spot_1530_nowcast_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DZ", "fixed_policy": POLICY,
        "payload_sha256": digest, "decision_time": "15:30:00 Europe/Moscow",
        "strict_asof": "10-minute candle end < signal date 15:30",
        "scores": RAW_NAMES, "sign": "positive spot-current-CBR basis is favourable",
        "sign_fitted": False, "blend_weights": BLEND_WEIGHTS,
        "stale_rows": STALE_ROWS, "selection_period": 2024,
        "selected": selected, "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
