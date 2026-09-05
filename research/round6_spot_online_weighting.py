"""Packet-DV: frozen delayed online weighting of noon and signed-spot experts."""
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
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_perpetual_online_weighting import (
    OnlineSpec,
    online_scores,
    outcome_causality_check,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/spot_online_weighting")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
SPOT_PATH = Path("results/research/round6/spot_signed_nowcast/outputs.pkl")
SPEC = OnlineSpec("global", 250, 5.0)
RANK_WINDOW = 250
RANK_MINIMUM = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reaches = np.column_stack([
        target_reach_dates(index, series, h) for h in HORIZONS
    ])
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]

    noon_output = _load(NOON_PATH, "selected")
    spot_output = _load(SPOT_PATH, "selected")
    stale_spot_output = _load(SPOT_PATH, "matched_stale20")
    noon_rank = causal_percentiles(
        row_scores(noon_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    spot_rank = causal_percentiles(
        row_scores(spot_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    stale_spot_rank = causal_percentiles(
        row_scores(stale_spot_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    outcome_causality_check(
        SPEC, noon_rank, spot_rank, labels, reaches, dates, currencies,
    )
    online, noon_weight = online_scores(
        SPEC, noon_rank, spot_rank, labels, reaches, dates, currencies,
    )
    stale_online, stale_noon_weight = online_scores(
        SPEC, noon_rank, stale_spot_rank, labels, reaches, dates, currencies,
    )
    outputs = {
        "noon_consensus": noon_output,
        "signed_spot": spot_output,
        "static_equal": _outputs(.5 * noon_rank + .5 * spot_rank, y5, dates),
        SPEC.name: _outputs(online, y5, dates),
        "matched_stale20": _outputs(stale_online, y5, dates),
        "static_equal_stale20": _outputs(
            .5 * noon_rank + .5 * stale_spot_rank, y5, dates,
        ),
    }
    periods = (
        ("screen_2024", (2024,)),
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    )
    detail = []
    for period, years in periods:
        part = horizon_rows(
            outputs, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        detail.append(part)
    detail = pd.concat(detail, ignore_index=True)
    detail.to_csv(OUT / "by_horizon.csv", index=False)
    summary = pd.concat([
        summarize(part).assign(period=period)
        for period, part in detail.groupby("period", sort=False)
    ], ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    h5_rows = []
    for candidate, output in outputs.items():
        for period, years in periods:
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    finite = np.isfinite(online)
    pd.DataFrame({
        "date": dates[finite], "currency": currencies[finite],
        "noon_rank": noon_rank[finite], "signed_spot_rank": spot_rank[finite],
        "stale_spot_rank": stale_spot_rank[finite],
        "noon_weight": noon_weight[finite],
        "stale_system_noon_weight": stale_noon_weight[finite],
        "online_score": online[finite],
    }).to_csv(OUT / "weight_path.csv", index=False)

    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], outputs, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "spot_online_weighting_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in outputs.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DV", "fixed_policy": POLICY,
        "experts": ("packet-DO noon consensus", "packet-DU signed spot"),
        "online_spec": {
            "scope": SPEC.scope, "window": SPEC.window, "eta": SPEC.eta,
        },
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "feedback_loss": "equal-weight Brier events over h=1/3/5/10/20",
        "feedback_rule": "target reach date strictly before signal date",
        "matched_control": "signed spot delayed 20 target rows only",
        "future_outcome_corruption_check": True,
        "next_cbr_rate_used": False,
        "status": "retrospective causal regime-mechanism test",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY\n" + summary.to_string(index=False))
    print("\nCOMBINED BY HORIZON\n" + detail[
        detail.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
