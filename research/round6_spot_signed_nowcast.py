"""Packet-DU: label-free signed partial-fixing spot nowcast."""
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
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_moex_perpetual_hourly_features import load_hourly_history
from research.round6_moex_spot_hourly_features import (
    build_spot_features,
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


OUT = Path("results/research/round6/spot_signed_nowcast")
NOON_CONSENSUS_PATH = Path(
    "results/research/round6/three_view_futures_consensus/outputs.pkl"
)
BLEND_WEIGHTS = (.10, .25, .40)
STALE_ROWS = 20
RAW_NAMES = (
    "spot_usd_basis", "spot_cny_basis", "spot_basis_arithmetic",
    "spot_basis_minimum", "spot_basis_maximum", "spot_basis_lower_quartile",
)


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def raw_scores(matrix, names):
    cny = matrix[:, names.index("moex_hourly_spot_cnyrub_tom_cbr_basis")].astype(float)
    usd = matrix[:, names.index("moex_hourly_spot_usd000utstom_cbr_basis")].astype(float)
    pair = np.column_stack((usd, cny))
    return {
        "spot_usd_basis": usd,
        "spot_cny_basis": cny,
        "spot_basis_arithmetic": np.mean(pair, axis=1),
        "spot_basis_minimum": np.min(pair, axis=1),
        "spot_basis_maximum": np.max(pair, axis=1),
        "spot_basis_lower_quartile": np.quantile(pair, .25, axis=1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    spot_history, spot_digest = load_spot_history()
    perpetual_history, _perpetual_digest = load_hourly_history()
    matrix, names = build_spot_features(
        index, spot_history, references, perpetual_history,
    )
    raw = raw_scores(matrix, names)
    if tuple(raw) != RAW_NAMES:
        raise AssertionError("signed spot score order changed")

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
        )
        for name, score in raw.items()
    }

    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    noon_consensus = _load(NOON_CONSENSUS_PATH, "selected")
    candidates = {
        "incumbent": incumbent, "noon_consensus": noon_consensus,
        **aligned_raw,
    }
    matched_stale = dict(stale_raw)
    for name, output in aligned_raw.items():
        for base_name, base in (
            ("incumbent", incumbent), ("noon_consensus", noon_consensus),
        ):
            for weight in BLEND_WEIGHTS:
                candidate = (
                    f"{base_name}{int((1-weight)*100)}_"
                    f"{name}{int(weight*100)}"
                )
                candidates[candidate] = combine_causal(
                    (base, output), (1.0 - weight, weight), dates, currencies,
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
        "incumbent": incumbent, "noon_consensus": noon_consensus,
        "selected": candidates[selected],
    }
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
        y5, dates, currencies, valid, masks, "signed_spot_nowcast_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DU", "fixed_policy": POLICY,
        "spot_payload_sha256": spot_digest,
        "decision_time": "12:00:00 Europe/Moscow",
        "raw_scores": RAW_NAMES,
        "sign": "positive spot minus current CBR basis is favourable",
        "sign_fitted": False,
        "blend_bases": ("incumbent", "packet-DO noon_consensus"),
        "blend_weights": BLEND_WEIGHTS, "stale_rows": STALE_ROWS,
        "selection_period": 2024, "selected": selected,
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
