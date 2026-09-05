"""Packet-CM: causal shadow-rate signals from the National Bank of Kazakhstan."""
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
from research.round6_cny_error_regime import row_scores
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose, _forward, horizon_rows, summarize,
)
from research.round6_kazakh_central_bank_features import (
    build_kazakh_nbk_features, causality_check, load_kazakh_nbk,
)


OUT = Path("results/research/round6/kazakh_central_bank_models")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
RAW_ORDER = (
    "kazakh_nbk_direct_basis", "kazakh_nbk_negative_direct_basis",
    "kazakh_nbk_usd_basis", "kazakh_nbk_negative_usd_basis",
    "kazakh_nbk_cny_basis", "kazakh_nbk_negative_cny_basis",
    "kazakh_nbk_consensus_basis", "kazakh_nbk_negative_consensus_basis",
    "kazakh_nbk_inverse_rub_momentum_1", "kazakh_nbk_inverse_rub_momentum_2",
    "kazakh_nbk_inverse_rub_momentum_5", "kazakh_nbk_inverse_rub_momentum_10",
    "kazakh_nbk_direct_usd_disagreement", "kazakh_nbk_direct_cny_disagreement",
    "kazakh_nbk_usd_cny_disagreement", "kazakh_nbk_consensus_stale20",
)


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    local, digest = load_kazakh_nbk()
    matrix, names = build_kazakh_nbk_features(index, series, references, local)
    causality_check(index, series, references, local)
    col = {name: matrix[:, names.index(name)].astype(float) for name in names}
    raw = {
        "kazakh_nbk_direct_basis": col["kazakh_nbk_direct_basis_bps"],
        "kazakh_nbk_negative_direct_basis": -col["kazakh_nbk_direct_basis_bps"],
        "kazakh_nbk_usd_basis": col["kazakh_nbk_usd_basis_bps"],
        "kazakh_nbk_negative_usd_basis": -col["kazakh_nbk_usd_basis_bps"],
        "kazakh_nbk_cny_basis": col["kazakh_nbk_cny_basis_bps"],
        "kazakh_nbk_negative_cny_basis": -col["kazakh_nbk_cny_basis_bps"],
        "kazakh_nbk_consensus_basis": col["kazakh_nbk_consensus_basis_bps"],
        "kazakh_nbk_negative_consensus_basis": -col["kazakh_nbk_consensus_basis_bps"],
        "kazakh_nbk_inverse_rub_momentum_1": -col["kazakh_nbk_rub_quote_ret_1"],
        "kazakh_nbk_inverse_rub_momentum_2": -col["kazakh_nbk_rub_quote_ret_2"],
        "kazakh_nbk_inverse_rub_momentum_5": -col["kazakh_nbk_rub_quote_ret_5"],
        "kazakh_nbk_inverse_rub_momentum_10": -col["kazakh_nbk_rub_quote_ret_10"],
        "kazakh_nbk_direct_usd_disagreement": col["kazakh_nbk_direct_minus_usd_bps"],
        "kazakh_nbk_direct_cny_disagreement": col["kazakh_nbk_direct_minus_cny_bps"],
        "kazakh_nbk_usd_cny_disagreement": col["kazakh_nbk_usd_minus_cny_bps"],
        "kazakh_nbk_consensus_stale20": delayed_by_currency(
            col["kazakh_nbk_consensus_basis_bps"][:, None], index, rows=20,
        )[:, 0],
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    raw_outputs = {
        name: _outputs(score, targets["fav_h5"], dates)
        for name, score in raw.items()
    }
    raw_screen = horizon_rows(
        raw_outputs, (2024,), targets, forwards, dates, currencies,
    )
    raw_summary = summarize(raw_screen)
    selected_raw = _choose(raw_summary)
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    finalists = {"incumbent": incumbent, "kazakh_selected": raw_outputs[selected_raw]}
    for weight in (.10, .25, .40):
        finalists[f"incumbent{int((1-weight)*100)}_kazakh{int(weight*100)}"] = combine_causal(
            [incumbent, raw_outputs[selected_raw]], (1.0 - weight, weight),
            dates, currencies,
        )
    incumbent_score = row_scores(incumbent, len(index))
    overlay_score = incumbent_score.copy()
    overlay_score[currencies == "KZT"] = raw[selected_raw][currencies == "KZT"]
    finalists["incumbent_kzt_local_overlay"] = _outputs(
        overlay_score, targets["fav_h5"], dates,
    )
    finalist_screen = horizon_rows(
        finalists, (2024,), targets, forwards, dates, currencies,
    )
    finalist_summary = summarize(finalist_screen)
    selected_finalist = _choose(finalist_summary)
    raw_screen.to_csv(OUT / "raw_screen_2024_by_horizon.csv", index=False)
    raw_summary.to_csv(OUT / "raw_screen_2024_summary.csv", index=False)
    finalist_screen.to_csv(OUT / "finalist_screen_2024_by_horizon.csv", index=False)
    finalist_summary.to_csv(OUT / "finalist_screen_2024_summary.csv", index=False)
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(finalists, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    h5_rows = []
    for candidate, output in finalists.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, targets["fav_h5"], forwards[5],
                dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(finalists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], finalists, (2025, 2026),
        targets["fav_h5"], forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        targets["fav_h5"], dates, currencies, valid, masks,
        "kazakh_central_bank_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in finalists.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            targets["fav_h5"], forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CM", "fixed_policy": POLICY,
        "source": "National Bank of Kazakhstan official daily XLSX archive",
        "source_url": "https://nationalbank.kz/en/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut",
        "source_file": str(Path("data/external_kazakhstan_nbk_rub_usd_cny_2016_2026.xlsx")),
        "payload_sha256": digest, "selection_period": 2024,
        "raw_selected": selected_raw, "finalist_selected": selected_finalist,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20",
        "selection_constraints": "positive symmetric and future-only benefit at every horizon",
        "asof_rule": "local-NBK effective date strictly before signal date; CBR date <= signal date",
        "publication_time_assumed": False,
        "physical_future_corruption_check": True,
        "stale_control_rows_per_currency": 20,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected raw on 2024: {selected_raw}")
    print(f"Selected finalist on 2024: {selected_finalist}\n")
    print("RAW\n" + raw_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))
    print("\nFINALIST SCREEN\n" + finalist_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))


if __name__ == "__main__":
    main()
