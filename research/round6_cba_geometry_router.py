"""Packet-CP: causal router that decides when the Armenian signal is useful."""
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
from research.round6_armenian_central_bank_features import build_cba_features, load_cba
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multihorizon_error_router import (
    TAIL_WEIGHTS, outcome_causality_check, prequential_router,
    router_labels, router_matrix,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose, _forward, horizon_rows, summarize,
)


OUT = Path("results/research/round6/cba_geometry_router")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
GEOMETRY_PATH = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
GEOMETRY = "primary75_geometry_min75_max2525"


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reach = target_reach_dates(index, series, 20)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    geometry = _load(GEOMETRY_PATH, GEOMETRY)
    incumbent_score = row_scores(incumbent, len(index))
    geometry_score = row_scores(geometry, len(index))
    _broad, _broad_names, references = load_broad_features(index, series)
    cba_source, cba_digest = load_cba()
    cba, cba_names = build_cba_features(index, series, references, cba_source)
    moex, moex_digest = load_moex_history()
    wave, wave_names = build_waveform_features(index, moex)
    gate = router_matrix(
        incumbent_score, geometry_score, X, names,
        cba, cba_names, wave, wave_names,
    )
    stale_gate = delayed_by_currency(gate, index, rows=20)
    outcome_causality_check(
        gate, incumbent_score, geometry_score, labels, dates, reach,
    )
    route_targets = {
        "equal": router_labels(
            incumbent_score, geometry_score, labels,
            np.full(len(HORIZONS), .2),
        ),
        "tail": router_labels(
            incumbent_score, geometry_score, labels, TAIL_WEIGHTS,
        ),
    }
    outputs = {
        "incumbent": incumbent,
        "geometry": geometry,
        "incumbent50_geometry50": combine_causal(
            [incumbent, geometry], (.5, .5), dates, currencies,
        ),
    }
    logs = []
    weight_columns = {}
    for objective, route_y in route_targets.items():
        for kind in ("logit", "hist", "tree"):
            soft, hard, weights, part = prequential_router(
                kind, gate, route_y, incumbent_score, geometry_score,
                dates, reach,
            )
            base = f"router_{objective}_{kind}"
            outputs[base + "_soft"] = _outputs(soft, targets["fav_h5"], dates)
            outputs[base + "_hard"] = _outputs(hard, targets["fav_h5"], dates)
            weight_columns[base] = weights
            for row in part:
                row["objective"] = objective
                logs.append(row)
    stale_soft, _hard, stale_weights, stale_logs = prequential_router(
        "logit", stale_gate, route_targets["equal"], incumbent_score,
        geometry_score, dates, reach,
    )
    outputs["router_equal_logit_soft_stale20"] = _outputs(
        stale_soft, targets["fav_h5"], dates,
    )
    weight_columns["router_equal_logit_stale20"] = stale_weights
    for row in stale_logs:
        row["objective"] = "equal_stale20"
        logs.append(row)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)
    pd.DataFrame({
        "date": dates, "currency": currencies, **weight_columns,
    }).to_csv(OUT / "router_weights.csv", index=False)
    screen = horizon_rows(
        outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(outputs, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    h5_rows = []
    for candidate, output in outputs.items():
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
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], outputs, (2025, 2026),
        targets["fav_h5"], forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        targets["fav_h5"], dates, currencies, valid, masks,
        "cba_geometry_router_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in outputs.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            targets["fav_h5"], forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CP", "fixed_policy": POLICY,
        "experts": [INCUMBENT, GEOMETRY],
        "router_interpretation": "choose whether the CBA overlay is useful now",
        "selection_period": 2024, "selected": selected,
        "router_label": "second expert has lower weighted mean Brier loss over resolved h=1/3/5/10/20 outcomes",
        "objectives": {"equal": [.2] * 5, "tail": TAIL_WEIGHTS.tolist()},
        "training_start": "2023-01-01", "purge_horizon": 20,
        "all_training_labels_resolved": chronology_ok,
        "future_outcome_corruption_check": True,
        "stale_control_rows_per_currency": 20,
        "payload_sha256": {"CBA": cba_digest, "MOEX": moex_digest},
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("router chronology failed")
    print(f"Selected on 2024: {selected}\n")
    print("SCREEN\n" + screen_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.sort_values(
        "horizon_lift_min", ascending=False,
    ).to_string(index=False))


if __name__ == "__main__":
    main()
