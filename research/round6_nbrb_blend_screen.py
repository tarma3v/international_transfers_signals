"""Packet-CS: select one NBRB/incumbent blend on 2024, audit later once."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_belarus_nbrb_features import build_nbrb_features, load_nbrb
from research.round6_belarus_nbrb_models import INCUMBENT, INCUMBENT_PATH, raw_candidates
from research.round6_broad_cbr_features import load_broad_features
from research.round6_multiobjective_blend import combine_causal
from research.round6_uzbek_central_bank_models import _forward, horizon_rows, summarize


OUT = Path("results/research/round6/nbrb_blend_screen")
WEIGHTS = (.025, .05, .075, .10, .15, .20, .30, .40)


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def _select(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    incumbent = summary[summary.candidate == "incumbent"].iloc[0]
    feasible = summary[
        summary.symmetric_benefit_min.gt(0)
        & summary.future_benefit_min.gt(0)
        & summary.horizon_lift_min.ge(incumbent.horizon_lift_min)
    ].copy()
    if not len(feasible):
        return "incumbent"
    # Protect the short horizon explicitly, then maximize worst/mean lift.
    h1 = detail[detail.horizon == 1].set_index("candidate").case_lift
    feasible["h1_lift"] = feasible.candidate.map(h1)
    feasible = feasible[feasible.h1_lift.ge(
        float(h1.loc["incumbent"]) - .01
    )]
    if not len(feasible):
        return "incumbent"
    return str(feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0].candidate)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    nbrb, digest = load_nbrb()
    matrix, names = build_nbrb_features(index, references, nbrb)
    raw, _base = raw_candidates(matrix, names, index)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    raw_outputs = {
        name: _outputs(score, targets["fav_h5"], dates)
        for name, score in raw.items()
    }
    candidates = {"incumbent": incumbent}
    for raw_name, raw_output in raw_outputs.items():
        for weight in WEIGHTS:
            name = f"incumbent{int(round((1-weight)*1000)):03d}_{raw_name}_w{int(round(weight*1000)):03d}"
            candidates[name] = combine_causal(
                [incumbent, raw_output], (1.0 - weight, weight),
                dates, currencies,
            )
    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    summary = summarize(screen)
    selected = _select(summary, screen)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_csv(OUT / "screen_2024_summary.csv", index=False)

    finalists = {"incumbent": incumbent, "selected": candidates[selected]}
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
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(finalists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CS", "candidate_count": len(candidates),
        "formula_family": list(raw), "weights": WEIGHTS,
        "selection_period": 2024, "selected": selected,
        "selection_rule": "positive benefits; min lift >= incumbent; h1 within 0.01; maximize min then mean lift",
        "source": "NBRB official API", "payload_sha256": digest,
        "asof_rule": "NBRB effective date strictly before signal date; CBR date <= signal date",
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened only after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected on 2024: {selected}\n")
    print("SCREEN TOP\n" + summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
