"""Packet-AU stitch frozen causal CNY models into a full deployment lifecycle."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_lifecycle")
PRE = Path("results/research/round6/cny_pre2022/outputs.pkl")
BRIDGE = Path("results/research/round6/cny_shock_bridge/outputs.pkl")
LATER = Path("results/research/round6/cny_history_weighting/outputs.pkl")
YEARS = tuple(range(2017, 2027))
ORDER = ("always_expanding", "early_reset_700", "resolved2000_handoff")


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _stitch(*parts):
    result = {}
    for part in parts:
        overlap = set(result).intersection(part)
        if overlap:
            raise AssertionError(f"lifecycle years overlap: {sorted(overlap)}")
        result.update({year: value for year, value in part.items()})
    if tuple(sorted(result)) != YEARS:
        raise AssertionError(f"incomplete lifecycle: {sorted(result)}")
    return result


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pre, bridge, later = _load(PRE), _load(BRIDGE), _load(LATER)
    pre_consensus = pre["logit50_extra50"]
    expanding_bridge = bridge["expanding_consensus"]
    early_bridge = bridge["mechanical_reset_hybrid"]
    all_later = later["all_history"]
    reset_later = later["hard_reset"]
    outputs = {
        "always_expanding": _stitch(pre_consensus, expanding_bridge, all_later),
        "early_reset_700": _stitch(pre_consensus, early_bridge, reset_later),
        "resolved2000_handoff": _stitch(
            pre_consensus, expanding_bridge, reset_later,
        ),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    rows = []
    for candidate in ORDER:
        for year in YEARS:
            item = _evaluate(
                outputs[candidate], (year,), POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": str(year), **POLICY})
            rows.append(item)
        item = _evaluate(
            outputs[candidate], YEARS, POLICY, y, benefit, dates, currencies,
        )
        item.update({"candidate": candidate, "period": "2017_2026", **POLICY})
        rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)

    combined = results[results.period == "2017_2026"].copy()
    bootstrap, masks, valid = _bootstrap(
        combined, outputs, YEARS, y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2017_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "lifecycle_2017_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2017_2026.csv", index=False)

    source_hashes = {str(path): _digest(path) for path in (PRE, BRIDGE, LATER)}
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AU",
        "years": YEARS,
        "variants": ORDER,
        "fixed_policy": POLICY,
        "handoff_rule": (
            "use expanding consensus until a scheduled refit has at least "
            "2000 resolved post-2022 target rows; hard reset thereafter"
        ),
        "resolved2000_handoff_date": "2024-01-01",
        "model_refit": False,
        "source_output_sha256": source_hashes,
        "all_years_retained": True,
        "later_period_status": "retrospective composition of frozen causal scores",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "year_lift_min", "year_frequency_min",
        "corridor_lift_min", "quarter_frequency_min",
    ]
    print(results[columns].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
