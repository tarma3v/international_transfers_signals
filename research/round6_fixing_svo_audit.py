"""Packet-EH: descriptive 2022 SVO-boundary audit of fixed fixing decisions."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import HORIZONS, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_resolved_models import _fire
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/round6/fixing_svo_audit")
SOURCE = Path("results/research/round6/fixing_history/outputs.pkl")
BOUNDARY = dt.date(2022, 2, 24)
REGIMES = (
    ("pre_svo_2022", dt.date(2022, 1, 3), dt.date(2022, 2, 23)),
    ("post_svo_2022", BOUNDARY, dt.date(2022, 12, 31)),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    with SOURCE.open("rb") as handle:
        output = pickle.load(handle)["fixing_basis"]

    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        valid, fired = _fire(output, (2022,), POLICY, y, dates, currencies)
        for name, start, end in REGIMES:
            period = np.asarray([
                start <= day <= end for day in dates
            ])
            scope = valid & period
            active = scope & fired
            corridor_lifts = []
            for currency in CORRIDORS:
                corridor_scope = scope & (currencies == currency)
                corridor_active = active & (currencies == currency)
                if corridor_scope.any() and corridor_active.any():
                    corridor_lifts.append(float(
                        np.mean(y[corridor_active]) / np.mean(y[corridor_scope])
                    ))
            base = float(np.mean(y[scope]))
            hit = float(np.mean(y[active])) if active.any() else np.nan
            rows.append({
                "regime": name,
                "start": str(start),
                "end": str(end),
                "horizon": h,
                "n_scope": int(scope.sum()),
                "n_signals": int(active.sum()),
                "frequency": rate_per_week(
                    int(active.sum()), len(CORRIDORS), dates, scope,
                ),
                "base_rate": base,
                "hit_rate": hit,
                "pooled_lift": hit / base if active.any() and base > 0 else np.nan,
                "corridor_lift_min": (
                    float(min(corridor_lifts)) if corridor_lifts else np.nan
                ),
                "symmetric_benefit_bps": float(np.nanmean(
                    targets[f"benefit_h{h}"][active]
                )),
                "future_only_benefit_bps": float(np.nanmean(
                    forwards[h][active]
                )),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "svo_boundary_by_horizon.csv", index=False)
    summary = result.groupby("regime", as_index=False).agg(
        horizon_lift_min=("pooled_lift", "min"),
        horizon_lift_mean=("pooled_lift", "mean"),
        horizon_corridor_lift_min=("corridor_lift_min", "min"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_only_benefit_bps", "min"),
        signal_count_min=("n_signals", "min"),
        signal_count_max=("n_signals", "max"),
    )
    summary.to_csv(OUT / "svo_boundary_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EH",
        "source": "packet-EG fixed fixing_basis decisions",
        "boundary": str(BOUNDARY),
        "regimes": [
            {"name": name, "start": str(start), "end": str(end)}
            for name, start, end in REGIMES
        ],
        "fixed_policy": POLICY,
        "model_or_policy_refit": False,
        "selector_used": False,
        "next_cbr_rate_used": False,
        "interpretation": (
            "descriptive only; short pre-boundary sample cannot promote a model"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSUMMARY\n" + summary.to_string(index=False))
    print("\nDETAIL\n" + result.to_string(index=False))


if __name__ == "__main__":
    main()
