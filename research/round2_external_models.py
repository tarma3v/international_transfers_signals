"""Sensitivity experiment for release-aware external macro/market data."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    Candidate, _extra, _features, _future_objects, _hist, _logit,
    _metric_grid, _panel_features, _select, generate,
)

OUT = Path("results/research/round2")


def _join_external(X, names, index, file: Path):
    ext = pd.read_csv(file, parse_dates=["date"])
    ext["date"] = ext.date.dt.date
    ext = ext.set_index("date")
    numeric = [c for c in ext.columns if pd.api.types.is_numeric_dtype(ext[c])]
    rows = pd.DataFrame({"date": [d for _c, _i, d in index]})
    joined = rows.join(ext[numeric], on="date")
    values_only = joined[numeric]
    missing = values_only.isna().astype(float)
    missing.columns = [f"{c}_missing" for c in numeric]
    values = values_only.ffill().fillna(0.0)
    extra = pd.concat([values, missing], axis=1)
    return np.column_stack([X, extra.to_numpy(float)]), list(names) + list(extra.columns)


def main():
    X0, names0, index, series = load_or_build()
    X0, names0 = _panel_features(X0, names0, index)
    dates = np.asarray([d for _c, _i, d in index], dtype=object)
    currencies = np.asarray([c for c, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit, floor, alive = _future_objects(series, index)
    all_outputs = {}

    files = sorted(OUT.glob("external_features_b*_d*.csv"))
    for file in files:
        X, names = _join_external(X0, names0, index, file)
        base_sets = _features(names)
        external = [i for i, name in enumerate(names) if (
            name.startswith(("brent_", "broad_dollar_", "ruonia_", "key_rate_"))
            or name in ("ruonia", "key_rate", "ruonia_key_spread", "days_since_key_change")
        )]
        currency = [i for i, name in enumerate(names) if name.startswith("currency_")]
        sets = {
            "external": np.asarray(external + currency, dtype=int),
            "combined": np.asarray(list(dict.fromkeys(list(base_sets["compact"]) + external)), dtype=int),
        }
        tag = file.stem.replace("external_features_", "")
        candidates = [
            Candidate(f"{tag}__global_external_logit", _logit, "external"),
            Candidate(f"{tag}__global_combined_logit", _logit, "combined"),
            Candidate(f"{tag}__local_combined_logit", _logit, "combined", local=True),
            Candidate(f"{tag}__global_combined_hist", _hist, "combined", window_years=7),
            Candidate(f"{tag}__local_combined_hist", _hist, "combined", local=True, window_years=7),
            Candidate(f"{tag}__global_combined_extra", _extra, "combined", window_years=7),
        ]
        for candidate in candidates:
            print(candidate.name, flush=True)
            all_outputs[candidate.name] = generate(
                candidate, X, sets[candidate.feature_set], y, alive, floor,
                dates, currencies, reach,
            )

    with (OUT / "external_model_outputs.pkl").open("wb") as fh:
        pickle.dump(all_outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    general_rows = []
    for name, output in all_outputs.items():
        general_rows.extend(_metric_grid(output, y, dates, currencies, benefit, GENERAL_YEARS, name))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "external_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "external_stage1.csv", index=False)

    # Permit at most two architectures from each lag assumption before the
    # post-shock comparison, preventing one lag family from flooding selection.
    selected = stage1.assign(tag=stage1.candidate.str.split("__").str[0]).groupby("tag").head(2)
    shock_rows = []
    for row in selected.itertuples(index=False):
        result = evaluate(
            all_outputs[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
            float(row.rate_target), int(row.rolling_window) or None, int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate, "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "external_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(3).itertuples(index=False):
        result = evaluate(
            all_outputs[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None, int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective and external-vintage sensitivity only"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "external_final_2024_2026_retrospective.csv", index=False)
    print("\nGENERAL")
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nSHOCK")
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL")
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
