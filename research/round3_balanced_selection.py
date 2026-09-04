"""Frequency-balanced selection across all frozen model families.

Headline aggregate lift can be inflated when a policy fires mostly in years
with a high base rate.  This audit selects policies on 2017--2020 only while
requiring reasonable alert frequency in every year, then applies the locked
policy to 2022--2023 and finally 2024--2026.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round3_postshock_reset import _benefit

ROOT = Path("results/research")
R2 = ROOT / "round2"
R3 = ROOT / "round3"
GENERAL = (2017, 2018, 2019, 2020)
SHOCK = (2022, 2023)
FINAL = (2024, 2025, 2026)
RATES = (.20, .25, .30, .35, .40)
ROLLS = (None, 120, 250, 500)

CACHES = (
    ("round1", ROOT / "candidate_outputs_h5_v2.pkl"),
    ("diverse", R2 / "diverse_outputs.pkl"),
    ("router", R2 / "router_outputs.pkl"),
    ("recency", R2 / "recency_outputs.pkl"),
    ("ranker", R2 / "ranker_outputs.pkl"),
    ("tower", R2 / "tower_outputs.pkl"),
    ("external", R2 / "external_model_outputs.pkl"),
    ("barrier", R3 / "barrier_outputs.pkl"),
    ("delayed", R3 / "delayed_outputs.pkl"),
    ("cross", R3 / "cross_outputs.pkl"),
    ("hazard", R3 / "pooled_hazard_outputs.pkl"),
    ("online_mix", R3 / "online_mixture_outputs.pkl"),
    ("online_sgd", R3 / "online_sgd_outputs.pkl"),
)


def _load() -> dict:
    result = {}
    for family, path in CACHES:
        if not path.exists():
            continue
        with path.open("rb") as fh:
            part = pickle.load(fh)
        for name, output in part.items():
            if all(year in output for year in GENERAL + SHOCK + FINAL):
                result[f"{family}:{name}"] = output
    return result


def _metrics(output, years, rate, rolling, y, dates, currencies, benefit):
    overall = evaluate(output, y, dates, currencies, benefit, years, rate, rolling, 0)
    annual = [evaluate(output, y, dates, currencies, benefit, (year,), rate, rolling, 0)
              for year in years]
    overall["macro_year_lift"] = float(np.mean([row["lift"] for row in annual]))
    overall["year_frequency_min"] = float(np.min([row["frequency"] for row in annual]))
    overall["year_frequency_max"] = float(np.max([row["frequency"] for row in annual]))
    overall["simpson_gap"] = float(overall["lift"] - overall["macro_year_lift"])
    return overall


def _pick(part: pd.DataFrame) -> pd.Series:
    feasible = part[
        part.frequency.between(.90, 2.10)
        & part.year_frequency_min.ge(.75)
        & part.year_frequency_max.le(2.25)
        & part.corridor_freq_min.ge(.65)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    if not len(feasible):
        feasible = part.copy()
    feasible["balanced_robustness"] = feasible[
        ["macro_year_lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    return feasible.sort_values(
        ["balanced_robustness", "macro_year_lift", "lift"], ascending=False
    ).iloc[0]


def main() -> None:
    outputs = _load()
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = _benefit(series, index)

    grid_path = R3 / "balanced_general_grid.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        rows = []
        for name, output in outputs.items():
            for rate in RATES:
                for rolling in ROLLS:
                    row = _metrics(output, GENERAL, rate, rolling, y, dates, currencies, benefit)
                    row.update({"candidate": name, "rate": rate, "rolling": rolling or 0})
                    rows.append(row)
        grid = pd.DataFrame(rows)
        grid.to_csv(grid_path, index=False)
    stage1 = pd.DataFrame([_pick(part) for _name, part in grid.groupby("candidate")])
    stage1 = stage1.sort_values(
        ["balanced_robustness", "macro_year_lift", "lift"], ascending=False
    )
    stage1["family"] = stage1.candidate.str.split(":").str[0]
    stage1.to_csv(R3 / "balanced_stage1.csv", index=False)

    shock_rows = []
    # Prevent a large hyperparameter family from crowding every other
    # architecture out of the regime gate: advance at most two per family.
    advanced = stage1.groupby("family", sort=False, group_keys=False).head(2)
    for row in advanced.itertuples(index=False):
        metric = _metrics(
            outputs[row.candidate], SHOCK, float(row.rate), int(row.rolling) or None,
            y, dates, currencies, benefit,
        )
        metric.update({"candidate": row.candidate, "stage1_rate": row.rate,
                       "stage1_rolling": row.rolling})
        metric["balanced_robustness"] = min(
            metric["macro_year_lift"], metric["year_lift_min"], metric["corridor_lift_min"]
        )
        shock_rows.append(metric)
    shock = pd.DataFrame(shock_rows).sort_values(
        ["balanced_robustness", "macro_year_lift", "lift"], ascending=False
    )
    shock.to_csv(R3 / "balanced_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(6).itertuples(index=False):
        metric = _metrics(
            outputs[row.candidate], FINAL, float(row.stage1_rate),
            int(row.stage1_rolling) or None, y, dates, currencies, benefit,
        )
        metric.update({"candidate": row.candidate,
                       "status": "retrospective; final interval previously inspected"})
        metric["balanced_robustness"] = min(
            metric["macro_year_lift"], metric["year_lift_min"], metric["corridor_lift_min"]
        )
        final_rows.append(metric)
    final = pd.DataFrame(final_rows).sort_values(
        ["balanced_robustness", "macro_year_lift", "lift"], ascending=False
    )
    final.to_csv(R3 / "balanced_final_2024_2026_retrospective.csv", index=False)

    columns = ["candidate", "frequency", "year_frequency_min", "year_frequency_max",
               "lift", "macro_year_lift", "simpson_gap", "forward_benefit_bps",
               "year_lift_min", "corridor_lift_min", "balanced_robustness"]
    print("\nGENERAL", stage1[columns].head(20).to_string(index=False), sep="\n")
    print("\nSHOCK", shock[columns].to_string(index=False), sep="\n")
    print("\nFINAL", final[columns].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
