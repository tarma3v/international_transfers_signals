"""Per-currency expert selection with a shared product policy.

Each corridor chooses one model family using only 2017--2020 out-of-fold
predictions at a fixed 20% alert quota.  The five selected score streams are
then joined and evaluated unchanged on 2022--2023 and retrospectively on
2024--2026.  This directly tests whether corridor-specific models are more
stable than global routing.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import build_targets
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS, _metric_grid, _select,
)
from research.round3_postshock_reset import _benefit

ROOT = Path("results/research")
R2 = ROOT / "round2"
OUT = ROOT / "round3"
POOL = {
    "anchor_multiscale": (ROOT / "candidate_outputs_h5_v2.pkl", "anchor_multiscale"),
    "global_compact_extra": (R2 / "diverse_outputs.pkl", "global_compact_extra"),
    "local_floor_q35": (R2 / "diverse_outputs.pkl", "local_floor_q35"),
    "global_gmm3_hist": (R2 / "diverse_outputs.pkl", "global_gmm3_hist"),
    "global_survival_logit": (R2 / "diverse_outputs.pkl", "global_survival_logit"),
    "global_hist_window5": (R2 / "recency_outputs.pkl", "global_hist_window5"),
    "global_extra_window3": (R2 / "recency_outputs.pkl", "global_extra_window3"),
    "local_hist_window3": (R2 / "recency_outputs.pkl", "local_hist_window3"),
    "global_pairwise_quarter_w5": (R2 / "ranker_outputs.pkl", "global_pairwise_quarter_w5"),
    "delayed_global_hist_w5": (OUT / "delayed_outputs.pkl", "delayed_global_hist_w5"),
    "cross_extra_w5": (OUT / "cross_outputs.pkl", "cross_extra_w5"),
    "barrier_hist_window5": (OUT / "barrier_outputs.pkl", "barrier_hist_window5"),
    "pooled_hazard_local_hist_w5": (
        OUT / "pooled_hazard_outputs.pkl", "pooled_hazard_local_hist_w5"
    ),
}


def _load_pool() -> dict:
    caches, result = {}, {}
    for name, (path, key) in POOL.items():
        if path not in caches:
            with path.open("rb") as fh:
                caches[path] = pickle.load(fh)
        result[name] = caches[path][key]
    return result


def _currency_metric(output: dict, currency: str, y: np.ndarray,
                     dates: np.ndarray, currencies: np.ndarray,
                     benefit: np.ndarray) -> dict:
    hits, bases, benefits, year_lifts, fired_total, valid_total = [], [], [], [], 0, 0
    active_dates = []
    for year in GENERAL_YEARS:
        z = output[year]
        ca, te = np.asarray(z["calib_idx"]), np.asarray(z["test_idx"])
        cm = currencies[ca] == currency; tm = currencies[te] == currency
        reference = np.asarray(z["calib_score"])[cm]
        rows = te[tm]; scores = np.asarray(z["test_score"])[tm]
        threshold = float(np.quantile(reference, .80))
        fired = scores >= threshold
        valid_y = y[rows]
        base = float(np.mean(valid_y)); hit = float(np.mean(valid_y[fired]))
        year_lifts.append(hit / base)
        hits.extend(valid_y[fired]); bases.extend(valid_y)
        benefits.extend(benefit[rows[fired]])
        fired_total += int(fired.sum()); valid_total += len(rows)
        active_dates.extend(dates[rows[fired]])
    overall = float(np.mean(hits) / np.mean(bases))
    span_weeks = max(1.0, (max(dates) - min(dates)).days / 7.0)
    return {
        "currency": currency, "lift": overall,
        "year_lift_min": float(min(year_lifts)),
        "forward_benefit_bps": float(np.nanmean(benefits)),
        "selected_share": fired_total / max(1, valid_total),
        "robustness": float(min(overall, min(year_lifts))),
    }


def _hybrid(outputs: dict, selected: dict[str, str]) -> dict:
    result = {}
    years = sorted(set.intersection(*(set(outputs[name]) for name in selected.values())))
    for year in years:
        template = outputs[next(iter(selected.values()))][year]
        ca, te = np.asarray(template["calib_idx"]), np.asarray(template["test_idx"])
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        # Currencies are reconstructed from row ordering in main and attached below.
        result[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = _load_pool()
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = _benefit(series, index)

    rows = []
    for name, output in outputs.items():
        for currency in CORRIDORS:
            metric = _currency_metric(output, currency, y, dates, currencies, benefit)
            metric["candidate"] = name; rows.append(metric)
    selection = pd.DataFrame(rows)
    selection.to_csv(OUT / "currency_champion_selection_grid_2017_2020.csv", index=False)
    winners = {}
    for currency, part in selection.groupby("currency"):
        eligible = part[part.forward_benefit_bps > 0]
        if not len(eligible):
            eligible = part
        winners[currency] = str(eligible.sort_values(
            ["robustness", "lift"], ascending=False
        ).iloc[0].candidate)
    pd.DataFrame([
        selection[(selection.currency == currency) & (selection.candidate == name)].iloc[0]
        for currency, name in winners.items()
    ]).to_csv(OUT / "currency_champions.csv", index=False)

    hybrid = _hybrid(outputs, winners)
    for year, z in hybrid.items():
        ca, te = z["calib_idx"], z["test_idx"]
        for currency, name in winners.items():
            source = outputs[name][year]
            if not np.array_equal(ca, source["calib_idx"]) or not np.array_equal(te, source["test_idx"]):
                raise ValueError(f"unaligned output {name}, {year}")
            cm = currencies[ca] == currency; tm = currencies[te] == currency
            z["calib_score"][cm] = source["calib_score"][cm]
            z["test_score"][tm] = source["test_score"][tm]
    with (OUT / "currency_champion_outputs.pkl").open("wb") as fh:
        pickle.dump({"currency_champions": hybrid}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general = pd.DataFrame(_metric_grid(
        hybrid, y, dates, currencies, benefit, GENERAL_YEARS, "currency_champions"
    ))
    general.to_csv(OUT / "currency_champion_general_grid.csv", index=False)
    locked = _select(general)
    pd.DataFrame([locked]).to_csv(OUT / "currency_champion_stage1.csv", index=False)
    shock = evaluate(
        hybrid, y, dates, currencies, benefit, SHOCK_YEARS,
        float(locked.rate_target), int(locked.rolling_window) or None,
        int(locked.cooldown_days),
    )
    shock.update({"candidate": "currency_champions"})
    pd.DataFrame([shock]).to_csv(OUT / "currency_champion_stage2_2022_2023.csv", index=False)
    final = evaluate(
        hybrid, y, dates, currencies, benefit, FINAL_YEARS,
        float(locked.rate_target), int(locked.rolling_window) or None,
        int(locked.cooldown_days),
    )
    final.update({"candidate": "currency_champions",
                  "status": "retrospective; final interval previously inspected"})
    pd.DataFrame([final]).to_csv(OUT / "currency_champion_final_2024_2026_retrospective.csv", index=False)
    print("WINNERS", winners, sep="\n")
    print("\nGENERAL", locked.to_string(), sep="\n")
    print("\nSHOCK", pd.Series(shock).to_string(), sep="\n")
    print("\nFINAL", pd.Series(final).to_string(), sep="\n")


if __name__ == "__main__":
    main()

