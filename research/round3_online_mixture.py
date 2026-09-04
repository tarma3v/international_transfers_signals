"""Round-3 causal consensus and delayed-feedback online expert mixtures.

The base predictions are strictly out-of-fold outputs frozen in round two.  This
script introduces a new decision layer: expert scores are mapped to causal
year-ahead percentile ranks, then combined either by robust consensus or by an
online exponentially weighted forecaster.  Online weights are updated only when
the complete h=5 label has become observable.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    FINAL_YEARS,
    GENERAL_YEARS,
    SHOCK_YEARS,
    _metric_grid,
    _select,
)

ROUND2 = Path("results/research/round2")
OUT = Path("results/research/round3")

# Frozen before inspecting round-3 results.  The pool spans local/global,
# linear/tree/analogue/ranker, direct/survival/floor, and different windows.
EXPERT_SOURCES = {
    "global_compact_extra": "diverse_outputs.pkl",
    "local_compact_extra": "diverse_outputs.pkl",
    "global_gmm3_hist": "diverse_outputs.pkl",
    "global_path_knn300": "diverse_outputs.pkl",
    "local_spline_logit": "diverse_outputs.pkl",
    "global_survival_logit": "diverse_outputs.pkl",
    "local_floor_q35": "diverse_outputs.pkl",
    "global_hist_window5": "recency_outputs.pkl",
    "global_extra_window3": "recency_outputs.pkl",
    "local_hist_window3": "recency_outputs.pkl",
    "global_pairwise_quarter_w5": "ranker_outputs.pkl",
}
EXPERTS = tuple(EXPERT_SOURCES)
ALL_YEARS = GENERAL_YEARS + SHOCK_YEARS + FINAL_YEARS


@dataclass(frozen=True)
class HedgeSpec:
    scope: str
    eta: float
    rho: float

    @property
    def name(self) -> str:
        return f"hedge_{self.scope}_eta{self.eta:g}_rho{self.rho:g}".replace(".", "p")


def _load_experts() -> dict:
    caches = {}
    result = {}
    for expert, filename in EXPERT_SOURCES.items():
        if filename not in caches:
            with (ROUND2 / filename).open("rb") as fh:
                caches[filename] = pickle.load(fh)
        result[expert] = caches[filename][expert]
    return result


def _rank_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _ranked_parts(base: dict, currencies: np.ndarray) -> dict:
    result = {expert: {} for expert in EXPERTS}
    common_years = sorted(set.intersection(*(set(base[e]) for e in EXPERTS)))
    for expert in EXPERTS:
        for year in common_years:
            z = base[expert][year]
            ca = np.asarray(z["calib_idx"], dtype=int)
            te = np.asarray(z["test_idx"], dtype=int)
            ca_rank = np.full(len(ca), np.nan)
            te_rank = np.full(len(te), np.nan)
            for currency in CORRIDORS:
                cm = currencies[ca] == currency
                tm = currencies[te] == currency
                ca_rank[cm] = _rank_against(z["calib_score"][cm], z["calib_score"][cm])
                te_rank[tm] = _rank_against(z["calib_score"][cm], z["test_score"][tm])
            result[expert][year] = {
                "calib_idx": ca,
                "test_idx": te,
                "calib_score": ca_rank,
                "test_score": te_rank,
            }
    return result


def _matrix(ranked: dict, year: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    idx_key, score_key = f"{split}_idx", f"{split}_score"
    rows = np.asarray(ranked[EXPERTS[0]][year][idx_key], dtype=int)
    columns = []
    for expert in EXPERTS:
        part = ranked[expert][year]
        if not np.array_equal(rows, part[idx_key]):
            raise ValueError(f"unaligned {split} rows: {expert}, {year}")
        columns.append(np.asarray(part[score_key], dtype=float))
    return rows, np.column_stack(columns)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / e.sum()


def _consensus(kind: str, scores: np.ndarray) -> np.ndarray:
    if kind == "mean":
        return scores.mean(axis=1)
    if kind == "trimmed":
        return np.sort(scores, axis=1)[:, 1:-1].mean(axis=1)
    if kind == "lower_quartile":
        return np.quantile(scores, .25, axis=1)
    if kind == "median":
        return np.median(scores, axis=1)
    if kind == "geometric":
        return np.exp(np.log(np.clip(scores, 1e-4, 1.0)).mean(axis=1))
    if kind.startswith("disagreement_"):
        penalty = float(kind.rsplit("_", 1)[1])
        return scores.mean(axis=1) - penalty * scores.std(axis=1)
    raise KeyError(kind)


def build_consensus(kind: str, ranked: dict) -> dict:
    output = {}
    for year in ALL_YEARS:
        ca, ca_scores = _matrix(ranked, year, "calib")
        te, te_scores = _matrix(ranked, year, "test")
        output[year] = {
            "calib_idx": ca,
            "test_idx": te,
            "calib_score": _consensus(kind, ca_scores),
            "test_score": _consensus(kind, te_scores),
        }
    return output


def _online_sequence(
    spec: HedgeSpec,
    rows: np.ndarray,
    scores: np.ndarray,
    roles: np.ndarray,
    dates: np.ndarray,
    currencies: np.ndarray,
    y: np.ndarray,
    reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return combined calibration/test scores while respecting feedback delay."""
    original_roles = roles.copy()
    order = np.lexsort((currencies[rows], dates[rows]))
    rows, scores, roles = rows[order], scores[order], roles[order]
    n_experts = scores.shape[1]
    global_loss = np.full(n_experts, .25)
    local_loss = np.full((len(CORRIDORS), n_experts), .25)
    global_count = 0
    local_count = np.zeros(len(CORRIDORS), dtype=int)
    currency_pos = {currency: i for i, currency in enumerate(CORRIDORS)}
    pending: list[int] = []
    combined = np.full(len(rows), np.nan)

    def update(indices: list[int]) -> None:
        nonlocal global_loss, global_count
        if not indices:
            return
        losses = np.square(scores[indices] - y[rows[indices], None])
        batch = losses.mean(axis=0)
        if spec.rho < 1.0:
            global_loss = spec.rho * global_loss + (1.0 - spec.rho) * batch
        else:
            global_loss = (global_count * global_loss + len(indices) * batch) / (
                global_count + len(indices)
            )
        global_count += len(indices)
        for pos in indices:
            c = currency_pos[currencies[rows[pos]]]
            if spec.rho < 1.0:
                local_loss[c] = (
                    spec.rho * local_loss[c]
                    + (1.0 - spec.rho) * losses[indices.index(pos)]
                )
            else:
                local_loss[c] = (
                    local_count[c] * local_loss[c] + losses[indices.index(pos)]
                ) / (local_count[c] + 1)
            local_count[c] += 1

    start = 0
    while start < len(rows):
        day = dates[rows[start]]
        stop = start + 1
        while stop < len(rows) and dates[rows[stop]] == day:
            stop += 1
        ready = [pos for pos in pending if reach[rows[pos]] <= day]
        update(ready)
        if ready:
            ready_set = set(ready)
            pending = [pos for pos in pending if pos not in ready_set]
        for pos in range(start, stop):
            c = currency_pos[currencies[rows[pos]]]
            if spec.scope == "global":
                loss = global_loss
            elif spec.scope == "local":
                loss = local_loss[c]
            elif spec.scope == "hierarchical":
                # Local estimates start noisy, so shrink them toward the global
                # history with a fixed prior strength declared in the protocol.
                alpha = local_count[c] / (local_count[c] + 250.0)
                loss = alpha * local_loss[c] + (1.0 - alpha) * global_loss
            else:
                raise KeyError(spec.scope)
            weights = _softmax(-spec.eta * loss)
            combined[pos] = float(scores[pos] @ weights)
            pending.append(pos)
        start = stop

    # Restore the caller's row order.
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    return combined[inverse], original_roles


def build_hedge(spec: HedgeSpec, ranked: dict, dates: np.ndarray,
                currencies: np.ndarray, y: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    available = sorted(set.intersection(*(set(ranked[e]) for e in EXPERTS)))
    for year in available:
        rows_parts, score_parts, role_parts = [], [], []
        for old in available:
            if old >= year - 1:
                continue
            rows, scores = _matrix(ranked, old, "test")
            rows_parts.append(rows); score_parts.append(scores)
            role_parts.append(np.full(len(rows), "history", dtype=object))
        ca, ca_scores = _matrix(ranked, year, "calib")
        te, te_scores = _matrix(ranked, year, "test")
        rows_parts.extend([ca, te]); score_parts.extend([ca_scores, te_scores])
        role_parts.extend([
            np.full(len(ca), "calib", dtype=object),
            np.full(len(te), "test", dtype=object),
        ])
        rows = np.concatenate(rows_parts)
        scores = np.vstack(score_parts)
        roles = np.concatenate(role_parts)
        combined, restored_roles = _online_sequence(
            spec, rows, scores, roles, dates, currencies, y, reach
        )
        output[year] = {
            "calib_idx": ca,
            "test_idx": te,
            "calib_score": combined[restored_roles == "calib"],
            "test_score": combined[restored_roles == "test"],
        }
    return output


def _benefit(series, index) -> np.ndarray:
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None:
            result[row] = value
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series = load_or_build()
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)
    ranked = _ranked_parts(_load_experts(), currencies)

    consensus = (
        "mean", "trimmed", "median", "lower_quartile", "geometric",
        "disagreement_0.25", "disagreement_0.5", "disagreement_1.0",
    )
    hedge_specs = [
        HedgeSpec(scope, eta, rho)
        for scope in ("global", "local", "hierarchical")
        for eta in (2.0, 5.0, 10.0, 20.0)
        for rho in (.97, .99, 1.0)
    ]
    outputs = {f"consensus_{kind}": build_consensus(kind, ranked) for kind in consensus}
    for spec in hedge_specs:
        outputs[spec.name] = build_hedge(spec, ranked, dates, currencies, y, reach)
        print(spec.name, flush=True)
    with (OUT / "online_mixture_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for name, output in outputs.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, name
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "online_mixture_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "online_mixture_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.head(10).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({
            "candidate": row.candidate,
            "stage1_rate": row.rate_target,
            "stage1_rolling": row.rolling_window,
            "stage1_cooldown": row.cooldown_days,
        })
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "online_mixture_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(4).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.stage1_cooldown),
        )
        result.update({
            "candidate": row.candidate,
            "status": "retrospective; final interval previously inspected",
        })
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "online_mixture_final_2024_2026_retrospective.csv", index=False)

    (OUT / "online_mixture_protocol.json").write_text(json.dumps({
        "experts": EXPERT_SOURCES,
        "consensus": consensus,
        "hedge_specs": [spec.__dict__ for spec in hedge_specs],
        "loss": "Brier loss on preceding-year percentile-ranked expert scores",
        "feedback_rule": "update only when target_reach_date <= current publication date",
        "selection": {
            "general": GENERAL_YEARS,
            "shock_gate": SHOCK_YEARS,
            "final_retrospective": FINAL_YEARS,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "year_lift_min", "corridor_lift_min", "robustness"]
    print("\nGENERAL", stage1[columns].head(15).to_string(index=False), sep="\n")
    print("\nSHOCK", shock[columns].to_string(index=False), sep="\n")
    print("\nFINAL", final[[c for c in columns if c in final]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
