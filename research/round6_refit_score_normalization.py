"""Packet-T target-free normalization across quarterly score refits."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/refit_score_normalization")
SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
SOURCE_CANDIDATE = "stack50_benefit50"
MIN_QUARTER_HISTORY = 10


def full_score(output: dict, n_rows: int) -> np.ndarray:
    """Restore a row-aligned score vector without reading any target."""
    score = np.full(n_rows, np.nan, dtype=float)
    # Calibration copies can be normalized against a different reference than
    # the score actually emitted when that row was live. They are only a
    # fallback for rows without a saved test score; live test scores win.
    for year in sorted(output):
        item = output[year]
        rows = np.asarray(item["calib_idx"], dtype=int)
        values = np.asarray(item["calib_score"], dtype=float)
        if len(rows) != len(values):
            raise AssertionError("source calibration score/index length mismatch")
        missing = ~np.isfinite(score[rows])
        score[rows[missing]] = values[missing]
    for year in sorted(output):
        item = output[year]
        rows = np.asarray(item["test_idx"], dtype=int)
        values = np.asarray(item["test_score"], dtype=float)
        if len(rows) != len(values):
            raise AssertionError("source test score/index length mismatch")
        score[rows] = values
    return score


def causal_normalizers(
    raw: np.ndarray, dates: np.ndarray, currencies: np.ndarray,
    minimum_history: int = MIN_QUARTER_HISTORY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-quarter expanding ranks and same-day panel ranks.

    The current row is appended to its currency-quarter history only after its
    percentile is computed. Same-day ranks use only simultaneously observable
    target-currency scores and are therefore cross-sectional, not forward.
    """
    quarter_rank = np.full(len(raw), np.nan, dtype=float)
    day_rank = np.full(len(raw), np.nan, dtype=float)

    for currency in np.unique(currencies):
        rows = np.where(np.isfinite(raw) & (currencies == currency))[0]
        rows = rows[np.argsort(dates[rows])]
        history: list[float] = []
        quarter = None
        for row in rows:
            day = dates[row]
            key = (day.year, (day.month - 1) // 3 + 1)
            if key != quarter:
                history = []
                quarter = key
            if len(history) >= minimum_history:
                reference = np.sort(np.asarray(history, dtype=float))
                quarter_rank[row] = (
                    np.searchsorted(reference, raw[row], side="right")
                    / len(reference)
                )
            else:
                quarter_rank[row] = .5
            history.append(float(raw[row]))

    for day in np.unique(dates[np.isfinite(raw)]):
        rows = np.where(np.isfinite(raw) & (dates == day))[0]
        values = raw[rows]
        # Average tie ranks, scaled into (0, 1].
        day_rank[rows] = pd.Series(values).rank(method="average", pct=True).to_numpy()
    return quarter_rank, day_rank


def transformed_scores(raw, quarter_rank, day_rank) -> dict[str, np.ndarray]:
    result = {"source_stack50_benefit50": raw.copy()}
    for original_weight in (.75, .50, .25):
        q_weight = 1.0 - original_weight
        base = original_weight * raw + q_weight * quarter_rank
        stem = f"original{int(original_weight*100):02d}_qrank{int(q_weight*100):02d}"
        result[stem] = base
        result[f"{stem}_day20"] = .80 * base + .20 * day_rank
    result["qrank75_day25"] = .75 * quarter_rank + .25 * day_rank
    result["qrank50_day50"] = .50 * quarter_rank + .50 * day_rank
    return result


def future_score_check(output, dates, currencies) -> None:
    original_raw = full_score(output, len(dates))
    original_q, original_day = causal_normalizers(original_raw, dates, currencies)
    cut = dt.date(2025, 6, 30)
    changed_raw = original_raw.copy()
    future = np.asarray([day > cut for day in dates]) & np.isfinite(changed_raw)
    changed_raw[future] = np.linspace(-1000, 1000, int(future.sum()))
    changed_q, changed_day = causal_normalizers(changed_raw, dates, currencies)
    past = np.asarray([day <= cut for day in dates])
    if not np.array_equal(
        np.nan_to_num(original_q[past], nan=-999),
        np.nan_to_num(changed_q[past], nan=-999),
    ):
        raise AssertionError("future score changed an earlier quarter rank")
    if not np.array_equal(
        np.nan_to_num(original_day[past], nan=-999),
        np.nan_to_num(changed_day[past], nan=-999),
    ):
        raise AssertionError("future score changed an earlier same-day rank")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with SOURCE.open("rb") as handle:
        source = pickle.load(handle)[SOURCE_CANDIDATE]
    future_score_check(source, dates, currencies)

    raw = full_score(source, len(y))
    quarter_rank, day_rank = causal_normalizers(raw, dates, currencies)
    scores = transformed_scores(raw, quarter_rank, day_rank)
    outputs = {name: _outputs(score, y, dates) for name, score in scores.items()}
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Packet T predeclared the existing rolling grid, not weekly quota rules.
    policies = [row for row in _policy_rows() if row["policy_type"] == "rolling"]
    screen_rows = []
    for candidate, output in outputs.items():
        for policy in policies:
            item = _evaluate(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": candidate, **policy})
            screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in screen.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    later_rows = []
    for row in selected.itertuples(index=False):
        policy = _row_policy(row)
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[row.candidate], years, policy,
                             y, benefit, dates, currencies)
            item.update({"period": period, "candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            later_rows.append(item)
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_results.csv", index=False)

    finalists = selected.head(8)
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        finalists, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        finalists, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "retrospective_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown_rows = []
    for row in finalists.itertuples(index=False):
        breakdown_rows.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "source": SOURCE_CANDIDATE,
        "minimum_same_quarter_history": MIN_QUARTER_HISTORY,
        "same_quarter_rank": "strictly earlier scores, appended after transform",
        "same_day_rank": "simultaneously observable five-currency panel",
        "candidates": list(outputs),
        "target_used_by_transform": False,
        "physical_future_score_corruption_check": True,
        "policy_selected_on": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024\n" + selected[[
        "candidate", "rate", "rolling", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].to_string(index=False))
    print("\nLATER\n" + later[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "robustness",
    ]].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).to_string(index=False))


if __name__ == "__main__":
    main()
