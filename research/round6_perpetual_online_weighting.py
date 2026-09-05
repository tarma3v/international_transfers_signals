"""Packet-DK: delayed multi-horizon weighting of the MOEX futures expert."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
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
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_cny_decomposition import POLICY
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/perpetual_online_weighting")
FUTURES_PATH = Path("results/research/round6/moex_perpetual/outputs.pkl")
RANK_WINDOW = 250
RANK_MINIMUM = 20
PRIOR_STRENGTH = 250.0


@dataclass(frozen=True)
class OnlineSpec:
    scope: str
    window: int
    eta: float

    @property
    def name(self) -> str:
        return f"online_{self.scope}_w{self.window}_eta{self.eta:g}"


SPECS = tuple(
    OnlineSpec(scope, window, eta)
    for scope in ("global", "local", "hierarchical")
    for window in (250, 1000)
    for eta in (2.0, 5.0)
)


class LossWindow:
    """A fixed-size causal window of two-expert loss observations."""

    def __init__(self, maximum: int):
        self.values: deque[np.ndarray] = deque()
        self.maximum = maximum
        self.total = np.zeros(2, dtype=float)

    def append(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        self.values.append(value)
        self.total += value
        if len(self.values) > self.maximum:
            self.total -= self.values.popleft()

    @property
    def count(self) -> int:
        return len(self.values)

    def mean(self) -> np.ndarray:
        if not self.values:
            return np.full(2, .25, dtype=float)
        return self.total / len(self.values)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / exp.sum()


def online_scores(
    spec: OnlineSpec,
    incumbent: np.ndarray,
    futures: np.ndarray,
    labels: np.ndarray,
    reaches: np.ndarray,
    dates: np.ndarray,
    currencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine experts using only horizon outcomes resolved before each day."""
    combined = np.full(len(dates), np.nan, dtype=float)
    incumbent_weight = np.full(len(dates), np.nan, dtype=float)
    finite_score = np.isfinite(incumbent) & np.isfinite(futures)
    events = []
    for row in np.flatnonzero(finite_score):
        for column, _horizon in enumerate(HORIZONS):
            if np.isfinite(labels[row, column]):
                events.append((reaches[row, column], row, column))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    global_window = LossWindow(spec.window)
    local_windows = {
        currency: LossWindow(spec.window) for currency in np.unique(currencies)
    }
    pointer = 0
    ordered_rows = np.flatnonzero(finite_score)
    ordered_rows = ordered_rows[np.lexsort((currencies[ordered_rows], dates[ordered_rows]))]
    start = 0
    while start < len(ordered_rows):
        day = dates[ordered_rows[start]]
        stop = start + 1
        while stop < len(ordered_rows) and dates[ordered_rows[stop]] == day:
            stop += 1

        # Strict inequality is intentional: an outcome reaching the current
        # publication date is admitted only on the next signal date.
        while pointer < len(events) and events[pointer][0] < day:
            _reach, old_row, column = events[pointer]
            truth = labels[old_row, column]
            loss = np.square(
                np.asarray((incumbent[old_row], futures[old_row])) - truth
            )
            global_window.append(loss)
            local_windows[currencies[old_row]].append(loss)
            pointer += 1

        for position in range(start, stop):
            row = ordered_rows[position]
            local = local_windows[currencies[row]]
            if spec.scope == "global":
                loss = global_window.mean()
            elif spec.scope == "local":
                loss = local.mean()
            elif spec.scope == "hierarchical":
                alpha = local.count / (local.count + PRIOR_STRENGTH)
                loss = alpha * local.mean() + (1.0 - alpha) * global_window.mean()
            else:
                raise KeyError(spec.scope)
            weights = _softmax(-spec.eta * loss)
            incumbent_weight[row] = weights[0]
            combined[row] = (
                weights[0] * incumbent[row] + weights[1] * futures[row]
            )
        start = stop
    return combined, incumbent_weight


def outcome_causality_check(
    spec: OnlineSpec,
    incumbent: np.ndarray,
    futures: np.ndarray,
    labels: np.ndarray,
    reaches: np.ndarray,
    dates: np.ndarray,
    currencies: np.ndarray,
) -> bool:
    cutoff = pd.Timestamp("2025-06-30").date()
    original, original_weight = online_scores(
        spec, incumbent, futures, labels, reaches, dates, currencies,
    )
    changed = labels.copy()
    for column in range(changed.shape[1]):
        future = np.asarray([value > cutoff for value in reaches[:, column]])
        eligible = future & np.isfinite(changed[:, column])
        changed[eligible, column] = 1.0 - changed[eligible, column]
    altered, altered_weight = online_scores(
        spec, incumbent, futures, changed, reaches, dates, currencies,
    )
    past = (dates <= cutoff) & np.isfinite(original)
    np.testing.assert_array_equal(original[past], altered[past])
    np.testing.assert_array_equal(original_weight[past], altered_weight[past])
    return True


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reaches = np.column_stack([
        target_reach_dates(index, series, h) for h in HORIZONS
    ])
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]

    incumbent_output = _load(INCUMBENT_PATH, INCUMBENT)
    futures_output = _load(FUTURES_PATH, "selected")
    stale_output = _load(FUTURES_PATH, "matched_stale20")
    incumbent_rank = causal_percentiles(
        row_scores(incumbent_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    futures_rank = causal_percentiles(
        row_scores(futures_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    stale_rank = causal_percentiles(
        row_scores(stale_output, len(index)), dates, currencies,
        RANK_WINDOW, RANK_MINIMUM,
    )
    outcome_causality_check(
        SPECS[0], incumbent_rank, futures_rank, labels, reaches,
        dates, currencies,
    )

    outputs = {
        "incumbent": incumbent_output,
        "futures": futures_output,
        "static_equal": _outputs(
            .5 * incumbent_rank + .5 * futures_rank, y5, dates,
        ),
    }
    stale_outputs = {
        "incumbent": incumbent_output,
        "futures": stale_output,
        "static_equal": _outputs(
            .5 * incumbent_rank + .5 * stale_rank, y5, dates,
        ),
    }
    weights, stale_weights = {}, {}
    for spec in SPECS:
        score, weight = online_scores(
            spec, incumbent_rank, futures_rank, labels, reaches,
            dates, currencies,
        )
        stale_score, stale_weight = online_scores(
            spec, incumbent_rank, stale_rank, labels, reaches,
            dates, currencies,
        )
        outputs[spec.name] = _outputs(score, y5, dates)
        stale_outputs[spec.name] = _outputs(stale_score, y5, dates)
        weights[spec.name] = weight
        stale_weights[spec.name] = stale_weight
        print(f"built {spec.name}", flush=True)

    screen = horizon_rows(
        outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    selected_online = _choose(screen_summary[
        screen_summary.candidate.str.startswith("online_")
    ])
    comparison = {
        "incumbent": incumbent_output,
        "futures": futures_output,
        "selected": outputs[selected],
        "best_online": outputs[selected_online],
        "best_online_stale20": stale_outputs[selected_online],
    }
    if selected != "incumbent":
        comparison["matched_stale20"] = stale_outputs[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summarize(later[later.period == "combined_2025_2026"]).to_csv(
        OUT / "later_summary.csv", index=False,
    )

    h5_rows = []
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if selected == "incumbent":
        selected_weight = np.ones(len(index), dtype=float)
        selected_stale_weight = selected_weight.copy()
    elif selected == "futures":
        selected_weight = np.zeros(len(index), dtype=float)
        selected_stale_weight = selected_weight.copy()
    elif selected == "static_equal":
        selected_weight = np.full(len(index), .5, dtype=float)
        selected_stale_weight = selected_weight.copy()
    else:
        selected_weight = weights[selected]
        selected_stale_weight = stale_weights[selected]
    selected_score = row_scores(outputs[selected], len(index))
    best_online_score = row_scores(outputs[selected_online], len(index))
    weight_rows = np.isfinite(selected_score)
    pd.DataFrame({
        "date": dates[weight_rows],
        "currency": currencies[weight_rows],
        "incumbent_rank": incumbent_rank[weight_rows],
        "futures_rank": futures_rank[weight_rows],
        "stale20_rank": stale_rank[weight_rows],
        "incumbent_weight": selected_weight[weight_rows],
        "stale_system_incumbent_weight": selected_stale_weight[weight_rows],
        "selected_score": selected_score[weight_rows],
        "best_online_candidate": selected_online,
        "best_online_incumbent_weight": weights[selected_online][weight_rows],
        "best_online_score": best_online_score[weight_rows],
    }).to_csv(OUT / "selected_weight_path.csv", index=False)

    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], comparison, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "perpetual_online_weighting_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DK", "fixed_policy": POLICY,
        "incumbent": INCUMBENT,
        "futures_expert": "packet-DH selected futures_extra",
        "matched_stale_expert": "packet-DH matched_stale20",
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "specs": [asdict(spec) for spec in SPECS],
        "feedback_loss": "equal-weight Brier events over h=1/3/5/10/20",
        "feedback_rule": "target horizon reach date strictly before signal date",
        "hierarchical_prior_strength": PRIOR_STRENGTH,
        "selection_period": 2024, "selected": selected,
        "best_online_diagnostic": selected_online,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "physical_future_outcome_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("Selected on 2024:", selected)
    print("Best online on 2024:", selected_online)
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + summarize(
        later[later.period == "combined_2025_2026"]
    ).to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
