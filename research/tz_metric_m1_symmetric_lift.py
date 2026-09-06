"""Frozen AP37 metric sensitivity: future-only versus symmetric local min."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS, target_symmetric_local_minimum
from research.after_publication_ap1 import DATA, SEED, adjusted, scorecard
from research.after_publication_ap37_effective import CANDIDATE


SOURCE = Path("results/research/after_publication/ap37_effective")
OUT = Path("results/research/tz_metric/m1_symmetric_lift")
REGISTERED = Path("research/tz_metric_m1_symmetric_lift_registered.md")
BLOCKS = (20, 50)
DRAWS = 500


def symmetric_targets(series, panel: pd.DataFrame) -> dict[int, np.ndarray]:
    """Rebuild the literal ``+-h`` binary target from AP37 current indices."""
    out: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        values = np.full(len(panel), np.nan)
        for row, (currency, index) in enumerate(
            zip(panel.currency, panel.current_index)
        ):
            value = target_symmetric_local_minimum(
                series[str(currency)].values, int(index), h
            )
            if value is not None:
                values[row] = value
        out[h] = values
    return out


def score_target(
    panel: pd.DataFrame,
    saved: dict[str, np.ndarray],
    target: np.ndarray,
    h: int,
    fired: np.ndarray,
    scope: np.ndarray,
) -> dict[str, float | int | str]:
    outcomes = {key: saved[key] for key in saved if key.startswith(("sym", "forward"))}
    for horizon in HORIZONS:
        outcomes[f"y{horizon}"] = (
            target if horizon == h else saved[f"y{horizon}"]
        )
    row = scorecard(panel, outcomes, fired, scope, saved["groups"])[
        list(HORIZONS).index(h)
    ]
    return row


def slice_rows(
    panel: pd.DataFrame,
    target: np.ndarray,
    fired: np.ndarray,
    scope: np.ndarray,
    label: str,
    h: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    years = np.array([day.year for day in dates])
    for slice_kind, levels in (
        ("currency", CORRIDORS),
        ("year", sorted(set(years[scope]))),
    ):
        for level in levels:
            select = currencies == level if slice_kind == "currency" else years == level
            valid = scope & select & np.isfinite(target)
            active = valid & fired
            base = float(np.mean(target[valid])) if valid.any() else np.nan
            hit = float(np.mean(target[active])) if active.any() else np.nan
            rows.append({
                "target": label,
                "h": h,
                "slice_kind": slice_kind,
                "slice": str(level),
                "n_scope": int(valid.sum()),
                "n_signals": int(active.sum()),
                "hit_rate": hit,
                "base_rate": base,
                "pooled_lift": hit / base if base > 0 and active.any() else np.nan,
            })
    return rows


def block_bootstrap(
    panel: pd.DataFrame,
    target: np.ndarray,
    fired: np.ndarray,
    scope: np.ndarray,
    groups: np.ndarray,
    block: int,
    draws: int = DRAWS,
) -> np.ndarray:
    valid = scope & np.isfinite(target)
    dates = panel.date.to_numpy()
    unique_dates = np.unique(dates[valid])
    date_to_id = {day: index for index, day in enumerate(unique_dates)}
    row_day = np.array([date_to_id.get(day, -1) for day in dates])
    rng = np.random.default_rng(SEED + block)
    results = np.full(draws, np.nan)
    n_dates = len(unique_dates)
    n_blocks = int(np.ceil(n_dates / block))
    for draw in range(draws):
        starts = rng.integers(0, n_dates, size=n_blocks)
        chosen = ((starts[:, None] + np.arange(block)) % n_dates).ravel()[:n_dates]
        date_weights = np.bincount(chosen, minlength=n_dates).astype(float)
        weights = np.zeros(len(panel), dtype=float)
        inside = row_day >= 0
        weights[inside] = date_weights[row_day[inside]]
        results[draw] = adjusted(target, fired, valid, groups, weights)
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(SOURCE / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(SOURCE / "outputs.npz") as source:
        saved = {key: source[key] for key in source.files}
    fired = saved[f"signal__{CANDIDATE}"].astype(bool)
    sym = symmetric_targets(load(DATA), panel)

    # Match AP37's publication-reference end-of-series completeness convention.
    for h in HORIZONS:
        sym[h][~np.isfinite(saved[f"y{h}"])] = np.nan
        common = np.isfinite(sym[h]) & np.isfinite(saved[f"y{h}"])
        assert np.all(sym[h][common] <= saved[f"y{h}"][common])

    score_rows = []
    detail_rows = []
    bootstrap_rows = []
    for period in ("early", "later"):
        scope = saved[period].astype(bool)
        for label, targets in (
            ("future_only_hit", {h: saved[f"y{h}"] for h in HORIZONS}),
            ("symmetric_local_min", sym),
        ):
            for h in HORIZONS:
                row = score_target(panel, saved, targets[h], h, fired, scope)
                score_rows.append({"period": period, "target": label, **row})
                if period == "later":
                    detail_rows.extend(slice_rows(
                        panel, targets[h], fired, scope, label, h
                    ))
                    for block in BLOCKS:
                        draws = block_bootstrap(
                            panel, targets[h], fired, scope, saved["groups"], block
                        )
                        bootstrap_rows.append({
                            "target": label,
                            "h": h,
                            "block_dates": block,
                            "point_adjusted_lift": float(row["adjusted_lift"]),
                            "ci_lo": float(np.nanquantile(draws, 0.025)),
                            "ci_hi": float(np.nanquantile(draws, 0.975)),
                            "p_lift_le_1_30": float(
                                (np.sum(draws <= 1.30) + 1) / (np.isfinite(draws).sum() + 1)
                            ),
                        })

    scores = pd.DataFrame(score_rows)
    detail = pd.DataFrame(detail_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    scores.to_csv(OUT / "scorecard.csv", index=False)
    detail.to_csv(OUT / "later_slices.csv", index=False)
    bootstrap.to_csv(OUT / "later_block_bootstrap.csv", index=False)

    # Reproduce the frozen AP37 future-only table before interpreting sensitivity.
    original = pd.read_csv(SOURCE / "retrospective_all_horizons.csv")
    original = original[original.candidate == CANDIDATE].sort_values("h")
    rebuilt = scores[
        (scores.period == "later") & (scores.target == "future_only_hit")
    ].sort_values("h")
    for column in (
        "n_scope", "n_signals", "hit_rate", "base_rate", "pooled_lift",
        "adjusted_lift", "frequency", "currency_rate_min", "currency_rate_max",
        "symmetric_bps", "forward_bps",
    ):
        np.testing.assert_allclose(
            rebuilt[column].to_numpy(), original[column].to_numpy(),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    later_wide = scores[scores.period == "later"].pivot(
        index="h", columns="target", values="adjusted_lift"
    )
    summary = {
        "candidate": CANDIDATE,
        "period": "AP37 later 2024-2026",
        "future_only_adjusted_lift": {
            str(h): float(later_wide.loc[h, "future_only_hit"]) for h in HORIZONS
        },
        "symmetric_local_min_adjusted_lift": {
            str(h): float(later_wide.loc[h, "symmetric_local_min"]) for h in HORIZONS
        },
        "model_or_policy_changed": False,
        "selection_changed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    sources = (SOURCE / "outputs.npz", SOURCE / "announcement_panel.csv", DATA, REGISTERED)
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "M1",
        "candidate": CANDIDATE,
        "draws": DRAWS,
        "blocks": list(BLOCKS),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }, indent=2))
    (OUT / "audit_checks.json").write_text(json.dumps({
        "frozen_ap37_signal_used": True,
        "frozen_scopes_and_groups_used": True,
        "future_only_scorecard_reproduced_exactly": True,
        "symmetric_target_rebuilt_from_raw_cbr_and_current_index": True,
        "symmetric_success_is_subset_of_future_success": True,
        "no_selection_or_model_change": True,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
