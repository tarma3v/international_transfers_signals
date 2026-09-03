"""Point-in-time signal computation and full-history signal table export.

The case declares this a verifiability CONDITION, not an optional nicety
(problem.md:113): "код должен уметь считать сигналы на произвольную дату
среза... нужна функция «дай сигналы, как они выглядели бы на дату T»,
а не только прогон по всей истории целиком." Without this function the
no-look-ahead claim cannot be checked independently.

Guarantee here is structural, matching the sibling project's approach
(international_transfers_signals/ml/signals.py): series are physically cut
at T via `Series.truncate` BEFORE any feature is computed, so observations
after T do not exist in memory to leak from.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from transfer_lift.data import CORRIDORS, REFERENCE, Series
from transfer_lift.features import WARMUP, _row_features, published_next_features

# Metadata for each rule-based indicator: which base indicator from the case it
# implements ("моментум"/"уровень"/"разворот"/суточный лаг ЦБ), its reaction
# speed classification (used for the fast-vs-slow "цена ожидания" analysis)
# and which push message scenario it supports.
INDICATOR_META: dict[str, dict[str, str]] = {
    "momentum_down": {
        "base_indicator": "momentum",
        "speed": "fast",
        "scenario": "currently_favourable",
    },
    "level_low_percentile": {
        "base_indicator": "level",
        "speed": "slow",
        "scenario": "currently_favourable",
    },
    "reversal_from_low": {
        "base_indicator": "reversal",
        "speed": "slow",
        "scenario": "window_closing",
    },
    "published_tomorrow_worse": {
        "base_indicator": "published_next_rate",
        "speed": "fast",
        "scenario": "currently_favourable",
    },
    "published_next_low": {
        "base_indicator": "published_next_rate",
        "speed": "fast",
        "scenario": "currently_favourable",
    },
}

# Direction of the rate move each indicator flags. "down" = RUB/FX rate
# declining, favourable to the sender; "up" = rate recovering after a bottom,
# i.e. the favourable window is closing.
DIRECTION_BY_INDICATOR: dict[str, str] = {
    "momentum_down": "down",
    "level_low_percentile": "down",
    "reversal_from_low": "up",
    "published_tomorrow_worse": "down",
    "published_next_low": "down",
}

RULE_INDICATORS: tuple[str, ...] = tuple(INDICATOR_META)
PUBLISHED_NEXT_INDICATORS: frozenset[str] = frozenset(
    name for name, meta in INDICATOR_META.items() if meta["base_indicator"] == "published_next_rate"
)


def truncate_to(series: dict[str, Series], as_of: dt.date) -> dict[str, Series]:
    """Copy of series physically cut at date T (inclusive). No rows after T exist.

    This is the single place where the "no look-ahead" guarantee becomes
    structural rather than a promise: any feature computed from the result
    physically cannot see a value published after `as_of`.
    """
    out: dict[str, Series] = {}
    for code, s in series.items():
        n = int(np.searchsorted(np.array(s.dates, dtype=object), as_of, side="right"))
        if n == 0:
            raise ValueError(f"{code}: no publication on or before {as_of}")
        out[code] = s.truncate(n)
    return out


def signals_as_of(
    series: dict[str, Series],
    as_of: dt.date,
    corridors: tuple[str, ...] = CORRIDORS,
    refs: tuple[str, ...] = REFERENCE,
    indicators: tuple[str, ...] = RULE_INDICATORS,
    include_published_next: bool = False,
) -> list[dict[str, object]]:
    """Signal-layer state as of a single date T, for every corridor.

    Rule-based indicators only: no model fitting is required, so this works
    for ANY cut date with enough warmup history, independent of the
    walk-forward fold schedule used elsewhere for ML benchmarking.

    `include_published_next=False` (default) restricts the output to the
    3 indicators whose features are built strictly from `values[:idx+1]`,
    i.e. rows with effective date <= as_of, with the cleanest possible
    no-look-ahead story.

    `include_published_next=True` additionally scores the 2 indicators that
    use tomorrow's already-published CBR rate (problem.md:100: "курс на
    завтра публикуется сегодня"). This is legitimate only because the
    original (untruncated) series stores that rate at position idx+1 of the
    SAME array truncate_to() sliced from — it is not future market data, it
    is a rate CBR already announced on the cut date. Callers must ensure
    their product actually sends pushes after that publication before
    relying on these two indicators from this function.
    """
    from transfer_lift.models import make_model

    cut = truncate_to(series, as_of)
    ref_dates = {code: np.array(cut[code].dates, dtype=object) for code in refs}

    too_short = sorted(c for c in corridors if len(cut[c]) <= WARMUP)
    if too_short:
        raise ValueError(
            f"as_of={as_of} too early: corridors {too_short} do not have "
            f"{WARMUP} warmup observations yet"
        )

    out: list[dict[str, object]] = []
    for corridor in corridors:
        s = cut[corridor]
        idx = len(s) - 1
        day = s.dates[idx]
        ref_past: dict[str, np.ndarray] = {}
        for ref in refs:
            ref_idx = int(np.searchsorted(ref_dates[ref], day, side="right")) - 1
            ref_past[ref] = cut[ref].values[: ref_idx + 1]
        row = _row_features(s.values[: idx + 1], day, corridor, ref_past)

        has_published_next = False
        if include_published_next:
            original = series[corridor]
            if idx + 1 < len(original.values):
                row.update(published_next_features(original.values, idx))
                has_published_next = True

        row_frame = pd.DataFrame([row])
        rate = float(s.values[idx])
        for name in indicators:
            if name in PUBLISHED_NEXT_INDICATORS and not has_published_next:
                continue
            score = float(make_model(name).score(row_frame, [])[0])
            meta = INDICATOR_META[name]
            out.append(
                {
                    "as_of": as_of,
                    "date": day,
                    "corridor": corridor,
                    "rate": rate,
                    "indicator": name,
                    "base_indicator": meta["base_indicator"],
                    "direction": DIRECTION_BY_INDICATOR[name],
                    "strength": round(score, 4),
                    "speed": meta["speed"],
                    "scenario": meta["scenario"],
                }
            )
    return out


def build_signal_table(
    dataset: pd.DataFrame,
    indicators: tuple[str, ...] = RULE_INDICATORS,
) -> pd.DataFrame:
    """Full-history signal table required as a deliverable (problem.md:134):

    "на выходе формируется таблица сигналов: дата, коридор, индикатор,
    направление, сила, скорость индикатора и рекомендованный сценарий."

    Reuses the already-built past-only feature dataset from `build_dataset`,
    so results are consistent with the walk-forward benchmark: every row's
    strength is computed exactly the same way as in `evaluation.py`. This is
    the reproducible backtest artifact, not an as-of-T verifiability probe
    (see `signals_as_of` for that).
    """
    from transfer_lift.models import make_model

    parts: list[pd.DataFrame] = []
    for name in indicators:
        if name in PUBLISHED_NEXT_INDICATORS and not any(
            column.startswith("published_next_") for column in dataset.columns
        ):
            continue
        scores = make_model(name).score(dataset, [])
        meta = INDICATOR_META[name]
        parts.append(
            pd.DataFrame(
                {
                    "date": dataset["date"].to_numpy(),
                    "corridor": dataset["corridor"].to_numpy(),
                    "rate": dataset["rate"].to_numpy(),
                    "indicator": name,
                    "base_indicator": meta["base_indicator"],
                    "direction": DIRECTION_BY_INDICATOR[name],
                    "strength": np.asarray(scores, dtype=float).round(4),
                    "speed": meta["speed"],
                    "scenario": meta["scenario"],
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "corridor", "indicator"])
        .reset_index(drop=True)
    )
