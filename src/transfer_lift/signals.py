"""Point-in-time rule signals built from data physically truncated at date T."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from transfer_lift.data import CORRIDORS, REFERENCE, Series
from transfer_lift.features import WARMUP, _row_features
from transfer_lift.models import make_model

# Product metadata used in both point-in-time and historical exports.
INDICATOR_META: dict[str, dict[str, str]] = {
    "momentum_down": {
        "base_indicator": "momentum",
        "direction": "down",
        "speed": "fast",
        "scenario": "currently_favourable",
    },
    "level_low_percentile": {
        "base_indicator": "level",
        "direction": "down",
        "speed": "slow",
        "scenario": "currently_favourable",
    },
    "reversal_from_low": {
        "base_indicator": "reversal",
        "direction": "up",
        "speed": "slow",
        "scenario": "window_closing",
    },
}

RULE_INDICATORS: tuple[str, ...] = tuple(INDICATOR_META)


def truncate_to(series: dict[str, Series], as_of: dt.date) -> dict[str, Series]:
    """Copy all series through date T, making look-ahead structurally impossible."""
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
) -> list[dict[str, object]]:
    """Return rule-indicator states for every corridor as they looked on date T."""
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

        row_frame = pd.DataFrame([row])
        rate = float(s.values[idx])
        for name in indicators:
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
                    "direction": meta["direction"],
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
    """Export rule scores and product metadata for a past-only feature dataset."""
    parts: list[pd.DataFrame] = []
    for name in indicators:
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
                    "direction": meta["direction"],
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
