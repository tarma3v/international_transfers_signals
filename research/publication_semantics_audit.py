"""Read-only model replay: next effective date is not a publication timestamp.

No model is fitted and no old result is overwritten. The calendar screen uses
the ordinary CBR rule: a fixing becomes effective on the following calendar
day. Subtracting one day gives an inferred announcement date, NOT its time.
Even rows passing this screen still require point-in-time source evidence.
"""
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from research.extended_features import load_or_build
from research.round2_statistical_audit import _fired


OUT = Path("results/research/publication_applicability")


def main():
    _, _, index, series = load_or_build()
    dates = np.array([r[2] for r in index], dtype=object)
    currencies = np.array([r[0] for r in index])
    y = build_targets(series, index)["fav_h5"]
    with Path("results/research/round4/after_publication_outputs.pkl").open("rb") as f:
        outputs = pickle.load(f)
    chosen = pd.read_csv("results/research/round4/after_publication_final_2024_2026_retrospective.csv").iloc[0]
    valid, fired = _fired(outputs[chosen.candidate], (2024, 2025, 2026), dates,
                         currencies, y, float(chosen.selected_rate),
                         int(chosen.selected_rolling), int(chosen.selected_cooldown))
    rows = []
    for r, (currency, pos, day) in enumerate(index):
        if not valid[r] or pos + 1 >= len(series[currency].dates):
            continue
        next_effective = series[currency].dates[pos + 1]
        inferred_publication = next_effective - dt.timedelta(days=1)
        rows.append({"currency": currency, "signal_date": str(day),
                     "weekday": day.strftime("%A"), "next_effective_date": str(next_effective),
                     "inferred_publication_date": str(inferred_publication),
                     "announcement_after_signal_date": bool(inferred_publication > day),
                     "signal": bool(fired[r]), "outcome": float(y[r]),
                     "current_rate": float(series[currency].values[pos]),
                     "next_rate": float(series[currency].values[pos+1])})
    frame = pd.DataFrame(rows)
    active = frame[frame.signal]
    result = {
        "audit_date": "2026-09-06", "candidate": str(chosen.candidate),
        "period": "2024-2026 already explored retrospective",
        "rule": "inferred_publication_date = next_effective_date - one calendar day",
        "limitation": "Calendar consistency check only; actual publication timestamps not present",
        "n_rows": len(frame), "n_signals": len(active),
        "reproduced_pooled_lift": float(active.outcome.mean()/frame.outcome.mean()),
        "rows_requiring_later_announcement": int(frame.announcement_after_signal_date.sum()),
        "signals_requiring_later_announcement": int(active.announcement_after_signal_date.sum()),
        "signals_requiring_later_announcement_share": float(active.announcement_after_signal_date.mean()),
        "signal_weekdays": active.groupby("weekday").size().to_dict(),
        "flagged_signal_weekdays": active[active.announcement_after_signal_date].groupby("weekday").size().to_dict(),
        "examples": active[active.announcement_after_signal_date].head(8).to_dict(orient="records"),
        "conclusion": "Old 2.459 is conditional row-shift result, not verified 18:00 live lift",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT/"calendar_alignment.csv", index=False)
    (OUT/"calendar_alignment_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
