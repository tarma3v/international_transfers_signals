"""Export the mandatory case signal table at an arbitrary timestamp.

Example before a verified receipt:
    python run_case_output.py --as-of 2026-09-01T18:45:00+03:00 --horizon 5

Example after the ingestion layer recorded a same-day receipt:
    python run_case_output.py --as-of 2026-09-01T18:45:00+03:00 \
      --verified-receipt-at 2026-09-01T18:42:00+03:00 --horizon 5

The default artifact is the T17 observed-availability replay, but case queries
apply the T18 verified-receipt gate. Use `--output` to save a CSV; without it
the same table is written to stdout. Calendar-assumed after-publication replay
is available only through an explicit research flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.transfer_temperature import (
    case_output_runtime_table_as_of,
    case_output_table_as_of,
)


DEFAULT_SNAPSHOTS = Path(
    "results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"
)
REQUIRED_COLUMNS = (
    "date", "corridor", "indicator", "direction", "strength",
    "indicator_speed", "recommended_scenario",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the required no-future signal table for one as_of."
    )
    parser.add_argument(
        "--as-of", required=True,
        help="Timezone-aware ISO timestamp, for example 2026-09-01T21:15:00+03:00",
    )
    parser.add_argument(
        "--horizon", type=int, choices=(1, 3, 5, 10, 20), default=5,
    )
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument(
        "--verified-receipt-at",
        help="Observed timezone-aware receipt timestamp for the as_of Moscow day",
    )
    parser.add_argument(
        "--historical-calendar-assumption", action="store_true",
        help="Research only: allow the artifact's assumed 18:30 receipt",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def build_table(snapshot_path, as_of, horizon, verified_receipt_at=None,
                historical_calendar_assumption=False):
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    if verified_receipt_at is not None and historical_calendar_assumption:
        raise ValueError(
            "verified receipt and historical calendar assumption are exclusive")
    snapshots = pd.read_csv(snapshot_path)
    if historical_calendar_assumption:
        table = case_output_table_as_of(
            snapshots, timestamp.to_pydatetime(), horizon=horizon)
    else:
        table = case_output_runtime_table_as_of(
            snapshots, timestamp.to_pydatetime(), horizon=horizon,
            verified_receipt_at=verified_receipt_at)
    if table.empty:
        raise ValueError("no admissible snapshot at the requested as_of")
    ordered = [*REQUIRED_COLUMNS,
               *(column for column in table.columns
                 if column not in REQUIRED_COLUMNS)]
    return table[ordered]


def main(argv=None):
    args = parse_args(argv)
    table = build_table(
        args.snapshots, args.as_of, args.horizon,
        verified_receipt_at=args.verified_receipt_at,
        historical_calendar_assumption=args.historical_calendar_assumption,
    )
    if args.output is None:
        table.to_csv(sys.stdout, index=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(args.output)


if __name__ == "__main__":
    main()
