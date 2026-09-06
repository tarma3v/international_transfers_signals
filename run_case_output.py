"""Export the mandatory case signal table at an arbitrary timestamp.

Example:
    python run_case_output.py --as-of 2026-09-01T21:15:00+03:00 --horizon 5

The default artifact is the T17 observed-availability replay. Use `--output`
to save a CSV; without it the same table is written to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.transfer_temperature import case_output_table_as_of


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
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def build_table(snapshot_path, as_of, horizon):
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    snapshots = pd.read_csv(snapshot_path)
    table = case_output_table_as_of(
        snapshots, timestamp.to_pydatetime(), horizon=horizon)
    if table.empty:
        raise ValueError("no admissible snapshot at the requested as_of")
    ordered = [*REQUIRED_COLUMNS,
               *(column for column in table.columns
                 if column not in REQUIRED_COLUMNS)]
    return table[ordered]


def main(argv=None):
    args = parse_args(argv)
    table = build_table(args.snapshots, args.as_of, args.horizon)
    if args.output is None:
        table.to_csv(sys.stdout, index=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(args.output)


if __name__ == "__main__":
    main()
