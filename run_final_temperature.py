"""Query the frozen final temperature model for one timestamp.

Examples:
    python run_final_temperature.py --as-of 2026-09-01T15:45:00+03:00
    python run_final_temperature.py --as-of 2026-09-01T18:45:00+03:00 \
      --verified-receipt-at 2026-09-01T18:42:00+03:00 --currency KZT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml.final_temperature_model import (
    load_manifest,
    score_final_all_horizons_as_of,
    score_final_table_as_of,
    verify_manifest,
)


def parse_args(argv=None):
    manifest = load_manifest()
    parser = argparse.ArgumentParser(
        description="Query the hash-verified final any-time temperature model."
    )
    parser.add_argument("--as-of", help="Timezone-aware ISO timestamp")
    parser.add_argument("--currency", choices=manifest["corridors"])
    parser.add_argument(
        "--selected-horizon", type=int, choices=manifest["horizons"],
        default=manifest["default_horizon"],
    )
    parser.add_argument(
        "--verified-receipt-at",
        help="Observed same-day CBR receipt time; omit until it actually arrives",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify manifest and artifact hashes without scoring",
    )
    args = parser.parse_args(argv)
    if not args.verify_only and args.as_of is None:
        parser.error("--as-of is required unless --verify-only is used")
    return args


def build_payload(as_of, currency=None, selected_horizon=5,
                  verified_receipt_at=None):
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    query = timestamp.to_pydatetime()
    if currency is None:
        rows = score_final_table_as_of(
            query, selected_horizon=selected_horizon,
            verified_receipt_at=verified_receipt_at,
        )
        available = [row for row in rows if row is not None]
        if not available:
            raise ValueError("no admissible snapshot at the requested as_of")
        return available
    result = score_final_all_horizons_as_of(
        currency, query, selected_horizon=selected_horizon,
        verified_receipt_at=verified_receipt_at,
    )
    if result is None:
        raise ValueError("no admissible snapshot at the requested as_of")
    return result


def main(argv=None):
    args = parse_args(argv)
    manifest = verify_manifest()
    if args.verify_only:
        payload = {
            "model_version": manifest["model_version"],
            "status": manifest["status"],
            "artifact_hashes_verified": True,
        }
    else:
        payload = build_payload(
            args.as_of, currency=args.currency,
            selected_horizon=args.selected_horizon,
            verified_receipt_at=args.verified_receipt_at,
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        sys.stdout.write(rendered + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
