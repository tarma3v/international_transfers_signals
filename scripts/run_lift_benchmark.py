#!/usr/bin/env python3
"""Run walk-forward lift benchmark for candidate signal models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transfer_lift.data import load_rates
from transfer_lift.evaluation import benchmark_models, summarize_overall
from transfer_lift.features import build_dataset
from transfer_lift.models import default_model_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Path to normalized cbr_rates.json")
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 3, 5, 10, 20])
    parser.add_argument(
        "--target",
        default="target_fav",
        choices=["target_fav", "target_close", "target_pub_fav"],
    )
    parser.add_argument("--top-rate", type=float, default=0.15, help="Share of top-score days selected as signals")
    parser.add_argument("--models", nargs="*", default=default_model_names())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rates = load_rates(args.data)
    dataset = build_dataset(rates, horizon=args.horizon)
    metrics = benchmark_models(dataset, model_names=args.models, target_col=args.target, top_rate=args.top_rate)
    overall = summarize_overall(metrics)
    print("\n=== Model summary ===")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n=== Model x corridor metrics ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
