"""Independent T5 reconstruction and future-prefix audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap45_effective_audit import same
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_cutoff_frontier import (
    CUTOFFS,
    cutoff_causality_check,
    cutoff_name,
    cutoff_scores,
)
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE
from research.temperature_t5_market_grid import OUT


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    _X, _names, index, series, *_ = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, _digest = load_spot_1530_history()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    saved = loadz(OUT / 'outputs.npz')
    cut_date = dt.date(2025, 1, 6)
    past, future = dates < cut_date, dates >= cut_date
    rebuilt = 0
    for cutoff in CUTOFFS:
        cutoff_causality_check(index, history, references, cutoff, cut_date)
        candidate = cutoff_name(cutoff)
        raw = cutoff_scores(index, history, references, cutoff).astype(float)
        rank = causal_percentiles(raw, dates, currencies, 250, 20)
        same(saved['raw__' + candidate], raw)
        same(saved['rank__' + candidate], rank)
        for h in HORIZONS:
            target = targets['fav_h' + str(h)]
            maturity = target_reach_dates(index, series, h)
            probability, prior, count, _logs = fit_quarterly_calibrator(
                rank, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            prefix = candidate + '__h' + str(h)
            same(saved['prob__' + prefix], probability)
            same(saved['prior__' + prefix], prior)
            np.testing.assert_array_equal(saved['n_train__' + prefix], count)

            changed_rank = rank.copy()
            changed_target = target.copy()
            changed_maturity = maturity.copy()
            changed_rank[future] = np.where(
                np.isfinite(changed_rank[future]), 1. - changed_rank[future],
                np.nan)
            finite = future & np.isfinite(changed_target)
            changed_target[finite] = 1. - changed_target[finite]
            changed_maturity[future] = dt.date(1900, 1, 1)
            altered = fit_quarterly_calibrator(
                changed_rank, changed_target, changed_maturity, dates,
                currencies, min_train_date=MIN_TRAIN_DATE)[0]
            same(probability[past], altered[past])
            rebuilt += 1
    checks = {
        'source_hashes_verified': True,
        'eight_physical_future_candle_checks_passed': True,
        'forty_calibrators_rebuilt_exactly': rebuilt,
        'future_score_label_maturity_corruption_prefix_invariant': True,
        'post_2022_training_boundary_verified': True,
        'tomorrow_cbr_used': False,
        'historical_receipts_certified': False,
    }
    (OUT / 'audit_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
