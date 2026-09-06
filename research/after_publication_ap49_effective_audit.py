"""Independent four-head reconstruction and prefix-causality audit for AP49."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap45_effective_audit import (
    exact_saved,
    finish,
    loadz,
    panel_at,
    same,
    verify_sources,
)
from research.after_publication_ap46_effective_models import currency_features
from research.after_publication_ap49_effective import (
    BASE,
    CANDIDATE,
    CONTROLS,
    OUT,
    UNKNOWN_H,
    build_policies,
    fit_heads,
)
from research.after_publication_ap49_effective_models import geometric_consensus
from research.after_publication_panel import build_outcomes


def main():
    verify_sources(OUT)
    panel = panel_at(OUT)
    dates = panel.date.to_numpy()
    saved = loadz(OUT / 'outputs.npz')
    old = loadz(BASE / 'outputs.npz')
    cap = build_outcomes(load(DATA), panel, 'publication')
    features = currency_features(old['features'], panel.currency.to_numpy())
    heads, logs, importances = fit_heads(
        features, old, cap, dates, old['eligible_next'].astype(bool))
    probability = geometric_consensus(heads)
    same(saved['multihorizon_features'], features)
    same(saved['multihorizon_probability'], probability)
    for h in UNKNOWN_H:
        same(saved['head_probability_' + str(h)], heads[h])
        same(saved['head_feature_importance_' + str(h)], importances[h])
    pd.testing.assert_frame_equal(
        pd.read_csv(OUT / 'training_log.csv').fillna(''),
        pd.DataFrame(logs).fillna(''), check_dtype=False)
    signals, diagnostics = build_policies(panel, old, probability)
    exact_saved(saved, signals, diagnostics)

    cut = int(np.searchsorted(dates, dt.date(2025, 1, 6), side='left'))
    future = np.arange(len(dates)) >= cut
    bad_features = features.copy()
    bad_features[future] = np.where(
        np.isfinite(bad_features[future]), bad_features[future] + 29., -29.)
    bad_old = {key: value.copy() for key, value in old.items()}
    bad_cap = {key: value.copy() for key, value in cap.items()}
    for h in UNKNOWN_H:
        key = 'y' + str(h)
        finite = future & np.isfinite(bad_old[key])
        bad_old[key][finite] = 1. - bad_old[key][finite]
        bad_cap['mature' + str(h)][future] = dt.date(1900, 1, 1)
    bad_old['eligible_next'][future] = ~bad_old[
        'eligible_next'][future].astype(bool)
    bad_heads = fit_heads(
        bad_features, bad_old, bad_cap, dates,
        bad_old['eligible_next'].astype(bool))[0]
    for h in UNKNOWN_H:
        same(heads[h][:cut], bad_heads[h][:cut])
    bad_probability = geometric_consensus(bad_heads)
    same(probability[:cut], bad_probability[:cut])

    for key in ('input__core_signal', 'input__fallback_signal',
                'input__fallback_quality'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    bad_signals, bad_diagnostics = build_policies(
        panel, bad_old, bad_probability)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])

    finish(OUT, CANDIDATE, CONTROLS, panel, saved, signals, diagnostics, {
        'four_horizon_specific_targets_exact': True,
        'four_horizon_specific_maturities_exact': True,
        'sixty_eight_quarterly_meta_cat_fits_exact': True,
        'geometric_probability_consensus_exact': True,
        'annual_and_post2022_features_absent': True,
    })


if __name__ == '__main__':
    main()
