"""Independent reconstruction and future-prefix causality audit for AP46."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap44_effective_models import fit_quarterly_meta_cat
from research.after_publication_ap45_effective_audit import (
    exact_saved,
    finish,
    loadz,
    panel_at,
    same,
    verify_sources,
)
from research.after_publication_ap46_effective import (
    BASE,
    CANDIDATE,
    CONTROLS,
    OUT,
    build_policies,
)
from research.after_publication_ap46_effective_models import (
    currency_features,
    joint_survival_target,
)
from research.after_publication_panel import build_outcomes


def main():
    verify_sources(OUT)
    panel = panel_at(OUT)
    dates = panel.date.to_numpy()
    saved = loadz(OUT / 'outputs.npz')
    old = loadz(BASE / 'outputs.npz')
    cap = build_outcomes(load(DATA), panel, 'publication')

    outcomes = {
        'y' + str(h): old['y' + str(h)] for h in (3, 5, 10, 20)
    }
    target = joint_survival_target(outcomes)
    features = currency_features(old['features'], panel.currency.to_numpy())
    probability, logs, importance = fit_quarterly_meta_cat(
        features, target, cap['mature20'], dates,
        old['eligible_next'].astype(bool))
    same(saved['joint_target'], target)
    same(saved['joint_features'], features)
    same(saved['joint_probability'], probability)
    same(saved['joint_feature_importance'], importance)
    np.testing.assert_array_equal(
        saved['joint_target_maturity'],
        np.asarray([str(value) for value in cap['mature20']]))
    pd.testing.assert_frame_equal(
        pd.read_csv(OUT / 'training_log.csv').fillna(''),
        pd.DataFrame(logs).fillna(''), check_dtype=False)

    signals, diagnostics = build_policies(panel, old, probability)
    exact_saved(saved, signals, diagnostics)

    cut_day = dt.date(2025, 1, 6)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    future = np.arange(len(dates)) >= cut
    bad_features = features.copy()
    bad_features[future] = np.where(
        np.isfinite(bad_features[future]), bad_features[future] + 23., -23.)
    bad_target = target.copy()
    finite = future & np.isfinite(bad_target)
    bad_target[finite] = 1. - bad_target[finite]
    bad_maturity = cap['mature20'].copy()
    bad_maturity[future] = dt.date(1900, 1, 1)
    bad_eligible = old['eligible_next'].astype(bool).copy()
    bad_eligible[future] = ~bad_eligible[future]
    bad_probability = fit_quarterly_meta_cat(
        bad_features, bad_target, bad_maturity, dates, bad_eligible)[0]
    same(probability[:cut], bad_probability[:cut])

    bad_old = {key: value.copy() for key, value in old.items()}
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
        'joint_y3_y5_y10_y20_target_exact': True,
        'currency_only_augmentation_exact': True,
        'seventeen_quarterly_meta_cat_fits_exact': True,
        'joint_target_maturity_and_importances_exact': True,
        'annual_and_post2022_features_absent': True,
    })


if __name__ == '__main__':
    main()
