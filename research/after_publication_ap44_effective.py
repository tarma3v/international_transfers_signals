"""AP44-E: nonlinear meta-CatBoost for mature core quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap32_effective import SIMPLE
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap38_effective import CANDIDATE as AP38_CANDIDATE
from research.after_publication_ap40_effective import CANDIDATE as AP40_CANDIDATE
from research.after_publication_ap40_effective_models import optimal_stopping_router
from research.after_publication_ap44_effective_models import (
    EXTRA_FEATURE_NAMES,
    augment_features,
    fit_quarterly_meta_cat,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap44_effective')
BASE = OUT.parent / 'ap43_effective'
AP37 = OUT.parent / 'ap37_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'meta_y20_cat_core_ap37_fallback_cap2'
CONTROLS = (AP37_CANDIDATE, AP38_CANDIDATE, AP40_CANDIDATE)


def build_policies(panel, old, probability):
    eligible = old['eligible_next'].astype(bool)
    routed = optimal_stopping_router(
        old['input__core_signal'].astype(bool),
        old['input__fallback_signal'].astype(bool), probability,
        old['input__fallback_quality'].astype(bool),
        panel.date.to_numpy(), panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: routed[0]}
    for control in (*CONTROLS, SIMPLE):
        signals[control] = old['signal__' + control]
    return signals, {
        'trailing_rate': routed[1],
        'days_since_signal': routed[2],
        'reason': routed[3],
        'core_veto': routed[4],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(BASE / 'outputs.npz') as source:
        old = {key: source[key] for key in source.files}
    with np.load(AP37 / 'outputs.npz') as source:
        ap37 = {key: source[key] for key in source.files}
    series = load(DATA)
    cap = build_outcomes(series, panel, 'publication')
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    base_features = old['features']
    features = augment_features(base_features, dates, currencies)
    eligible = old['eligible_next'].astype(bool)
    probability, logs, importance = fit_quarterly_meta_cat(
        features, old['y20'], cap['mature20'], dates, eligible)
    signals, diagnostics = build_policies(panel, old, probability)
    outcomes = {kind + str(h): old[kind + str(h)]
                for h in HORIZONS
                for kind in ('y', 'sym', 'forward', 'floor')}

    early_rows = pd.DataFrame([
        {'candidate': key, **row}
        for key, signal in signals.items()
        for row in scorecard(panel, outcomes, signal, old['early'], old['groups'])])
    uncertainty = benefit_bootstrap(panel, outcomes, signals, old['early'])
    early = unknown_summary(early_rows)
    early['min_benefit_lower_ci'] = uncertainty[
        uncertainty.h.isin(UNKNOWN_H)].groupby('candidate').ci_lo.min()
    early['max_weekly_signals'] = [
        weekly_max(panel, signals[key], old['early']) for key in early.index]
    forward = early_rows[early_rows.h.isin(UNKNOWN_H)].pivot(
        index='candidate', columns='h', values='forward_bps')
    early['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    early = early.join(clustering_stats(
        panel, signals, old['early'], np.isfinite(outcomes['y5'])))
    early['rate_cap_pass'] = ((early.min_rate >= 1) & (early.max_rate <= 2)
                              & (early.max_weekly_signals <= 2))
    early['joint_early_pass'] = (
        early.rate_cap_pass & (early.min_lift >= 1.3)
        & (early.min_benefit_lower_ci > 0)
        & (early.early_forward_ratio_min >= .8)
        & (early.empty_complete_months == 0))
    selected = CANDIDATE if bool(
        early.loc[CANDIDATE, 'joint_early_pass']) else AP37_CANDIDATE
    selection = {
        'selected': selected,
        'selected_simple': SIMPLE,
        'fresh_candidates': 1,
        'fresh_joint_early_pass_count': int(
            early.loc[CANDIDATE, 'joint_early_pass']),
        'used_registered_fallback': selected != CANDIDATE,
        'registered_fallback': AP37_CANDIDATE,
        'selection_year': 2023,
        'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True,
        'h1_known_validity_only': True,
        'reference': 'today-effective CBR',
        'fresh_holdout': False,
    }
    early_rows.to_csv(OUT / 'early_all_horizons.csv', index=False)
    uncertainty.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    early.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    final = pd.DataFrame([
        {'candidate': key, **row}
        for key, signal in signals.items()
        for row in scorecard(panel, outcomes, signal, old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    names = tuple(old['feature_names'].astype(str)) + EXTRA_FEATURE_NAMES
    pd.DataFrame({'feature': names, 'mean_importance': importance}).sort_values(
        'mean_importance', ascending=False).to_csv(
            OUT / 'feature_importance.csv', index=False)

    arrays = {key: old[key] for key in old}
    arrays.update({
        'meta_features': features,
        'meta_feature_names': np.asarray(names),
        'meta_y20_probability': probability,
        'meta_feature_importance': importance,
    })
    for name, value in diagnostics.items():
        arrays['policy__' + name] = value
    for key, value in signals.items():
        arrays['signal__' + key] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [
        DATA,
        *(folder / name for folder in (BASE, AP37)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap44_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP44-E',
        'model': 'quarterly mature-only nonlinear y20 meta-CatBoost',
        'n_features': len(names),
        'n_fresh_models': 1,
        'n_fresh_policies': 1,
        'selected': selected,
        'reference': 'today-effective CBR',
        'h1_known_after_receipt': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(final[final.h == 5].sort_values(
        'adjusted_lift', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
