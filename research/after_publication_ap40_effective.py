"""AP40-E: causal weekly optimal-stopping logistic model and router."""
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
from research.after_publication_ap32_effective import (
    AP21_STRICT,
    AP23_BEST,
    AP26_BEST,
    CANDIDATE as AP32_CANDIDATE,
    SIMPLE,
)
from research.after_publication_ap33_effective import CANDIDATE as AP33_CANDIDATE
from research.after_publication_ap34_effective import CANDIDATE as AP34_CANDIDATE
from research.after_publication_ap35_effective import CANDIDATE as AP35_CANDIDATE
from research.after_publication_ap36_effective import CANDIDATE as AP36_CANDIDATE
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap37_effective_models import EXPERTS
from research.after_publication_ap38_effective import CANDIDATE as AP38_CANDIDATE
from research.after_publication_ap39_effective import CANDIDATE as AP39_CANDIDATE
from research.after_publication_ap40_effective_models import (
    FEATURE_NAMES,
    build_stopping_features,
    build_weekly_take_target,
    fit_quarterly_logit,
    optimal_stopping_router,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap40_effective')
BASE = OUT.parent / 'ap39_effective'
AP37 = OUT.parent / 'ap37_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'weekly_optimal_stopping_logit_ap37_fallback_cap2'
FROZEN_CONTROLS = (
    AP37_CANDIDATE, AP38_CANDIDATE, AP39_CANDIDATE, AP33_CANDIDATE,
    AP36_CANDIDATE, AP34_CANDIDATE, AP35_CANDIDATE, AP32_CANDIDATE,
    AP23_BEST, AP26_BEST, AP21_STRICT,
)


def build_policies(panel, core, fallback, prediction, fallback_quality,
                   eligible, old):
    routed = optimal_stopping_router(
        core, fallback, prediction, fallback_quality,
        panel.date.to_numpy(), panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: routed[0]}
    for control in (*FROZEN_CONTROLS, SIMPLE):
        signals[control] = old['signal__' + control]
    assert all(not value[~eligible].any() for value in signals.values())
    return signals, {
        'trailing_rate': routed[1],
        'days_since_signal': routed[2],
        'reason': routed[3],
        'core_veto': routed[4],
    }


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, early, groups)])
    uncertainty = benefit_bootstrap(panel, outcomes, signals, early)
    summary = unknown_summary(frame)
    summary['min_benefit_lower_ci'] = uncertainty[
        uncertainty.h.isin(UNKNOWN_H)].groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [
        weekly_max(panel, signals[key], early) for key in summary.index]
    forward = frame[frame.h.isin(UNKNOWN_H)].pivot(
        index='candidate', columns='h', values='forward_bps')
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    summary = summary.join(clustering_stats(
        panel, signals, early, np.isfinite(outcomes['y5'])))
    summary['rate_cap_pass'] = ((summary.min_rate >= 1) & (summary.max_rate <= 2)
                                & (summary.max_weekly_signals <= 2))
    summary['joint_early_pass'] = (
        summary.rate_cap_pass & (summary.min_lift >= 1.3)
        & (summary.min_benefit_lower_ci > 0)
        & (summary.early_forward_ratio_min >= .8)
        & (summary.empty_complete_months == 0))
    selected = CANDIDATE if bool(
        summary.loc[CANDIDATE, 'joint_early_pass']) else AP37_CANDIDATE
    selection = {
        'selected': selected,
        'selected_simple': SIMPLE,
        'fresh_candidates': 1,
        'fresh_joint_early_pass_count': int(
            summary.loc[CANDIDATE, 'joint_early_pass']),
        'used_registered_fallback': selected != CANDIDATE,
        'registered_fallback': AP37_CANDIDATE,
        'selection_year': 2023,
        'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True,
        'h1_known_validity_only': True,
        'reference': 'today-effective CBR',
        'fresh_holdout': False,
    }
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    uncertainty.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    loaded = []
    for folder in (BASE, AP37):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap37 = loaded
    eligible = old['eligible_next'].astype(bool)
    core = old['input__core_signal'].astype(bool)
    fallback = old['input__fallback_signal'].astype(bool)
    opportunity = eligible & (core | fallback)
    ranks = {name: old['rank__' + name] for name in EXPERTS}
    leader = old['signal__' + AP37_CANDIDATE].astype(bool)
    features = build_stopping_features(
        panel, ranks, core, fallback,
        ap37['local_stratum_precision'], ap37['overall_precision'],
        ap37['policy__trailing_rate'], ap37['policy__days_since_signal'],
        leader)
    target = build_weekly_take_target(
        panel.date.to_numpy(), panel.currency.to_numpy(), opportunity,
        outcomes, cap['mature20'])
    take_y, take_maturity, utility, future_count, wait_gain = target
    prediction, logs = fit_quarterly_logit(
        features, take_y, take_maturity, panel.date.to_numpy(), opportunity)
    signals, diagnostics = build_policies(
        panel, core, fallback, prediction, ap37['quality_gate'].astype(bool),
        eligible, old)
    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal,
                                               old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    valid_target = opportunity & np.isfinite(take_y)
    pd.DataFrame({
        'weekday': [panel.date.iloc[i].isoweekday() for i in np.flatnonzero(valid_target)],
        'take_target': take_y[valid_target],
        'future_opportunity_count': future_count[valid_target],
        'wait_gain': wait_gain[valid_target],
    }).groupby('weekday').agg(
        n=('take_target', 'size'), take_rate=('take_target', 'mean'),
        mean_future_opportunities=('future_opportunity_count', 'mean'),
        mean_wait_gain=('wait_gain', 'mean')).to_csv(OUT / 'target_by_weekday.csv')
    arrays = {key: old[key] for key in ('dates', 'currencies', 'early', 'later',
                                        'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays.update({
        'input__core_signal': core,
        'input__fallback_signal': fallback,
        'input__ap37_signal': leader,
        'input__fallback_quality': ap37['quality_gate'].astype(bool),
        'features': features,
        'feature_names': np.asarray(FEATURE_NAMES),
        'opportunity': opportunity,
        'take_target': take_y,
        'take_target_maturity': np.asarray([
            str(value) if value is not None else '' for value in take_maturity]),
        'utility': utility,
        'future_opportunity_count': future_count,
        'wait_gain': wait_gain,
        'take_probability': prediction,
    })
    for name in EXPERTS:
        arrays['rank__' + name] = ranks[name]
    arrays.update({f'policy__{name}': value
                   for name, value in diagnostics.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [
        DATA,
        *(folder / name for folder in (BASE, AP37)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap40_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP40-E',
        'n_rows': len(panel),
        'model': 'standardized logistic weekly optimal stopping',
        'n_features': len(FEATURE_NAMES),
        'n_fresh_models': 1,
        'n_fresh_policies': 1,
        'n_frozen_controls': len(FROZEN_CONTROLS),
        'selection_horizons': list(UNKNOWN_H),
        'selected': selection['selected'],
        'reference': 'today-effective CBR',
        'h1_known_after_receipt': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }, indent=2))
    print(final[final.h == 5].sort_values(
        'adjusted_lift', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
