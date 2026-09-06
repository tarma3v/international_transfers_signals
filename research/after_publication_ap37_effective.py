"""AP37-E: mature disagreement-tail precision gate around frozen AP33."""
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
from research.after_publication_ap37_effective_models import (
    EXPERTS,
    mature_support_precision,
    precision_calendar_router,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap37_effective')
BASE = OUT.parent / 'ap36_effective'
AP33 = OUT.parent / 'ap33_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'ap26_core_mature_precision_calendar_fallback_cap2'
FROZEN_CONTROLS = (
    AP33_CANDIDATE, AP36_CANDIDATE, AP34_CANDIDATE, AP35_CANDIDATE,
    AP32_CANDIDATE, AP23_BEST, AP26_BEST, AP21_STRICT,
)


def build_policies(panel, core, fallback, quality, eligible, old):
    signal, rate, silence, reason = precision_calendar_router(
        core, fallback, quality, panel.date.to_numpy(),
        panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: signal}
    for control in (*FROZEN_CONTROLS, SIMPLE):
        signals[control] = old['signal__' + control]
    assert all(not value[~eligible].any() for value in signals.values())
    return signals, {
        'trailing_rate': rate,
        'days_since_signal': silence,
        'reason': reason,
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
        summary.loc[CANDIDATE, 'joint_early_pass']) else AP33_CANDIDATE
    selection = {
        'selected': selected,
        'selected_simple': SIMPLE,
        'fresh_candidates': 1,
        'fresh_joint_early_pass_count': int(
            summary.loc[CANDIDATE, 'joint_early_pass']),
        'used_registered_fallback': selected != CANDIDATE,
        'registered_fallback': AP33_CANDIDATE,
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
    for folder in (BASE, AP33):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap33 = loaded
    eligible = old['eligible_next'].astype(bool)
    core = ap33['input__core_signal'].astype(bool)
    fallback = old['input__fallback_signal'].astype(bool)
    ranks = {name: old['rank__' + name] for name in EXPERTS}
    precision = mature_support_precision(
        ranks, outcomes, cap['mature20'], panel.date.to_numpy(),
        panel.currency.to_numpy(), eligible, core)
    (support, quality, overall_p, global_p, local_p,
     overall_n, global_n, local_n) = precision
    signals, diagnostics = build_policies(
        panel, core, fallback, quality, eligible, old)
    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal,
                                               old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    arrays = {key: old[key] for key in ('dates', 'currencies', 'early', 'later',
                                        'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays['mature20'] = np.array([str(day) for day in cap['mature20']])
    arrays.update({
        'input__core_signal': core,
        'input__fallback_signal': fallback,
        'support_stratum': support,
        'quality_gate': quality,
        'overall_precision': overall_p,
        'global_stratum_precision': global_p,
        'local_stratum_precision': local_p,
        'overall_count': overall_n,
        'global_stratum_count': global_n,
        'local_stratum_count': local_n,
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
        *(folder / name for folder in (BASE, AP33)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap37_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP37-E',
        'n_rows': len(panel),
        'experts': list(EXPERTS),
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
