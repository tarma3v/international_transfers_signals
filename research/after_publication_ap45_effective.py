"""AP45-E: dual-agreement veto with a conservative rate floor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap32_effective import SIMPLE
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap38_effective import CANDIDATE as AP38_CANDIDATE
from research.after_publication_ap40_effective import CANDIDATE as AP40_CANDIDATE
from research.after_publication_ap40_effective_models import optimal_stopping_router
from research.after_publication_ap44_effective import CANDIDATE as AP44_CANDIDATE


OUT = Path('results/research/after_publication/ap45_effective')
BASE = OUT.parent / 'ap44_effective'
AP38 = OUT.parent / 'ap38_effective'
UNKNOWN_H = (3, 5, 10, 20)
RATE_FLOOR = 1.20
CANDIDATE = 'dual_gate_meta_precision_rate120_cap2'
CONTROLS = (AP37_CANDIDATE, AP38_CANDIDATE, AP40_CANDIDATE, AP44_CANDIDATE)


def build_policies(panel, old, ap38):
    eligible = old['eligible_next'].astype(bool)
    precision_gate = ap38['core__quality_gate'].astype(bool)
    meta = old['meta_y20_probability'].astype(float)
    combined = np.where(precision_gate, 1., meta)
    routed = optimal_stopping_router(
        old['input__core_signal'].astype(bool),
        old['input__fallback_signal'].astype(bool), combined,
        old['input__fallback_quality'].astype(bool),
        panel.date.to_numpy(), panel.currency.to_numpy(), eligible,
        rate_floor=RATE_FLOOR)
    signals = {CANDIDATE: routed[0]}
    for control in (*CONTROLS, SIMPLE):
        signals[control] = old['signal__' + control]
    return signals, {
        'trailing_rate': routed[1],
        'days_since_signal': routed[2],
        'reason': routed[3],
        'core_veto': routed[4],
        'combined_probability': combined,
        'precision_gate': precision_gate,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(BASE / 'outputs.npz') as source:
        old = {key: source[key] for key in source.files}
    with np.load(AP38 / 'outputs.npz') as source:
        ap38 = {key: source[key] for key in source.files}
    signals, diagnostics = build_policies(panel, old, ap38)
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
    arrays = {key: old[key] for key in old}
    for name, value in diagnostics.items():
        arrays['policy__' + name] = value
    for key, value in signals.items():
        arrays['signal__' + key] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [
        DATA,
        *(folder / name for folder in (BASE, AP38)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap45_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP45-E',
        'model': 'AP38 and AP44 conservative dual-gate veto',
        'rate_floor': RATE_FLOOR,
        'n_fresh_models': 0,
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
