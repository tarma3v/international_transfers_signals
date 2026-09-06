"""Independent AP21 dual-expert state, cadence and uncertainty audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap21_effective import (
    AP12,
    AP12_CONTROL,
    AP13,
    AP17_CONTROL,
    AP18,
    AP18_CONTROL,
    AP19,
    AP19_CAT,
    AP20_SELECTED,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    PAIRINGS,
    build_policies,
    candidate_key,
)
from research.after_publication_ap21_effective_models import pairing_scores
from research.after_publication_panel import build_outcomes


ROLL_CAT = 'roll_cat_dual_pace_month24_cap2'
LOCAL_CAT = 'local_cat_dual_pace_month24_cap2'


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    folders = (OUT, BASE, AP12, AP13, AP18, AP19)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap12, ap13, ap18, ap19 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    experts = {
        'rolling': ap13['prediction__extra_roll2'],
        'local': ap13['prediction__local_extra'],
        'ap12': ap12['prediction__extra_h5'],
        'ap18': ap18['prediction__full_recent50'],
        'cat': ap19['prediction__cat_mean_utility'],
    }
    for name, value in experts.items():
        np.testing.assert_array_equal(saved['expert__' + name], value)
    pairs = pairing_scores(experts['rolling'], experts['local'], experts['ap12'],
                           experts['ap18'], experts['cat'])
    for name, (primary, pace) in pairs.items():
        same(saved['primary_score__' + name], primary)
        same(saved['pace_score__' + name], pace)
    eligible = saved['eligible_next'].astype(bool)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, pairs, reserve, eligible, old)
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)
        reason = detail['reason']
        assert np.array_equal(signals[candidate], reason > 0)
        assert (detail['primary_rank'][reason == 1] > .70).all()
        pace = reason == 2
        assert (detail['pace_rank'][pace] > .55).all()
        assert (detail['reserve_rank'][pace] > .70).all()
        assert (detail['trailing_rate'][pace] < 1.).all()
        assert all(day.day >= 24 for day in dates[reason == 3])
        for i in np.flatnonzero(reason == 3):
            prior = ((currencies == currencies[i])
                     & np.array([(day.year, day.month) ==
                                 (dates[i].year, dates[i].month) for day in dates])
                     & (np.arange(len(dates)) < i))
            assert not signals[candidate][prior].any()

    cut = int(np.searchsorted(dates, dt.date(2025, 1, 1), side='left'))
    bad = {name: value.copy() for name, value in experts.items()}
    for j, value in enumerate(bad.values(), start=1):
        value[cut:] = (1000 * j) * (-1 if j % 2 else 1)
    bad_pairs = pairing_scores(bad['rolling'], bad['local'], bad['ap12'],
                               bad['ap18'], bad['cat'])
    bad_signals, bad_detail = build_policies(
        panel, bad_pairs, reserve, eligible, old)
    for name in PAIRINGS:
        key = candidate_key(name)
        for original, changed in zip(pairs[name], bad_pairs[name]):
            same(original[:cut], changed[:cut])
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in PAIRINGS]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in dict.fromkeys([selected, ROLL_CAT, LOCAL_CAT])
        for control in (AP20_SELECTED, AP19_CAT, AP18_CONTROL, AP17_CONTROL,
                        AP12_CONTROL, 'extra_roll2_primary_cap2',
                        'local_extra_primary_cap2')
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    strict = []
    for name in fresh:
        row = later.loc[name]
        if (row.min_lift > 2.4 and row.min_rate >= 1 and row.max_rate <= 2
                and row.empty_complete_months == 0):
            strict.append(name)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selected,
        'selected_min_unknown_h_lift': float(later.loc[selected, 'min_lift']),
        'strict_later_successes': strict,
        'roll_cat_min_unknown_h_lift': float(later.loc[ROLL_CAT, 'min_lift']),
        'roll_cat_min_rate': float(later.loc[ROLL_CAT, 'min_rate']),
        'local_cat_h5_is_diagnostic_only': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'all_five_experts_exact': True, 'all_six_pairings_exact': True,
        'all_policy_states_rebuilt': True,
        'primary_pace_reserve_thresholds_checked': True,
        'month_rescue_first_in_currency_month': True,
        'future_all_expert_corruption_prefix_invariant': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
