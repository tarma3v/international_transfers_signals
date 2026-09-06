"""Independent AP27 backstop state, cadence and uncertainty audit."""
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
from research.after_publication_ap27_effective import (
    AP17_CONTROL,
    AP18_CONTROL,
    AP21_STRICT,
    AP23_BEST,
    AP24,
    AP26_BEST,
    AP26_SELECTED,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    candidate_key,
    policy_inputs,
)
from research.after_publication_ap27_effective_models import POLICIES
from research.after_publication_panel import build_outcomes


SELECTED = candidate_key('s200_cat95_r70')
STRICT = candidate_key('s200_cat95_r60')


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    with np.load(OUT / 'outputs.npz') as source, \
            np.load(BASE / 'outputs.npz') as old_source, \
            np.load(AP24 / 'outputs.npz') as ap24_source:
        saved, old, ap24 = ({key: obj[key] for key in obj.files}
                            for obj in (source, old_source, ap24_source))
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    inputs = policy_inputs(old, ap24['expert__cat'])
    for name, value in inputs.items():
        np.testing.assert_array_equal(saved['input__' + name], value)
    eligible = saved['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(panel, inputs, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, config in POLICIES.items():
        key = candidate_key(name)
        detail = diagnostics[key]
        for field, value in detail.items():
            same(saved[f'{field}__{key}'], value)
        reason = detail['reason']
        assert np.array_equal(signals[key], reason > 0)
        assert (detail['primary_rank'][reason == 1] > .70).all()
        pace = reason == 2
        assert (detail['pace_rank'][pace] > .55).all()
        assert (detail['reserve_rank'][pace] > .70).all()
        backstop = reason == 3
        _, rate, threshold, silence = config
        assert (detail['backstop_rank'][backstop] > threshold).all()
        assert (detail['reserve_rank'][backstop] > .70).all()
        if silence is None:
            assert (detail['trailing_rate'][backstop] < rate).all()
        else:
            assert (detail['days_since'][backstop] >= silence).all()
        assert all(day.day >= 24 for day in dates[reason == 4])

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_inputs = {name: value.copy() for name, value in inputs.items()}
    for j, name in enumerate(('primary', 's200', 's100', 'cat', 'reserve'), start=1):
        bad_inputs[name][cut:] = j * 999. * (-1 if j % 2 else 1)
    bad_signals, bad_detail = build_policies(
        panel, bad_inputs, eligible, old)
    for name in POLICIES:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in POLICIES]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert int(early.loc[fresh, 'joint_early_pass'].sum()) == 5
    assert selection['selected'] == SELECTED
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    strict = []
    for name in fresh:
        row = later.loc[name]
        if (row.min_lift > 2.4 and row.min_rate >= 1 and row.max_rate <= 2
                and row.empty_complete_months == 0):
            strict.append(name)
    assert strict == [STRICT]
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in (SELECTED, STRICT)
        for control in (AP26_BEST, AP26_SELECTED, AP23_BEST, AP21_STRICT,
                        AP18_CONTROL, AP17_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    reason_rows = []
    for key in fresh:
        for period in ('early', 'later'):
            scope = saved[period].astype(bool)
            reason = diagnostics[key]['reason'][scope]
            reason_rows.append({
                'candidate': key, 'period': period,
                'primary': int((reason == 1).sum()),
                'pace': int((reason == 2).sum()),
                'backstop': int((reason == 3).sum()),
                'month': int((reason == 4).sum()),
            })
    pd.DataFrame(reason_rows).to_csv(OUT / 'reason_counts.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': SELECTED,
        'selected_late_min_lift': float(later.loc[SELECTED, 'min_lift']),
        'selected_late_min_rate': float(later.loc[SELECTED, 'min_rate']),
        'strict_later_successes': strict,
        'strict_late_min_lift': float(later.loc[STRICT, 'min_lift']),
        'strict_late_min_rate': float(later.loc[STRICT, 'min_rate']),
        'strict_is_registered_late_challenger_not_early_selected': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'all_five_inputs_and_twenty_ranks_exact': True,
        'all_five_policy_states_and_reasons_rebuilt': True,
        'backstop_rate_rank_reserve_and_silence_gates_checked': True,
        'future_all_input_corruption_prefix_invariant': True,
        'five_fresh_early_joint_passes_confirmed': True,
        'one_strict_late_success_confirmed': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
