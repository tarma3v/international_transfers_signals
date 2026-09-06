"""Independent AP23 maturity, competence, state and uncertainty audit."""
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
from research.after_publication_ap23_effective import (
    AP12,
    AP12_CONTROL,
    AP17_CONTROL,
    AP18,
    AP18_CONTROL,
    AP21_STRICT,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    PAIRINGS,
    build_policies,
    candidate_key,
)
from research.after_publication_ap23_effective_models import (
    EXPERTS,
    competence_pairings,
)
from research.after_publication_panel import build_outcomes


BEST_FRESH = candidate_key('soft730_pace')


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
    folders = (OUT, BASE, AP12, AP18)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap12, ap18 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    experts = {
        'rolling': old['expert__rolling'],
        'local': old['expert__local'],
        'ap18': ap18['prediction__full_recent50'],
        'cat': old['expert__cat'],
    }
    assert tuple(experts) == EXPERTS
    for name, value in experts.items():
        np.testing.assert_array_equal(saved['expert__' + name], value)
    result = competence_pairings(
        experts, outcomes, cap['mature20'], dates, currencies,
        saved['eligible_next'].astype(bool))
    pairs, ranks, competence, counts, weights, utility = result
    same(saved['unknown_utility'], utility)
    for name, value in ranks.items():
        same(saved['inner_rank__' + name], value)
    for (window, name), value in competence.items():
        same(saved[f'competence{window}__{name}'], value)
        np.testing.assert_array_equal(saved[f'local_count{window}__{name}'],
                                      counts[(window, name, 'local')])
        np.testing.assert_array_equal(saved[f'global_count{window}__{name}'],
                                      counts[(window, name, 'global')])
    for name, value in weights.items():
        same(saved['weight__' + name], value)
        assert np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all()
    for name, (primary, pace) in pairs.items():
        same(saved['primary_score__' + name], primary)
        same(saved['pace_score__' + name], pace)

    eligible = saved['eligible_next'].astype(bool)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, pairs, reserve, eligible, old)
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

    # Change all future expert values and every target not fully mature at the
    # cut. Earlier router values and policy state must remain byte-for-byte equal.
    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_experts = {name: value.copy() for name, value in experts.items()}
    for j, value in enumerate(bad_experts.values(), start=1):
        value[cut:] = j * 1234. * (-1 if j % 2 else 1)
    bad_outcomes = {key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in outcomes.items()}
    unresolved = np.asarray(cap['mature20'], dtype=object) >= (
        cut_day - dt.timedelta(days=2))
    for h in (3, 5, 10, 20):
        value = bad_outcomes[f'y{h}']
        finite = unresolved & np.isfinite(value)
        value[finite] = 1 - value[finite]
    bad_mature = np.asarray(cap['mature20'], dtype=object).copy()
    bad_mature[cut:] = dt.date.max
    bad = competence_pairings(
        bad_experts, bad_outcomes, bad_mature, dates, currencies, eligible)
    for name in PAIRINGS:
        for original, changed in zip(pairs[name], bad[0][name]):
            same(original[:cut], changed[:cut])
    for name in ranks:
        same(ranks[name][:cut], bad[1][name][:cut])
    for key in competence:
        same(competence[key][:cut], bad[2][key][:cut])
    for key in weights:
        same(weights[key][:cut], bad[4][key][:cut])
    bad_signals, bad_detail = build_policies(
        panel, bad[0], reserve, eligible, old)
    for name in PAIRINGS:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in PAIRINGS]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert not early.loc[fresh, 'joint_early_pass'].any()
    assert selection['selected'] == AP21_STRICT
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    best_fresh = str(later.loc[fresh].sort_values(
        ['min_lift', 'mean_lift'], ascending=False).index[0])
    assert best_fresh == BEST_FRESH
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, best_fresh, control, block)
        for control in (AP21_STRICT, AP18_CONTROL, AP17_CONTROL, AP12_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)

    weight_rows = []
    for period in ('early', 'later'):
        scope = saved[period].astype(bool)
        for name, value in weights.items():
            weight_rows.append({
                'period': period, 'weight': name, 'n': int(scope.sum()),
                'mean_first_expert_weight': float(np.mean(value[scope])),
                'q10': float(np.quantile(value[scope], .1)),
                'q50': float(np.quantile(value[scope], .5)),
                'q90': float(np.quantile(value[scope], .9)),
                'first_expert_majority_share': float(np.mean(value[scope] >= .5)),
            })
    pd.DataFrame(weight_rows).to_csv(OUT / 'competence_weight_summary.csv', index=False)
    strict = []
    for name in fresh:
        row = later.loc[name]
        if (row.min_lift > 2.4 and row.min_rate >= 1 and row.max_rate <= 2
                and row.empty_complete_months == 0):
            strict.append(name)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_control_after_no_fresh_early_pass': selection['selected'],
        'fresh_early_pass_count': 0,
        'best_later_fresh': best_fresh,
        'best_later_fresh_min_lift': float(later.loc[best_fresh, 'min_lift']),
        'best_later_fresh_h5_lift': float(pd.read_csv(
            OUT / 'retrospective_all_horizons.csv').query(
                'candidate == @best_fresh and h == 5').iloc[0].adjusted_lift),
        'strict_later_successes': strict,
        'late_success_is_not_early_selected_or_fresh_holdout': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'all_four_experts_exact': True,
        'all_unknown_utility_and_maturity_inputs_rebuilt': True,
        'all_eight_competence_streams_and_counts_exact': True,
        'all_six_weights_and_five_pairings_exact': True,
        'all_policy_states_rebuilt': True,
        'primary_pace_reserve_thresholds_checked': True,
        'future_and_unmatured_expert_target_corruption_prefix_invariant': True,
        'no_fresh_early_joint_pass_confirmed': True,
        'known_down_veto': True, 'h1_excluded_from_router_and_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
