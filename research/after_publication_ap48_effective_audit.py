"""Independent policy and prefix-causality audit for AP48."""
from __future__ import annotations

import datetime as dt

import numpy as np

from research.after_publication_ap45_effective_audit import (
    exact_saved,
    finish,
    loadz,
    panel_at,
    same,
    verify_sources,
)
from research.after_publication_ap48_effective import (
    AP47,
    BASE,
    CANDIDATE,
    CONTROLS,
    OUT,
    build_policies,
)


def main():
    verify_sources(OUT)
    panel = panel_at(OUT)
    saved = loadz(OUT / 'outputs.npz')
    old = loadz(BASE / 'outputs.npz')
    ap47 = loadz(AP47 / 'outputs.npz')
    signals, diagnostics = build_policies(panel, old, ap47)
    exact_saved(saved, signals, diagnostics)

    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 6), side='left'))
    future = np.arange(len(dates)) >= cut
    bad_old = {key: value.copy() for key, value in old.items()}
    bad_ap47 = {key: value.copy() for key, value in ap47.items()}
    for key in ('signal__joint_allh_cat_core_ap37_fallback_cap2',
                'policy__core_veto', 'input__fallback_signal'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    for key in [key for key in bad_ap47 if key.startswith('signal__')]:
        bad_ap47[key][future] = ~bad_ap47[key][future].astype(bool)
    bad_signals, bad_diagnostics = build_policies(
        panel, bad_old, bad_ap47)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])

    finish(OUT, CANDIDATE, CONTROLS, panel, saved, signals, diagnostics, {
        'frozen_ap46_primary_veto_exact': True,
        'same_week_ap23_substitution_state_exact': True,
        'sequential_cap_two_exact': True,
    })


if __name__ == '__main__':
    main()
