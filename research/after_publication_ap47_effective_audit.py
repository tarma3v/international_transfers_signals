"""Independent policy and prefix-causality audit for AP47."""
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
from research.after_publication_ap45_effective import (
    CANDIDATE as AP45_CANDIDATE,
    OUT as AP45_OUT,
)
from research.after_publication_ap47_effective import (
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
    signals, diagnostics = build_policies(panel, old)
    exact_saved(saved, signals, diagnostics)
    ap45 = loadz(AP45_OUT / 'outputs.npz')
    late = saved['later'].astype(bool)
    np.testing.assert_array_equal(
        signals[CANDIDATE][late],
        ap45['signal__' + AP45_CANDIDATE][late])

    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 6), side='left'))
    future = np.arange(len(dates)) >= cut
    bad_old = {key: value.copy() for key, value in old.items()}
    for key in ('input__core_signal', 'input__fallback_signal',
                'input__fallback_quality'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    finite = future & np.isfinite(bad_old['joint_probability'])
    bad_old['joint_probability'][finite] = (
        1. - bad_old['joint_probability'][finite])
    bad_signals, bad_diagnostics = build_policies(panel, bad_old)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])

    finish(OUT, CANDIDATE, CONTROLS, panel, saved, signals, diagnostics, {
        'frozen_ap46_probability_exact': True,
        'registered_rate120_buffer_exact': True,
        'late_ap45_decision_equivalence_exact': True,
    })


if __name__ == '__main__':
    main()
