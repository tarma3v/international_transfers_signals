"""Independent mask, controller and uncertainty audit for AP12-E."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap12_effective import (
    ANCHOR,
    AP10,
    BASE,
    FROZEN_CONTROLS,
    HAZARD,
    MODEL_SCORES,
    OUT,
    build_policies,
)
from research.after_publication_ap12_effective_models import (
    compact_feature_indices,
    first_failure_class,
)
from research.after_publication_panel import build_outcomes


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
    with np.load(OUT / 'outputs.npz') as z, np.load(BASE / 'outputs.npz') as b, \
            np.load(AP10 / 'outputs.npz') as a:
        saved, old, ap10 = ({k: obj[k] for k in obj.files} for obj in (z, b, a))
    for key in ('dates', 'currencies', 'early', 'later', 'groups', 'eligible_next',
                'conditional_labels'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])

    names = json.loads((BASE.parent / 'ap10_effective' / 'metadata.json').read_text())['cbr_feature_names']
    compact = compact_feature_indices(names)
    np.testing.assert_array_equal(saved['compact_feature_indices'], compact)
    np.testing.assert_array_equal(saved['features'], old['features'])

    logs = pd.read_csv(OUT / 'training_log.csv')
    assert len(logs) == 17 * len(MODEL_SCORES)
    audit_rows = []
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    eligible = saved['eligible_next'].astype(bool)
    conditional = saved['conditional_labels']
    failure = first_failure_class(conditional)
    for origin_text in sorted(logs.origin.unique()):
        origin = dt.date.fromisoformat(origin_text)
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        shared_hash = hashlib.sha256(np.packbits(shared).tobytes()).hexdigest()
        eligible_hash = hashlib.sha256(np.packbits(train).tobytes()).hexdigest()
        group = logs[logs.origin == origin_text]
        assert set(group.family) == set(MODEL_SCORES)
        assert (group.n_shared_train == shared.sum()).all()
        assert (group.n_eligible_train == train.sum()).all()
        assert (group.shared_mask_sha256 == shared_hash).all()
        assert (group.eligible_mask_sha256 == eligible_hash).all()
        assert (group.last_shared_mature20 == str(max(cap['mature20'][shared]))).all()
        local = json.loads(group[group.family == 'local_hist_h5'].iloc[0].local_counts)
        expected_local = {c: int((train & (currencies == c)).sum()) for c in CORRIDORS}
        assert local == expected_local
        counts = json.loads(group[group.family == 'first_failure_extra_h5'].iloc[0].class_counts)
        expected_counts = {str(i): int(np.sum(failure[train] == i)) for i in sorted(set(failure[train].astype(int)))}
        assert counts == expected_counts
        audit_rows.append({'origin': origin_text, 'shared': int(shared.sum()),
                           'eligible': int(train.sum()), 'last_shared': str(max(cap['mature20'][shared]))})
    pd.DataFrame(audit_rows).to_csv(OUT / 'training_audit.csv', index=False)

    accuracy = []
    for name in MODEL_SCORES:
        p = saved['prediction__' + name]
        finite = np.isfinite(p)
        assert ((p[finite] >= 0) & (p[finite] <= 1)).all()
        for period in ('early', 'later'):
            mask = saved[period] & eligible & np.isfinite(saved['y5']) & finite
            accuracy.append({'model': name, 'period': period, 'n': int(mask.sum()),
                'brier_h5': float(np.mean((p[mask] - saved['y5'][mask]) ** 2)),
                'mean_probability': float(p[mask].mean()),
                'event_rate': float(saved['y5'][mask].mean())})
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)

    prediction = {k: saved['prediction__' + k] for k in MODEL_SCORES}
    scores, signals = build_policies(panel, prediction, old, ap10, names)
    for key, value in scores.items():
        np.testing.assert_array_equal(saved['score__' + key], value)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
    assert all(not value[~eligible].any() for key, value in signals.items()
               if key != 'ap1_change_z_r25_exact')

    keys = list(signals)
    fresh = [k for k in keys if k not in FROZEN_CONTROLS]
    comparisons = [
        (ANCHOR, fresh),
        (HAZARD, fresh),
        ('ap1_change_z_r25_exact', fresh),
        ('ap1_change_z_r25_exact_cap2', fresh),
    ]
    common_audit(OUT, keys, comparisons)

    selection = json.loads((OUT / 'selection.json').read_text())
    selected = selection['selected']
    extra = 'extra_h5_r30_nogap_cap2'
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in (selected, extra)
        for control in (ANCHOR, HAZARD, 'ap1_change_z_r25_exact', 'ap1_change_z_r25_exact_cap2')
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'quarterly_origins_checked': len(audit_rows),
        'training_log_rows_checked': len(logs),
        'same_shared_maturity_cap': True,
        'conditional_training_only_on_known_non_down': True,
        'compact_feature_subset_rebuilt': True,
        'local_and_first_failure_counts_rebuilt': True,
        'all_scores_and_signals_rebuilt': True,
        'all_non_exact_controls_veto_known_down': True,
        'h1_excluded_from_selection': selection['selection_horizons'] == [3, 5, 10, 20],
        'historical_receipts_certified': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
