"""Independent AP13 masks, router state, cadence and uncertainty audit."""
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
from research.after_publication_ap12_effective_models import compact_feature_indices, extra_factory
from research.after_publication_ap13_effective import (
    ANCHOR,
    BASE,
    EXTRA_CONTROL,
    FROZEN_CONTROLS,
    FEATURE_META,
    HAZARD,
    LOCAL_CONTROL,
    MODEL_SCORES,
    OUT,
    build_policies,
)
from research.after_publication_ap13_effective_models import brier365_router
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
    with np.load(OUT / 'outputs.npz') as z, np.load(BASE / 'outputs.npz') as b:
        saved, old = ({k: obj[k] for k in obj.files} for obj in (z, b))
    for key in ('dates', 'currencies', 'early', 'later', 'groups', 'eligible_next',
                'conditional_labels', 'features'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])

    names = json.loads(FEATURE_META.read_text())['feature_names']
    assert len(names) == saved['features'].shape[1]
    compact = compact_feature_indices(names)
    np.testing.assert_array_equal(saved['compact_feature_indices'], compact)
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    eligible = saved['eligible_next'].astype(bool)
    y5 = saved['conditional_labels'][:, 1]
    extra = old['prediction__extra_h5']
    local = old['prediction__local_hist_h5']

    logs = pd.read_csv(OUT / 'training_log.csv')
    assert len(logs) == 17 * len(MODEL_SCORES)
    audit = []
    for origin_text in sorted(logs.origin.unique()):
        origin = dt.date.fromisoformat(origin_text)
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        group = logs[logs.origin == origin_text]
        assert set(group.family) == set(MODEL_SCORES)
        shared_hash = hashlib.sha256(np.packbits(shared).tobytes()).hexdigest()
        eligible_hash = hashlib.sha256(np.packbits(train).tobytes()).hexdigest()
        assert (group.n_shared_train == shared.sum()).all()
        assert (group.n_eligible_train == train.sum()).all()
        assert (group.shared_mask_sha256 == shared_hash).all()
        assert (group.eligible_mask_sha256 == eligible_hash).all()
        assert (group.last_shared_mature20 == str(max(cap['mature20'][shared]))).all()
        roll2 = train & (dates >= origin - dt.timedelta(days=730))
        roll3 = train & (dates >= origin - dt.timedelta(days=1095))
        assert group[group.family == 'extra_roll2'].iloc[0].n_model_train == roll2.sum()
        assert group[group.family == 'extra_roll3'].iloc[0].n_model_train == roll3.sum()
        age = np.array([(origin - day).days for day in dates], dtype=float)
        decay = np.exp2(-age / 730.)
        np.testing.assert_allclose(group[group.family == 'extra_decay730'].iloc[0].weight_sum,
                                   decay[train].sum(), rtol=1e-12, atol=1e-12)
        local_counts = json.loads(group[group.family == 'local_extra'].iloc[0].local_counts)
        assert local_counts == {c: int((train & (currencies == c)).sum()) for c in CORRIDORS}
        finite = np.isfinite(extra) & np.isfinite(local) & np.isfinite(y5)
        meta_train = train & finite
        prefer = (extra - y5) ** 2 < (local - y5) ** 2
        meta_row = group[group.family == 'router_extra_local'].iloc[0]
        assert meta_row.meta_n == meta_train.sum()
        assert meta_row.meta_prefer_extra == prefer[meta_train].sum()
        audit.append({'origin': origin_text, 'shared': int(shared.sum()),
                      'eligible': int(train.sum()), 'roll2': int(roll2.sum()),
                      'roll3': int(roll3.sum()), 'meta': int(meta_train.sum())})
    pd.DataFrame(audit).to_csv(OUT / 'training_audit.csv', index=False)

    brier, weights, counts = brier365_router(
        dates, outcomes['mature5'], y5, eligible, extra, local)
    np.testing.assert_array_equal(saved['prediction__brier365_extra_local'], brier)
    np.testing.assert_array_equal(saved['brier_weights'], weights)
    np.testing.assert_array_equal(saved['brier_counts'], counts)
    for day in sorted(set(dates)):
        same = dates == day
        assert np.unique(weights[same], axis=0).shape[0] == 1
        assert np.unique(counts[same]).size == 1
    assert np.allclose(weights.sum(axis=1), 1.)

    prediction = {name: saved['prediction__' + name] for name in MODEL_SCORES}
    accuracy = []
    for name, p in prediction.items():
        finite = np.isfinite(p)
        assert ((p[finite] >= 0) & (p[finite] <= 1)).all()
        for period in ('early', 'later'):
            mask = saved[period] & eligible & np.isfinite(y5) & finite
            accuracy.append({'model': name, 'period': period, 'n': int(mask.sum()),
                'brier_h5': float(np.mean((p[mask] - y5[mask]) ** 2)),
                'mean_probability': float(p[mask].mean()), 'event_rate': float(y5[mask].mean())})
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)

    signals = build_policies(panel, prediction, old)
    for key, signal in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], signal)
    assert all(not signal[~eligible].any() for key, signal in signals.items()
               if key != 'ap1_change_z_r25_exact')

    keys = list(signals)
    fresh = [k for k in keys if k not in FROZEN_CONTROLS]
    comparisons = [(control, fresh) for control in
                   (EXTRA_CONTROL, LOCAL_CONTROL, HAZARD, ANCHOR,
                    'ap1_change_z_r25_exact', 'ap1_change_z_r25_exact_cap2')]
    common_audit(OUT, keys, comparisons)

    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    late_h5 = pd.read_csv(OUT / 'retrospective_all_horizons.csv').query('h == 5')
    best_late = str(late_h5[late_h5.candidate.isin(fresh)].sort_values(
        'adjusted_lift', ascending=False).iloc[0].candidate)
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in (selected, best_late, 'extra_roll2_reserve7_cap2')
        for control in (EXTRA_CONTROL, LOCAL_CONTROL, HAZARD, ANCHOR,
                        'ap1_change_z_r25_exact', 'ap1_change_z_r25_exact_cap2')
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)

    # Train-only impurity importance at the latest frozen origin; explanatory only.
    latest_origin = dt.date(2026, 7, 1)
    latest = matured_mask(panel, cap, latest_origin) & eligible
    latest &= dates >= latest_origin - dt.timedelta(days=730)
    importance_model = extra_factory()
    importance_model.fit(saved['features'][latest], y5[latest])
    pd.DataFrame({'feature': names, 'importance': importance_model.feature_importances_}).sort_values(
        'importance', ascending=False).to_csv(OUT / 'latest_roll2_feature_importance.csv', index=False)

    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'quarterly_origins_checked': len(audit), 'training_log_rows_checked': len(logs),
        'old_ap12_expert_predictions_unchanged': True,
        'rolling_decay_local_and_meta_masks_rebuilt': True,
        'brier365_same_date_mature_only_weights_rebuilt': True,
        'all_predictions_bounded': True, 'all_signals_rebuilt': True,
        'all_non_exact_controls_veto_known_down': True,
        'h1_excluded_from_selection': True,
        'best_late_diagnostic': best_late,
        'historical_receipts_certified': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
