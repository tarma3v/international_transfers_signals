"""Independent labels/masks/risk rows and date-block evidence for AP11-E."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, SEED, adjusted, paired_bootstrap, scorecard
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective import (OUT, BASE, ANCHOR, AP1_EXACT,
    old_ap1_signal, policies)
from research.after_publication_ap11_effective_models import (FAMILIES, conditional_targets,
    effective_scale, eligible_next, margin_scores, margin_targets, person_period)
from research.after_publication_effective_information import SIMPLE
from research.after_publication_panel import build_outcomes


def selected_bootstrap(panel, saved, selected, control, block):
    idx = np.flatnonzero(saved['later'])
    _, date_id = np.unique(panel.date.to_numpy()[idx], return_inverse=True)
    nd = int(date_id.max() + 1)
    rng = np.random.default_rng(SEED)
    weights = []
    for _ in range(1000):
        starts = rng.integers(0, nd, size=int(np.ceil(nd / block)))
        chosen = ((starts[:, None] + np.arange(block)) % nd).ravel()[:nd]
        weights.append(np.bincount(chosen, minlength=nd)[date_id])
    weights = np.array(weights)
    rows = []
    for h in HORIZONS:
        valid = np.isfinite(saved['y' + str(h)][idx])
        for metric in ('lift', 'sym', 'forward'):
            samples, points = [], []
            for key in (selected, control):
                signal = saved['signal__' + key][idx]
                if metric == 'lift':
                    values = saved['y' + str(h)][idx]
                    estimates = np.array([adjusted(values, signal, valid, saved['groups'][idx], w) for w in weights])
                    point = adjusted(values, signal, valid, saved['groups'][idx])
                else:
                    values = saved[metric + str(h)][idx]
                    use = signal & valid & np.isfinite(values)
                    n = weights[:, use].sum(axis=1)
                    estimates = np.divide(weights[:, use] @ values[use], n,
                        out=np.full(len(weights), np.nan), where=n > 0)
                    point = float(values[use].mean())
                samples.append(estimates); points.append(point)
            delta = samples[0] - samples[1]
            rows.append({'candidate': selected, 'control': control, 'h': h, 'metric': metric,
                'block_dates': block, 'draws': len(weights), 'estimate': points[0],
                'ci_lo': float(np.nanquantile(samples[0], .025)), 'ci_hi': float(np.nanquantile(samples[0], .975)),
                'difference': points[0] - points[1], 'delta_lo': float(np.nanquantile(delta, .025)),
                'delta_hi': float(np.nanquantile(delta, .975))})
    return pd.DataFrame(rows)


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
    for key in ('dates', 'currencies', 'early', 'later', 'groups'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])
    eligible = eligible_next(panel)
    np.testing.assert_array_equal(saved['eligible_next'], eligible)
    labels = conditional_targets(outcomes)
    np.testing.assert_array_equal(saved['conditional_labels'], labels)
    names = json.loads((BASE.parent / 'ap10_effective' / 'metadata.json').read_text())['cbr_feature_names']
    scale = effective_scale(saved['announced_features'], names)
    np.testing.assert_array_equal(saved['effective_scale'], scale)
    margins = margin_targets(series, panel, scale)
    np.testing.assert_array_equal(saved['margin_targets'], margins)
    np.testing.assert_array_equal(saved['signal__' + AP1_EXACT], old_ap1_signal(panel))

    logs = pd.read_csv(OUT / 'training_log.csv')
    assert len(logs) == 17 * len(FAMILIES)
    audit_rows = []
    for origin_text in sorted(logs.origin.unique()):
        origin = dt.date.fromisoformat(origin_text)
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        assert np.isfinite(labels[train]).all() and np.isfinite(margins[train]).all()
        shared_hash = hashlib.sha256(np.packbits(shared).tobytes()).hexdigest()
        eligible_hash = hashlib.sha256(np.packbits(train).tobytes()).hexdigest()
        group = logs[logs.origin == origin_text]
        assert set(group.family) == set(FAMILIES)
        assert (group.n_shared_train == shared.sum()).all() and (group.n_eligible_train == train.sum()).all()
        assert (group.shared_mask_sha256 == shared_hash).all() and (group.eligible_mask_sha256 == eligible_hash).all()
        assert (group.last_shared_mature20 == str(max(cap['mature20'][shared]))).all()
        assert (group.last_effective_mature20 == str(max(outcomes['mature20'][shared]))).all()
        design, target, rows, intervals = person_period(saved['features'][train], labels[train])
        risk_hash = hashlib.sha256(np.column_stack([rows, intervals]).astype(np.int32).tobytes()).hexdigest()
        hazard = group[group.family.isin(['hazard_hist', 'hazard_logit'])]
        assert (hazard.n_risk == len(target)).all() and (hazard.n_failures == target.sum()).all()
        assert (hazard.risk_hash == risk_hash).all()
        expected_counts = {c: int((train & (panel.currency.to_numpy() == c)).sum()) for c in CORRIDORS}
        for family in ('local_logit', 'margin_ridge_local'):
            assert json.loads(group[group.family == family].iloc[0].local_counts) == expected_counts
        audit_rows.append({'origin': origin_text, 'shared': int(shared.sum()), 'eligible': int(train.sum()),
            'risk_rows': len(target), 'failures': int(target.sum()), 'last_shared': str(max(cap['mature20'][shared]))})
    pd.DataFrame(audit_rows).to_csv(OUT / 'training_audit.csv', index=False)

    classifier_families = set(FAMILIES) - {'margin_q25', 'margin_ridge_local'}
    accuracy = []
    for family in FAMILIES:
        curve = saved['curve__' + family]
        if family in classifier_families:
            finite = np.isfinite(curve).all(axis=1)
            assert ((curve[finite] >= 0) & (curve[finite] <= 1)).all()
            assert (np.diff(curve[finite], axis=1) <= 1e-12).all()
            assert (curve[eligible & finite, 0] == 1).all() and (curve[~eligible & finite] == 0).all()
            for period in ('early', 'later'):
                for j, h in enumerate(HORIZONS[1:], start=1):
                    mask = saved[period] & eligible & np.isfinite(saved['y' + str(h)])
                    accuracy.append({'family': family, 'period': period, 'h': h, 'n': int(mask.sum()),
                        'brier': float(np.mean((curve[mask, j] - saved['y' + str(h)][mask]) ** 2)),
                        'mean_probability': float(curve[mask, j].mean()),
                        'event_rate': float(saved['y' + str(h)][mask].mean())})
        else:
            prediction = saved['margin_prediction__' + family]
            finite = np.isfinite(prediction).all(axis=1)
            expected = margin_scores(prediction[finite],
                saved['announced_features'][finite, names.index('known_change_z')], eligible[finite])
            np.testing.assert_array_equal(curve[finite], expected)
            assert np.isnan(curve[~finite]).all()
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)

    predictions = {f: saved['curve__' + f] for f in FAMILIES}
    raw, signals = policies(panel, predictions, old)
    for key, value in raw.items():
        np.testing.assert_array_equal(saved['score__' + key], value)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
    keys = list(signals)
    comparisons = []
    for control in (ANCHOR, AP1_EXACT, AP1_EXACT + '_cap2'):
        comparisons.append((control, [k for k in keys if k != control]))
    for family in FAMILIES:
        comparisons.append((family + '_h5_urgent_cap2', [family + '_mean_urgent_cap2']))
    comparisons.extend([('hist_h5_urgent_cap2', ['known75_hist25_h5_urgent_cap2']),
                        ('hazard_hist_h5_urgent_cap2', ['known75_hazard_hist25_h5_urgent_cap2'])])
    common_audit(OUT, keys, comparisons)

    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    sensitivity = pd.concat([selected_bootstrap(panel, saved, selected, control, block)
        for control in (ANCHOR, AP1_EXACT, AP1_EXACT + '_cap2') for block in (20, 50)])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    detail = []
    years = np.array([d.year for d in panel.date])
    focus = list(dict.fromkeys([selected, ANCHOR, AP1_EXACT, AP1_EXACT + '_cap2',
                               'hist_h5_urgent_cap2', 'hazard_hist_h5_urgent_cap2']))
    for key in focus:
        for year in (2024, 2025, 2026):
            for currency in CORRIDORS:
                scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == currency)
                detail.extend({'candidate': key, 'year': year, 'currency': currency, **row} for row in
                    scorecard(panel, outcomes, saved['signal__' + key], scope, saved['groups']))
    pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
    (OUT / 'audit_checks.json').write_text(json.dumps({'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel), 'fit_log_rows_checked': len(logs),
        'quarterly_origins_checked': len(audit_rows), 'conditional_training_excludes_step1': True,
        'all_masks_risk_rows_failures_and_local_counts_match': True,
        'all_old_controls_and_signals_exact': True, 'all_scores_and_signals_rebuilt': True,
        'all_classifier_curves_valid_and_gated': True, 'margin_targets_and_scores_rebuilt': True,
        'historical_receipts_certified': False, 'fresh_holdout': False}, indent=2))


if __name__ == '__main__':
    main()
