"""Path consistency, mature library membership, exact neighbors and comparisons."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap3_policy import known_past_sums
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_ap5_learning import available_mask
from research.after_publication_ap7 import OUT, BASE, CONTROLS, INCUMBENT, AP4_CONTROL
from research.after_publication_ap7_paths import FAMILIES, normalized_paths, chronological_split, scenario_summary, SplitPathModel
from research.after_publication_panel import build_outcomes


def main():
    meta = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'publication')
    training = pd.read_csv(OUT / 'training_log.csv')
    log = pd.read_csv(OUT / 'scenario_log.csv')
    assert (log.min_weight >= 0).all() and np.allclose(log.weight_sum, 1.)
    assert (log.effective_scenarios >= 1 - 1e-10).all()
    assert (log.effective_scenarios <= log.n_scenarios + 1e-8).all()
    neighbor_rows, checked_forest = 0, 0
    with np.load(OUT / 'outputs.npz') as saved, np.load(BASE / 'outputs.npz') as previous:
        # NpzFile lazily decompresses on each lookup. Cache once for the full replay.
        saved = {k: saved[k] for k in saved.files}
        previous = {k: previous[k] for k in previous.files}
        paths, core = saved['paths'], saved['core']
        np.testing.assert_array_equal(paths, normalized_paths(series, panel, saved['scale']))
        np.testing.assert_array_equal(saved['known_past_sums'], known_past_sums(series, panel))
        complete = np.isfinite(paths).all(axis=1)
        for h in HORIZONS:
            for prefix in ('y', 'sym', 'forward', 'floor'):
                np.testing.assert_array_equal(saved[prefix + str(h)], outcomes[prefix + str(h)])
            np.testing.assert_array_equal((paths[complete, :h].min(axis=1) >= 0).astype(float), outcomes[f'y{h}'][complete])
        max_utility_error = 0.
        for i in np.flatnonzero(complete):
            _, forward, symmetric, clipped = scenario_summary(paths[i:i+1], np.ones(1), saved['scale'][i], saved['known_past_sums'][i])
            assert clipped == 0
            for j, h in enumerate(HORIZONS):
                max_utility_error = max(max_utility_error, abs(forward[j] - outcomes[f'forward{h}'][i]),
                                        abs(symmetric[j] - outcomes[f'sym{h}'][i]))
        assert max_utility_error < 1e-8
        for key in ('early', 'later', 'dates', 'currencies', 'groups'):
            np.testing.assert_array_equal(saved[key], previous[key])
        for key in CONTROLS:
            np.testing.assert_array_equal(saved['signal__' + key], previous['signal__' + key])
        for row in training.itertuples():
            origin = dt.date.fromisoformat(row.origin)
            mask = available_mask(dates, outcomes['mature20'], origin, complete)
            if row.currency != 'ALL':
                mask &= currencies == row.currency
            ids = np.flatnonzero(mask)
            structure, estimation, boundary = chronological_split(ids, dates, outcomes['mature20'])
            assert (len(ids), len(structure), len(estimation)) == (row.n_library, row.n_structure, row.n_estimation)
            assert str(boundary) == row.split_date
            assert str(max(outcomes['mature20'][ids])) == row.last_library_maturity
            assert str(max(dates[ids])) == row.last_library_date
            assert str(max(outcomes['mature20'][structure])) == row.last_structure_maturity
            if row.currency != 'ALL':
                continue
            month_end = (pd.Timestamp(origin).to_period('M') + 1).start_time.date()
            queries = np.flatnonzero((dates >= origin) & (dates < month_end))
            for key, count in (('global128', 128), ('local32', 32)):
                for i in queries:
                    library = ids if key == 'global128' else ids[currencies[ids] == currencies[i]]
                    assert len(library) >= count  # Actual packet needs no local fallback.
                    X = core[library]
                    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
                    d2 = np.sum(((X - mean) / scale - (core[i] - mean) / scale) ** 2, axis=1)
                    nearest = np.argsort(d2, kind='stable')[:count]
                    expected_ids = library[nearest]
                    weights = np.exp(-d2[nearest] / max(float(np.median(d2[nearest])), 1e-8))
                    weights /= weights.sum()
                    np.testing.assert_array_equal(saved['neighbor_ids__' + key][i], expected_ids)
                    np.testing.assert_allclose(saved['neighbor_weights__' + key][i], weights, atol=1e-13)
                    p, f, s, _ = scenario_summary(paths[expected_ids], weights, saved['scale'][i], saved['known_past_sums'][i])
                    for metric, value in zip(('probability', 'forward', 'symmetric'), (p, f, s)):
                        np.testing.assert_allclose(saved[metric + '__' + key][i], value, atol=1e-10)
                    neighbor_rows += 1
            if row.fitted:
                model = SplitPathModel(core, paths, structure, estimation, 'forest')
                i = queries[0]
                scenario, weight = model.query(core[i])
                p, f, s, _ = scenario_summary(scenario, weight, saved['scale'][i], saved['known_past_sums'][i])
                for metric, value in zip(('probability', 'forward', 'symmetric'), (p, f, s)):
                    np.testing.assert_allclose(saved[metric + '__forest'][i], value, atol=1e-10)
                checked_forest += 1
            print(f'AP7 audit {row.origin}: library and neighbor replay verified', flush=True)
        accuracy = []
        for family in FAMILIES:
            p = saved['probability__' + family]
            finite = np.isfinite(p).all(axis=1)
            assert ((p[finite] >= -1e-12) & (p[finite] <= 1 + 1e-12)).all()
            assert (np.diff(p[finite], axis=1) <= 1e-12).all()
            for period in ('early', 'later'):
                for j, h in enumerate(HORIZONS):
                    m = saved[period] & np.isfinite(saved[f'y{h}'])
                    accuracy.append({'family': family, 'period': period, 'h': h, 'n': int(m.sum()),
                        'brier': float(np.mean((p[m, j] - saved[f'y{h}'][m]) ** 2)),
                        'forward_mse': float(np.mean((saved['forward__' + family][m, j] - saved[f'forward{h}'][m]) ** 2)),
                        'forward_bias': float(np.mean(saved['forward__' + family][m, j] - saved[f'forward{h}'][m]))})
        pd.DataFrame(accuracy).to_csv(OUT / 'probability_and_utility_accuracy.csv', index=False)
        selected = json.loads((OUT / 'selection.json').read_text())['selected']
        pd.concat([selected_bootstrap(panel, saved, selected, b) for b in (20, 50)]).to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
        detail = []
        years = np.array([d.year for d in dates])
        for key in dict.fromkeys([selected, INCUMBENT, AP4_CONTROL]):
            for year in (2024, 2025, 2026):
                for c in sorted(set(currencies)):
                    scope = saved['later'] & (years == year) & (currencies == c)
                    detail.extend({'candidate': key, 'year': year, 'currency': c, **r} for r in
                        scorecard(panel, outcomes, saved['signal__' + key], scope, saved['groups']))
        pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
        keys = [k.removeprefix('signal__') for k in saved if k.startswith('signal__')]
    suffix = '_urgent_cap2'
    pairs = [(AP4_CONTROL, [k for k in keys if k != AP4_CONTROL]),
        ('path_global128_h5' + suffix, ['path_' + k + '_h5' + suffix for k in ('global64', 'wide128', 'local32', 'shrink', 'recent128', 'forest')]),
        ('path_ridge_empirical_h5' + suffix, ['path_ridge_gaussian_h5' + suffix, 'path_ridge_local_h5' + suffix])]
    for family in FAMILIES:
        pairs.append(('path_' + family + '_h5' + suffix,
            ['path_' + family + '_' + variant + suffix for variant in ('mean', 'forward25', 'symmetric25')]))
    common_audit(OUT, keys, pairs)
    log.groupby('model').agg(rows=('row', 'size'), clips=('clipped_log_ratios', 'sum'),
        effective_min=('effective_scenarios', 'min'), effective_mean=('effective_scenarios', 'mean'),
        scenario_min=('n_scenarios', 'min'), scenario_max=('n_scenarios', 'max')).to_csv(OUT / 'scenario_summary.csv')
    (OUT / 'audit_checks.json').write_text(json.dumps({'source_hashes_verified': True,
        'train_log_rows_rebuilt': len(training), 'monthly_origins': int(training.origin.nunique()),
        'neighbor_queries_independently_rebuilt': neighbor_rows, 'forest_monthly_samples_rebuilt': checked_forest,
        'all_paths_and_outcomes_rebuilt': True, 'max_path_utility_error_bps': max_utility_error,
        'same_controls_and_support': True, 'all_survival_curves_monotone': True,
        'all_scenario_weights_checked': True, 'total_clipped_log_ratios': int(log.clipped_log_ratios.sum())}, indent=2))


if __name__ == '__main__':
    main()
