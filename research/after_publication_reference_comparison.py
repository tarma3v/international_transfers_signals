"""Re-score every saved AP1--AP9 policy from both CBR reference prices.

No fitting, threshold changes, or later winner selection. Each horizon uses
the intersection of the two reference supports, so the fired dates match.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, sign_policy
from research.after_publication_panel import build_outcomes

ROOT = Path('results/research/after_publication')
OUT = ROOT / 'reference_comparison'
PACKETS = ('ap1/publication', 'ap1/effective', 'ap2', 'ap2_delay20',
           'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'ap9')


def validate_references(series, panel, outcomes):
    """Independent indexing and label checks, including the fully known h1."""
    for row, event in enumerate(panel.itertuples()):
        s = series[event.currency]
        current = int(np.searchsorted(s.dates, event.date, side='right')) - 1
        announced = int(event.announced_index)
        assert current == event.current_index
        assert s.dates[current] <= event.date < s.dates[announced]
        assert current + 1 == announced, 'Do not assume one known step on a different panel'
        assert np.isclose(s.values[current], event.current_price, rtol=1e-13, atol=0)
        assert np.isclose(s.values[announced], event.announced_price, rtol=1e-13, atol=0)
        assert outcomes['effective']['y1'][row] == float(event.announced_price >= event.current_price)
        for reference, start in (('effective', current), ('publication', announced)):
            for h in HORIZONS:
                actual = outcomes[reference]['y' + str(h)][row]
                expected = float(all(s.values[j] >= s.values[start] for j in range(start + 1, start + h + 1))) if start + h < len(s) else np.nan
                assert (np.isnan(actual) and np.isnan(expected)) or actual == expected


def matched_scorecard(panel, outcomes, fired, scope, groups):
    rows = []
    for h in HORIZONS:
        common = scope & np.isfinite(outcomes['effective']['y' + str(h)]) & np.isfinite(outcomes['publication']['y' + str(h)])
        for reference in ('effective', 'publication'):
            row = next(r for r in scorecard(panel, outcomes[reference], fired, common, groups) if r['h'] == h)
            active = common & fired
            row.update(reference=reference,
                known_next_lower_signal_share=float((panel.announced_price.to_numpy()[active] < panel.current_price.to_numpy()[active]).mean()) if active.any() else np.nan)
            rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series = load(DATA)
    all_rows, selected_rows, known_rows, source_rows, checks = [], [], [], [], []
    cache = {}
    for packet in PACKETS:
        folder = ROOT / packet
        panel_path = (folder.parent if packet.startswith('ap1/') else folder) / 'announcement_panel.csv'
        panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
        panel = pd.read_csv(panel_path)
        panel.date = pd.to_datetime(panel.date).dt.date
        signature = tuple(zip(panel.currency, panel.date, panel.current_index, panel.announced_index))
        if signature not in cache:
            outcomes = {ref: build_outcomes(series, panel, ref) for ref in ('effective', 'publication')}
            validate_references(series, panel, outcomes)
            cache[signature] = outcomes
        outcomes = cache[signature]
        selection = json.loads((folder / 'selection.json').read_text())
        with np.load(folder / 'outputs.npz') as z:
            prefix = 'fired_' if packet.startswith('ap1/') else 'signal__'
            signals = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
            early = z['early_scope' if packet.startswith('ap1/') else 'early']
            later = z['later_scope' if packet.startswith('ap1/') else 'later']
            groups = z['groups'] if 'groups' in z.files else np.unique([f'{c}-{d.year}' for c, d in zip(panel.currency, panel.date)], return_inverse=True)[1]
            old_reference = 'effective' if packet == 'ap1/effective' else 'publication'
            for h in HORIZONS:
                np.testing.assert_array_equal(z['y' + str(h)], outcomes[old_reference]['y' + str(h)])
        for scope_name, scope in (('early', early), ('later', later)):
            for key, fired in signals.items():
                rows = [{'packet': packet, 'period': scope_name, 'candidate': key,
                         'previously_selected': key == selection['selected'], **r}
                        for r in matched_scorecard(panel, outcomes, fired, scope, groups)]
                all_rows.extend(rows)
                if key == selection['selected']:
                    selected_rows.extend(rows)
            known = sign_policy(panel.announced_price.to_numpy() - panel.current_price.to_numpy(),
                                panel.date.to_numpy(), panel.currency.to_numpy(), 3)
            known_rows.extend({'packet': packet, 'period': scope_name, 'candidate': 'known_next_not_lower_cd3', **r}
                              for r in matched_scorecard(panel, outcomes, known, scope, groups))
        for file in (panel_path, folder / 'selection.json', folder / 'outputs.npz'):
            source_rows.append({'packet': packet, 'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        checks.append({'packet': packet, 'n_events': len(panel), 'n_policies': len(signals),
                       'index_gap_always1': True, 'effective_h1_fully_known': True,
                       'original_targets_exact': True, 'unchanged_signals': True,
                       'panel_sha256': panel_hash})
        print(packet, len(signals), 'policies rescored on matched dates', flush=True)
    frame = pd.DataFrame(all_rows)
    matching = frame.groupby(['packet', 'period', 'candidate', 'h'])
    assert (matching.n_scope.nunique() == 1).all()
    assert (matching.n_signals.nunique() == 1).all()
    frame.to_csv(OUT / 'all_policies_both_references.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / 'previously_selected_both_references.csv', index=False)
    pd.DataFrame(known_rows).to_csv(OUT / 'known_next_baseline.csv', index=False)
    pd.DataFrame(source_rows).to_csv(OUT / 'source_hashes.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'packet_checks.csv', index=False)
    (OUT / 'metadata.json').write_text(json.dumps({'requested_on': '2026-09-06',
        'scope': 'all saved policies AP1 through AP9, including both AP2 delays and AP1 targets',
        'n_packets': len(PACKETS), 'n_policy_instances': sum(r['n_policies'] for r in checks),
        'n_score_rows': len(frame), 'training_performed': False, 'signals_changed': False,
        'winner_reselected': False, 'same_dates_and_signals_within_h': True,
        'h_units': 'next observations starting at the chosen reference; endpoints differ by one observation',
        'effective_h1': 'already known after receipt, not a prediction of unknown prices',
        'bank_execution_validated': False, 'historical_receipts_certified': False,
        'fresh_holdout': False, 'data_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print(pd.DataFrame(selected_rows).query('period == "later" and h == 5').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
