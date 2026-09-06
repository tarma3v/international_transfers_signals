"""Verify AP8 information deadlines, compatible controls and paired timing gains."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap2_features import load_market_frames, market_features, append_features
from research.after_publication_ap3_policy import sequential_policy
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_ap4_survival import person_period
from research.after_publication_ap8 import OUT, BASE, CLOCKS, FAMILIES, INCUMBENT, AP4_CONTROL, clock_policies
from research.after_publication_panel import build_features, build_outcomes


def main():
    meta = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    series = load(DATA)
    panel, base_x, names = build_features(series)
    keep = panel.date.to_numpy() >= dt.date(2022, 1, 1)
    panel, base_x = panel[keep].reset_index(drop=True), base_x[keep]
    outcomes = build_outcomes(series, panel, 'publication')
    frames = load_market_frames()
    with np.load(OUT / 'outputs.npz') as packet, np.load(BASE / 'outputs.npz') as old:
        saved, previous = ({k: obj[k] for k in obj.files} for obj in (packet, old))
    for key in ('dates', 'currencies', 'early', 'later', 'groups'):
        np.testing.assert_array_equal(saved[key], previous[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])
    dates = panel.date.to_numpy()
    training = pd.read_csv(OUT / 'training_log.csv', dtype={'clock': str})
    survival = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
    for row in training.itertuples():
        tr = matured_mask(panel, outcomes, dt.date.fromisoformat(row.origin))
        assert tr.sum() == row.n_train
        assert str(max(dates[tr])) == row.last_train_date
        assert str(max(outcomes['mature20'][tr])) == row.last_mature20
        assert hashlib.sha256(np.packbits(tr).tobytes()).hexdigest() == row.mask_sha256
        _, failure, _, _ = person_period(saved['features__' + row.clock][tr], survival[tr])
        assert len(failure) == row.n_at_risk and failure.sum() == row.n_failures
    markets, availability = {}, []
    for clock, cutoff in CLOCKS.items():
        market = market_features(panel, series, frames, 20, cutoff)
        disk = pd.read_csv(OUT / f'market_{clock}.csv')
        disk.date = pd.to_datetime(disk.date).dt.date
        # CSV uses NaN where in-memory source metadata uses None.
        actual = market.astype(object).where(market.notna(), None)
        expected = disk.astype(object).where(disk.notna(), None)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12)
        xx, full_names = append_features(base_x, names, market)
        hx, _ = append_features(base_x, names, disk)
        np.testing.assert_array_equal(xx, saved['features__' + clock])
        np.testing.assert_array_equal(hx, saved['hazard_features__' + clock])
        assert full_names == meta['feature_names']
        decisions = pd.to_datetime(market.decision_at, utc=True)
        assert (pd.to_datetime(panel.source_max_received_at, utc=True) <= decisions).all()
        assert (pd.to_datetime(market.cny_source_received_at, utc=True) <= decisions).all()
        for source in ('cny_max_source', 'local_max_source'):
            source_times = pd.to_datetime(market[source]).dt.tz_localize('Europe/Moscow').dt.tz_convert('UTC')
            usable = source_times.notna()
            assert (source_times[usable] <= decisions[usable]).all()
        curve = saved['survival__' + clock]
        finite = np.isfinite(curve).all(axis=1)
        assert ((curve[finite] >= 0) & (curve[finite] <= 1)).all()
        assert (np.diff(curve[finite], axis=1) <= 1e-12).all()
        cny = np.where(market.cny_last_missing == 0, market.cny_basis_last_z,
                       base_x[:, names.index('known_change_z')])
        raw = clock_policies(panel, cny, saved['hist__' + clock], curve)
        for family, score in raw.items():
            key = f't{clock}_{family}'
            np.testing.assert_array_equal(score, saved['score__' + key])
            np.testing.assert_array_equal(sequential_policy(score, dates, panel.currency.to_numpy(), 'urgent_cap2'),
                                          saved['signal__' + key + '_urgent_cap2'])
        # Independently check selected bar boundaries, including actual end time.
        day_frames = {ticker: dict(tuple(f.groupby(f.begin.dt.date))) for ticker, f in frames.items()}
        for ticker, by_day in day_frames.items():
            for day in sorted(set(dates)):
                if day not in by_day:
                    continue
                frame = by_day[day]
                stop = pd.Timestamp(dt.datetime.combine(day, cutoff))
                start = pd.Timestamp(dt.datetime.combine(day, dt.time(10)))
                nominal = frame.begin + pd.Timedelta(minutes=30)
                actual = frame.end + pd.Timedelta(minutes=20)
                mask = (frame.begin >= start) & (nominal <= stop) & (actual < stop)
                if not mask.any():
                    continue
                selected = frame[mask].sort_values('begin')
                availability.append({'clock': clock, 'ticker': ticker, 'date': str(day),
                    'bars': int(mask.sum()), 'last_bar_begin': selected.begin.iloc[-1].isoformat(),
                    'last_bar_end': selected.end.iloc[-1].isoformat(),
                    'max_nominal_delayed': nominal[mask].max().isoformat(),
                    'max_actual_delayed': actual[mask].max().isoformat(),
                    'decision': stop.isoformat()})
        markets[clock] = market
        print(f'AP8 audit {clock}: full snapshot and policy reconstruction passed', flush=True)
    for clock in ('1850', '1930'):
        for family in FAMILIES:
            for prefix, suffix in (('score__', ''), ('signal__', '_urgent_cap2')):
                np.testing.assert_array_equal(saved[f'{prefix}t{clock}_frozen1830_{family}{suffix}'],
                                              saved[f'{prefix}t1830_{family}{suffix}'])
    for key in (INCUMBENT, AP4_CONTROL):
        np.testing.assert_array_equal(saved['signal__' + key], previous['signal__' + key])
    for clock, market in markets.items():
        for col in ('cny_announced_price', 'cny_announced_effective_date', 'cny_source_received_at'):
            np.testing.assert_array_equal(market[col], markets['1830'][col])
    pd.DataFrame(availability).to_csv(OUT / 'source_deadlines.csv', index=False)
    coverage, accuracy = [], []
    base = markets['1830']
    for clock, market in markets.items():
        for period in ('early', 'later'):
            for currency in ('ALL', *sorted(panel.currency.unique())):
                mask = saved[period] & np.isfinite(saved['y5'])
                if currency != 'ALL':
                    mask &= panel.currency.to_numpy() == currency
                for src in ('cny', 'local'):
                    available = market[src + '_last_missing'].to_numpy() == 0
                    newer = market[src + '_n'].to_numpy() > base[src + '_n'].to_numpy()
                    changed = ~np.isclose(market[src + '_basis_last_z'], base[src + '_basis_last_z'], rtol=1e-12, atol=1e-12)
                    coverage.append({'clock': clock, 'period': period, 'currency': currency, 'source': src,
                        'n': int(mask.sum()), 'available_share': float(available[mask].mean()),
                        'mean_bar_count': float(market.loc[mask, src + '_n'].mean()),
                        'median_age_minutes': float(market.loc[mask & available, src + '_age'].median()),
                        'new_bar_share_vs1830': float(newer[mask].mean()),
                        'changed_basis_share_vs1830': float(changed[mask].mean())})
            for kind in ('hist', 'survival'):
                for j, h in enumerate(HORIZONS):
                    if kind == 'hist' and h != 5:
                        continue
                    pred = saved[f'hist__{clock}'] if kind == 'hist' else saved[f'survival__{clock}'][:, j]
                    mask = saved[period] & np.isfinite(saved[f'y{h}'])
                    accuracy.append({'clock': clock, 'model': kind, 'period': period, 'h': h, 'n': int(mask.sum()),
                        'brier': float(np.mean((pred[mask] - saved[f'y{h}'][mask]) ** 2))})
    pd.DataFrame(coverage).to_csv(OUT / 'timing_coverage.csv', index=False)
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    pd.concat([selected_bootstrap(panel, saved, selected, block) for block in (20, 50)]).to_csv(
        OUT / 'selected_block_sensitivity.csv', index=False)
    detail = []
    years = np.array([d.year for d in dates])
    for key in dict.fromkeys([selected, INCUMBENT, AP4_CONTROL]):
        for year in (2024, 2025, 2026):
            for c in sorted(panel.currency.unique()):
                scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == c)
                detail.extend({'candidate': key, 'year': year, 'currency': c, **row} for row in
                    scorecard(panel, outcomes, saved['signal__' + key], scope, saved['groups']))
    pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
    keys = [k.removeprefix('signal__') for k in saved if k.startswith('signal__')]
    pairs = [(AP4_CONTROL, keys)]
    for family in FAMILIES:
        pairs.append((f't1830_{family}_urgent_cap2',
                      [f't{c}_{family}_urgent_cap2' for c in CLOCKS if c != '1830']))
    common_audit(OUT, keys, pairs)
    (OUT / 'audit_checks.json').write_text(json.dumps({'source_hashes_verified': True,
        'snapshots_rebuilt': len(CLOCKS), 'n_snapshot_events': len(CLOCKS) * len(panel),
        'training_masks_rebuilt': len(training), 'n_origins': int(training.origin.nunique()),
        'same_cbr_references_and_outcomes': True, 'same_evaluation_support': True,
        'all_frozen_policies_exact': True, 'named_controls_exact': True,
        'all_scores_and_signals_rebuilt': True, 'all_survival_curves_valid': True,
        'raw_source_deadline_rows': len(availability), 'historical_receipts_certified': False}, indent=2))


if __name__ == '__main__':
    main()
