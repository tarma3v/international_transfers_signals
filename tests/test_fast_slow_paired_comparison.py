import json
from pathlib import Path

import pandas as pd


OUT = Path('results/research/fast_slow_paired')


def test_fast_slow_uses_exact_common_targets_and_all_horizons():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    score = pd.read_csv(OUT / 'scorecard.csv')
    assert metadata['targets_exact_on_common_support'] is True
    assert metadata['common_rows'] == 1600
    assert set(score.h) == {1, 3, 5, 10, 20}
    assert set(score.target) == {'now_favourable', 'window_closing'}


def test_after_receipt_h5_improvement_has_positive_paired_interval():
    bootstrap = pd.read_csv(OUT / 'paired_bootstrap.csv')
    chosen = bootstrap[
        bootstrap.target.eq('now_favourable')
        & bootstrap.h.eq(5)
    ]
    assert set(chosen.metric) == {'lift', 'symmetric', 'future'}
    assert chosen.ci_lo.gt(0).all()


def test_waiting_cost_is_only_a_market_proxy():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    waiting = pd.read_csv(OUT / 'cny_waiting_cost_summary.csv')
    assert metadata['bank_execution_validated'] is False
    assert waiting.loc[
        waiting.scope.eq('all_common_dates'), 'n_dates'
    ].iloc[0] == 320
