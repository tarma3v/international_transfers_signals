import numpy as np

from research.after_publication_ap20_effective_models import SCORE_NAMES, expert_scores


def test_fixed_raw_blends_and_all_names():
    left = np.linspace(.1, .9, 120)
    right = np.linspace(.8, .2, 120)
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 24)
    scores, _, _ = expert_scores(left, right, currencies)
    assert tuple(scores) == SCORE_NAMES
    np.testing.assert_allclose(scores['raw_ap18_75_cat25'], .75 * left + .25 * right)
    np.testing.assert_allclose(scores['raw_ap18_50_cat50'], .50 * left + .50 * right)
    np.testing.assert_allclose(scores['raw_ap18_25_cat75'], .25 * left + .75 * right)


def test_future_expert_corruption_does_not_change_past_scores():
    rng = np.random.default_rng(24)
    left, right = rng.normal(size=160), rng.normal(size=160)
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 32)
    original, lr, rr = expert_scores(left, right, currencies)
    bad_left, bad_right = left.copy(), right.copy()
    bad_left[110:] = 9999
    bad_right[110:] = -9999
    changed, lr_bad, rr_bad = expert_scores(bad_left, bad_right, currencies)
    np.testing.assert_array_equal(lr[:110], lr_bad[:110])
    np.testing.assert_array_equal(rr[:110], rr_bad[:110])
    for name in SCORE_NAMES:
        np.testing.assert_array_equal(original[name][:110], changed[name][:110])


def test_missing_expert_falls_back_to_available_score():
    left = np.linspace(.1, .9, 250)
    right = np.linspace(.8, .2, 250)
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 50)
    left[210] = np.nan
    right[211] = np.nan
    scores, left_rank, right_rank = expert_scores(left, right, currencies)
    assert scores['raw_ap18_50_cat50'][210] == right[210]
    assert scores['raw_ap18_50_cat50'][211] == left[211]
    if np.isfinite(right_rank[210]):
        assert scores['rank_agreement_min'][210] == right_rank[210]
    if np.isfinite(left_rank[211]):
        assert scores['rank_agreement_geom'][211] == left_rank[211]
