import numpy as np

from research.after_publication_ap22_effective_models import PAIRINGS, consensus_pairings


def test_consensus_pairings_are_complete_and_prefix_invariant():
    rng = np.random.default_rng(22)
    n = 220
    values = [rng.normal(size=n) for _ in range(4)]
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 44)
    pairs, rr, lr = consensus_pairings(*values, currencies)
    assert tuple(pairs) == PAIRINGS
    changed = [value.copy() for value in values]
    for j, value in enumerate(changed):
        value[160:] = 1000 * (j + 1)
    pairs_bad, rr_bad, lr_bad = consensus_pairings(*changed, currencies)
    np.testing.assert_array_equal(rr[:160], rr_bad[:160])
    np.testing.assert_array_equal(lr[:160], lr_bad[:160])
    for name in PAIRINGS:
        for original, bad in zip(pairs[name], pairs_bad[name]):
            np.testing.assert_array_equal(original[:160], bad[:160])


def test_weighted_and_agreement_geometries_are_exact():
    rng = np.random.default_rng(2)
    n = 250
    rolling, local, cat, ap12 = [rng.normal(size=n) for _ in range(4)]
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 50)
    pairs, rr, lr = consensus_pairings(rolling, local, cat, ap12, currencies)
    finite = np.isfinite(rr) & np.isfinite(lr)
    np.testing.assert_allclose(pairs['rank75_cat'][0][finite],
                               .75 * rr[finite] + .25 * lr[finite])
    np.testing.assert_allclose(pairs['rank50_cat'][0][finite],
                               .50 * rr[finite] + .50 * lr[finite])
    np.testing.assert_allclose(pairs['rankmin_cat'][0][finite],
                               np.minimum(rr[finite], lr[finite]))
    np.testing.assert_allclose(pairs['rankgeom_cat'][0][finite],
                               np.sqrt(rr[finite] * lr[finite]))
    np.testing.assert_array_equal(pairs['rank75_cat'][1], cat)
    np.testing.assert_array_equal(pairs['rank75_ap12'][1], ap12)
