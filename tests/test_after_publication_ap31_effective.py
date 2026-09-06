import numpy as np

from research.after_publication_ap31_effective_models import factorized_score


def test_factorized_score_is_bounded_product():
    p3 = np.array([-.2, .4, .8, 1.2])
    continuation = np.array([.5, .75, .25, 2.])
    np.testing.assert_allclose(
        factorized_score(p3, continuation), [0., .3, .2, 1.])
