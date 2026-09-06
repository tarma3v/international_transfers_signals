import numpy as np

from research.after_publication_ap30_effective_models import (
    SPECIALIST_WEIGHT,
    rank_space_blend,
)


def test_rank_space_blend_has_fixed_weight_and_causal_warmup():
    n = 300
    currencies = np.array(['AMD'] * n)
    specialist = np.arange(n, dtype=float)
    competence = specialist[::-1]
    score, srank, crank = rank_space_blend(
        specialist, competence, currencies)
    assert np.isnan(score[:40]).all()
    usable = np.isfinite(srank) & np.isfinite(crank)
    np.testing.assert_allclose(
        score[usable],
        SPECIALIST_WEIGHT * srank[usable]
        + (1. - SPECIALIST_WEIGHT) * crank[usable])
