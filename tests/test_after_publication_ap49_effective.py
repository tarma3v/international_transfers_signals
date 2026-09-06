import numpy as np

from research.after_publication_ap49_effective_models import geometric_consensus


def test_geometric_consensus_uses_all_heads_and_preserves_missingness():
    probabilities = {
        3: np.array([.25, .5, np.nan]),
        5: np.array([.25, .5, .5]),
        10: np.array([1., .5, .5]),
        20: np.array([1., .5, .5]),
    }
    result = geometric_consensus(probabilities)
    assert abs(result[0] - .5) < 1e-5
    assert result[1] == .5
    assert np.isnan(result[2])
