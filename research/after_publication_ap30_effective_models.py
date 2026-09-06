"""Fixed causal rank-space interpolation for AP30."""
from __future__ import annotations

import numpy as np

from research.after_publication_ap12_effective_models import causal_rank_percentile


SPECIALIST_WEIGHT = .75


def rank_space_blend(specialist, competence, currencies):
    specialist_rank = causal_rank_percentile(specialist, currencies)
    competence_rank = causal_rank_percentile(competence, currencies)
    score = np.full(len(specialist_rank), np.nan)
    both = np.isfinite(specialist_rank) & np.isfinite(competence_rank)
    score[both] = (SPECIALIST_WEIGHT * specialist_rank[both]
                   + (1. - SPECIALIST_WEIGHT) * competence_rank[both])
    score[np.isfinite(specialist_rank) & ~np.isfinite(competence_rank)] = (
        specialist_rank[np.isfinite(specialist_rank) & ~np.isfinite(competence_rank)])
    score[np.isfinite(competence_rank) & ~np.isfinite(specialist_rank)] = (
        competence_rank[np.isfinite(competence_rank) & ~np.isfinite(specialist_rank)])
    return score, specialist_rank, competence_rank
