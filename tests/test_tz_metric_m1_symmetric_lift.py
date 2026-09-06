import numpy as np

from ml.targets import target_symmetric_local_minimum
from research.tz_metric_m1_symmetric_lift import (
    CANDIDATE,
    OUT,
    SOURCE,
)


def test_symmetric_local_minimum_requires_both_sides_and_accepts_ties():
    values = np.array([3.0, 2.0, 1.0, 1.0, 4.0])
    assert target_symmetric_local_minimum(values, 0, 1) is None
    assert target_symmetric_local_minimum(values, 2, 2) == 1.0
    assert target_symmetric_local_minimum(values, 3, 1) == 1.0
    assert target_symmetric_local_minimum(values, 1, 1) == 0.0


def test_metric_packet_preserves_frozen_candidate_and_inputs():
    assert CANDIDATE == "ap26_core_mature_precision_calendar_fallback_cap2"
    assert (SOURCE / "outputs.npz").exists()
    if OUT.exists():
        assert (OUT / "audit_checks.json").exists()
