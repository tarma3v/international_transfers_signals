from research.round6_fixing_cutoff_frontier import CUTOFFS


def test_market_grid_is_strictly_ordered_and_fixed():
    assert [value.isoformat() for value in CUTOFFS] == [
        '10:30:00', '11:30:00', '12:30:00', '13:30:00',
        '14:30:00', '15:00:00', '15:20:00', '15:30:00',
    ]
