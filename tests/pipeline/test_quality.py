"""quality.py: per-race counts (the golden numbers are checked through the backfill report)."""

import pytest

from src.pipeline.quality import lap_ranges


@pytest.mark.parametrize("laps, expected", [
    ([], ""),
    ([40], "40"),
    ([1, 2, 3, 4, 34, 35, 36, 40], "1-4, 34-36, 40"),
    ([3, 1, 2, 2, 3], "1-3"),        # every driver's lap 1-3: unordered, repeated
])
def test_lap_ranges(laps, expected):
    assert lap_ranges(laps) == expected
