"""The lap-time chart HUD counts tyre changes, not every trip through the pit lane."""

import pytest

chart = pytest.importorskip("src.insights.lap_time_chart_window")

MEDIUM, HARD = 1, 2


def entry(lap, tyre, life, pit_entry=False):
    return {"lap": lap, "tyre": tyre, "tyre_life": life, "is_pit_entry": pit_entry}


def test_safety_car_pass_throughs_are_not_stops():
    # Verstappen, Qatar 2024: one real stop (lap 35), then two passes through the pit lane
    entries = ([entry(l, MEDIUM, l, pit_entry=(l == 35)) for l in range(30, 36)]
               + [entry(36, HARD, 1, True), entry(37, HARD, 2, True), entry(38, HARD, 3)])
    assert chart._count_tyre_changes(entries) == 1


def test_fresh_set_of_the_same_compound_is_a_stop():
    entries = [entry(26, MEDIUM, 10, True), entry(27, MEDIUM, 1)]
    assert chart._count_tyre_changes(entries) == 1


def test_pit_entry_on_the_latest_lap_counts():
    # The car is in the pits right now; the next lap has not happened yet
    assert chart._count_tyre_changes([entry(10, MEDIUM, 10, True)]) == 1
