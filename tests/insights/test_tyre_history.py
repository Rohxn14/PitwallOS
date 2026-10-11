"""Tests for the rewind-safe tyre history behind the tyre strategy window."""

from src.insights.tyre_history import TyreHistory

SOFT, MEDIUM, HARD, INTER = 1, 2, 3, 4


def drive(history, first_lap, last_lap, compound, first_life):
    for lap in range(first_lap, last_lap + 1):
        history.observe(lap, compound, first_life + (lap - first_lap))


def test_one_stint_while_tyres_age_normally():
    h = TyreHistory()
    drive(h, 1, 10, MEDIUM, 1)
    assert h.stints() == [{"tyre": MEDIUM, "start_lap": 1, "end_lap": None}]


def test_compound_change_starts_a_stint():
    h = TyreHistory()
    drive(h, 1, 27, MEDIUM, 1)
    drive(h, 28, 30, INTER, 1)
    assert h.stints() == [
        {"tyre": MEDIUM, "start_lap": 1, "end_lap": 28},
        {"tyre": INTER, "start_lap": 28, "end_lap": None},
    ]


def test_same_compound_stop_is_visible():
    # British GP 2024: several drivers swapped worn inters for fresh inters
    h = TyreHistory()
    drive(h, 19, 27, INTER, 1)
    drive(h, 28, 32, INTER, 1)
    assert [s["start_lap"] for s in h.stints()] == [19, 28]


def test_pass_through_without_new_tyres_is_not_a_stint():
    # Qatar 2024: the safety car led the field through the pit lane; tyre age kept counting
    h = TyreHistory()
    drive(h, 36, 40, HARD, 1)
    assert len(h.stints()) == 1


def test_rewind_forgets_the_future():
    h = TyreHistory()
    drive(h, 1, 27, MEDIUM, 1)
    drive(h, 28, 40, INTER, 1)
    h.observe(20, MEDIUM, 20)        # rewound to lap 20
    assert h.stints() == [{"tyre": MEDIUM, "start_lap": 1, "end_lap": None}]
    assert len(h) == 20


def test_repeated_samples_of_a_lap_do_not_add_anything():
    h = TyreHistory()
    for _ in range(100):
        h.observe(5, SOFT, 5)
    assert len(h) == 1 and len(h.stints()) == 1


def test_forward_seek_over_unseen_laps_keeps_a_continuing_stint():
    h = TyreHistory()
    drive(h, 1, 5, MEDIUM, 1)
    h.observe(15, MEDIUM, 15)        # skipped laps 6-14, same tyres
    assert len(h.stints()) == 1


def test_missing_tyre_age_never_invents_a_stop():
    h = TyreHistory()
    h.observe(1, MEDIUM, None)
    h.observe(2, MEDIUM, None)
    assert len(h.stints()) == 1
