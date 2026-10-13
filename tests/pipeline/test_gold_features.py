"""feature_store.build_lap_features: Silver laps -> Gold features."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.cleaner import clean_laps
from src.pipeline.feature_store import (DIRTY_AIR_THRESHOLD_SEC, FUEL_END_KG, FUEL_START_KG, car_ahead_on_track,
                                        build_lap_features)
from tests.helpers import make_bronze_laps


def gold(rows, total_laps=10):
    silver, _ = clean_laps(make_bronze_laps(rows), pd.DataFrame(), 2024, 1, "R")
    return build_lap_features(silver, total_laps).set_index(["driver_code", "lap_number"])


def stint(driver, lap_times, first_lap=1, start=0.0, position=1.0, **extra):
    """Consecutive laps for one driver with the given lap times."""
    rows, t = [], start
    for i, lap_time in enumerate(lap_times):
        rows.append(dict(driver=driver, lap=first_lap + i, start=t, end=t + lap_time, position=position, **extra))
        t += lap_time
    return rows


def test_fuel_starts_full_and_burns_linearly():
    g = gold(stint("AAA", [90] * 10), total_laps=10)
    assert g.loc[("AAA", 1), "fuel_mass_est"] == FUEL_START_KG
    burn = (FUEL_START_KG - FUEL_END_KG) / 10
    assert g.loc[("AAA", 6), "fuel_mass_est"] == pytest.approx(FUEL_START_KG - 5 * burn)
    assert g["fuel_mass_est"].is_monotonic_decreasing


def test_laps_remaining_and_stint_lap():
    # Mediums for 3 laps, stop at the end of lap 3, hards (new set) from lap 4
    rows = [
        dict(driver="AAA", lap=1, start=0, end=90, tyre_life=1.0),
        dict(driver="AAA", lap=2, start=90, end=180, tyre_life=2.0),
        dict(driver="AAA", lap=3, start=180, end=290, tyre_life=3.0, pit_in=285),
        dict(driver="AAA", lap=4, start=290, end=400, tyre_life=1.0, compound="HARD", pit_out=310),
        dict(driver="AAA", lap=5, start=400, end=490, tyre_life=2.0, compound="HARD"),
    ]
    g = gold(rows, total_laps=5)
    assert g["laps_remaining"].tolist() == [4, 3, 2, 1, 0]
    assert g["stint_number"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert g["stint_lap"].tolist() == [1, 2, 3, 1, 2]
    assert g["is_tyre_change"].tolist() == [False, False, True, False, False]


def test_race_gaps_are_measured_at_the_finish_line():
    rows = (stint("AAA", [90, 90], position=1.0) + stint("BBB", [92, 91], position=2.0)
            + stint("CCC", [95, 93], position=3.0))
    g = gold(rows)
    assert pd.isna(g.loc[("AAA", 2), "gap_ahead_sec"])            # the leader has nobody ahead
    assert g.loc[("AAA", 2), "gap_to_leader_sec"] == 0.0
    assert g.loc[("BBB", 2), "gap_ahead_sec"] == pytest.approx(3.0)    # 183 - 180
    assert g.loc[("CCC", 2), "gap_ahead_sec"] == pytest.approx(5.0)    # 188 - 183
    assert g.loc[("CCC", 2), "gap_to_leader_sec"] == pytest.approx(8.0)


def test_on_track_gap_sees_a_lapped_car_that_the_race_gap_misses():
    # AAA leads and is about to lap CCC. On the road, CCC is 1 s ahead of AAA.
    rows = (stint("AAA", [90, 90, 90], position=1.0)
            + stint("BBB", [100, 100, 100], position=2.0)
            + stint("CCC", [134.5, 134.5], position=3.0))       # CCC crosses at 134.5 and 269.0
    g = gold(rows)
    # AAA ends lap 3 at t=270; the previous crossing by anyone is CCC at t=269.
    assert g.loc[("AAA", 3), "gap_on_track_end_sec"] == pytest.approx(1.0)
    assert pd.isna(g.loc[("AAA", 3), "gap_ahead_sec"])          # in the race, nobody is ahead of P1


def test_car_ahead_on_track_names_the_car_and_the_lap_it_is_on():
    rows = (stint("AAA", [90, 90, 90], position=1.0)
            + stint("CCC", [134.5, 134.5], position=3.0))       # CCC crosses at 134.5 and 269.0
    silver, _ = clean_laps(make_bronze_laps(rows), pd.DataFrame(), 2024, 1, "R")
    ahead = car_ahead_on_track(silver).join(silver[["driver_code", "lap_number"]]).set_index(
        ["driver_code", "lap_number"])
    assert ahead.loc[("AAA", 3)].tolist() == [pytest.approx(1.0), "CCC", 2.0]   # 270 - 269.0, CCC on its lap 2
    assert ahead.loc[("CCC", 2)].tolist() == [pytest.approx(89.0), "AAA", 2.0]  # 269.0 - 180, AAA on its lap 2
    assert ahead.loc[("AAA", 1)].isna().all()                   # nobody crossed before


def test_dirty_air_uses_the_average_of_start_and_end_gap():
    lead = stint("AAA", [90, 90, 90], position=1.0)
    follow = stint("BBB", [91.0, 90.0, 92.0], position=2.0)     # gaps at lap end: 1.0, 1.0, 3.0
    g = gold(lead + follow)
    assert g.loc[("BBB", 2), "gap_on_track_start_sec"] == pytest.approx(1.0)
    assert g.loc[("BBB", 2), "gap_on_track_end_sec"] == pytest.approx(1.0)
    assert g.loc[("BBB", 2), "in_dirty_air"]                    # mean 1.0 < threshold
    assert not g.loc[("BBB", 3), "in_dirty_air"]                # mean (1.0 + 3.0) / 2 = 2.0
    assert DIRTY_AIR_THRESHOLD_SEC == 1.5


def test_in_lap_crossing_is_not_a_car_ahead_on_track():
    lead = stint("AAA", [90, 90])
    lead[1]["pit_in"] = 178                                     # AAA crosses the line in the pit lane
    g = gold(lead + stint("BBB", [91, 90], position=2.0))
    assert pd.isna(g.loc[("AAA", 2), "gap_on_track_end_sec"])   # AAA itself is in the pits
    assert g.loc[("BBB", 2), "gap_on_track_end_sec"] == pytest.approx(91.0)   # previous ON-TRACK crossing: t=90


def test_recent_pace_ignores_pit_laps():
    rows = stint("AAA", [100, 90, 91, 115, 92, 93])
    rows[3]["pit_in"] = 380                                     # lap 4 is an in-lap
    g = gold(rows)
    # lap 1 is never clean (standing start), lap 4 is a pit lap
    assert pd.isna(g.loc[("AAA", 2), "last_clean_lap_time_sec"])
    assert g.loc[("AAA", 4), "last_clean_lap_time_sec"] == 91.0
    assert g.loc[("AAA", 5), "last_clean_lap_time_sec"] == 91.0     # skips the pit lap
    assert g.loc[("AAA", 5), "prev_lap_time_sec"] == 115.0          # the literal previous lap
    assert g.loc[("AAA", 5), "roll3_clean_mean_sec"] == pytest.approx((91 + 92) / 2)   # laps 3,4,5 minus the pit lap
    assert g.loc[("AAA", 6), "roll3_clean_mean_sec"] == pytest.approx((92 + 93) / 2)


def test_spread_is_null_not_zero_when_fewer_than_two_clean_laps():
    g = gold(stint("AAA", [100, 90, 91]))
    assert pd.isna(g.loc[("AAA", 2), "roll3_clean_std_sec"])    # only one clean lap so far
    assert g.loc[("AAA", 3), "roll3_clean_std_sec"] > 0


def test_target_is_the_next_lap_and_is_null_on_the_final_lap():
    rows = stint("AAA", [100, 90, 120, 91])
    rows[2]["pit_in"] = 300
    g = gold(rows)
    assert g["target_next_lap_time_sec"].tolist()[:3] == [90.0, 120.0, 91.0]
    assert pd.isna(g.loc[("AAA", 4), "target_next_lap_time_sec"])
    assert g["target_next_is_clean"].tolist() == [True, False, True, False]


def test_target_never_crosses_drivers_or_skips_a_missing_lap():
    rows = stint("AAA", [90, 90]) + stint("BBB", [95, 95], position=2.0) + stint("AAA", [90], first_lap=4, start=270)
    g = gold(rows)
    assert pd.isna(g.loc[("AAA", 2), "target_next_lap_time_sec"])   # lap 3 is missing; lap 4 is not "next"
    assert pd.isna(g.loc[("BBB", 2), "target_next_lap_time_sec"])
    assert pd.isna(g.loc[("AAA", 4), "prev_lap_time_sec"])


def test_features_do_not_depend_on_the_future():
    """
    The leakage test. Build features from the whole race, then again from only
    the laps finished by time T. Every feature for those laps must be identical;
    only the target is allowed to look ahead.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i, drv in enumerate(["AAA", "BBB", "CCC"]):
        rows += stint(drv, list(90 + i + rng.normal(0, 0.5, 12)), position=float(i + 1))
    rows[5]["pit_in"] = 1.0
    full_silver, _ = clean_laps(make_bronze_laps(rows), pd.DataFrame(), 2024, 1, "R")

    cutoff = 600.0
    past_silver = full_silver[full_silver["lap_end_sec"] <= cutoff]
    full = build_lap_features(full_silver, 12).set_index(["driver_code", "lap_number"])
    past = build_lap_features(past_silver, 12).set_index(["driver_code", "lap_number"])

    feature_cols = [c for c in past.columns if not c.startswith("target_")]
    pd.testing.assert_frame_equal(full.loc[past.index, feature_cols], past[feature_cols])


def test_invalid_total_laps_is_rejected():
    silver, _ = clean_laps(make_bronze_laps(stint("AAA", [90, 90])), pd.DataFrame(), 2024, 1, "R")
    with pytest.raises(ValueError):
        build_lap_features(silver, 0)
