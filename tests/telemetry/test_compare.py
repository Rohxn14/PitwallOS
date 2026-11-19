"""
compare.py: two drivers, corner by corner, on the laps both ran clean.

A made-up race first, where every answer is known: three corners, four laps. A gains 0.1 s in corner 1 on every lap by
braking 10 m later; B gains 0.05 s in corner 3 carrying 4 km/h more; corner 2 is level. Lap 4 is not clean for B, so
it is not compared. Then the golden race: Qatar 2024 from OpenF1 (the shared test lake), where the paired lap gap must
be the official one.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.telemetry import compare as cp
from src.telemetry import corner_features as cf
from src.telemetry import track_model as tm
from tests.conftest import QATAR
from tests.telemetry.test_track_model import LUSAIL

CORNERS = [("1", 500.0, 90.0), ("2", 1500.0, 200.0), ("3", 2500.0, 140.0)]     # name, distance, speed at the corner


def made_up_race():
    rows, laps = [], []
    for driver in ("AAA", "BBB"):
        for lap in (1, 2, 3, 4):
            wobble = 0.01 * lap                                    # a little noise, the same for both drivers
            for order, (name, at, speed) in enumerate(CORNERS, start=1):
                time = 10.0 + wobble
                brake, corner_speed = 100.0, speed
                if order == 1 and driver == "AAA":
                    time, brake = time - 0.1, 90.0                 # A brakes 10 m later and gains 0.1 s
                if order == 3 and driver == "BBB":
                    time, corner_speed = time - 0.05, speed + 4    # B carries 4 km/h more and gains 0.05 s
                rows.append({"driver_code": driver, "lap_number": lap, "corner_order": order, "corner": name,
                             "corner_distance_m": at, "segment_time_sec": time, "brake_point_m": brake,
                             "corner_speed_kmh": corner_speed, "exit_speed_kmh": 200.0, "throttle_on_m": 20.0})
            total = sum(r["segment_time_sec"] for r in rows[-3:])
            laps.append({"driver_code": driver, "lap_number": lap, "is_clean_lap": not (driver == "BBB" and lap == 4),
                         "is_complete": True, "lap_time_sec": total, "segments_time_sum_sec": total})
    return pd.DataFrame(rows), pd.DataFrame(laps)


def test_only_laps_both_ran_clean_are_compared():
    _, laps = made_up_race()
    assert cp.paired_laps(laps, "AAA", "BBB") == [1, 2, 3]


def test_every_corner_says_who_gains_how_much_and_how():
    features, laps = made_up_race()
    c = cp.compare(features, laps, "AAA", "BBB")
    k = c.corners.set_index("corner")
    assert k["gap_median_sec"].round(6).tolist() == [-0.1, 0.0, 0.05]          # negative: A faster
    assert k["winner"].tolist() == ["AAA", "", "BBB"]
    assert k.loc["1", "how"] == "brakes 10 m later"
    assert k.loc["3", "how"] == "4 km/h faster at the corner"
    assert k.loc["2", "how"] == "level"
    assert k.loc["1", "a_faster_share"] == 1.0 and k.loc["3", "a_faster_share"] == 0.0
    assert c.corners_won_by_a == 1


def test_the_corners_mean_gaps_add_up_to_the_lap_and_the_build_up_ends_there():
    features, laps = made_up_race()
    c = cp.compare(features, laps, "AAA", "BBB")
    assert c.lap_gap_mean_sec == pytest.approx(-0.05)
    assert c.corners["gap_mean_sec"].sum() == pytest.approx(c.lap_gap_mean_sec)
    assert c.build_up["median_sec"].tolist() == pytest.approx([-0.1, -0.1, -0.05])
    assert c.build_up["distance_m"].iloc[:2].tolist() == [1000.0, 2000.0]      # halfway between the corners
    assert np.isnan(c.build_up["distance_m"].iloc[-1])                         # the last one ends at the line
    # corners under 150 km/h (1 and 3) hold all of the gap: -0.1 + 0.05 of -0.05
    assert c.slow_share() == pytest.approx(1.0)


def test_one_bad_moment_moves_the_mean_not_the_median():
    """A locks up in corner 2 on lap 3 and loses 0.6 s. A typical lap (the median) is still level there; the mean,
    which must add up to the lap, carries the 0.6 s shared over 3 laps."""
    features, laps = made_up_race()
    lockup = (features["driver_code"] == "AAA") & (features["corner_order"] == 2) & (features["lap_number"] == 3)
    features.loc[lockup, "segment_time_sec"] += 0.6
    laps.loc[(laps["driver_code"] == "AAA") & (laps["lap_number"] == 3), "lap_time_sec"] += 0.6
    c = cp.compare(features, laps, "AAA", "BBB")
    k = c.corners.set_index("corner")
    assert k.loc["2", "gap_median_sec"] == pytest.approx(0.0) and k.loc["2", "gap_mean_sec"] == pytest.approx(0.2)
    assert c.corners["gap_mean_sec"].sum() == pytest.approx(c.lap_gap_mean_sec)
    assert c.lap_gap_median_sec == pytest.approx(-0.05)


def test_from_the_other_side_everything_flips():
    features, laps = made_up_race()
    c = cp.compare(features, laps, "BBB", "AAA")
    assert c.lap_gap_mean_sec == pytest.approx(0.05)
    assert c.corners.set_index("corner")["how"].tolist() == ["brakes 10 m later", "level", "4 km/h faster at the corner"]


def test_a_difference_seen_on_only_some_laps_is_not_called_how():
    features, laps = made_up_race()
    later = (features["driver_code"] == "AAA") & (features["corner_order"] == 1) & (features["lap_number"] >= 2)
    features.loc[later, "brake_point_m"] = 100.0                               # A brakes later on lap 1 only
    k = cp.compare(features, laps, "AAA", "BBB").corners.set_index("corner")
    assert k.loc["1", "brake_point_m_a_better"] == pytest.approx(1 / 3)
    assert k.loc["1", "how"] == "no single difference"


def test_two_drivers_without_a_common_clean_lap_or_the_same_driver_twice_are_refused():
    features, laps = made_up_race()
    laps.loc[laps["driver_code"] == "BBB", "is_clean_lap"] = False
    with pytest.raises(cp.NoPairedLaps):
        cp.compare(features, laps, "AAA", "BBB")
    with pytest.raises(ValueError, match="two different"):
        cp.compare(features, laps, "AAA", "AAA")


def test_teammates_are_the_two_drivers_of_a_team():
    silver = pd.DataFrame({"team": ["X", "X", "Y", "Y", "Z"], "driver_code": ["BBB", "AAA", "CCC", "DDD", "EEE"]})
    assert cp.teammates(silver) == [("X", "AAA", "BBB"), ("Y", "CCC", "DDD")]


def test_laps_spent_following_a_car_are_counted_for_each_driver():
    traffic = pd.DataFrame({"driver_code": ["AAA"] * 3 + ["BBB"] * 3, "lap_number": [1, 2, 3] * 2,
                            "gap_on_track_start_sec": [0.8, 2.5, None, 1.49, 1.5, 3.0]})
    assert cp.following_laps(traffic, "AAA", [1, 2, 3]) == 1                   # unknown gap: not counted
    assert cp.following_laps(traffic, "BBB", [1, 2, 3]) == 1                   # under 1.5 s, not at it
    assert cp.following_laps(traffic, "AAA", [2, 3]) == 0                      # only the paired laps


def test_tyre_age_gap_is_what_pairing_by_lap_does_not_equalise():
    silver = pd.DataFrame({"driver_code": ["AAA", "AAA", "BBB", "BBB"], "lap_number": [1, 2, 1, 2],
                           "tyre_life": [5, 6, 2, 3]})
    assert cp.tyre_age_gap(silver, "AAA", "BBB", [1, 2]) == 3.0


# --- the golden race ---------------------------------------------------------------------------------------------------

def test_qatar_verstappen_against_norris_on_their_one_paired_lap(openf1_qatar_lake, tmp_path):
    root = str(tmp_path / "lake")
    shutil.copytree(openf1_qatar_lake, root)
    tm.build_track(*QATAR, root=root, circuit=LUSAIL)
    cf.build_corner_features(*QATAR, root=root)
    c = cp.load_comparison(*QATAR, "VER", "NOR", root=root)
    assert c.laps == [6] and c.notes == ["only 1 paired laps: read with care"]
    assert c.lap_gap_median_sec == pytest.approx(86.502 - 86.683)              # the official lap times, to the ms
    assert c.corners["gap_mean_sec"].sum() == pytest.approx(86.502 - 86.683)   # the corners share it out exactly
    assert c.build_up["median_sec"].iloc[-1] == pytest.approx(86.502 - 86.683)
    assert len(c.corners) == 16 and c.tyre_age_gap == 0
