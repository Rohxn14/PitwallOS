"""
telemetry_quality.py: how complete and trustworthy each race's telemetry is.

Each measure worked by hand on small made-up tables, then the whole row on the Qatar 2024 fixture race with OpenF1's
telemetry for VER and NOR on laps 1-6 only: the report must say exactly that (18 cars missing, the rest of the race a
hole, the complete laps as long as Lusail).
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import lake, telemetry_quality as quality
from src.pipeline.cleaner import build_silver
from tests.pipeline.test_telemetry_clock import LUSAIL_M, QATAR, add_openf1_fixture


def ticks(driver, lap, n, start=0.0, gap=None, speed=180.0, x_step=50.0):
    """n ticks of one driver on one lap at 10 Hz, driving straight along x; `gap`: tick indices inside a hole."""
    t = start + np.arange(n) / 10
    holes = np.zeros(n, dtype=bool)
    if gap is not None:
        holes[gap] = True
    return pd.DataFrame({"driver_number": driver, "lap_number": lap, "session_time_sec": t, "speed_kmh": speed,
                         "x": np.arange(n) * x_step, "y": 0.0, "is_data_gap": holes,
                         "lap_distance_m": np.cumsum(np.full(n, speed / 3.6 / 10))})


def test_runs_of_true():
    assert quality.runs(np.array([False, True, True, False, True])).tolist() == [2, 1]
    assert quality.runs(np.array([False, False])).tolist() == []
    assert quality.runs(np.array([True, True, True])).tolist() == [3]


def test_holes_are_counted_per_driver():
    tel = pd.concat([ticks("1", 1, 100, gap=slice(10, 13)),            # a hole of 3 ticks = 0.3 s
                     ticks("4", 1, 100, gap=slice(70, 95))])           # a hole of 25 ticks = 2.5 s
    out = quality.hole_summary(tel)
    assert out == {"hole_share": 28 / 200, "holes": 2, "longest_hole_sec": 2.5, "drivers_with_holes": 2}


def test_red_flag_laps_and_the_lap_a_car_stopped_on_are_not_racing():
    laps = pd.DataFrame({"driver_number": ["1", "1", "1", "63", "63"], "lap_number": [1, 2, 3, 1, 2],
                         "is_red": [False, True, False, False, False],
                         "lap_time_sec": [90.0, 400.0, 89.0, 91.0, np.nan]})     # 63 stopped on lap 2
    out = quality.not_racing(laps)
    assert sorted(zip(out["driver_number"], out["lap_number"])) == [("1", 2), ("63", 2)]


def test_a_lap_is_measured_by_speed_and_by_position():
    # 100 ticks at 180 km/h = 50 m/s for 10 s: 500 m by speed; x moves 50 tenths (5 m) a tick: 99 steps = 495 m
    lap = quality.per_lap(ticks("1", 2, 100))
    assert lap[["ticks", "hole_ticks"]].iloc[0].tolist() == [100, 0]
    assert lap["length_m"].iloc[0] == pytest.approx(500.0) and lap["path_m"].iloc[0] == pytest.approx(495.0)
    # the step from one lap to the next is not part of either lap's path
    two = quality.per_lap(pd.concat([ticks("1", 2, 100), ticks("1", 3, 100, start=10.0)]))
    assert two["path_m"].tolist() == pytest.approx([495.0, 495.0])


def test_the_line_speed_is_the_tick_nearest_the_end_of_the_lap():
    tel = ticks("1", 1, 100, speed=300.0)
    tel.loc[50, "speed_kmh"] = 310.0                                   # the tick at 5.0 s
    loop = pd.DataFrame({"DriverNumber": ["1", "1", "1"], "LapNumber": [1, 2, 3],
                         "Time": pd.to_timedelta([5.04, 9.9, 60.0], unit="s"), "SpeedFL": [306.0, 296.0, 290.0]})
    assert quality.line_speeds(tel, loop).tolist() == [4.0, 4.0]      # lap 3 ended after the last tick: no value


def test_the_finish_line_is_where_the_cars_are_when_their_laps_end():
    tel = pd.concat([ticks("1", 1, 100), ticks("4", 1, 100)])        # x = 0, 5, 10 ... m along y = 0
    laps = pd.DataFrame({"driver_number": ["1", "4", "1"], "lap_end_sec": [5.0, 5.2, 9.0],
                         "is_pit_lap": [False, False, True]})        # the pit lap's end is left out
    out = quality.finish_line(tel, laps)
    assert out["line_x_m"] == pytest.approx(255.0) and out["line_y_m"] == 0.0     # ticks 50 and 52: 250 and 260 m
    assert out["line_spread_m"] == pytest.approx(5.0)


def test_a_race_whose_line_is_far_from_its_circuits_other_races_is_named():
    report = pd.DataFrame({"location": ["Las Vegas", "Las Vegas", "Las Vegas", "Lusail"],
                           "line_x_m": [201.3, 205.0, 237.0, 10.0], "line_y_m": [-68.3, -68.0, -68.3, 0.0],
                           "slowest_x_m": [0.0, 0.0, 0.0, 0.0], "slowest_y_m": [0.0, 0.0, 0.0, 0.0]})
    offsets = quality.line_offsets(report)
    assert offsets.iloc[:3].round(1).tolist() == [3.7, 0.3, 32.0] and np.isnan(offsets.iloc[3])   # alone: no judge
    # the whole map moved by (-8.8, +30.1) m (Suzuka 2025): seen from the slowest point, the line has not moved
    moved = report.copy()
    moved.loc[1, ["line_x_m", "line_y_m", "slowest_x_m", "slowest_y_m"]] = [205.0 - 8.8, -68.0 + 30.1, -8.8, 30.1]
    assert quality.line_offsets(moved).iloc[1] == pytest.approx(0.3, abs=0.05)


def test_the_slowest_point_is_found_without_the_clock():
    lap = ticks("1", 2, 100)                                          # x = 0, 5, 10 ... m
    lap.loc[30, "speed_kmh"] = 60.0                                   # the hairpin at x = 150 m
    other = ticks("4", 2, 100)
    other.loc[32, "speed_kmh"] = 62.0                                 # 160 m
    laps = pd.DataFrame({"driver_number": ["1", "4"], "lap_number": [2, 2], "is_clean_lap": [True, True]})
    out = quality.slowest_point(pd.concat([lap, other]), laps)
    assert out["slowest_x_m"] == pytest.approx(155.0) and out["slowest_spread_m"] == pytest.approx(5.0)


def test_a_race_that_passes_every_check_has_no_problems():
    good = {"drivers_missing": "", "racing_hole_share": 0.0, "longest_hole_sec": 0.0, "full_lap_share": 1.0,
            "off_length_share": 0.0, "speed_to_position": 1.0, "line_speed_iqr_kmh": 2.0}
    assert quality.problems(good) == []
    bad = dict(good, drivers_missing="44", racing_hole_share=0.01, longest_hole_sec=30.0, full_lap_share=0.5,
               off_length_share=0.2, speed_to_position=0.95, line_speed_iqr_kmh=12.0)
    assert len(quality.problems(bad)) == 6
    assert quality.problems(dict(good, full_lap_share=np.nan)) == ["only nan % of clean laps complete"]


def test_qatar_with_two_cars_on_six_laps_says_exactly_that(golden_lake, capsys):
    add_openf1_fixture(golden_lake)
    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=True)
    row = quality.race_quality(*QATAR, root=golden_lake)
    assert row["source"] == lake.OPENF1 and row["openf1_session_key"] == 9655 and row["clock_drivers"] == 20
    assert row["drivers"] == 20 and row["drivers_with_telemetry"] == 2
    assert row["drivers_missing"].split() == ["10", "11", "14", "16", "18", "20", "22", "23", "24", "27", "30", "31",
                                              "43", "44", "55", "63", "77", "81"]
    assert row["drivers_with_holes"] == 2 and row["longest_hole_sec"] > 3000      # laps 7-57: no samples in the fixture
    assert 0 < row["full_laps"] <= 4 and row["full_lap_share"] < 0.01
    assert abs(row["lap_length_m"] / LUSAIL_M - 1) < 0.02 and 0.99 <= row["speed_to_position"] <= 1.01
    assert row["line_speed_laps"] >= 10 and row["line_speed_iqr_kmh"] < 10 and row["clock_method"] == "lap 1"   # 12 crossings, 8 behind the SC
    assert row["racing_hole_share"] == row["hole_share"]                        # Qatar had no red flag
    assert np.isfinite(row["line_x_m"]) and row["line_spread_m"] < 10
    assert "no telemetry for car 10 11" in row["problems"] and "holes" in row["problems"]

    assert quality.main(["--season", "2024", "--lake", golden_lake]) == 0
    assert "1 races, 0 pass every check" in capsys.readouterr().out
    saved = lake.read_report(quality.REPORT, golden_lake)
    assert saved[["season", "round"]].values.tolist() == [[2024, 23]]
