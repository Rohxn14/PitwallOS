"""
track_model.py: the reference line every car shares, and the corners found on it.

Made-up circuits first, drawn from straights and arcs, so every corner's place is known exactly: a rectangle with four
round corners, then the same with one corner turned into a double apex. Then the golden race: Qatar 2024 from OpenF1
(two cars' telemetry, in the shared test lake), scored against Lusail's official corner list
(tests/fixtures/multiviewer/circuit_150_2024.json, MultiViewer's, 16 corners).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import contracts, lake
from src.telemetry import track_model as tm
from tests.conftest import LUSAIL_M, QATAR

LUSAIL = json.loads((Path(__file__).parent.parent / "fixtures" / "multiviewer" / "circuit_150_2024.json")
                    .read_text(encoding="utf-8"))


# --- a made-up circuit ---------------------------------------------------------------------------------------------

def draw(pieces):
    """A closed path from (length m, radius m or None for straight, + = left) pieces, one point a metre:
    distance, x, y (tenths of a metre, as the telemetry) and the curvature at each point."""
    heading, x, y, rows = 0.0, 0.0, 0.0, []
    for length, radius in pieces:
        n = int(np.ceil(length))
        step = length / n                                  # exact lengths: the loop closes on itself
        k = 0.0 if radius is None else 1.0 / radius
        for _ in range(n):
            rows.append((x, y, k))
            heading += k * step
            x += step * np.cos(heading)
            y += step * np.sin(heading)
    path = pd.DataFrame(rows, columns=["x", "y", "k"])
    steps = np.hypot(np.diff(path["x"]), np.diff(path["y"]))
    path["distance"] = np.concatenate([[0.0], np.cumsum(steps)])
    path[["x", "y"]] *= 10
    return path


QUARTER = np.pi / 2 * 60                     # a 90-degree corner of radius 60 m: 94.2 m long
RECTANGLE = [(800, None), (QUARTER, 60), (300, None), (QUARTER, 60), (800, None), (QUARTER, 60), (300, None),
             (QUARTER, 60)]
# the first and third corners as double apexes: 39.3 degrees, a gentle 11.5 degrees (radius 300 m), 39.3 degrees.
# Opposite corners of the same shape: the loop still closes.
APEXES = [(np.radians(39.27) * 60, 60), (60, 300), (np.radians(39.27) * 60, 60)]
DOUBLE = RECTANGLE[:1] + APEXES + RECTANGLE[2:5] + APEXES + RECTANGLE[6:]


def corners_of(pieces):
    """Where each corner's middle is, from the start (metres)."""
    out, at = [], 0.0
    for length, radius in pieces:
        if radius is not None and abs(radius) < 100:
            out.append(at + length / 2)
        at += length
    return out


def laps_on(path, cars=3, noise_m=0.1, seed=1):
    """Telemetry and a lap table for one clean lap per car along the path: fast on the straights, slow in the
    corners, a little noise on the position."""
    rng = np.random.default_rng(seed)
    k = np.abs(path["k"].to_numpy())
    wrapped = pd.Series(np.concatenate([k[-150:], k, k[:150]]))         # round the loop, across the line
    bend = wrapped.rolling(150, center=True, min_periods=1).max().to_numpy()[150:-150]
    speed = 300 - 200 * bend / bend.max()
    rows = []
    for car in range(1, cars + 1):
        rows.append(pd.DataFrame({
            "driver_number": str(car), "lap_number": 5,
            "x": path["x"] + rng.normal(0, noise_m * 10, len(path)), "y": path["y"] + rng.normal(0, noise_m * 10, len(path)),
            "speed_kmh": speed, "lap_distance_m": path["distance"] * (0.99 + 0.005 * car),     # each car's own scale
            "is_data_gap": False}))
    telemetry = pd.concat(rows, ignore_index=True)
    laps = pd.DataFrame({"driver_number": [str(c) for c in range(1, cars + 1)], "lap_number": 5, "is_clean_lap": True})
    return laps, telemetry


@pytest.fixture(scope="module")
def rectangle():
    path = draw(RECTANGLE)
    laps, telemetry = laps_on(path)
    return path, tm.reference_line(telemetry, tm.good_laps(laps, telemetry))


def test_the_line_is_the_path_and_as_long_as_it(rectangle):
    path, line = rectangle
    assert tm.lap_length(line) == pytest.approx(len(path), rel=0.003)          # 2,577 m, though every car's own differs
    assert line["laps"].iloc[0] == 3 and len(line) == pytest.approx(len(path) / tm.GRID_STEP_M, abs=1)
    contracts.validate(line, contracts.GOLD_TRACK_LINE, "Gold track line")
    on_path = [np.hypot(path["x"] - x, path["y"] - y).min() / 10 for x, y in zip(line["x"], line["y"])]
    assert np.median(on_path) < 0.3                                            # the noise of each lap averages out


def test_the_four_corners_are_found_both_ways_where_they_are(rectangle):
    _, line = rectangle
    truth = corners_of(RECTANGLE)
    for found in (tm.corners_from_map(line), tm.corners_from_speed(line)):
        assert len(found) == 4
        assert np.abs(np.sort(found["distance_m"].to_numpy()) - truth).max() < 15
    turns = tm.corners_from_map(line)["curvature"]
    assert (turns > 0).all() and turns.mean() == pytest.approx(1 / 60, rel=0.25)   # four left turns, radius ~60 m


def test_a_double_apex_is_two_corners_on_the_map_and_one_for_the_speed():
    path = draw(DOUBLE)
    laps, telemetry = laps_on(path)
    line = tm.reference_line(telemetry, tm.good_laps(laps, telemetry))
    assert tm.lap_length(line) == pytest.approx(path["distance"].iloc[-1], rel=0.003)
    bends = np.sort(tm.corners_from_map(line)["distance_m"].to_numpy())
    truth = corners_of(DOUBLE)                                                 # 2 + 1 + 2 + 1 apexes
    assert len(bends) == len(truth) == 6 and np.abs(bends - truth).max() < 15
    assert len(tm.corners_from_speed(line)) == 4                               # the cars slow down once for each pair


# the long straights with a slight S in them, 6 degrees each way at radius 500 m: no corner (the same on both sides,
# so the loop still closes)
KINK = [(400, None), (np.radians(6) * 500, 500), (np.radians(6) * 500, -500), (400 - 2 * np.radians(6) * 500, None)]
S_BENDS = KINK + RECTANGLE[1:4] + KINK + RECTANGLE[5:]


def test_a_slight_kink_on_a_straight_is_no_corner():
    path = draw(S_BENDS)
    laps, telemetry = laps_on(path)
    line = tm.reference_line(telemetry, tm.good_laps(laps, telemetry))
    found = np.sort(tm.corners_from_map(line)["distance_m"].to_numpy())
    assert len(found) == 4 and np.abs(found - corners_of(S_BENDS)).max() < 15


def test_one_car_running_wide_does_not_move_the_line():
    path = draw(RECTANGLE)
    laps, telemetry = laps_on(path, cars=5)
    wide = (telemetry["driver_number"] == "5") & telemetry["lap_distance_m"].between(300, 500)
    telemetry.loc[wide, "y"] += 200                                         # 20 m off the track for 200 m
    line = tm.reference_line(telemetry, tm.good_laps(laps, telemetry))
    stretch = line[line["distance_m"].between(320, 480)]
    on_path = [np.hypot(path["x"] - x, path["y"] - y).min() / 10 for x, y in zip(stretch["x"], stretch["y"])]
    assert max(on_path) < 1.0                                                  # the median ignores one car


def test_one_tight_bend_is_never_split_by_its_own_noise():
    curvature = np.concatenate([np.zeros(20), np.full(30, 1 / 60) + np.random.default_rng(3).normal(0, 0.001, 30),
                                np.zeros(20)])
    assert len(tm._apexes(curvature[20:50], 5.0, tm.MIN_TURN_DEG)) == 1


# --- the shared distance -----------------------------------------------------------------------------------------------

def test_any_point_gets_the_distance_of_the_nearest_point_of_the_line(rectangle):
    path, line = rectangle
    picks = path.iloc[[0, 500, 1000, 2000]]
    shared = tm.project(line, picks["x"], picks["y"])
    assert shared == pytest.approx([0, 500, 1000, 2000], abs=tm.GRID_STEP_M)
    off = tm.project(line, picks["x"] + 50, picks["y"])                          # five metres off the line: same place
    assert np.abs(off - shared).max() <= 2 * tm.GRID_STEP_M


# --- scoring -----------------------------------------------------------------------------------------------------------

def test_each_official_corner_is_matched_once_closest_first_and_round_the_line():
    official = pd.DataFrame({"number": [1, 2, 3], "distance_m": [100.0, 160.0, 990.0]})
    found = pd.DataFrame({"distance_m": [120.0, 130.0, 5.0, 600.0]})            # 5 m is 15 m from 990 across the line
    s = tm.score(found, official, length=1000.0)
    assert (s["matched"], s["missed"], s["extra"]) == (3, 0, 1)
    assert s["median_error_m"] == 20.0 and s["exact_count"] is False and s["all_right"] is False
    assert tm.loop_gap(5, 990, 1000) == 15 and tm.loop_gap(990, 5, 1000) == 15
    s = tm.score(found.iloc[[0, 2]], official, length=1000.0)
    assert s["missed_numbers"] == "2" and s["found"] == 2


def test_only_whole_clean_laps_of_the_usual_length_make_the_line():
    laps = pd.DataFrame({"driver_number": ["1", "1", "1", "1"], "lap_number": [1, 2, 3, 4],
                         "is_clean_lap": [True, True, True, False]})
    tick = lambda lap, length, gap=False: pd.DataFrame({"driver_number": "1", "lap_number": lap, "x": 0.0, "y": 0.0,
                                                        "lap_distance_m": [0.0, length], "is_data_gap": [False, gap]})
    telemetry = pd.concat([tick(1, 5000), tick(2, 5000, gap=True), tick(3, 5400), tick(4, 5000)])
    telemetry = pd.concat([telemetry, tick(5, 5010)])
    laps = pd.concat([laps, pd.DataFrame({"driver_number": ["1"], "lap_number": [5], "is_clean_lap": [True]})])
    assert tm.good_laps(laps, telemetry)["lap_number"].tolist() == [1, 5]       # a hole, 8 % long, not clean: out


# --- the golden race: Qatar 2024 ------------------------------------------------------------------------------------------

def test_qatar_against_the_official_corners(openf1_qatar_lake, tmp_path):
    import shutil
    shutil.copytree(openf1_qatar_lake, tmp_path / "lake")                     # the shared lake stays read-only
    root = str(tmp_path / "lake")
    row = tm.build_track(*QATAR, root=root, circuit=LUSAIL)
    assert row["good_laps"] == 2                                               # VER's and NOR's lap 6
    assert abs(row["lap_length_m"] / LUSAIL_M - 1) < 0.01                     # Lusail is 5,419 m
    assert row["official_off_line_m"] < 5                                      # the official corners lie on our line
    assert row["map_official"] == 16 and row["map_matched"] >= 14 and row["map_extra"] <= 1
    assert row["speed_matched"] >= 9 and row["speed_extra"] == 0               # what the cars brake for, and only that
    corners = lake.read_table(lake.table_path(lake.GOLD, tm.CORNERS_TABLE, *QATAR, "R", root))
    official = corners[corners["method"] == "official"].sort_values("number")
    assert official["number"].tolist() == list(range(1, 17))
    assert official["speed_kmh"].min() < 120                                   # the slowest corner, turn 6 (~96 km/h)
