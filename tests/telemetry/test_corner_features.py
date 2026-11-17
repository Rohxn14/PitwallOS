"""
corner_features.py: what each car did in each corner of each lap.

A made-up lap first, driven round the made-up rectangle of test_track_model.py with a speed, brake and throttle chosen
by hand, so every feature has a known answer: the cars brake 150 m before corners 1, 2 and 4 from 288 to 108 km/h, and
only lift for corner 3. Then the golden race: Qatar 2024 from OpenF1 on Lusail's 16 official corners, where the
segment times of a lap must add up to its official lap time (which the code never reads) and turn 6 is the slowest.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.pipeline import contracts, lake
from src.telemetry import corner_features as cf
from src.telemetry import track_model as tm
from tests.conftest import QATAR
from tests.telemetry.test_track_model import LUSAIL, RECTANGLE, corners_of, draw, laps_on

FAST, SLOW, ZONE = 80.0, 30.0, 150.0          # m/s on the straights and at the apex; braking and accelerating distance
LIFT_ONLY = 3                                 # the corner taken without the brakes
SLOW_THROTTLE_M = 30.0                        # after each apex the throttle stays at 20 % for this far


def speed_at(d, apexes):
    """m/s at distance d: FAST, falling in a straight line to SLOW at each apex over ZONE metres and rising again."""
    gap = np.min(np.abs(np.subtract.outer(d, apexes)), axis=1)
    return SLOW + (FAST - SLOW) * np.minimum(gap / ZONE, 1.0)


def drive(path, apexes, start_sec=1000.0):
    """One lap of 10 Hz telemetry along the path: a fine simulation (1 ms), then every 100th step."""
    length = path["distance"].iloc[-1]
    d, t, rows = 0.0, 0.0, []
    while d < length:
        rows.append((t, d))
        d += speed_at(np.array([d]), apexes)[0] * 0.001
        t += 0.001
    fine = np.array(rows)
    ticks = fine[::100]
    dist = ticks[:, 1]
    before = np.subtract.outer(apexes, dist)                       # > 0: the apex is still ahead
    braking = ((before > 0) & (before <= ZONE)).T                  # one column per corner
    braking[:, LIFT_ONLY - 1] = False
    after = -before
    slow_throttle = ((after >= 0) & (after < SLOW_THROTTLE_M)).any(axis=0)
    approaching = ((before > 0) & (before <= ZONE)).any(axis=0)
    lap = pd.DataFrame({
        "driver_number": "1", "driver_code": "AAA", "lap_number": 5,
        "session_time_sec": start_sec + ticks[:, 0],
        "x": np.interp(dist, path["distance"], path["x"]), "y": np.interp(dist, path["distance"], path["y"]),
        "speed_kmh": speed_at(dist, apexes) * 3.6,
        "throttle_pct": np.where(approaching, 0.0, np.where(slow_throttle, 20.0, 100.0)),
        "brake": braking.any(axis=1).astype(float),
        "lap_distance_m": dist * 0.99,                             # the car's own distance: 1 % short, as in Silver
        "is_data_gap": False})
    return lap, fine


@pytest.fixture(scope="module")
def made_up():
    path = draw(RECTANGLE)
    laps, telemetry = laps_on(path)
    line = tm.reference_line(telemetry, tm.good_laps(laps, telemetry))
    apexes = np.array(corners_of(RECTANGLE))
    corners = pd.DataFrame({"order": [1, 2, 3, 4], "label": ["1", "2", "3", "4"], "distance_m": apexes,
                            "source": "official"})
    lap, fine = drive(path, apexes)
    return line, corners, lap, fine, apexes


def test_every_feature_of_a_lap_driven_by_hand(made_up):
    line, corners, lap, fine, apexes = made_up
    rows, lap_row = cf.lap_features(lap, line, corners, tm.lap_length(line))
    f = pd.DataFrame(rows).set_index("corner_order")
    assert f["is_complete"].all() and len(f) == 4
    assert f["min_speed_kmh"].to_numpy() == pytest.approx([108] * 4, abs=3)             # 30 m/s at each apex
    assert f["corner_speed_kmh"].to_numpy() == pytest.approx([108] * 4, abs=4)
    assert f["min_speed_distance_m"].to_numpy() == pytest.approx(apexes, abs=8)         # on the shared line
    braked = f.drop(LIFT_ONLY)
    assert braked["brake_point_m"].to_numpy() == pytest.approx([ZONE] * 3, abs=10)      # one tick at 288 km/h is 8 m
    assert braked["entry_speed_kmh"].to_numpy() == pytest.approx([288] * 3, abs=12)       # the first tick on the brakes: up to 8 m in
    assert np.isnan(f.loc[LIFT_ONLY, "brake_point_m"])                                  # no brakes: no braking point,
    assert f.loc[LIFT_ONLY, "entry_speed_kmh"] == pytest.approx(228, abs=6)             # entry 100 m before: 63.3 m/s
    assert f["exit_speed_kmh"].to_numpy()[:3] == pytest.approx([228] * 3, abs=6)        # and 100 m after
    to_the_line = fine[-1, 1] - apexes[3]                                               # corner 4: the lap ends 47 m
    assert f.loc[4, "exit_speed_kmh"] == pytest.approx(3.6 * (SLOW + (FAST - SLOW) * to_the_line / ZONE), abs=8)   # after
    assert f["throttle_on_m"].to_numpy() == pytest.approx([SLOW_THROTTLE_M] * 4, abs=5)
    # the share of a segment's time at full throttle: all but the braking zone and the 30 m after the apex. Corners 2
    # and 3: 6.2 s of 9.9 s (worked out by hand). Corner 4's segment ends at the line, 47 m after its apex
    assert f.loc[[2, 3], "full_throttle_share"].to_numpy() == pytest.approx([0.63, 0.63], abs=0.03)
    assert f.loc[4, "full_throttle_share"] < 0.3


def test_segment_times_are_the_time_between_the_cuts_and_add_up_to_the_lap(made_up):
    line, corners, lap, fine, apexes = made_up
    length = tm.lap_length(line)
    rows, lap_row = cf.lap_features(lap, line, corners, length, 1000.0, 1000.0 + fine[-1, 0])   # official ends
    times = np.array([r["segment_time_sec"] for r in rows])
    cuts = cf.segment_edges(apexes, fine[-1, 1])
    truth = np.diff(np.interp(cuts, fine[:, 1], fine[:, 0]))                            # from the 1 ms simulation
    assert times == pytest.approx(truth, abs=0.03)
    assert times.sum() == pytest.approx(fine[-1, 0], abs=1e-9)                          # exactly the lap
    no_ends, _ = cf.lap_features(lap, line, corners, length)                            # unknown: first and last
    assert sum(r["segment_time_sec"] for r in no_ends) == pytest.approx(fine[-1, 0], abs=0.1)   # sample, to a tick
    assert lap_row["segments_time_sum_sec"] == pytest.approx(times.sum())
    assert lap_row["top_speed_kmh"] == pytest.approx(288) and lap_row["is_complete"]
    assert lap_row["brake_share"] == pytest.approx(lap["brake"].mean())
    assert lap_row["coast_share"] > 0.02                                                # the lift for corner 3
    assert lap_row["coast_share"] == pytest.approx(((lap["throttle_pct"] < 5) & (lap["brake"] < 0.5)).mean())


def test_a_hole_in_a_corner_leaves_that_corner_empty_and_the_others_whole(made_up):
    line, corners, lap, fine, apexes = made_up
    holed = lap.copy()
    in_two = (holed["lap_distance_m"] / 0.99).between(apexes[1] - 20, apexes[1] + 20)
    holed.loc[in_two, "is_data_gap"] = True
    rows, lap_row = cf.lap_features(holed, line, corners, tm.lap_length(line))
    f = pd.DataFrame(rows).set_index("corner_order")
    assert f["is_complete"].tolist() == [True, False, True, True]
    assert f.loc[2, ["segment_time_sec", "min_speed_kmh", "brake_point_m", "exit_speed_kmh"]].isna().all()
    assert f.loc[1, "min_speed_kmh"] == pytest.approx(108, abs=3)
    assert not lap_row["is_complete"] and np.isnan(lap_row["top_speed_kmh"]) and np.isnan(lap_row["segments_time_sum_sec"])


def test_a_sample_without_a_position_is_a_hole_too(made_up):
    line, corners, lap, fine, apexes = made_up
    lost = lap.copy()
    lost.loc[(lost["lap_distance_m"] / 0.99).between(apexes[1] - 20, apexes[1] + 20), ["x", "y"]] = np.nan
    rows, lap_row = cf.lap_features(lost, line, corners, tm.lap_length(line))
    assert [r["is_complete"] for r in rows] == [True, False, True, True] and not lap_row["is_complete"]


def test_a_lap_that_stops_after_the_first_corner_has_only_that_corner(made_up):
    """A car that retires: its telemetry ends 100 m after corner 1. (Measured against its own last distance, it would
    look like a whole lap and its last samples would be put past the line.)"""
    line, corners, lap, fine, apexes = made_up
    stopped = lap[lap["lap_distance_m"] / 0.99 < apexes[0] + 100]
    rows, lap_row = cf.lap_features(stopped, line, corners, tm.lap_length(line))
    f = pd.DataFrame(rows).set_index("corner_order")
    assert f["is_complete"].tolist() == [True, False, False, False]
    assert f.loc[2:, ["segment_time_sec", "min_speed_kmh", "corner_speed_kmh"]].isna().all().all()
    assert f.loc[1, "min_speed_kmh"] == pytest.approx(108, abs=3)
    assert cf.lap_on_line(line, stopped, tm.lap_length(line)).max() == pytest.approx(apexes[0] + 100, abs=10)


def test_the_distance_never_goes_backwards_and_crosses_the_line_at_both_ends(made_up, monkeypatch):
    """The map alone (no fusion: a single sample that jumps is exactly what fusion smooths away)."""
    monkeypatch.setattr(cf, "FUSE_TICKS", 1)
    line, corners, lap, fine, apexes = made_up
    length = tm.lap_length(line)
    path = draw(RECTANGLE)
    at = lambda d: (np.interp(d, path["distance"], path["x"]), np.interp(d, path["distance"], path["y"]))
    first, last = lap.iloc[[0]].copy(), lap.iloc[[-1]].copy()
    first[["x", "y"]], first["lap_distance_m"] = at(length - 4), 0.0            # 4 m before the line as the lap starts
    last[["x", "y"]], last["lap_distance_m"] = at(3.0), length * 0.99           # 3 m past it as the lap ends
    middle = lap.iloc[5:-5].copy()
    back = middle.index[200]
    middle.loc[back, ["x", "y"]] = middle.loc[middle.index[198], ["x", "y"]].to_numpy()   # one sample jumps back
    d = cf.lap_on_line(line, pd.concat([first, middle, last]), length)
    assert d[0] == pytest.approx(-4, abs=4) and d[-1] == pytest.approx(length + 3, abs=4)
    assert (np.diff(d) >= 0).all()


def test_stale_positions_are_smoothed_by_the_cars_own_distance(made_up, monkeypatch):
    """OpenF1 sends about 3.7 positions a second: at 10 Hz a position is often repeated. Here every position is held
    for 3 samples. Fused with the car's own distance (1 % short), the cuts land closer to the truth than the map alone
    places them; and a lap with official crossings adds up to exactly the time between them."""
    line, corners, lap, fine, apexes = made_up
    length = tm.lap_length(line)
    stale = lap.copy()
    held = np.minimum(((np.arange(len(stale)) + 1) // 3) * 3, len(stale) - 1)   # the same reading 3 samples running
    stale[["x", "y"]] = stale[["x", "y"]].to_numpy()[held]
    cuts = cf.segment_edges(apexes, fine[-1, 1])
    truth = np.diff(np.interp(cuts, fine[:, 1], fine[:, 0]))

    def worst(fuse):
        monkeypatch.setattr(cf, "FUSE_TICKS", fuse)
        rows, _ = cf.lap_features(stale, line, corners, length, 1000.0, 1000.0 + fine[-1, 0])
        return np.abs(np.array([r["segment_time_sec"] for r in rows]) - truth).max()

    alone, fused = worst(1), worst(31)
    print(f"worst cut: map alone {alone:.3f} s, fused {fused:.3f} s")
    assert fused < alone / 2
    rows, lap_row = cf.lap_features(stale, line, corners, length, 1000.0, 1000.0 + fine[-1, 0])
    assert lap_row["segments_time_sum_sec"] == pytest.approx(fine[-1, 0], abs=1e-9)


def test_official_lap_ends_are_used_only_where_they_agree_with_the_telemetry():
    t = np.arange(0.0, 90.0, 0.1) + 1000.0
    assert cf.lap_ends(t, 999.97, 1090.04) == (999.97, 1090.04)
    assert cf.lap_ends(t) == (1000.0, pytest.approx(1090.0))                       # unknown: the samples
    assert cf.lap_ends(t, 999.97, 1092.0) == (999.97, pytest.approx(1090.0))       # 2 s off: not this lap's end


def test_the_sector_check_turns_distance_into_time():
    """Sector line at 1,000 m on most laps; a lap 8 m off at 80 m/s is 100 ms off."""
    points = [(1000.0, 80.0, 2000.0, 50.0)] * 30 + [(1008.0, 80.0, 2000.0, 50.0)]
    median, p95 = cf.sector_check(points)
    assert median == 0.0 and p95 == pytest.approx(0.0)
    points = [(1000.0 + k, 80.0, 2000.0 + k, 80.0) for k in range(-10, 11)]       # off by 0 to 10 m, both sectors
    assert cf.sector_check(points)[0] == pytest.approx(5 / 80 * 1000)                # the median: 5 m = 62.5 ms
    assert np.isnan(cf.sector_check([(1000.0, 80.0, 2000.0, 50.0)] * 5)[0])          # too few laps


def test_the_lap_is_cut_halfway_between_corners_and_at_the_line():
    assert cf.segment_edges([100.0, 300.0, 900.0], 1000.0).tolist() == [0.0, 200.0, 600.0, 1000.0]


def test_official_corners_are_used_where_there_are_any_else_the_maps_bends_numbered():
    both = pd.DataFrame({"method": ["map", "map", "official", "official", "speed"],
                         "distance_m": [700.0, 90.0, 400.0, 100.0, 95.0], "number": [np.nan, np.nan, 2.0, 1.0, np.nan]})
    used = cf.corner_list(both)
    assert used["label"].tolist() == ["1", "2"] and used["distance_m"].tolist() == [100.0, 400.0]
    assert used["order"].tolist() == [1, 2]
    assert (used["source"] == "official").all()
    used = cf.corner_list(both[both["method"] != "official"])
    assert used["label"].tolist() == ["1", "2"] and used["distance_m"].tolist() == [90.0, 700.0]
    assert (used["source"] == "map").all()


def test_a_lettered_corner_keeps_its_letter_and_its_place_in_the_lap():
    """The Hungaroring: MultiViewer has a turn 1 and a turn 1A. Both are kept, one row each, in the order driven."""
    hungary = pd.DataFrame({"method": "official", "distance_m": [819.0, 635.0, 1132.0], "number": [1.0, 1.0, 2.0],
                            "label": ["1A", "1", "2"]})
    used = cf.corner_list(hungary)
    assert used["label"].tolist() == ["1", "1A", "2"] and used["order"].tolist() == [1, 2, 3]
    circuit = {"corners": [{"number": 1, "trackPosition": {"x": 0, "y": 0}},
                           {"number": 1, "letter": "A", "trackPosition": {"x": 0, "y": 0}}]}
    line = pd.DataFrame({"distance_m": [0.0, 5.0], "x": [0.0, 50.0], "y": [0.0, 0.0], "speed_kmh": [100.0, 100.0]})
    assert tm.official_corners(circuit, line)["label"].tolist() == ["1", "1A"]


# --- a circuit that passes close to itself ------------------------------------------------------------------------------

def test_where_two_stretches_run_side_by_side_a_sample_lands_on_the_one_the_car_is_on():
    """Out along y = 0 for 1,000 m, round a tight hairpin, back along y = 10 m. A car 6 m to the side of the way out is
    nearer the way back (4 m): the nearest point of the line is 1,500 m further on. Its own distance says which."""
    out = np.arange(0, 1000, 5.0)
    turn = np.linspace(0, np.pi, 5)[1:-1]
    x = np.concatenate([out, 1000 + 5 * np.sin(turn), out[::-1] + 5])
    y = np.concatenate([np.zeros(len(out)), 5 - 5 * np.cos(turn), np.full(len(out), 10.0)])
    steps = np.hypot(np.diff(x), np.diff(y))
    line = pd.DataFrame({"distance_m": np.concatenate([[0.0], np.cumsum(steps)]), "x": x * 10, "y": y * 10})
    length = tm.lap_length(line)
    car_x, car_y = np.array([500.0]) * 10, np.array([6.0]) * 10
    assert tm.project(line, car_x, car_y)[0] == pytest.approx(1515, abs=10)             # the wrong stretch
    assert tm.project_lap(line, car_x, car_y, [500 / length])[0] == pytest.approx(500, abs=5)
    assert tm.project_lap(line, car_x, car_y, [1515 / length])[0] == pytest.approx(1515, abs=10)   # on the way back
    # the way out, 6 m to the side all the way, on the car's own distance (1 % short): the shared distance is right
    way = out[20:-20]
    lap = pd.DataFrame({"x": way * 10, "y": 60.0, "lap_distance_m": way * 0.99})
    lap.loc[len(lap)] = [50.0, 100.0, length * 0.99]                                    # ... and the lap's last sample
    d = cf.lap_on_line(line, lap, length)
    assert d[:-1] == pytest.approx(way, abs=5) and (np.diff(d) >= 0).all()


def test_a_sample_between_two_points_of_the_line_gets_the_distance_between_them():
    """The line has a point every 5 m; a sample is placed on the stretch between two points, not snapped to one
    (snapping put up to 2.5 m, 0.03 s at 300 km/h, into every distance)."""
    x = np.arange(0.0, 1000.0, 5.0)
    line = pd.DataFrame({"distance_m": x, "x": x * 10, "y": 0.0})
    line = pd.concat([line, pd.DataFrame({"distance_m": [1000.0 + k * 5 for k in range(200)],
                                          "x": (1000.0 - np.arange(200) * 5.0) * 10, "y": 500.0})], ignore_index=True)
    on = tm.project_lap(line, [122.5 * 10, 377.0 * 10], [1.0 * 10, -2.0 * 10], [122.5 / 2000, 377.0 / 2000])
    assert on == pytest.approx([122.5, 377.0], abs=0.01)


# --- the golden race: Qatar 2024 ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qatar(openf1_qatar_lake, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("corner_features") / "lake")
    shutil.copytree(openf1_qatar_lake, root)                                    # the shared lake stays read-only
    tm.build_track(*QATAR, root=root, circuit=LUSAIL)
    row = cf.build_corner_features(*QATAR, root=root)
    features = lake.read_table(lake.table_path(lake.GOLD, cf.CORNER_TABLE, *QATAR, "R", root))
    lap_table = lake.read_table(lake.table_path(lake.GOLD, cf.LAP_TABLE, *QATAR, "R", root))
    return row, features, lap_table


def test_qatar_segment_times_add_up_to_the_official_lap_times(qatar):
    row, features, lap_table = qatar
    assert row["corners"] == 16 and row["corner_source"] == "official"
    assert (features["corner"] == features["corner_order"].astype(str)).all()      # Lusail: no letters, 1 to 16
    assert len(features) == 16 * len(lap_table) and lap_table["driver_code"].nunique() == 2      # VER, NOR, laps 1-7
    contracts.validate(features, contracts.GOLD_CORNER_FEATURES, "Gold corner features")
    contracts.validate(lap_table, contracts.GOLD_LAP_TELEMETRY, "Gold lap telemetry")
    clean = lap_table[lap_table["is_clean_lap"] & lap_table["is_complete"]]
    assert len(clean) == 2                                     # lap 6 of each (1-4 safety car; lap 7's positions stop)
    assert (clean["segments_time_sum_sec"] - clean["lap_time_sec"]).abs().max() < 0.06        # under one tick
    summed = features.groupby(["driver_number", "lap_number"])["segment_time_sec"].sum(min_count=16)
    both = clean.set_index(["driver_number", "lap_number"]).join(summed)
    assert (both["segment_time_sec"] - both["segments_time_sum_sec"]).abs().max() < 1e-6       # the rows are the lap
    # lap 7: the fixture's positions stop early in the lap. Nothing after that is made up
    cut = features[features["lap_number"] == 7]
    assert not cut["is_complete"].any() and cut["min_speed_kmh"].isna().all() and cut["segment_time_sec"].isna().all()
    assert not lap_table.loc[lap_table["lap_number"] == 7, "is_complete"].any()


def test_a_race_whose_map_was_refused_keeps_no_old_map_and_no_corner_features(openf1_qatar_lake, tmp_path, monkeypatch):
    """Hungary 2026: 5.1's contract refuses its line (OpenF1's positions are faulty). A line written before that
    contract existed was still in Gold, and 5.2 built on it. Now neither survives the refusal."""
    root = str(tmp_path / "lake")
    shutil.copytree(openf1_qatar_lake, root)
    tm.build_track(*QATAR, root=root, circuit=LUSAIL)
    cf.build_corner_features(*QATAR, root=root)
    gold = lambda table: lake.table_path(lake.GOLD, table, *QATAR, "R", root)
    assert all(gold(t).exists() for t in (tm.LINE_TABLE, tm.CORNERS_TABLE, cf.CORNER_TABLE, cf.LAP_TABLE))

    def refuse(df, schema, name):
        raise contracts.ContractError(f"{name}: refused")
    monkeypatch.setattr(contracts, "validate", refuse)
    with pytest.raises(contracts.ContractError):
        tm.build_track(*QATAR, root=root, circuit=LUSAIL)
    assert not gold(tm.LINE_TABLE).exists() and not gold(tm.CORNERS_TABLE).exists()
    with pytest.raises(FileNotFoundError, match="no track model"):
        cf.build_corner_features(*QATAR, root=root)
    assert not gold(cf.CORNER_TABLE).exists() and not gold(cf.LAP_TABLE).exists()


def test_qatar_turn_6_is_the_slowest_and_turn_1_the_big_stop(qatar):
    _, features, lap_table = qatar
    clean = features[features["lap_number"] == 6]
    by_corner = clean[clean["is_complete"]].groupby("corner_order")
    slowest = by_corner["min_speed_kmh"].median()
    assert slowest.idxmin() == 6 and 80 < slowest[6] < 110                     # the hairpin: about 92 km/h
    braked = by_corner["brake_point_m"].apply(lambda s: s.notna().mean())
    assert braked[[1, 2, 4, 6, 7, 10, 16]].eq(1).all()                         # the corners both brake for
    assert braked[[3, 5, 8, 11, 12, 13]].eq(0).all()                           # no brakes: the fast ones
    assert 100 < by_corner["brake_point_m"].median()[1] < 180                  # from 280 to 145 km/h in ~140 m
    assert by_corner["entry_speed_kmh"].median()[1] > 260
    top = lap_table.loc[lap_table["lap_number"] == 6, "top_speed_kmh"]
    assert 290 < top.min() and top.max() < 350
