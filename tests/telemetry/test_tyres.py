"""
tyres.py: where tyre wear costs time, and which signal sees it first.

Golden: the 2024 Qatar GP's real laps (the FastF1-shaped fixture), each lap's time split into four corners in fixed
shares: the corners' wear slopes must add up to exactly the wear slope Phase 2 measures for each compound (step 2.3,
itself golden-tested), and the slow corners must carry exactly their share. Then made-up stints where every answer is
known: a clean trend seen at once, noise never seen as wear, a drop in speed counted as worse, the jump rule needing two
laps, and the fuel effect recovered for both signals by one method.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.degradation import FUEL_EFFECT_SEC_PER_KG, USED, compound_slopes, degradation_laps
from src.analytics.laps import classify_laps
from src.telemetry import tyres as ty

QATAR = (2024, 23)
SHARES = (0.2, 0.3, 0.1, 0.4)                 # four corners' shares of every lap
SPEEDS = (90.0, 200.0, 120.0, 280.0)          # corners 1 and 3 are slow (under 150 km/h): 0.3 of the lap


@pytest.fixture(scope="module")
def qatar_used(golden_silver_laps):
    deg = degradation_laps(classify_laps(golden_silver_laps[QATAR]))
    return deg, deg[deg["deg_status"] == USED]


def split_into_corners(used: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (share, speed) in enumerate(zip(SHARES, SPEEDS), start=1):
        rows.append(used.assign(corner_order=k, corner=str(k), corner_speed_kmh=speed,
                                segment_time_sec=used["lap_time_sec"] * share,
                                corrected_sec=used["lap_time_sec"] * share
                                - FUEL_EFFECT_SEC_PER_KG * used["fuel_kg"] * share))
    return pd.concat(rows, ignore_index=True)


def test_qatar_corner_slopes_add_up_to_phase_2s_wear_of_each_compound(qatar_used):
    deg, used = qatar_used
    wear = ty.corner_wear(split_into_corners(used))
    phase2 = compound_slopes(deg).set_index("compound")["slope"]
    for compound, g in wear.groupby("compound"):
        assert g["lap_slope"].iloc[0] == pytest.approx(phase2[compound], abs=1e-9)
        assert g["slope"].sum() == pytest.approx(phase2[compound], abs=1e-9)
        assert g.sort_values("corner_order")["slope"].to_numpy() == pytest.approx(np.array(SHARES) * phase2[compound])
    # a lap missing one corner is left out of every corner's slope, so the corners still add up to the lap exactly
    segments = split_into_corners(used)
    holed = segments.drop(segments.index[(segments["corner_order"] == 2)][5])
    for _, g in ty.corner_wear(holed).groupby("compound"):
        assert g["slope"].sum() == pytest.approx(g["lap_slope"].iloc[0], abs=1e-9)
    share = ty.slow_share(wear).set_index("compound")
    assert (share["slow_slope"] / share["lap_slope"]).to_numpy() == pytest.approx([0.3] * len(share))
    assert share["slow_time_share"].to_numpy() == pytest.approx([0.3] * len(share))


def test_a_clean_trend_is_clear_as_soon_as_it_can_be_judged():
    age = np.arange(1.0, 16.0)
    assert ty.detection_lap(age, 80.0 + 0.1 * age, worse=+1) == (ty.MIN_TREND_LAPS, False)
    assert ty.detection_lap(age, 200.0 - 0.3 * age, worse=-1) == (ty.MIN_TREND_LAPS, False)   # slower corners
    assert ty.detection_lap(age, 200.0 + 0.3 * age, worse=-1) == (None, True)                 # faster: the wrong way


def test_noise_takes_longer_to_see_through_and_pure_noise_is_rarely_called_wear():
    rng = np.random.default_rng(7)
    age = np.arange(1.0, 26.0)
    quiet = [ty.detection_lap(age, 80 + 0.05 * age + rng.normal(0, 0.05, 25), +1)[0] for _ in range(200)]
    loud = [ty.detection_lap(age, 80 + 0.05 * age + rng.normal(0, 0.30, 25), +1)[0] for _ in range(200)]
    assert np.median([q for q in quiet if q]) < np.median([q for q in loud if q])
    flat = [ty.detection_lap(age, 80 + rng.normal(0, 0.3, 25), +1)[0] for _ in range(400)]
    assert np.mean([f is not None for f in flat]) < 0.3                                      # some, never most


def test_which_signal_saw_it_first_is_counted_on_stints_measured_on_every_lap():
    stints = pd.DataFrame({"laps": [10, 10, 10, 10, 10], "time_laps": [10, 10, 10, 10, 9],
                           "speed_laps": [10] * 5, "time_clear_lap": [8, 6, 7, None, 5],
                           "speed_clear_lap": [5, 6, 9, 6, 5], "time_wrong_first": False, "speed_wrong_first": False})
    r = ty.who_first(stints)
    assert r["stints"] == 4 and r["both_saw"] == 3                       # the last stint misses a lap time
    assert r["speed_first"] == pytest.approx(1 / 3) and r["same_lap"] == pytest.approx(1 / 3)
    assert r["speed_3_earlier"] == pytest.approx(1 / 3) and r["only_speed"] == 1


def test_going_off_needs_two_laps_in_a_row_above_the_trend_so_far():
    lap = np.arange(1, 16)
    times = 90.0 + 0.05 * (lap - 1)
    stint = pd.DataFrame({"lap_number": lap, "age": lap.astype(float), "corrected_sec": times})
    assert ty.going_off(stint) is None
    jump = stint.assign(corrected_sec=times + np.where(np.isin(lap, [9, 10]), 0.6, 0.0))
    assert ty.going_off(jump) == 9
    once = stint.assign(corrected_sec=times + np.where(lap == 9, 0.6, 0.0))
    assert ty.going_off(once) is None


def test_teams_reacting_is_judged_against_chance():
    stints = pd.DataFrame({"final": [False, False, True], "first_lap": [1, 1, 30], "last_lap": [20, 20, 57],
                           "event_lap": [18.0, 8.0, 40.0]})
    r = ty.pit_reaction(stints)
    assert r["events_in_stopped_stints"] == 2 and r["stop_soon"] == 0.5     # lap 18 is 2 laps before the stop
    # possible event laps 6..19 in each stopped stint: laps 17, 18, 19 are within 3 laps of the stop
    assert r["chance"] == pytest.approx(3 / 14)


def test_the_fuel_effect_comes_out_of_both_signals_by_one_method():
    rows = []
    for d, driver in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for lap in range(2, 51):
            stop = 20 + 3 * d
            first = 2 if lap <= stop else stop + 1
            age = lap - first + 1.0
            fuel = 110.0 - (lap - 1) * 105.0 / 50
            rows.append({"driver_code": driver, "lap_number": lap, "deg_status": USED, "age": age,
                         "stint": f"{driver}|{int(lap > stop)}", "compound": "MEDIUM" if lap <= stop else "HARD",
                         "fuel_kg": fuel, "lap_time_sec": 90 + d * 0.2 + 0.05 * age + 0.03 * fuel,
                         "slow_speed_kmh": 150 - d - 0.2 * age - 0.04 * fuel})
    laps, coefficients = ty.fuel_corrected(pd.DataFrame(rows))
    assert coefficients[ty.TIME] == pytest.approx(0.03) and coefficients[ty.SPEED] == pytest.approx(-0.04)
    one = laps[laps["stint"] == "AAA|0"]
    assert np.polyfit(one["age"], one["time_fc"], 1)[0] == pytest.approx(0.05)      # only the wear is left
    assert np.polyfit(one["age"], one["speed_fc"], 1)[0] == pytest.approx(-0.2)


def test_early_slope_is_the_first_laps_only_and_needs_four_of_them():
    age = np.arange(1.0, 11.0)
    assert ty.early_slope(age, 2.0 * age) == pytest.approx(2.0)
    late_wear = np.where(age <= ty.EARLY_LAPS, 80.0, 80.0 + 0.5 * (age - ty.EARLY_LAPS))
    assert ty.early_slope(age, late_wear) == pytest.approx(0.0)                       # fresh: no fade yet
    assert ty.early_slope(age, late_wear, laps=len(age)) > 0.1                         # the whole stint wears
    assert np.isnan(ty.early_slope(age[:3], age[:3]))


def test_each_stint_says_when_each_signal_saw_the_wear():
    lap = np.arange(10, 25)
    age = lap - 9.0
    laps = pd.DataFrame({"season": 2024, "round": 1, "stint": "AAA|2|HARD", "driver_code": "AAA", "compound": "HARD",
                         "lap_number": lap, "age": age, "time_fc": 80 + 0.05 * age, "speed_fc": 150 - 0.3 * age,
                         "corrected_sec": 80 + 0.05 * age})
    row = ty.stint_detection(laps).iloc[0]
    assert row["time_clear_lap"] == row["speed_clear_lap"] == ty.MIN_TREND_LAPS        # slower laps, slower corners
    assert not row["time_wrong_first"] and not row["speed_wrong_first"] and bool(row["final"])
    assert row["speed_early_slope"] == pytest.approx(-0.3) and row["time_slope"] == pytest.approx(0.05)


def test_slow_corners_are_judged_on_clean_laps_and_their_speed_is_averaged_only_on_whole_laps():
    rows = []
    for lap in (2, 3, 4):
        for order, speed in ((1, 95.0), (2, 210.0), (3, 140.0)):
            rows.append({"driver_code": "AAA", "lap_number": lap, "corner_order": order, "is_complete": True,
                         "corner_speed_kmh": speed + lap, "exit_speed_kmh": speed + 20 + lap})
    features = pd.DataFrame(rows)
    features.loc[(features["lap_number"] == 4) & (features["corner_order"] == 3), "is_complete"] = False
    behind_safety_car = pd.DataFrame([{"driver_code": "AAA", "lap_number": lap, "corner_order": 2, "is_complete": True,
                                       "corner_speed_kmh": 60.0, "exit_speed_kmh": 80.0} for lap in (9, 10, 11, 12)])
    features = pd.concat([features, behind_safety_car], ignore_index=True)       # four slow laps, none clean
    clean = pd.DataFrame({"driver_code": ["AAA"] * 3, "lap_number": [2, 3, 4]})
    assert ty.slow_corners(features, clean) == [1, 3]               # laps 9-12 (not clean) do not make corner 2 slow
    speed = ty.slow_speed(features, [1, 3]).set_index("lap_number")["slow_speed_kmh"]
    assert speed.index.tolist() == [2, 3]                           # lap 4 has a slow corner incomplete: no value
    assert speed[2] == pytest.approx(((95 + 2 + 115 + 2) / 2 + (140 + 2 + 160 + 2) / 2) / 2)
