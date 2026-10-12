"""
Neutralised periods (src/analytics/neutralisations.py), Phase 2 step 1.

Golden facts come from Race Control's own log for the 2024 Qatar GP (independent of the lap
flags the code reads): "SAFETY CAR DEPLOYED" lap 1, "IN THIS LAP" lap 4; deployed lap 35, in
lap 39; "VIRTUAL SAFETY CAR DEPLOYED" lap 40 and "SAFETY CAR DEPLOYED" lap 40, in lap 42. The
2024 British GP had none. Unit tests pin the reading of the per-lap status codes.
"""

import pandas as pd

from src.analytics.neutralisations import (
    RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR, describe_periods, leader_laps, neutralised_periods,
)

BRITISH, QATAR = (2024, 12), (2024, 23)


def periods_as_tuples(periods):
    return list(zip(periods["kind"], periods["first_lap"], periods["last_lap"]))


# --- golden ---------------------------------------------------------------------------

def test_qatar_periods_match_race_control(golden_silver_laps):
    periods = neutralised_periods(golden_silver_laps[QATAR])
    assert periods_as_tuples(periods) == [
        (SAFETY_CAR, 1, 4), (SAFETY_CAR, 35, 39), (VIRTUAL_SAFETY_CAR, 40, 40), (SAFETY_CAR, 40, 42),
    ]
    assert periods["laps"].tolist() == [4, 5, 1, 3]
    assert describe_periods(periods) == "SC 1-4, SC 35-39, VSC 40, SC 40-42"


def test_qatar_second_safety_car_starts_on_the_leaders_lap_35_not_a_lapped_cars_34(golden_silver_laps):
    laps = golden_silver_laps[QATAR]
    # Pooling every driver's lap numbers says 34: a lapped car was on its lap 34 when the safety car came out
    assert laps.loc[laps["is_sc"], "lap_number"].between(30, 34).any()
    leaders = leader_laps(laps).set_index("lap_number")
    assert leaders.loc[34, "track_status"] == "1"                 # the leader's lap 34 was green
    assert leaders.loc[35, "driver_code"] == "VER"


def test_british_had_no_neutralisation(golden_silver_laps):
    periods = neutralised_periods(golden_silver_laps[BRITISH])
    assert periods.empty and describe_periods(periods) == "none"


def test_the_leader_is_the_first_car_to_complete_each_lap(golden_silver_laps):
    leaders = leader_laps(golden_silver_laps[QATAR])
    assert leaders["lap_number"].tolist() == list(range(1, 58))   # one row per race lap, 57 laps
    assert set(leaders["driver_code"]) == {"VER"}                 # Verstappen led every lap


# --- unit: reading the status codes -------------------------------------------------------

def laps_of(statuses, driver="AAA", lap_time=90.0, others=()):
    """Silver-like laps: the leader's statuses lap by lap, plus optional (code, delay, statuses) cars."""
    rows = []
    for i, status in enumerate(statuses, start=1):
        rows.append({"driver_code": driver, "lap_number": i, "lap_end_sec": i * lap_time, "position": 1,
                     "track_status": status, "is_generated": False})
    for code, lag, other_statuses in others:
        for i, status in enumerate(other_statuses, start=1):
            rows.append({"driver_code": code, "lap_number": i, "lap_end_sec": i * lap_time + lag, "position": 2,
                         "track_status": status, "is_generated": False})
    df = pd.DataFrame(rows)
    df["position"] = df["position"].astype("Int64")
    df["track_status"] = df["track_status"].astype("string")
    return df


def test_a_safety_car_that_starts_and_ends_mid_lap_covers_both_laps():
    periods = neutralised_periods(laps_of(["1", "14", "4", "41", "1"]))
    assert periods_as_tuples(periods) == [(SAFETY_CAR, 2, 4)]


def test_vsc_deployed_and_ending_is_one_period():
    periods = neutralised_periods(laps_of(["1", "16", "7", "71", "1"]))
    assert periods_as_tuples(periods) == [(VIRTUAL_SAFETY_CAR, 2, 4)]


def test_a_vsc_upgraded_to_a_safety_car_in_the_same_lap_is_two_periods():
    periods = neutralised_periods(laps_of(["1", "1264", "4", "41"]))       # Qatar 2024 lap 40
    assert periods_as_tuples(periods) == [(VIRTUAL_SAFETY_CAR, 2, 2), (SAFETY_CAR, 2, 4)]


def test_a_safety_car_followed_by_a_red_flag():
    periods = neutralised_periods(laps_of(["1", "124", "45", "1"]))
    assert periods_as_tuples(periods) == [(SAFETY_CAR, 2, 3), (RED_FLAG, 3, 3)]


def test_green_between_two_safety_cars_makes_two_periods():
    periods = neutralised_periods(laps_of(["4", "41", "1", "14", "41"]))
    assert periods_as_tuples(periods) == [(SAFETY_CAR, 1, 2), (SAFETY_CAR, 4, 5)]


def test_yellow_flags_are_not_neutralisations():
    assert neutralised_periods(laps_of(["1", "12", "21", "1"])).empty


def test_a_lapped_cars_status_does_not_count():
    # the lapped car sees the safety car on its lap 2, which it completes long after the leader's lap 2
    laps = laps_of(["1", "1", "14", "41"], others=[("BBB", 200.0, ["1", "14", "4", "1"])])
    assert periods_as_tuples(neutralised_periods(laps)) == [(SAFETY_CAR, 3, 4)]


def test_without_end_times_the_car_in_p1_stands_in_and_unknown_laps_are_skipped():
    laps = laps_of(["1", "14", "4", "41"])
    laps["lap_end_sec"] = float("nan")
    laps.loc[laps["lap_number"] == 3, "position"] = pd.NA          # nobody known to lead lap 3
    leaders = leader_laps(laps)
    assert leaders["lap_number"].tolist() == [1, 2, 4]
    assert periods_as_tuples(neutralised_periods(laps)) == [(SAFETY_CAR, 2, 4)]


def test_generated_laps_are_never_the_leader():
    laps = laps_of(["1", "1"], others=[("ZZZ", -50.0, ["4", "4"])])   # earlier end time, but made up
    laps.loc[laps["driver_code"] == "ZZZ", "is_generated"] = True
    assert set(leader_laps(laps)["driver_code"]) == {"AAA"}
    assert neutralised_periods(laps).empty
