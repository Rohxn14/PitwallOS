"""
lake_qualifying.py: a qualifying session for the replay, read from the lake instead of FastF1.

The lake is a temporary one holding real OpenF1 data: Baku 2026 qualifying (tests/fixtures/openf1/
baku_2026_qualifying): every small table as delivered, and the telemetry of three drivers' fastest laps (RUS, pole;
LEC, second; STR, out in Q1). The facts checked are the official ones: Russell's pole lap of 1:42.526, the three
parts' start and finish, Stroll's exit in Q1. No network, no real lake, and FastF1 is never asked.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import src.f1_data as f1_data
from src.lib import lake_qualifying
from src.lib.lake_qualifying import LakeQualifying
from src.pipeline import lake

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openf1" / "baku_2026_qualifying"
BAKU = (2026, 15)
DAY = "2026-09-25T"


@pytest.fixture(scope="module")
def baku_lake(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    shutil.copytree(FIXTURE, lake.session_dir(lake.BRONZE, *BAKU, "Q", root))
    return str(root)


@pytest.fixture(scope="module")
def session(baku_lake):
    return LakeQualifying(*BAKU, "Q", baku_lake)


def messages(*rows):
    return pd.DataFrame([{"date": DAY + t, "message": m, "qualifying_phase": p} for t, m, p in rows])


# --- the three parts ----------------------------------------------------------------------------------------------

def test_the_three_parts_are_the_started_finished_pairs_of_race_control(session):
    assert [(str(a.time())[:8], str(b.time())[:8]) for a, b in session.windows] == [
        ("12:04:00", "12:22:00"), ("12:29:00", "12:44:00"), ("12:51:00", "13:04:00")]


def test_a_part_stopped_by_a_red_flag_and_started_again_is_one_part():
    rc = messages(("12:00:00+00:00", "SESSION STARTED", 1), ("12:05:00+00:00", "SESSION FINISHED", 1),   # red flag
                  ("12:15:00+00:00", "SESSION STARTED", 1), ("12:28:00+00:00", "SESSION FINISHED", 1),
                  ("12:35:00+00:00", "SESSION STARTED", 2), ("12:50:00+00:00", "SESSION FINISHED", 2))
    windows = lake_qualifying.segment_windows(rc)
    assert [(str(a.time()), str(b.time())) for a, b in windows] == [("12:00:00", "12:28:00"), ("12:35:00", "12:50:00")]
    assert lake_qualifying.segment_windows(messages(("12:00:00+00:00", "GREEN LIGHT", 1))) == []


def test_a_lap_belongs_to_the_part_it_started_in(session):
    russell = lambda segment: sorted(session.segment_laps(segment).query("driver_number == 63")["lap_number"])
    assert russell("Q1") == list(range(1, 10)) and russell("Q2") == list(range(10, 18)) and russell("Q3") == list(range(19, 26))
    # lap 18 started between Q2 and Q3 (driving back to the pits), lap 26 two tenths after the chequered flag of Q3
    with pytest.raises(ValueError, match="does not exist"):
        LakeQualifying.segment_laps(type("Short", (), {"windows": session.windows[:2], "_laps": session._laps})(), "Q3")


# --- results and fastest laps -----------------------------------------------------------------------------------------

def test_the_results_are_the_official_ones(session):
    pole, last = session.results.iloc[0], session.results.iloc[-1]
    assert (pole["Abbreviation"], pole["FullName"], pole["Position"]) == ("RUS", "George Russell", 1.0)
    assert [pole[s].total_seconds() for s in ("Q1", "Q2", "Q3")] == [103.615, 103.462, 102.526]     # pole: 1:42.526
    stroll = session.results[session.results["Abbreviation"] == "STR"].iloc[0]
    assert stroll["Q1"].total_seconds() == 107.337 and pd.isna(stroll["Q2"]) and pd.isna(stroll["Q3"])
    assert len(session.results) == 22 and last["Position"] == 22.0
    listed = f1_data.get_qualifying_results(session)
    assert listed[0]["code"] == "RUS" and listed[0]["Q3"] == "102.526" and listed[0]["color"] == (0, 215, 182)


def test_a_drivers_fastest_lap_of_a_part_is_his_official_best(session):
    lap = session.fastest_lap("RUS", "Q3")
    assert (lap["LapNumber"], lap["LapTime"].total_seconds(), lap["Compound"]) == (25, 102.526, "SOFT")
    assert lap["Sector1Time"].total_seconds() + lap["Sector2Time"].total_seconds() + lap["Sector3Time"].total_seconds() == pytest.approx(102.526, abs=0.002)
    assert session.fastest_lap(63, "Q1")["LapTime"].total_seconds() == 103.615        # by car number too
    assert session.fastest_lap("STR", "Q2") is None                                    # he was out in Q1
    assert session.get_driver("RUS") == session.get_driver("63") == {
        "Abbreviation": "RUS", "DriverNumber": "63", "FullName": "George Russell"}


def test_a_faster_lap_whose_time_was_deleted_is_not_the_fastest(session, baku_lake):
    """Give Russell a made-up 1:40.000 in Q3: the result table still says 1:42.526, so that is his lap."""
    cheat = LakeQualifying(*BAKU, "Q", baku_lake)
    row = cheat._laps["driver_number"].eq(63) & cheat._laps["lap_number"].eq(23)
    cheat._laps.loc[row, "lap_duration"] = 100.0
    assert cheat.fastest_lap("RUS", "Q3")["LapTime"].total_seconds() == 102.526
    cheat.results.loc[cheat.results["Abbreviation"] == "RUS", "Q3"] = pd.NaT           # no official time: the shortest
    assert cheat.fastest_lap("RUS", "Q3")["LapTime"].total_seconds() == 100.0


# --- one lap's telemetry -------------------------------------------------------------------------------------------------

def test_a_laps_telemetry_starts_at_zero_and_covers_the_lap(session):
    lap, tel = session.segment_fastest_lap("RUS", "Q3")
    assert list(tel.columns) == ["Time", "X", "Y", "Distance", "RelativeDistance", "Speed", "nGear", "Throttle", "Brake", "DRS"]
    seconds = tel["Time"].dt.total_seconds()
    assert 0 <= seconds.iloc[0] < 0.5 and 102.0 < seconds.iloc[-1] <= 102.526 and seconds.is_monotonic_increasing
    assert 5850 < tel["Distance"].max() < 6100                           # Baku is 6.003 km
    assert tel["RelativeDistance"].iloc[-1] == pytest.approx(1.0) and tel["Speed"].max() == 331.0
    assert tel["Throttle"].between(0, 100).all() and tel["nGear"].between(0, 8).all() and tel["Brake"].dtype == bool
    assert not tel[["X", "Y", "Speed"]].isna().any().any()
    start_to_end = ((tel["X"].iloc[0] - tel["X"].iloc[-1]) ** 2 + (tel["Y"].iloc[0] - tel["Y"].iloc[-1]) ** 2) ** 0.5 / 10
    assert start_to_end < 60                                             # the lap ends where it began (metres)
    none, nothing = session.segment_fastest_lap("STR", "Q3")              # he set no time in Q3
    assert none is None and nothing.empty
    assert session.example_lap().equals(tel)                             # the pole lap draws the track


def test_a_lap_openf1_has_no_samples_for_gives_no_trace(session):
    lap = session.fastest_lap("VER", "Q1")                               # the fixture holds no telemetry for him
    assert lap is not None and session.lap_telemetry(lap).empty


# --- the replay's data ----------------------------------------------------------------------------------------------------

def test_the_qualifying_replays_data_is_built_from_the_lake(session):
    trace = f1_data.get_driver_quali_telemetry(session, "RUS", "Q3")
    assert trace["frames"][-1]["t"] == 102.526 and trace["frames"][0]["t"] == 0.0
    assert trace["max_speed"] == 331.0 and trace["compound"] == 0        # 0 = soft
    assert trace["sector_times"] == {"sector1": 36.009, "sector2": 41.659, "sector3": 24.858}
    assert {"x", "y", "speed", "gear", "throttle", "brake", "drs"}.issubset(trace["frames"][10]["telemetry"])
    with pytest.raises(ValueError, match="No valid laps"):
        f1_data.get_driver_quali_telemetry(session, "STR", "Q2")
    with pytest.raises(ValueError, match="must be 'Q1', 'Q2', or 'Q3'"):
        f1_data.get_driver_quali_telemetry(session, "RUS", "Q4")


# --- where the session comes from -------------------------------------------------------------------------------------------

def test_load_session_takes_qualifying_from_the_lake_and_never_asks_fastf1(baku_lake, monkeypatch):
    def no_fastf1(*args, **kwargs):
        raise AssertionError("FastF1 must not be asked for a session the lake has")

    monkeypatch.setattr(f1_data.fastf1, "get_session", no_fastf1)
    loaded = f1_data.load_session(*BAKU, "Q", lake_root=baku_lake)
    assert isinstance(loaded, LakeQualifying) and "Qualifying - lake" in str(loaded) and ":" not in str(loaded)
    assert lake_qualifying.available(*BAKU, "Q", baku_lake) and not lake_qualifying.available(*BAKU, "SQ", baku_lake)
    assert not lake_qualifying.available(*BAKU, "R", baku_lake)          # a race is LakeSession's
