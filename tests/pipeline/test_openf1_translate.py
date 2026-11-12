"""
openf1_translate.py: OpenF1's tables turned into the tables the cleaner reads.

Two kinds of test. Small made-up tables for each rule (the clock, chained laps, track status, tyres, pits). And the
GOLDEN race: OpenF1's real tables for Qatar 2024 (tests/fixtures/openf1/qatar_2024, the small tables as delivered),
checked against facts that do not come from this code: the official race time, Race Control's safety-car laps.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics import race_overview
from src.pipeline import cleaner, feature_store, lake, openf1_translate as tr

QATAR_FIXTURE = Path(__file__).parent.parent / "fixtures" / "openf1" / "qatar_2024"
QATAR = (2024, 23)
DAY = "2024-12-01T"


def stamp(seconds):
    """A UTC stamp `seconds` after 16:00:00, as OpenF1 writes them."""
    return (pd.Timestamp(DAY + "16:00:00") + pd.Timedelta(seconds=seconds)).isoformat() + "+00:00"


def messages(*rows):
    """Race Control rows from (seconds, message, flag, scope, sector)."""
    return pd.DataFrame([{"date": stamp(t), "message": m, "flag": f, "scope": s, "sector": k}
                         for t, m, f, s, k in rows])


START = (100.0, tr.SESSION_STARTED, None, None, None)
T0 = pd.Timestamp(DAY + "16:01:40") + pd.Timedelta(seconds=tr.LIGHTS_AFTER_MESSAGE_SEC)     # lights out in these tests


def laps_of(driver, starts, durations, out_laps=()):
    """OpenF1 lap rows: lap n's stamped start (seconds after 16:00, None = missing) and duration."""
    return pd.DataFrame([{"driver_number": driver, "lap_number": n + 1,
                          "date_start": None if s is None else stamp(s), "lap_duration": d,
                          "is_pit_out_lap": (n + 1) in out_laps}
                         for n, (s, d) in enumerate(zip(starts, durations))])


@pytest.fixture(scope="module")
def qatar():
    tables = {e: pd.read_parquet(QATAR_FIXTURE / f"openf1_{e}.parquet") for e in tr.SMALL_TABLES}
    return tables, tr.translate(tables)


# --- the clock -------------------------------------------------------------------------------------------------

def test_time_zero_is_the_first_session_started_message_plus_the_measured_delay():
    rc = messages(START, (2500.0, tr.SESSION_STARTED, None, None, None))          # a restart after a red flag
    restart_laps = laps_of(1, [2500.0, 2600.0], [None, 90.0])                      # OpenF1 dates lap 1 from the restart
    t0, report = tr.lights_out(rc, restart_laps)
    assert t0 == T0 and report["clock_method"] == "session started" and report["clock_lap1_gap_sec"] == 2400.0


def test_a_session_started_from_another_session_is_ignored():
    """Mexico 2026: the race's messages include a SESSION STARTED from the day before."""
    rc = messages((100.0 - 86400.0, tr.SESSION_STARTED, None, None, None), START)
    scheduled = pd.Timestamp(DAY + "16:00:00")
    assert tr.lights_out(rc, laps_of(1, [100.0], [95.0]), scheduled)[0] == T0
    assert tr.lights_out(rc, laps_of(1, [100.0], [95.0]))[0] < T0 - pd.Timedelta(hours=23)     # without the schedule


def test_without_the_message_lap_one_votes_and_without_both_there_is_no_clock():
    laps = pd.concat([laps_of(1, [100.0], [95.0]), laps_of(2, [100.0], [96.0]), laps_of(3, [1700.0], [97.0])])
    t0, report = tr.lights_out(messages((50.0, "DRS ENABLED", None, None, None)), laps)
    assert t0 == T0 and report["clock_method"] == "lap 1"                           # two of three drivers agree
    with pytest.raises(tr.TranslationError, match="lights out is unknown"):
        tr.lights_out(messages((50.0, "DRS ENABLED", None, None, None)), laps_of(1, [None], [95.0]))


# --- lap starts and ends ------------------------------------------------------------------------------------------

def test_laps_are_chained_by_their_exact_durations_and_placed_by_the_median_of_their_stamps():
    lag, zero = tr.LATER_LAP_LAG_SEC, 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    true_starts = [zero, zero + 95.0, zero + 185.0, zero + 275.5]                   # lap 1 took 95 s, then 90, 90.5, 91
    noise = [0.0, 0.04, -0.03, 0.0]                                                 # OpenF1's stamps jitter
    stamps = [100.0] + [t + lag + e for t, e in zip(true_starts[1:], noise[1:])]
    times = tr.lap_times(laps_of(44, stamps, [95.3, 90.0, 90.5, 91.0]), T0)
    assert times["start_sec"].tolist() == pytest.approx([0.0, 95.0, 185.0, 275.5], abs=1e-6)   # the jitter is gone
    assert times["end_sec"].tolist() == pytest.approx([95.0, 185.0, 275.5, 366.5], abs=1e-6)
    # lap 1 ends where lap 2 starts, not at its own stamped duration (95.3: OpenF1 times lap 1 from elsewhere)


def test_a_lap_without_a_duration_ends_where_the_next_starts_and_the_chain_starts_again():
    lag, zero = tr.LATER_LAP_LAG_SEC, 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    stamps = [100.0, zero + 95 + lag, zero + 185 + lag, zero + 1000 + lag, zero + 1090 + lag]
    times = tr.lap_times(laps_of(1, stamps, [95.0, 90.0, None, 90.0, None]), T0)    # lap 3: a red flag, no time
    assert times["start_sec"].tolist() == pytest.approx([0.0, 95.0, 185.0, 1000.0, 1090.0], abs=1e-6)
    assert times["end_sec"].tolist()[:4] == pytest.approx([95.0, 185.0, 1000.0, 1090.0], abs=1e-6)
    assert np.isnan(times["end_sec"].iloc[4])                                       # the last lap never ended: unknown


# --- track status ----------------------------------------------------------------------------------------------------

def test_track_status_is_rebuilt_from_race_controls_messages():
    rc = messages(
        (40.0, "YELLOW IN TRACK SECTOR 9", "YELLOW", "Sector", 9),                  # on the way to the grid, never cleared
        START,
        (200.0, "DOUBLE YELLOW IN TRACK SECTOR 3", "DOUBLE YELLOW", "Sector", 3),
        (230.0, "SAFETY CAR DEPLOYED", None, None, None),
        (300.0, "CLEAR IN TRACK SECTOR 3", "CLEAR", "Sector", 3),
        (500.0, "SAFETY CAR IN THIS LAP", None, None, None),
        (560.0, "TRACK CLEAR", "CLEAR", "Track", None),
        (700.0, "VIRTUAL SAFETY CAR DEPLOYED", None, None, None),
        (760.0, "VIRTUAL SAFETY CAR ENDING", None, None, None),
        (772.0, "TRACK CLEAR", "CLEAR", "Track", None),
        (900.0, "RED FLAG", "RED", "Track", None),
        (1500.0, "TRACK CLEAR", "CLEAR", "Track", None),                            # the track is clear, the race is not on
        (2500.0, tr.SESSION_STARTED, None, None, None),
    )
    timeline = tr.status_timeline(rc, T0)
    after_start = timeline[timeline["t_sec"] > -1]
    assert after_start["code"].tolist() == ["1", "2", "4", "1", "6", "7", "1", "5", "1"]
    zero = 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    assert after_start["t_sec"].tolist() == pytest.approx([t - zero for t in (100, 200, 230, 560, 700, 760, 772, 900, 2500)])
    starts = np.array([0.0, 150.0, 480.0, 590.0, np.nan])        # four laps in session seconds, and one with no start
    ends = np.array([150.0, 480.0, 590.0, 700.0, 50.0])
    # yellow then safety car | safety car, then clear | green all the way | VSC deployed, ending, then green again
    assert tr.lap_status(starts, ends, timeline) == ["124", "41", "1", "167", None]


# --- tyres ------------------------------------------------------------------------------------------------------------

def stints_of(*rows):
    return pd.DataFrame([{"driver_number": 1, "stint_number": k + 1, "lap_start": a, "lap_end": b, "compound": c,
                          "tyre_age_at_start": age} for k, (a, b, c, age) in enumerate(rows)])


def out_laps(*laps, last=80):
    """A lap table of laps 1..last for car 1, with these laps as out-laps."""
    return pd.DataFrame({"driver_number": 1, "lap_number": range(1, last + 1),
                         "is_pit_out_lap": [n in laps for n in range(1, last + 1)]})


def tyre(table, lap):
    row = table[table["lap_number"] == lap].iloc[0]
    return row["Compound"], row["TyreLife"], row["Stint"]


def test_tyre_life_counts_up_from_the_age_when_fitted():
    tyres = tr.tyres_by_lap(stints_of((1, 20, "MEDIUM", 0), (21, 50, "HARD", 3)), out_laps(21))
    assert tyre(tyres, 1) == ("MEDIUM", 1, 1) and tyre(tyres, 20) == ("MEDIUM", 20, 1)
    assert tyre(tyres, 21) == ("HARD", 4, 2)                         # a used set: 3 laps done before, this is its 4th
    assert tyres[tyres["lap_number"] == 1]["FreshTyre"].iloc[0] and not tyres[tyres["lap_number"] == 21]["FreshTyre"].iloc[0]


def test_a_stint_that_starts_on_the_in_lap_is_moved_to_the_out_lap():
    """Monaco 2023, VER: OpenF1 starts the intermediates on lap 55, the in-lap; the car left the pits on lap 56."""
    tyres = tr.tyres_by_lap(stints_of((1, 54, "MEDIUM", 0), (55, 78, "INTERMEDIATE", 0)), out_laps(56))
    assert tyre(tyres, 55) == ("MEDIUM", 55, 1) and tyre(tyres, 56) == ("INTERMEDIATE", 1, 2)


def test_an_empty_first_stint_after_a_lap_one_red_flag_covers_lap_one():
    """Monaco 2024, VER: stints 1..0 HARD, 1..51 MEDIUM, 52..78 HARD (age 1); the out-laps are 2 and 53."""
    tyres = tr.tyres_by_lap(stints_of((1, 0, "HARD", 0), (1, 51, "MEDIUM", 0), (52, 78, "HARD", 1)), out_laps(2, 53))
    assert tyre(tyres, 1) == ("HARD", 1, 1) and tyre(tyres, 2) == ("MEDIUM", 1, 2)
    assert tyre(tyres, 52) == ("MEDIUM", 51, 2) and tyre(tyres, 53) == ("HARD", 2, 3)


def test_a_set_carried_through_the_pit_lane_stays_the_same_stint():
    """Australia 2025, VER: three trips through the pit lane behind the safety car, OpenF1 opens a stint for each."""
    tyres = tr.tyres_by_lap(stints_of((1, 1, "INTERMEDIATE", 0), (2, 2, "INTERMEDIATE", 2), (3, 3, "INTERMEDIATE", 3),
                                      (4, 33, "INTERMEDIATE", 4), (34, 45, "MEDIUM", 0)), out_laps(3, 4, 5, 35))
    assert [tyre(tyres, n)[1] for n in (1, 2, 3, 4, 5, 34)] == [1, 2, 3, 4, 5, 34]
    assert {tyre(tyres, n)[2] for n in range(1, 35)} == {1} and tyre(tyres, 35) == ("MEDIUM", 1, 5)


def test_without_a_pit_passage_there_are_no_new_tyres_whatever_the_stint_says():
    """Belgium 2025, PIA: OpenF1 has intermediates on lap 1 only and mediums from lap 2; he pitted on lap 12."""
    stints = stints_of((1, 1, "INTERMEDIATE", 0), (2, 12, "MEDIUM", 0), (13, 44, "MEDIUM", 0))
    tyres = tr.tyres_by_lap(stints, out_laps(13, last=44))
    assert tyre(tyres, 2) == ("INTERMEDIATE", 2, 1) and tyre(tyres, 12) == ("INTERMEDIATE", 12, 1)
    assert tyre(tyres, 13) == ("MEDIUM", 1, 3)
    # with laps missing from the lap table the missing out-lap proves nothing: the stint is taken as given
    holes = out_laps(13, last=44)
    holes = holes[~holes["lap_number"].isin([1, 2])]
    assert tyre(tr.tyres_by_lap(stints, holes), 3) == ("MEDIUM", 2, 2)


def test_a_start_announced_behind_the_safety_car_runs_under_it_until_the_rolling_start():
    rc = messages((10.0, "RACE WILL START BEHIND THE SAFETY CAR", None, None, None), START,
                  (700.0, "ROLLING START", None, None, None))
    timeline = tr.status_timeline(rc, T0)
    assert timeline["code"].tolist() == ["1", "4", "1"]
    zero = 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    assert timeline["t_sec"].tolist()[1:] == pytest.approx([100.0 - zero, 700.0 - zero])


def test_a_red_flag_restart_runs_under_the_safety_car_only_if_race_control_spoke_of_it():
    zero = 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    stopped = [START, (900.0, "RED FLAG", "RED", "Track", None), (1500.0, "TRACK CLEAR", "CLEAR", "Track", None)]
    with_sc = messages(*stopped, (1800.0, "ALL CARS MAY OVERTAKE THE SAFETY CAR", None, None, None),
                       (2500.0, tr.SESSION_STARTED, None, None, None), (2560.0, "STANDING START", None, None, None),
                       (2700.0, "EXTRA FORMATION LAP", None, None, None), (2750.0, "STANDING START", None, None, None))
    assert tr.status_timeline(with_sc, T0)["code"].tolist() == ["1", "5", "4", "1", "4", "1"]
    no_start_message = messages(*stopped, (1800.0, "ALL CARS MAY OVERTAKE THE SAFETY CAR", None, None, None),
                                (2500.0, tr.SESSION_STARTED, None, None, None))           # Sao Paulo 2023
    timeline = tr.status_timeline(no_start_message, T0)
    assert timeline["code"].tolist() == ["1", "5", "4", "1"]
    assert timeline["t_sec"].iloc[-1] == pytest.approx(2500.0 - zero + tr.RESTART_SC_MAX_SEC)
    rolling = messages(*stopped, (2500.0, tr.SESSION_STARTED, None, None, None), (2540.0, "ROLLING START", None, None, None))
    timeline = tr.status_timeline(rolling, T0)
    assert timeline["code"].tolist() == ["1", "5", "1"] and timeline["t_sec"].iloc[-1] == pytest.approx(2500.0 - zero)


def test_a_lap_that_would_end_before_it_starts_has_no_end():
    """Netherlands 2026, LIN: a 32-minute lap, a lap without a time, then a lap stamped 10 ms too early."""
    lag, zero = tr.LATER_LAP_LAG_SEC, 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    times = tr.lap_times(laps_of(43, [100.0, zero + 95 + lag, zero + 1000 + lag - 0.01], [95.0, 905.0, 90.0]), T0)
    times = tr.lap_times(laps_of(43, [100.0, zero + 95 + lag, None, zero + 1000 + lag - 0.01],
                                 [95.0, 905.0, None, 90.0]), T0)
    assert np.isnan(times["end_sec"].iloc[2]) and times["end_sec"].iloc[1] == pytest.approx(1000.0)


# --- pits, deleted laps, results -------------------------------------------------------------------------------------

def test_a_pit_rows_date_is_the_exit_and_the_entry_is_the_lane_time_before_it():
    pit = pd.DataFrame([{"driver_number": 1, "lap_number": 2, "date": stamp(100.27 + 200.0), "lane_duration": 24.0}])
    passage = tr.pit_by_lap(pit, T0).iloc[0]
    assert (passage["in_lap"], passage["pit_in_sec"], passage["pit_out_sec"]) == pytest.approx((2, 176.0, 200.0))
    assert tr.pit_by_lap(None, T0).empty


def test_without_a_pit_table_the_pit_laps_are_flagged_and_their_times_stay_unknown():
    tables = {
        "laps": laps_of(1, [100.0, 195.4, 285.4, 400.4], [95.0, 90.0, 115.0, 100.0], out_laps=(4,)),
        "drivers": pd.DataFrame([{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}]),
        "stints": stints_of((1, 3, "MEDIUM", 0), (4, 4, "HARD", 0)),
        "race_control": messages(START),
        "position": pd.DataFrame([{"driver_number": 1, "date": stamp(90.0), "position": 1}]),
    }
    for name in ("duration_sector_1", "duration_sector_2", "duration_sector_3", "i1_speed", "i2_speed", "st_speed"):
        tables["laps"][name] = np.nan
    laps, report = tr.lap_table(tables, T0)
    assert laps["PitInLap"].tolist() == [False, False, True, False] and laps["PitOutLap"].tolist() == [False, False, False, True]
    assert laps["PitInTime"].isna().all() and laps["PitOutTime"].isna().all() and report["pit_laps_without_time"] == 1
    weather = pd.DataFrame({"Time": pd.to_timedelta([10.0], unit="s"), "AirTemp": [20.0]})
    silver, _ = cleaner.clean_laps(laps, weather, 2023, 1, "R")
    assert silver["is_in_lap"].tolist() == [False, False, True, False] and silver["is_out_lap"].tolist()[3]
    assert silver["pit_kind"].iloc[2] == cleaner.PIT_TYRE_CHANGE and silver["pit_in_sec"].isna().all()


def test_a_laps_position_is_the_last_one_recorded_up_to_a_second_after_the_line():
    zero = 100.0 + tr.LIGHTS_AFTER_MESSAGE_SEC
    position = pd.DataFrame([{"driver_number": 1, "date": stamp(zero + t), "position": p}
                             for t, p in ((-600.0, 2), (90.5, 1), (150.0, 3))])   # the grid, a pass timed at the line, later
    drivers, ends = np.array([1, 1, 1, 7]), np.array([90.0, 180.0, np.nan, 90.0])
    assert tr.positions_at(position, T0, drivers, ends).tolist()[:2] == [1.0, 3.0]
    assert np.isnan(tr.positions_at(position, T0, drivers, ends)[2:]).all()        # no lap end, or no record: unknown


def test_deleted_lap_times_come_from_race_control_and_a_reinstated_one_is_not_deleted():
    rc = messages((10.0, "CAR 14 (ALO) TIME 1:32.386 DELETED - TRACK LIMITS AT TURN 13 LAP 5 19:13:48", None, None, None),
                  (20.0, "CAR 81 (PIA) TIME 1:26.474 DELETED - TRACK LIMITS AT TURN 16 LAP 16 19:29:51", None, None, None),
                  (30.0, "CAR 81 (PIA) TIME 1:26.474 REINSTATED - TRACK LIMITS AT TURN 16 LAP 16", None, None, None))
    assert list(tr.deleted_laps(rc)) == [(14, 5)]


# --- the golden race: Qatar 2024 ----------------------------------------------------------------------------------

def test_qatar_the_winner_crosses_the_line_at_the_official_race_time(qatar):
    """1:31:05.323 on F1's results sheet. The lap chain never reads that number."""
    tables, t = qatar
    ver = t.laps[t.laps["Driver"] == "VER"]
    assert len(ver) == 57 and t.total_laps == 57
    assert ver["Time"].iloc[-1].total_seconds() == pytest.approx(5465.323, abs=0.1)
    assert ver["LapStartTime"].iloc[0].total_seconds() == 0.0
    ends, starts = ver["Time"].dt.total_seconds().to_numpy(), ver["LapStartTime"].dt.total_seconds().to_numpy()
    assert np.allclose(ends[:-1], starts[1:])                                       # no hole and no overlap between laps
    assert t.report["clock_method"] == "session started" and t.report["clock_lap1_gap_sec"] == 0.0


def test_qatar_the_safety_cars_fall_on_the_laps_race_control_gave_them(qatar):
    """The leader's laps: safety car 1-4, 35-39 and 40-42, a virtual safety car on lap 40."""
    ver = qatar[1].laps[qatar[1].laps["Driver"] == "VER"].set_index(qatar[1].laps[qatar[1].laps["Driver"] == "VER"]["LapNumber"].astype(int))
    status = ver["TrackStatus"]
    assert [n for n in status.index if "4" in status[n]] == [1, 2, 3, 4] + list(range(35, 43))
    assert [n for n in status.index if "6" in status[n]] == [40]
    assert status[1] == "124" and status[20] == "1" and not any("5" in s for s in status)


def test_qatar_positions_are_the_ones_held_at_the_line(qatar):
    """Verstappen led lap 1 from second on the grid, Norris passed Russell for second; Piastri ran third on lap 30."""
    laps = qatar[1].laps
    order = lambda lap: laps[laps["LapNumber"] == lap].sort_values("Position")["Driver"].tolist()
    assert order(1)[:3] == ["VER", "NOR", "RUS"] and order(30)[:3] == ["VER", "NOR", "PIA"]
    assert sorted(laps[laps["LapNumber"] == 30]["Position"]) == list(range(1, 18))       # 17 cars left, no place twice


def test_qatar_through_the_cleaner_stops_tyres_and_deleted_laps(qatar):
    tables, t = qatar
    silver, report = cleaner.clean_laps(t.laps, t.weather, *QATAR, "R", pit_stops=t.pit_stops)
    assert report["output_rows"] == 944 and report["dropped_null_key"] == 0
    assert report["pit_tyre_change"] == 26                                          # as counted from FastF1's data
    assert report["pit_stops_matched"] == 60
    nor = silver[silver["driver_code"] == "NOR"].set_index("lap_number")
    assert nor.loc[35, "compound"] == "MEDIUM" and nor.loc[35, "tyre_life"] == 35 and nor.loc[36, "compound"] == "HARD"
    assert nor.loc[45, "pit_lane_time_sec"] == pytest.approx(36.1, abs=0.2)        # his 10 s stop-and-go penalty
    assert bool(silver[(silver["driver_code"] == "ALO") & (silver["lap_number"] == 5)]["is_deleted"].iloc[0])
    assert 640 <= report["clean_laps"] <= 680
    assert silver["air_temp_c"].notna().mean() > 0.99


def test_qatar_the_classification(qatar):
    result = race_overview.classification(qatar[1].results)
    assert result.iloc[0][["driver_code", "grid", "result", "points"]].tolist() == ["VER", 2, "1:31:05.323", 25.0]
    assert result.iloc[1][["driver_code", "result"]].tolist() == ["LEC", "+6.031s"]
    assert result[result["driver_code"] == "RUS"]["grid"].iloc[0] == 1              # pole, after Verstappen's penalty
    assert result[result["driver_code"] == "ALB"]["result"].iloc[0] == "+1 lap"
    retired = result[result["status"] == "Retired"]
    assert set(retired["driver_code"]) == {"HUL", "PER", "STR", "COL", "OCO"} and not retired["finished"].any()


def test_qatar_builds_silver_and_gold_from_an_openf1_bronze(tmp_path):
    bronze = lake.session_dir(lake.BRONZE, *QATAR, "R", tmp_path)
    shutil.copytree(QATAR_FIXTURE, bronze)
    out = cleaner.build_silver(*QATAR, "R", str(tmp_path), telemetry=False)
    assert out["clean_rows"] == 944 and out["lap_report"]["clock_method"] == "session started"
    gold = feature_store.build_gold(*QATAR, "R", str(tmp_path))
    assert gold["status"] == "success"
    features = lake.read_table(lake.gold_features_path(*QATAR, "R", tmp_path))
    assert features["total_race_laps"].iloc[0] == 57
    assert lake.results_path(*QATAR, "R", tmp_path).parts[-5] == lake.SILVER        # the classification is in Silver
    overview = race_overview.load_overview(*QATAR, root=str(tmp_path))
    assert overview.classification.iloc[0]["driver_code"] == "VER" and len(overview.stints) > 40
