"""
telemetry_clock.py (typed by Rohan, Phase 5 step 5.0): OpenF1's samples put on the session clock.

Golden facts from a source other than the code: FastF1's own t0 for the four 2026 races it loaded with telemetry
(Australia, Barcelona, Baku, Bahrain), and FastF1's own samples of Baku 2026 driver 63, the same feed (fixtures in
tests/fixtures/openf1/, cut from real downloads; no test touches the network). Then Silver built from OpenF1 on the
Qatar 2024 fixture race: VER and NOR, laps 1-6, the lap as long as Lusail's.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

clock = pytest.importorskip("src.pipeline.telemetry_clock", reason="src/pipeline/telemetry_clock.py: Rohan types it")

from src.pipeline import lake  # noqa: E402
from src.pipeline.cleaner import build_silver  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"
QATAR = (2024, 23)
LUSAIL_M = 5419.0          # the official length of the Lusail circuit


def lap_one(race: str):
    """(OpenF1's lap table, Bronze's lap table, FastF1's own t0) for lap 1 of one 2026 race."""
    rows = pd.read_csv(FIXTURES / "lap1_2026.csv")
    rows = rows[(rows["race"] == race) & rows["lap_start_sec"].notna()]
    openf1_laps = pd.DataFrame({"driver_number": rows["driver_number"], "lap_number": 1,
                                "date_start": rows["date_start"]})
    bronze_laps = pd.DataFrame({"DriverNumber": rows["driver_number"].astype(str), "LapNumber": 1.0,
                                "LapStartTime": pd.to_timedelta(rows["lap_start_sec"], unit="s")})
    return openf1_laps, bronze_laps, pd.Timestamp(rows["fastf1_t0"].iloc[0])


# --- the offset: lap 1 in both sources ---------------------------------------------------------

@pytest.mark.parametrize("race", ["Australian Grand Prix", "Barcelona Grand Prix", "Azerbaijan Grand Prix",
                                  "Bahrain Grand Prix"])
def test_t0_from_lap_one_is_fastf1s_own(race):
    openf1_laps, bronze_laps, truth = lap_one(race)
    t0, report = clock.session_offset(openf1_laps, bronze_laps)
    assert abs((t0 - truth).total_seconds()) <= 0.002                    # measured: 1 ms, every race
    assert report["clock_drivers"] >= 20 and report["clock_spread_ms"] <= 1.0


def test_a_driver_missing_from_one_source_is_left_out():
    openf1_laps, bronze_laps, truth = lap_one("Azerbaijan Grand Prix")
    _, full = clock.session_offset(openf1_laps, bronze_laps)
    t0, report = clock.session_offset(openf1_laps, bronze_laps.iloc[1:])
    assert report["clock_drivers"] == full["clock_drivers"] - 1 and abs((t0 - truth).total_seconds()) <= 0.002


def late_by(openf1_laps, rows, seconds):
    """The same lap table with these rows' lap-1 start moved later."""
    late = openf1_laps.copy()
    for i in late.index[rows]:
        late.loc[i, "date_start"] = str(clock.utc(late["date_start"]).loc[i] + pd.Timedelta(seconds=seconds))
    return late


def test_a_few_wrong_lap_ones_are_outvoted_and_named():
    openf1_laps, bronze_laps, truth = lap_one("Azerbaijan Grand Prix")
    t0, report = clock.session_offset(late_by(openf1_laps, [0, 1], 0.5), bronze_laps)
    assert abs((t0 - truth).total_seconds()) <= 0.002
    late = sorted(openf1_laps["driver_number"].iloc[:2])
    assert report["clock_outliers"] == " ".join(map(str, late)) and report["clock_spread_ms"] <= 1.0


def test_a_split_vote_or_too_few_drivers_stop_the_clock():
    openf1_laps, bronze_laps, _ = lap_one("Azerbaijan Grand Prix")
    half = len(openf1_laps) // 2
    with pytest.raises(clock.ClockError, match="need more than half"):
        clock.session_offset(late_by(openf1_laps, slice(0, half), 30.0), bronze_laps)
    with pytest.raises(clock.ClockError, match="only 2 of 2"):
        clock.session_offset(openf1_laps, bronze_laps.iloc[:2])


@pytest.mark.parametrize("race, t0, outliers", [
    ("Mexico City Grand Prix", "2023-10-29 19:01:00.080", "11"),          # Perez, out at the first corner
    ("Sao Paulo Grand Prix", "2023-11-05 16:01:00.002", "3 20 23 81"),    # caught in the lap-1 crash: the restart
])
def test_real_red_flag_races_where_openf1_gives_a_few_cars_the_wrong_lap_one(race, t0, outliers):
    rows = pd.read_csv(FIXTURES / "lap1_2023_red_flags.csv")
    rows = rows[(rows["race"] == race) & rows["lap_start_sec"].notna()]
    openf1_laps = pd.DataFrame({"driver_number": rows["driver_number"], "lap_number": 1,
                                "date_start": rows["date_start"]})
    bronze_laps = pd.DataFrame({"DriverNumber": rows["driver_number"].astype(str), "LapNumber": 1.0,
                                "LapStartTime": pd.to_timedelta(rows["lap_start_sec"], unit="s")})
    found, report = clock.session_offset(openf1_laps, bronze_laps)
    assert abs((found - pd.Timestamp(t0)).total_seconds()) < 1e-5           # the 15-18 who agree, to the ms
    assert report["clock_outliers"] == outliers and report["clock_spread_ms"] <= 0.001


# --- the later laps: when every car's lap 1 is off -------------------------------------------------

def case(race):
    """OpenF1's and Bronze's lap tables (laps 1-15; Miami all) of a real race (fixture laps_clock_cases.csv)."""
    rows = pd.read_csv(FIXTURES / "laps_clock_cases.csv")
    rows = rows[rows["race"] == race]
    openf1_laps = rows[["driver_number", "lap_number", "date_start"]].dropna()
    ours = rows.dropna(subset=["lap_start_sec"])
    bronze_laps = pd.DataFrame({"DriverNumber": ours["driver_number"].astype(str), "LapNumber": ours["lap_number"] * 1.0,
                                "LapStartTime": pd.to_timedelta(ours["lap_start_sec"], unit="s")})
    return openf1_laps, bronze_laps


@pytest.mark.parametrize("race, method, t0", [
    ("Qatar 2024", "lap 1", "2024-12-01 15:08:43.187"),            # sound: the later laps agree within 0.01 s
    ("Japan 2024", "later laps", "2024-04-07 04:01:01.666"),       # red flag on lap 1: lap 1 is the restart, 28 min
    ("Monaco 2024", "later laps", "2024-05-26 12:08:05.403"),      # the same, 41 min
    ("Zandvoort 2023", "later laps", "2023-08-27 12:01:02.462"),   # every lap 1 2.1 s off
    ("Las Vegas 2024", "later laps", "2024-11-24 05:07:13.502"),   # every lap 1 0.45 s off
    ("Miami 2025", "later laps", "2025-05-04 19:06:26.855"),       # OpenF1 has no lap-1 start at all
])
def test_lap_one_is_checked_against_the_later_laps(race, method, t0):
    # The expected t0 are the whole race's. Las Vegas 2024 and Zandvoort 2023 were checked against the finish line in
    # another year (35.6 m away on the lap-1 clock, 4.3 m on the later laps'; 174 m against 7.2 m).
    found, report = clock.session_offset(*case(race))
    assert report["clock_method"] == method
    assert abs((found - pd.Timestamp(t0)).total_seconds()) < (0.002 if method == "lap 1" else 0.05)


def test_with_no_lap_one_and_no_later_lap_the_clock_refuses():
    openf1_laps, bronze_laps = case("Miami 2025")
    early = openf1_laps["lap_number"] <= 15                       # OpenF1 has none of Miami's first 15 lap starts
    with pytest.raises(clock.ClockError, match="0 of 0"):
        clock.session_offset(openf1_laps[early], bronze_laps)


# --- the samples: OpenF1's = FastF1's, on the same clock ------------------------------------------

def twins(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Each FastF1 sample with its OpenF1 twin: the two round the millisecond differently, so within 1 ms."""
    theirs = theirs.assign(Date=pd.to_datetime(theirs["Date"])).sort_values("Date")
    return pd.merge_asof(theirs, ours.sort_values("Date"), on="Date", by="DriverNumber", direction="nearest",
                         tolerance=pd.Timedelta(milliseconds=1), suffixes=("_ff1", "")).dropna(subset=["SessionTime"])


def baku_t0():
    openf1_laps, bronze_laps, _ = lap_one("Azerbaijan Grand Prix")
    return clock.session_offset(openf1_laps, bronze_laps)[0]


def test_openf1_car_samples_are_fastf1s_on_the_session_clock():
    raw = pd.read_parquet(FIXTURES / "baku_63_openf1_car_data.parquet")
    theirs = pd.read_parquet(FIXTURES / "baku_63_fastf1_car_telemetry.parquet")
    ours = clock.car_samples(raw, baku_t0())
    assert list(ours.columns) == ["DriverNumber", "Date", "SessionTime", "Speed", "RPM", "nGear", "Throttle", "Brake",
                                  "DRS"]
    both = twins(ours, theirs)
    assert len(both) == len(theirs) == 661                                    # every FastF1 sample has its twin
    assert (both["SessionTime"] - both["SessionTime_ff1"]).abs().max() <= pd.Timedelta(milliseconds=2)
    for col in ("Speed", "RPM", "nGear", "Throttle"):
        assert (both[col].astype(float) == both[f"{col}_ff1"].astype(float)).all(), col
    known = both["Brake"].notna()
    assert (both.loc[known, "Brake"] == both.loc[known, "Brake_ff1"]).all()  # FastF1's brake where there is a reading


def test_openf1_positions_are_fastf1s_on_the_session_clock():
    raw = pd.read_parquet(FIXTURES / "baku_63_openf1_location.parquet")
    theirs = pd.read_parquet(FIXTURES / "baku_63_fastf1_pos_telemetry.parquet")
    ours = clock.position_samples(raw, baku_t0())
    assert list(ours.columns) == ["DriverNumber", "Date", "SessionTime", "X", "Y", "Z"]
    both = twins(ours, theirs)
    assert len(both) == len(theirs) == 670
    assert (both["SessionTime"] - both["SessionTime_ff1"]).abs().max() <= pd.Timedelta(milliseconds=2)
    for col in ("X", "Y", "Z"):
        assert (both[col] == both[f"{col}_ff1"].astype(float)).all(), col


def test_104_on_the_brake_means_no_reading():
    raw = pd.DataFrame({"date": ["2024-12-01T16:00:00.100000+00:00", "2024-12-01T16:00:00.350000+00:00",
                                 "2024-12-01T16:00:00.600000+00:00"],
                        "driver_number": [1, 1, 1], "speed": [300, 120, 0], "rpm": [11000, 9000, 0],
                        "n_gear": [8, 3, 0], "throttle": [100, 0, 104], "brake": [0, 100, 104], "drs": [12, 0, None]})
    t0 = pd.Timestamp("2024-12-01 15:00:00")
    ours = clock.car_samples(raw, t0)
    assert ours["Brake"].tolist()[:2] == [False, True] and pd.isna(ours["Brake"].iloc[2])
    assert str(ours["nGear"].dtype) == "Int16"                             # raw codes such as 128 fit
    assert ours["Throttle"].tolist() == [100.0, 0.0, 104.0]                 # left for the resampler to blank
    assert ours["DriverNumber"].tolist() == ["1", "1", "1"] and pd.isna(ours["DRS"].iloc[2])
    assert ours["SessionTime"].dt.total_seconds().round(3).tolist() == [3600.1, 3600.35, 3600.6]


# --- Silver from OpenF1: the Qatar fixture race --------------------------------------------------

def add_openf1_fixture(root: str) -> None:
    """Put the Qatar OpenF1 fixture (VER and NOR, laps 1-6) into a lake's Bronze, as openf1.add_telemetry would."""
    for name in lake.BRONZE_OPENF1_TABLES:
        table = pd.read_parquet(FIXTURES / f"qatar_{name}.parquet")
        lake.write_table(table, lake.table_path(lake.BRONZE, name, *QATAR, "R", root))
    manifest = lake.read_manifest(*QATAR, root=root)
    manifest[lake.OPENF1] = {"session_key": 9655, "drivers": [1, 4], "missing": {"car_data": [], "location": []}}
    lake.write_manifest(manifest, *QATAR, root=root)


def test_silver_telemetry_from_openf1_on_qatar(golden_lake):
    add_openf1_fixture(golden_lake)
    assert lake.bronze_telemetry_source(*QATAR, root=golden_lake) == lake.OPENF1
    report = build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=True)["telemetry_report"]
    assert report["telemetry_source"] == lake.OPENF1 and report["clock_drivers"] == 20
    assert report["clock_spread_ms"] <= 5.0 and report["t0_utc"].startswith("2024-12-01 15:08:43.18")
    assert len(report["drivers_skipped"]) == 18                              # only VER and NOR are in the fixture
    tel = lake.read_table(lake.table_path(lake.SILVER, "telemetry", *QATAR, "R", golden_lake))
    assert set(tel["driver_number"]) == {"1", "4"}
    laps = tel[tel["lap_number"].between(2, 6)].groupby(["driver_number", "lap_number"])["lap_distance_m"].max()
    assert len(laps) == 10 and ((laps / LUSAIL_M - 1).abs() < 0.02).all()     # a whole lap is a lap of Lusail
    assert not tel.loc[tel["lap_number"].between(1, 6), "is_data_gap"].any()
