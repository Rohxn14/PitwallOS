"""
telemetry_clock.py: OpenF1's samples put on the session clock.

Golden facts from a source other than the code: FastF1's own samples of Baku 2026 driver 63, the same live feed, and
FastF1's own t0 for that race (fixtures in tests/fixtures/openf1/, cut from real downloads made while FastF1 could
still load them; no test touches the network). OpenF1's samples must be FastF1's, sample for sample. Then Silver
telemetry built from an OpenF1 Bronze on the Qatar 2024 fixture race: VER and NOR, laps 1-6, each lap as long as Lusail.
"""

from pathlib import Path

import pandas as pd

from src.pipeline import lake, telemetry_clock as clock
from tests.conftest import LUSAIL_M, QATAR

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"


def lap_one(race: str):
    """(OpenF1's lap table, Bronze's lap table, FastF1's own t0) for lap 1 of one 2026 race."""
    rows = pd.read_csv(FIXTURES / "lap1_2026.csv")
    rows = rows[(rows["race"] == race) & rows["lap_start_sec"].notna()]
    openf1_laps = pd.DataFrame({"driver_number": rows["driver_number"], "lap_number": 1,
                                "date_start": rows["date_start"]})
    bronze_laps = pd.DataFrame({"DriverNumber": rows["driver_number"].astype(str), "LapNumber": 1.0,
                                "LapStartTime": pd.to_timedelta(rows["lap_start_sec"], unit="s")})
    return openf1_laps, bronze_laps, pd.Timestamp(rows["fastf1_t0"].iloc[0])


# --- the samples: OpenF1's = FastF1's, on the same clock ------------------------------------------

def twins(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Each FastF1 sample with its OpenF1 twin: the two round the millisecond differently, so within 1 ms."""
    theirs = theirs.assign(Date=pd.to_datetime(theirs["Date"])).sort_values("Date")
    return pd.merge_asof(theirs, ours.sort_values("Date"), on="Date", by="DriverNumber", direction="nearest",
                         tolerance=pd.Timedelta(milliseconds=1), suffixes=("_ff1", "")).dropna(subset=["SessionTime"])


def baku_t0():
    return lap_one("Azerbaijan Grand Prix")[2]                 # FastF1's own t0 for that race


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

def test_silver_telemetry_from_openf1_on_qatar(openf1_qatar_lake):
    assert lake.bronze_telemetry_source(*QATAR, root=openf1_qatar_lake) == lake.OPENF1
    tel = lake.read_table(lake.table_path(lake.SILVER, "telemetry", *QATAR, "R", openf1_qatar_lake))
    assert set(tel["driver_number"]) == {"1", "4"}                            # only VER and NOR are in the fixture
    assert tel["session_time_sec"].min() == 0.0                               # the clock starts at lights out
    laps = tel[tel["lap_number"].between(2, 6)].groupby(["driver_number", "lap_number"])["lap_distance_m"].max()
    assert len(laps) == 10 and ((laps / LUSAIL_M - 1).abs() < 0.02).all()     # a whole lap is a lap of Lusail
    assert not tel.loc[tel["lap_number"].between(1, 6), "is_data_gap"].any()
