"""
lake_session.py: a race for the replay, read from the lake instead of FastF1.

The lake is a temporary one built from real OpenF1 data: Qatar 2024's small tables (tests/fixtures/openf1/qatar_2024)
and the telemetry of two cars, VER and NOR, for laps 1-6 (tests/fixtures/openf1/qatar_openf1_*). The other 18 cars
have laps but no telemetry, as a car OpenF1 lacks. No network, no real lake, and FastF1 is never asked.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.f1_data as f1_data
from src.lib import lake_session
from src.lib.lake_session import LakeSession
from src.lib.running_order import official_positions
from src.pipeline import cleaner, feature_store, lake

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"
QATAR = (2024, 23)


def bronze_with_telemetry(root):
    bronze = lake.session_dir(lake.BRONZE, *QATAR, "R", root)
    shutil.copytree(FIXTURES / "qatar_2024", bronze)
    manifest = json.loads((bronze / lake.MANIFEST_NAME).read_text(encoding="utf-8"))
    for table in ("openf1_car_data", "openf1_location"):
        shutil.copy(FIXTURES / f"qatar_{table}.parquet", bronze / f"{table}.parquet")
        manifest["tables"][table] = {"rows": len(pd.read_parquet(bronze / f"{table}.parquet"))}
    manifest["telemetry"] = True
    (bronze / lake.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def qatar_lake(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("lake"))
    bronze_with_telemetry(root)
    cleaner.build_silver(*QATAR, "R", root, telemetry=True)
    feature_store.build_gold(*QATAR, "R", root)
    return root


@pytest.fixture(scope="module")
def session(qatar_lake):
    return LakeSession(*QATAR, root=qatar_lake)


# --- which sessions the lake can replay ------------------------------------------------------------------------

def test_a_race_is_replayable_once_silver_has_its_telemetry(tmp_path, qatar_lake):
    assert lake_session.available(*QATAR, root=qatar_lake)
    assert not lake_session.available(*QATAR, session_type="Q", root=qatar_lake)        # races only so far
    assert not lake_session.available(2024, 12, root=qatar_lake)                         # not in this lake
    bronze_with_telemetry(tmp_path)
    cleaner.build_silver(*QATAR, "R", str(tmp_path), telemetry=False)
    assert not lake_session.available(*QATAR, root=tmp_path)                             # laps only: nothing to draw
    with pytest.raises(FileNotFoundError, match="not in the lake with telemetry"):
        LakeSession(*QATAR, root=tmp_path)


def test_load_session_takes_a_race_from_the_lake_and_never_asks_fastf1(qatar_lake, monkeypatch):
    def no_fastf1(*args, **kwargs):
        raise AssertionError("FastF1 must not be asked for a race the lake has")

    monkeypatch.setattr(f1_data.fastf1, "get_session", no_fastf1)
    loaded = f1_data.load_session(*QATAR, "R", lake_root=qatar_lake)
    assert isinstance(loaded, LakeSession) and loaded.event["EventName"] == "Qatar Grand Prix"


# --- the attributes the replay reads -----------------------------------------------------------------------------

def test_it_carries_what_the_replay_reads_from_a_fastf1_session(session):
    assert len(session.drivers) == 20 and session.get_driver("1")["Abbreviation"] == "VER"
    assert "lake" in str(session) and ":" not in str(session)          # its own cache file, and a legal file name
    assert session.event["Location"] == "Lusail" and session.event["EventDate"].strftime("%Y") == "2024"
    assert {"Driver", "LapNumber", "LapTime", "Time", "PitInTime", "PitOutTime", "Compound", "TyreLife",
            "Position"}.issubset(session.laps.columns)
    assert session.results.iloc[0]["Abbreviation"] == "VER" and session.total_laps == 57
    status = session.track_status
    assert status["Time"].dtype.kind == "m" and "4" in set(status["Status"])     # timedeltas; the safety car is there
    deployed = session.race_control_messages[session.race_control_messages["Message"] == "SAFETY CAR DEPLOYED"]
    assert deployed["Time"].iloc[0].total_seconds() == pytest.approx(59.65, abs=0.05)   # a minute after lights out
    assert session.weather_data["AirTemp"].notna().all()
    assert session.driver_colors()["VER"] == (0x36, 0x71, 0xC6) and session.driver_colors()["NOR"] == (255, 128, 0)


def test_a_drivers_race_comes_as_arrays_on_the_session_clock(session):
    ver = session.driver_samples("1")
    data = ver["data"]
    assert ver["code"] == "VER" and ver["max_lap"] in (6, 7) and ver["t_min"] == pytest.approx(0.0, abs=0.11)
    assert all(len(values) == len(data["t"]) for values in data.values())
    assert np.all(np.diff(data["t"]) > 0) and not np.isnan(data["x"]).any() and not np.isnan(data["speed"]).any()
    assert np.all(np.diff(data["dist"]) >= -1e-6) and 5 * 5300 < data["dist"].max() < 7 * 5500     # Lusail: 5.42 km
    assert data["rel_dist"].min() >= 0 and data["rel_dist"].max() <= 1
    assert np.all(np.diff(data["progress"]) > -0.02)                    # laps done never go backwards
    assert set(np.unique(data["tyre"])) == {1.0} and data["tyre_life"].max() in (6, 7)   # mediums, six laps old
    assert session.driver_samples("44") is None                         # a car with laps but no telemetry


def test_the_timing_stream_gives_the_official_order_and_gaps(session):
    stream = session.timing_stream()
    assert list(stream.columns) == ["Time", "Driver", "Position", "GapToLeader", "IntervalToPositionAhead"]
    codes = ["VER", "NOR", "RUS"]
    numbers = {"1": "VER", "4": "NOR", "63": "RUS"}
    at = official_positions(stream, numbers, codes, np.array([-600.0, 120.0]))
    assert at[:, 0].tolist() == [2.0, 3.0, 1.0]                         # the grid: Russell on pole
    assert at[:, 1].tolist() == [1.0, 2.0, 3.0]                         # two minutes in: Verstappen leads
    gaps = stream[stream["GapToLeader"].notna()]
    assert len(gaps) > 250 and gaps["GapToLeader"].str.match(r"^(\+\d+\.\d{3}|\d+L)$").all()
    assert lake_session._gap_text("+1 LAP") == "1L" and lake_session._gap_text(6.031) == "+6.031"
    assert pd.isna(lake_session._gap_text(None))


def test_the_example_lap_is_a_whole_fast_lap_for_the_tracks_outline(session):
    lap = session.example_lap()
    assert {"X", "Y", "Distance", "RelativeDistance", "Speed", "DRS"}.issubset(lap.columns)
    assert 5300 < lap["Distance"].max() < 5500 and lap["RelativeDistance"].max() == pytest.approx(1.0)
    closed = np.hypot(lap["X"].iloc[0] - lap["X"].iloc[-1], lap["Y"].iloc[0] - lap["Y"].iloc[-1]) / 10
    assert closed < 60                                                  # it ends where it began (metres)
    assert f1_data.example_lap_telemetry(session).equals(lap)


# --- the frames ---------------------------------------------------------------------------------------------------

def test_the_replays_frames_are_built_from_the_lake(session, tmp_path, monkeypatch):
    monkeypatch.setattr(f1_data.get_settings().__class__, "computed_data_location",
                        property(lambda self: str(tmp_path)), raising=False)
    data = f1_data.get_race_telemetry(session)
    frames = data["frames"]
    assert set(frames[0]["drivers"]) == {"VER", "NOR"} and data["running_order_source"] == "official timing"
    assert frames[0]["t"] == 0.0 and len(frames) > 25 * 60 * 10          # 25 frames a second, over ten minutes
    late = frames[25 * 300]["drivers"]                                   # five minutes in, behind the safety car
    assert late["VER"]["position"] == 1 and late["NOR"]["position"] == 2 and late["VER"]["lap"] == 3
    assert late["VER"]["tyre"] == 1.0 and 0 < late["VER"]["speed"] < 250
    assert [s["status"] for s in data["track_statuses"] if s["start_time"] > 0][:2] == ["2", "4"]
    assert data["driver_colors"]["NOR"] == (255, 128, 0) and data["t0_session_s"] == pytest.approx(0.0, abs=0.11)
    assert any("lake" in p.name for p in tmp_path.iterdir())             # cached under its own name
    assert f1_data.get_race_telemetry(session)["frames"][100] == frames[100]      # and read back from that cache


# --- the telemetry of a lap a car never finished -------------------------------------------------------------------

def car(times_and_speeds):
    return pd.DataFrame({"SessionTime": pd.to_timedelta([t for t, _ in times_and_speeds], unit="s"),
                         "Speed": [v for _, v in times_and_speeds]})


def test_a_retired_cars_telemetry_ends_where_it_came_to_rest():
    drive = [(t, 200.0) for t in range(100, 160)] + [(t, 0.0) for t in range(160, 400)]
    assert cleaner.stopped_at(car(drive), 100.0, 5000.0) == 160.0 + cleaner.STOPPED_MARGIN_SEC
    # an hour later the car moves again, on a recovery truck: not its race any more
    truck = drive + [(t, 30.0) for t in range(3600, 3700)]
    assert cleaner.stopped_at(car(truck), 100.0, 5000.0) == 160.0 + cleaner.STOPPED_MARGIN_SEC
    # a pit stop is not coming to rest
    stop = [(t, 80.0) for t in range(100, 120)] + [(t, 0.0) for t in range(120, 140)] + [(t, 60.0) for t in range(140, 150)]
    assert cleaner.stopped_at(car(stop), 100.0, 5000.0) == 149.0 + cleaner.STOPPED_MARGIN_SEC
    assert cleaner.stopped_at(car(drive), 100.0, 162.0) == 162.0                      # never past the race's end
    assert cleaner.stopped_at(car([(t, 0.0) for t in range(100, 200)]), 100.0, 5000.0) == 100.0   # it never moved
    assert cleaner.stopped_at(car(drive), 4800.0, 4400.0) == 4400.0   # a lap row dated after the flag adds nothing
