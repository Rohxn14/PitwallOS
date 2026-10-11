"""extractor.py: Bronze ingestion, using a fake session so no network is needed."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipeline import lake
from src.pipeline.extractor import extract_session
from tests.helpers import make_bronze_laps, make_weather


def fake_session(laps, car_drivers=("1", "2")):
    """An object with the same attributes extract_session reads from a FastF1 session."""
    tel = lambda: pd.DataFrame({"SessionTime": pd.to_timedelta([0.0, 0.25], unit="s"), "Speed": [100.0, 110.0]})
    return SimpleNamespace(
        drivers=["1", "2"],
        laps=laps,
        car_data={d: tel() for d in car_drivers},
        pos_data={d: tel() for d in ("1", "2")},
        weather_data=make_weather([(0, 30.0, False)]),
        track_status=pd.DataFrame({"Time": pd.to_timedelta([0], unit="s"), "Status": ["1"], "Message": ["AllClear"]}),
        race_control_messages=pd.DataFrame({"Message": ["GREEN"]}),
        # ClassifiedPosition mixes numbers and letters in real data
        results=pd.DataFrame({"Abbreviation": ["AAA", "BBB"], "ClassifiedPosition": [1, "R"]}, index=["1", "2"]),
        session_status=pd.DataFrame({"Status": ["Started"]}),
        event={"EventName": "Test GP", "Location": "Testville", "Country": "Testland", "EventDate": "2024-01-01"},
        total_laps=2,
        session_start_time=pd.Timedelta(seconds=10),
    )


LAPS = [dict(driver="AAA", lap=1, start=10, end=100), dict(driver="BBB", lap=1, start=10, end=101)]


def test_writes_all_eight_tables_and_a_manifest(tmp_path):
    result = extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS)))

    folder = lake.session_dir(lake.BRONZE, 2024, 1, "R", tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == [
        "_manifest.json", "car_telemetry.parquet", "laps.parquet", "pos_telemetry.parquet",
        "race_control.parquet", "results.parquet", "session_status.parquet",
        "track_status.parquet", "weather.parquet"]

    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    assert manifest["event_name"] == "Test GP"
    assert manifest["total_laps"] == 2
    assert manifest["session_start_sec"] == 10.0
    assert manifest["tables"]["laps"]["rows"] == 2
    assert result["lap_rows"] == 2 and result["telemetry_rows"] == 8


def test_bronze_keeps_raw_names_and_types(tmp_path):
    extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS)))
    laps = lake.read_table(lake.table_path(lake.BRONZE, "laps", 2024, 1, root=tmp_path))
    assert "LapTime" in laps.columns and "lap_time_sec" not in laps.columns
    assert pd.api.types.is_timedelta64_dtype(laps["LapTime"])


def test_mixed_type_column_survives_as_text(tmp_path):
    extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS)))
    results = lake.read_table(lake.table_path(lake.BRONZE, "results", 2024, 1, root=tmp_path))
    assert results["ClassifiedPosition"].tolist() == ["1", "R"]


def test_driver_without_telemetry_becomes_a_warning_not_a_crash(tmp_path):
    session = fake_session(make_bronze_laps(LAPS), car_drivers=("1",))
    result = extract_session(2024, 1, "R", lake_root=str(tmp_path), session=session)
    assert result["warnings"] == ["No car telemetry for driver 2"]
    assert lake.read_manifest(2024, 1, root=tmp_path)["warnings"] == result["warnings"]


def test_session_with_no_laps_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="No lap data"):
        extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS).iloc[0:0]))
    assert not (tmp_path / "bronze").exists()
