"""extractor.py: Bronze ingestion, using a fake session so no network is needed."""

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipeline import extractor, lake
from src.pipeline.extractor import (
    NOT_PUBLISHED, PUBLISHED, ExtractionError, FastF1Log, add_pit_stops, download_pit_stops, extract_session,
    parse_pit_stops,
)
from tests.helpers import make_bronze_laps, make_weather

API_PATH = "/static/2024/2024-01-01_Test_Grand_Prix/2024-01-01_Race/"
# The shape of F1's PitStopSeries, as the feed delivers it (values are text).
PIT_STOP_SERIES = {"PitTimes": {
    "1": [{"Timestamp": "2024-01-01T14:30:00.000Z",
           "PitStop": {"RacingNumber": "1", "PitStopTime": "2.4", "PitLaneTime": "21.283", "Lap": "10"}}],
    "2": {"0": {"Timestamp": "2024-01-01T14:31:00.000Z",          # numbered items instead of a list
                "PitStop": {"RacingNumber": "2", "PitStopTime": "10.8", "PitLaneTime": "31.069", "Lap": "11"}}},
}}


def fake_session(laps, car_drivers=("1", "2")):
    """An object with the same attributes extract_session reads from a FastF1 session."""
    tel = lambda: pd.DataFrame({"SessionTime": pd.to_timedelta([0.0, 0.25], unit="s"), "Speed": [100.0, 110.0]})
    return SimpleNamespace(
        api_path=API_PATH,
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
LAPS_ONLY_FILES = ["_manifest.json", "laps.parquet", "race_control.parquet", "results.parquet",
                   "session_status.parquet", "track_status.parquet", "weather.parquet"]


def test_writes_all_eight_tables_and_a_manifest(tmp_path):
    result = extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS)))

    folder = lake.session_dir(lake.BRONZE, 2024, 1, "R", tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        LAPS_ONLY_FILES + ["car_telemetry.parquet", "pos_telemetry.parquet"])

    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    assert manifest["event_name"] == "Test GP"
    assert manifest["total_laps"] == 2
    assert manifest["session_start_sec"] == 10.0
    assert manifest["tables"]["laps"]["rows"] == 2
    assert manifest["telemetry"] is True
    assert result["lap_rows"] == 2 and result["telemetry_rows"] == 8
    assert lake.has_bronze(2024, 1, root=tmp_path) and lake.bronze_has_telemetry(2024, 1, root=tmp_path)


def test_laps_only_writes_no_telemetry_tables(tmp_path):
    """No telemetry file at all, rather than an empty one that would claim there was no telemetry."""
    session = fake_session(make_bronze_laps(LAPS))
    result = extract_session(2024, 1, "R", lake_root=str(tmp_path), session=session, telemetry=False)

    folder = lake.session_dir(lake.BRONZE, 2024, 1, "R", tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == LAPS_ONLY_FILES
    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    assert manifest["telemetry"] is False
    assert "car_telemetry" not in manifest["tables"]
    assert result["telemetry"] is False and result["telemetry_rows"] == 0
    assert lake.has_bronze(2024, 1, root=tmp_path) and not lake.bronze_has_telemetry(2024, 1, root=tmp_path)
    assert manifest["pit_stops"] == NOT_PUBLISHED and not lake.bronze_has_pit_stops(2024, 1, root=tmp_path)


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


# --- F1's pit-stop record ------------------------------------------------------------

def test_pit_stop_record_is_kept_as_delivered():
    stops = parse_pit_stops(PIT_STOP_SERIES)
    assert stops.columns.tolist() == ["RacingNumber", "Lap", "PitStopTime", "PitLaneTime", "Timestamp"]
    assert stops.values.tolist() == [["1", "10", "2.4", "21.283", "2024-01-01T14:30:00.000Z"],
                                     ["2", "11", "10.8", "31.069", "2024-01-01T14:31:00.000Z"]]
    assert all(pd.api.types.is_string_dtype(stops[c]) for c in stops.columns)      # text, as in the feed
    assert parse_pit_stops({"PitTimes": {}}).empty


def test_a_published_pit_stop_record_is_written_with_the_session(tmp_path):
    extract_session(2024, 1, "R", lake_root=str(tmp_path), session=fake_session(make_bronze_laps(LAPS)),
                    pit_stops=parse_pit_stops(PIT_STOP_SERIES))
    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    assert manifest["pit_stops"] == PUBLISHED and manifest["tables"]["pit_stops"]["rows"] == 2
    stops = lake.read_table(lake.table_path(lake.BRONZE, "pit_stops", 2024, 1, root=tmp_path))
    assert stops["PitStopTime"].tolist() == ["2.4", "10.8"]


class Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.content = ("﻿" + json.dumps(payload or {})).encode("utf-8")   # the feed starts with a BOM


def test_download_tells_not_published_from_a_failed_download(monkeypatch):
    from fastf1.req import Cache

    asked = []
    answers = iter([Response(200, PIT_STOP_SERIES), Response(403), Response(500)])
    monkeypatch.setattr(Cache, "requests_get", lambda url, **kw: asked.append(url) or next(answers))
    assert len(download_pit_stops(API_PATH)) == 2
    assert asked[0] == "https://livetiming.formula1.com" + API_PATH + "PitStopSeries.json"
    assert download_pit_stops(API_PATH) is None                    # 403: F1 published no record
    with pytest.raises(ExtractionError, match="HTTP 500"):
        download_pit_stops(API_PATH)                               # a failed download is retried, never "none"

    def broken(url, **kw):
        raise ConnectionError("reset")

    monkeypatch.setattr(Cache, "requests_get", broken)
    with pytest.raises(ExtractionError, match="ConnectionError: reset"):
        download_pit_stops(API_PATH)


def test_pit_stops_can_be_added_to_older_bronze(tmp_path, monkeypatch):
    """Bronze ingested before the extractor fetched F1's record: only that file is added, the rest untouched."""
    session = fake_session(make_bronze_laps(LAPS))
    extract_session(2024, 1, "R", lake_root=str(tmp_path), session=session, telemetry=False)
    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    del manifest["pit_stops"]                                       # what a Bronze from before looks like
    lake.write_manifest(manifest, 2024, 1, root=tmp_path)
    laps_file = lake.table_path(lake.BRONZE, "laps", 2024, 1, root=tmp_path)
    written = laps_file.stat().st_mtime_ns

    monkeypatch.setattr(extractor, "_enable_cache", lambda cache_dir: None)          # leave FastF1's cache alone
    monkeypatch.setattr(extractor, "download_pit_stops", lambda api_path: parse_pit_stops(PIT_STOP_SERIES))
    assert add_pit_stops(2024, 1, "R", lake_root=str(tmp_path), api_path=API_PATH) == {"pit_stops": PUBLISHED, "rows": 2}
    manifest = lake.read_manifest(2024, 1, root=tmp_path)
    assert manifest["pit_stops"] == PUBLISHED and manifest["tables"]["pit_stops"]["rows"] == 2
    assert manifest["tables"]["laps"]["rows"] == 2 and laps_file.stat().st_mtime_ns == written
    assert lake.bronze_has_pit_stops(2024, 1, root=tmp_path)

    monkeypatch.setattr(extractor, "download_pit_stops", lambda api_path: None)
    lake.table_path(lake.BRONZE, "pit_stops", 2024, 1, root=tmp_path).unlink()
    assert add_pit_stops(2024, 1, "R", lake_root=str(tmp_path), api_path=API_PATH)["pit_stops"] == NOT_PUBLISHED


# --- listening to FastF1 while it loads ------------------------------------------
#
# FastF1 logs a part it failed to load in two records: the message at WARNING, then the
# exception at DEBUG. A data warning about a part it did load is one WARNING record.

ALIGN_WARNING = (logging.WARNING, "Failed to align laps for drivers: ['10']", None)
WEATHER_FAILED = [(logging.WARNING, "Failed to load weather data!", None),
                  (logging.DEBUG, "Traceback for failure in weather data", TimeoutError("read timed out"))]


def loading(session, messages=()):
    """Give a fake session a load() that logs like FastF1 and records the arguments it was called with."""
    session.load_args = {}

    def load(**kwargs):
        session.load_args = kwargs
        for level, text, err in messages:
            logging.getLogger("fastf1.core").log(level, text, exc_info=err)

    session.load = load
    return session


@pytest.fixture
def fake_fastf1(monkeypatch):
    """
    FastF1 with get_session faked and the cache untouched, and F1's pit-stop record faked (not
    published unless a test says so). Call it with the session to hand out.
    """
    import fastf1

    holder = {"pit_stops": None, "asked": []}
    monkeypatch.setattr(fastf1, "get_session", lambda *args: holder["session"])
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(extractor, "download_pit_stops",
                        lambda api_path: holder["asked"].append(api_path) or holder["pit_stops"])
    return lambda session, pit_stops=None: holder.update(session=session, pit_stops=pit_stops) or holder


def extract_via_fastf1(tmp_path, **kwargs):
    return extract_session(2024, 1, "R", lake_root=str(tmp_path / "lake"), cache_dir=str(tmp_path / "cache"), **kwargs)


def test_laps_only_does_not_ask_fastf1_for_telemetry(tmp_path, fake_fastf1):
    session = loading(fake_session(make_bronze_laps(LAPS)))
    fake_fastf1(session)
    extract_via_fastf1(tmp_path, telemetry=False)
    assert session.load_args["telemetry"] is False and session.load_args["laps"] is True


def test_the_pit_stop_record_is_downloaded_with_the_session(tmp_path, fake_fastf1):
    holder = fake_fastf1(loading(fake_session(make_bronze_laps(LAPS))), parse_pit_stops(PIT_STOP_SERIES))
    extract_via_fastf1(tmp_path, telemetry=False)
    assert holder["asked"] == [API_PATH]                             # the session's own path on F1's service
    assert lake.read_manifest(2024, 1, root=tmp_path / "lake")["pit_stops"] == PUBLISHED


def test_a_part_fastf1_failed_to_load_writes_nothing(tmp_path, fake_fastf1):
    """FastF1 swallows loader errors; an empty weather table must not pass as real Bronze."""
    fake_fastf1(loading(fake_session(make_bronze_laps(LAPS)), WEATHER_FAILED))
    with pytest.raises(ExtractionError, match=r"Failed to load weather data! \(TimeoutError: read timed out\)"):
        extract_via_fastf1(tmp_path)
    assert not (tmp_path / "lake").exists()


def test_a_fastf1_data_warning_is_kept_not_fatal(tmp_path, fake_fastf1):
    """Seen on 2024 round 2: FastF1 could not align one driver's laps. It loaded them: a warning."""
    fake_fastf1(loading(fake_session(make_bronze_laps(LAPS)), [ALIGN_WARNING]))
    result = extract_via_fastf1(tmp_path)
    assert result["fastf1_warnings"] == ["Failed to align laps for drivers: ['10']"]
    assert lake.read_manifest(2024, 1, root=tmp_path / "lake")["fastf1_warnings"] == result["fastf1_warnings"]


def test_failures_are_heard_even_when_the_app_silenced_fastf1(tmp_path, fake_fastf1):
    """main.py sets FastF1's logger to CRITICAL. The check must still work, and change nothing after."""
    logger = logging.getLogger("fastf1")
    before = logger.level
    handler_levels = [h.level for h in logger.handlers]
    logger.setLevel(logging.CRITICAL)
    try:
        fake_fastf1(loading(fake_session(make_bronze_laps(LAPS)), WEATHER_FAILED))
        with pytest.raises(ExtractionError, match="Failed to load weather data"):
            extract_via_fastf1(tmp_path)
        assert logger.level == logging.CRITICAL
        assert [h.level for h in logger.handlers] == handler_levels
    finally:
        logger.setLevel(before)


def test_fastf1_log_tells_failures_from_warnings():
    import fastf1  # noqa: F401  (real FastF1 logging setup: its logger passes DEBUG records on)

    log = FastF1Log()
    root = logging.getLogger("fastf1")
    root.addHandler(log)
    try:
        core = logging.getLogger("fastf1.core")
        core.info("Using cached data for weather_data")
        core.warning("Failed to align laps for drivers: ['10']")
        core.warning("Failed to load race control messages!")
        core.debug("Traceback for failure in race control messages", exc_info=ConnectionError("reset"))
        core.debug("an ordinary debug line")
    finally:
        root.removeHandler(log)
    assert log.failures == ["Failed to load race control messages! (ConnectionError: reset)"]
    assert log.warnings == ["Failed to align laps for drivers: ['10']"]
