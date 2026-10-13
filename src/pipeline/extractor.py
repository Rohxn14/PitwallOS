"""
Bronze layer: raw ingestion (src/pipeline/extractor.py)

Copies everything FastF1 knows about one session into the lake, unchanged.
Bronze applies NO business logic: no filtering, no renaming, no unit changes.
The only changes are the ones Parquet forces on us (see _to_storable).

Why keep it raw: FastF1 downloads can fail, be rate-limited, or change between
library versions. Once a session is in Bronze, every later stage can be rebuilt
offline, and a bug in cleaning never requires downloading again.

Laps only: with telemetry=False, FastF1 is asked for timing, weather, track status
and race control, but not the car and position streams (about 95 % of the download).
Silver laps and Gold need nothing else. The one lap column FastF1 can then not fill
is LapStartDate (a wall-clock time it derives from telemetry); nothing downstream uses it.

Partial loads: FastF1 catches its own loading errors, logs a warning ("Failed to load
weather data!") and carries on, so a dropped connection can leave an empty weather table
that looks complete. The extractor listens to FastF1's log while a session loads and
refuses to write Bronze if any part failed (rule 2: never invent a value, and an empty
table silently standing in for missing data is an invented value). Every other FastF1
warning is kept in the manifest.

Where the downloads live: FastF1 keeps its own download cache. The replay and this
extractor ask FastF1 for exactly the same data (laps, car and position telemetry,
weather, race control), so they share ONE cache folder: the "FastF1 Cache Location"
in the app settings (default `.fastf1-cache/`). A race replayed once is ingested
here without touching the network, and the other way round.

Pit stops: F1's timing feed also publishes its own record of every pit stop, with the time
the car stood still in its box (PitStopTime) and the time from pit entry to pit exit
(PitLaneTime). FastF1 does not read that part of the feed, so the extractor downloads it
itself, through FastF1's download cache and rate limiter, into the `pit_stops` table. F1
has published it since the 2024 United States GP (round 19); for earlier races there is
no such file, and the manifest says "not published".
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from src.lib.settings import get_settings
from src.pipeline import lake

FASTF1_LOGGER = "fastf1"
# What FastF1's soft_exceptions wrapper logs at DEBUG, with the exception, right after the
# warning of a part it failed to load. This pair is what tells a failure from a warning.
FASTF1_FAILURE_TRACE = "Traceback for failure in"

# F1's pit-stop record (see the module docstring). The file server answers 403 for a file that
# does not exist, i.e. a session F1 published no pit-stop record for.
LIVETIMING_URL = "https://livetiming.formula1.com"
PIT_STOP_PAGE = "PitStopSeries.json"
PIT_STOPS_TABLE = lake.BRONZE_PIT_STOPS_TABLE
PIT_STOP_COLUMNS = ["RacingNumber", "Lap", "PitStopTime", "PitLaneTime", "Timestamp"]
NOT_PUBLISHED_HTTP = (403, 404)
PUBLISHED, NOT_PUBLISHED = "published", "not published"   # the manifest's "pit_stops"


class ExtractionError(RuntimeError):
    """FastF1 could not load part of a session. Nothing was written to Bronze."""


def fastf1_cache_dir() -> str:
    """The shared FastF1 download cache (the same setting the replay reads in f1_data.enable_cache)."""
    return get_settings().cache_location


def _to_storable(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Make a FastF1 table safe to write as Parquet without changing its meaning.

    - FastF1 returns DataFrame subclasses (Laps, Telemetry); convert to plain pandas.
    - Drop the index (FastF1 results are indexed by driver number, already a column).
    - 'object' columns can mix types (e.g. ClassifiedPosition holds '1' and 'R');
      Parquet needs one type per column, so they become nullable strings.
    Timedeltas and datetimes are stored natively.
    """
    if df is None:
        return pd.DataFrame()
    out = pd.DataFrame(df).reset_index(drop=True)
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def _per_driver(frames: Dict[str, pd.DataFrame], drivers: List[str], warnings: List[str], label: str) -> pd.DataFrame:
    """Stack per-driver telemetry frames into one table with a DriverNumber column."""
    parts = []
    for drv in drivers:
        frame = frames.get(drv) if frames is not None else None
        if frame is None or len(frame) == 0:
            warnings.append(f"No {label} for driver {drv}")
            continue
        part = _to_storable(frame)
        part.insert(0, "DriverNumber", str(drv))
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --- listening to FastF1 -------------------------------------------------------

class FastF1Log(logging.Handler):
    """
    Collects what FastF1 logs while one session loads.

    failures: parts FastF1 failed to load. Its loaders are wrapped in soft_exceptions, which
              logs the failure at WARNING and then, at DEBUG, FASTF1_FAILURE_TRACE with the
              exception. That pair turns the warning into a failure, with the cause appended
              (a timeout, the hourly rate limit, ...).
    warnings: every other warning, e.g. "Failed to align laps for drivers: ['10']": FastF1
              flagging a data problem in a part it did load.

    FastF1 logs most data warnings while it parses a download. A session read back from its
    own cache is not parsed again and logs none, so no warnings does not prove a clean session.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.warnings.append(message)
        elif message.startswith(FASTF1_FAILURE_TRACE) and record.exc_info and record.exc_info[1] is not None:
            err = record.exc_info[1]
            failed = self.warnings.pop() if self.warnings else message
            self.failures.append(f"{failed} ({type(err).__name__}: {err})")


@contextmanager
def listening_to_fastf1() -> Iterator[FastF1Log]:
    """
    Attach a FastF1Log to FastF1's logger for the duration of a block.

    The app may have silenced FastF1 (main.py sets its logger to CRITICAL), and a failure is
    only recognisable from a DEBUG record. So for the block the logger passes every record on,
    and each of its other handlers is raised to the level the logger had, so the console shows
    exactly what it showed before. Everything is restored afterwards.
    """
    logger = logging.getLogger(FASTF1_LOGGER)
    log = FastF1Log()
    old_level = logger.level
    handler_levels = [(h, h.level) for h in logger.handlers]
    for handler, level in handler_levels:
        handler.setLevel(max(level, old_level))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log)
    try:
        yield log
    finally:
        logger.removeHandler(log)
        logger.setLevel(old_level)
        for handler, level in handler_levels:
            handler.setLevel(level)


def quiet_fastf1_console(level: int = logging.WARNING) -> None:
    """Show only FastF1 warnings on the console (FastF1 prints ~12 INFO lines per session)."""
    import fastf1  # noqa: F401  (importing FastF1 installs its console handler)

    for handler in logging.getLogger(FASTF1_LOGGER).handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, FastF1Log):
            handler.setLevel(level)


def _enable_cache(cache_dir: str) -> None:
    import fastf1  # imported here so Silver/Gold never need FastF1 installed or online

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def _load_fastf1_session(year: int, round_number: int, session_type: str, cache_dir: str,
                         telemetry: bool = True):
    """
    Download (or read from the shared cache) one FastF1 session.

    Returns (session, warnings). Raises ExtractionError if FastF1 failed to load any part.
    """
    import fastf1

    _enable_cache(cache_dir)
    with listening_to_fastf1() as log:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=telemetry, weather=True, messages=True)

    if log.failures:
        raise ExtractionError(f"FastF1 could not load part of {year} round {round_number} '{session_type}': "
                              + "; ".join(log.failures))
    return session, log.warnings


def race_rounds(year: int, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Every championship race of a season, from FastF1's event schedule (testing left out).

    Returns [{'round', 'event_name', 'race_start_utc'}]. race_start_utc is a timezone-aware
    Timestamp, or None if the schedule has no race session for the event. Deciding which races
    have been run is the caller's job (see backfill.py).
    """
    import fastf1

    _enable_cache(cache_dir or fastf1_cache_dir())
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = []
    for rnd in schedule["RoundNumber"]:
        event = schedule.get_event_by_round(int(rnd))
        try:
            start = pd.Timestamp(event.get_session_date("R", utc=True))
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        except ValueError:
            start = None
        races.append({"round": int(rnd), "event_name": str(event["EventName"]),
                      "race_start_utc": None if start is None or pd.isna(start) else start})
    return races


def collect_tables(session, telemetry: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Pull every raw table out of a loaded session object. Returns {table_name: DataFrame}.

    With telemetry=False the two telemetry tables are left out entirely: no table at all is
    the truth ("not ingested"); an empty table would claim the session had no telemetry.
    """
    warnings: List[str] = []
    drivers = [str(d) for d in session.drivers]
    tables = {"laps": _to_storable(session.laps)}
    if telemetry:
        tables["car_telemetry"] = _per_driver(getattr(session, "car_data", None), drivers, warnings, "car telemetry")
        tables["pos_telemetry"] = _per_driver(getattr(session, "pos_data", None), drivers, warnings, "position telemetry")
    tables.update({
        "weather": _to_storable(getattr(session, "weather_data", None)),
        "track_status": _to_storable(getattr(session, "track_status", None)),
        "race_control": _to_storable(getattr(session, "race_control_messages", None)),
        "results": _to_storable(getattr(session, "results", None)),
        "session_status": _to_storable(getattr(session, "session_status", None)),
    })
    tables["_warnings"] = warnings  # type: ignore[assignment]
    return tables


# --- F1's pit-stop record ---------------------------------------------------------

def parse_pit_stops(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    F1's PitStopSeries (the final state of that part of the feed) -> one row per pit stop, every
    field as delivered (text), in PIT_STOP_COLUMNS. Nothing is converted, renamed or dropped here.
    """
    rows = []
    for number, entries in (payload.get("PitTimes") or {}).items():
        if isinstance(entries, dict):        # the feed can number a list's items instead of listing them
            entries = [entries[k] for k in sorted(entries, key=int)]
        for entry in entries or []:
            stop = entry.get("PitStop") or {}
            rows.append({"RacingNumber": stop.get("RacingNumber", number), "Lap": stop.get("Lap"),
                         "PitStopTime": stop.get("PitStopTime"), "PitLaneTime": stop.get("PitLaneTime"),
                         "Timestamp": entry.get("Timestamp")})
    return pd.DataFrame(rows, columns=PIT_STOP_COLUMNS).astype("string")


def download_pit_stops(api_path: str) -> Optional[pd.DataFrame]:
    """
    F1's pit-stop record for one session (api_path: FastF1's Session.api_path), as delivered.

    Returns None when F1 published none for the session. Raises ExtractionError when the download
    fails, so a broken connection is retried rather than written down as "not published".
    """
    from fastf1.req import Cache   # FastF1's download cache and rate limiter (enable the cache first)

    url = LIVETIMING_URL + api_path + PIT_STOP_PAGE
    try:
        response = Cache.requests_get(url)
    except Exception as err:
        raise ExtractionError(f"Failed to download F1's pit-stop record ({type(err).__name__}: {err})") from err
    if response.status_code in NOT_PUBLISHED_HTTP:
        return None
    if response.status_code != 200:
        raise ExtractionError(f"Failed to download F1's pit-stop record (HTTP {response.status_code}: {url})")
    return parse_pit_stops(json.loads(response.content.decode("utf-8-sig")))


def _write_pit_stops(pit_stops: pd.DataFrame, identity: Dict[str, Any], year: int, round_number: int,
                     session_type: str, lake_root: str) -> Dict[str, Any]:
    path = lake.table_path(lake.BRONZE, PIT_STOPS_TABLE, year, round_number, session_type, lake_root)
    lake.write_table(pit_stops, path, metadata=identity)
    return {"rows": int(len(pit_stops)), "bytes": int(path.stat().st_size), "columns": list(pit_stops.columns)}


def add_pit_stops(year: int, round_number: int, session_type: str = "R",
                  lake_root: str = lake.DEFAULT_LAKE_ROOT, cache_dir: Optional[str] = None,
                  api_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Add F1's pit-stop record to a session already in Bronze, ingested before the extractor fetched
    it. Only that one file is downloaded; the manifest is rewritten last, as in extract_session.

    api_path: the session's path on F1's timing service (default: from FastF1's event schedule).
    Returns {'pit_stops': PUBLISHED or NOT_PUBLISHED, 'rows': stops written}.
    """
    import fastf1

    manifest = lake.read_manifest(year, round_number, session_type, lake_root)
    _enable_cache(cache_dir or fastf1_cache_dir())
    if api_path is None:
        api_path = fastf1.get_session(year, round_number, session_type).api_path
    pit_stops = download_pit_stops(api_path)

    tables = dict(manifest.get("tables", {}))
    if pit_stops is not None:
        identity = {k: manifest[k] for k in ("season", "round", "session", "event_name", "location", "country")}
        tables[PIT_STOPS_TABLE] = _write_pit_stops(pit_stops, identity, year, round_number, session_type, lake_root)
    manifest.update(pit_stops=PUBLISHED if pit_stops is not None else NOT_PUBLISHED, tables=tables)
    lake.write_manifest(manifest, year, round_number, session_type, lake_root)
    return {"pit_stops": manifest["pit_stops"], "rows": 0 if pit_stops is None else int(len(pit_stops))}


def extract_session(year: int, round_number: int, session_type: str = "R",
                    lake_root: str = lake.DEFAULT_LAKE_ROOT,
                    cache_dir: Optional[str] = None,
                    session: Any = None,
                    telemetry: bool = True,
                    pit_stops: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Ingest one session into Bronze.

    Args:
        year, round_number, session_type: which session ('R' race, 'Q' qualifying, 'S' sprint).
        lake_root: root folder of the lake.
        cache_dir: FastF1 download cache. None (the default) means the shared cache from the
            app settings, the one the replay uses.
        session: an already-loaded session object. Used by tests to avoid the network.
        telemetry: False = laps only (no car / position telemetry downloaded or written).
        pit_stops: with `session` only: F1's pit-stop record for it, as delivered (None: not
            published). Without `session` it is downloaded along with the session.

    Returns:
        Summary dict (paths, row counts, sizes, timing).
    """
    start = time.time()
    fastf1_warnings: List[str] = []
    if session is None:
        session, fastf1_warnings = _load_fastf1_session(year, round_number, session_type,
                                                        cache_dir or fastf1_cache_dir(), telemetry=telemetry)
        pit_stops = download_pit_stops(session.api_path)

    tables = collect_tables(session, telemetry=telemetry)
    warnings = tables.pop("_warnings")
    if pit_stops is not None:
        tables[PIT_STOPS_TABLE] = pit_stops

    if tables["laps"].empty:
        raise ValueError(f"No lap data for {year} round {round_number} session '{session_type}'. Nothing written.")

    event = session.event
    total_laps = getattr(session, "total_laps", None)
    start_time = getattr(session, "session_start_time", None)
    identity = {
        "season": int(year),
        "round": int(round_number),
        "session": session_type,
        "event_name": str(event.get("EventName", f"Round {round_number}")),
        "location": str(event.get("Location", "")),
        "country": str(event.get("Country", "")),
    }

    table_stats = {}
    total_bytes = 0
    for name, df in tables.items():
        path = lake.table_path(lake.BRONZE, name, year, round_number, session_type, lake_root)
        lake.write_table(df, path, metadata=identity)
        size = path.stat().st_size
        total_bytes += size
        table_stats[name] = {"rows": int(len(df)), "bytes": int(size), "columns": list(df.columns)}

    try:
        import fastf1
        fastf1_version = fastf1.__version__
    except Exception:
        fastf1_version = "unknown"

    manifest = {
        **identity,
        "event_date": str(event.get("EventDate", "")),
        "total_laps": int(total_laps) if total_laps else None,
        "session_start_sec": start_time.total_seconds() if start_time is not None and pd.notna(start_time) else None,
        "driver_count": len(session.drivers),
        "telemetry": bool(telemetry),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fastf1_version": fastf1_version,
        "warnings": warnings,
        "fastf1_warnings": fastf1_warnings,
        "pit_stops": PUBLISHED if pit_stops is not None else NOT_PUBLISHED,
        "tables": table_stats,
    }
    # Written last: the manifest is what marks this Bronze as complete (lake.has_bronze).
    manifest_path = lake.write_manifest(manifest, year, round_number, session_type, lake_root)

    telemetry_rows = sum(table_stats[t]["rows"] for t in lake.BRONZE_TELEMETRY_TABLES if t in table_stats)
    return {
        "status": "success",
        "file_path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "year": int(year),
        "round_number": int(round_number),
        "event_name": identity["event_name"],
        "lap_rows": table_stats["laps"]["rows"],
        "telemetry": bool(telemetry),
        "telemetry_rows": telemetry_rows,
        "file_size_mb": round(total_bytes / 1024 / 1024, 2),
        "driver_count": len(session.drivers),
        "warnings": warnings,
        "fastf1_warnings": fastf1_warnings,
        "elapsed_sec": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    print(extract_session(2024, 12, "R"))
