"""
Bronze layer: raw ingestion (src/pipeline/extractor.py)

Copies everything FastF1 knows about one session into the lake, unchanged.
Bronze applies NO business logic: no filtering, no renaming, no unit changes.
The only changes are the ones Parquet forces on us (see _to_storable).

Why keep it raw: FastF1 downloads can fail, be rate-limited, or change between
library versions. Once a session is in Bronze, every later stage can be rebuilt
offline, and a bug in cleaning never requires downloading again.

Where the downloads live: FastF1 keeps its own download cache. The replay and this
extractor ask FastF1 for exactly the same data (laps, car and position telemetry,
weather, race control), so they share ONE cache folder: the "FastF1 Cache Location"
in the app settings (default `.fastf1-cache/`). A race replayed once is ingested
here without touching the network, and the other way round.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.lib.settings import get_settings
from src.pipeline import lake


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


def _load_fastf1_session(year: int, round_number: int, session_type: str, cache_dir: str):
    """Download (or read from the shared cache) a fully loaded FastF1 session."""
    import os
    import fastf1  # imported here so Silver/Gold never need FastF1 installed or online

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, round_number, session_type)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    return session


def collect_tables(session) -> Dict[str, pd.DataFrame]:
    """Pull every raw table out of a loaded session object. Returns {table_name: DataFrame}."""
    warnings: List[str] = []
    drivers = [str(d) for d in session.drivers]
    tables = {
        "laps": _to_storable(session.laps),
        "car_telemetry": _per_driver(getattr(session, "car_data", None), drivers, warnings, "car telemetry"),
        "pos_telemetry": _per_driver(getattr(session, "pos_data", None), drivers, warnings, "position telemetry"),
        "weather": _to_storable(getattr(session, "weather_data", None)),
        "track_status": _to_storable(getattr(session, "track_status", None)),
        "race_control": _to_storable(getattr(session, "race_control_messages", None)),
        "results": _to_storable(getattr(session, "results", None)),
        "session_status": _to_storable(getattr(session, "session_status", None)),
    }
    tables["_warnings"] = warnings  # type: ignore[assignment]
    return tables


def extract_session(year: int, round_number: int, session_type: str = "R",
                    lake_root: str = lake.DEFAULT_LAKE_ROOT,
                    cache_dir: Optional[str] = None,
                    session: Any = None) -> Dict[str, Any]:
    """
    Ingest one session into Bronze.

    Args:
        year, round_number, session_type: which session ('R' race, 'Q' qualifying, 'S' sprint).
        lake_root: root folder of the lake.
        cache_dir: FastF1 download cache. None (the default) means the shared cache from the
            app settings, the one the replay uses.
        session: an already-loaded session object. Used by tests to avoid the network.

    Returns:
        Summary dict (paths, row counts, sizes, timing).
    """
    start = time.time()
    if session is None:
        session = _load_fastf1_session(year, round_number, session_type, cache_dir or fastf1_cache_dir())

    tables = collect_tables(session)
    warnings = tables.pop("_warnings")

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
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fastf1_version": fastf1_version,
        "warnings": warnings,
        "tables": table_stats,
    }
    manifest_path = lake.write_manifest(manifest, year, round_number, session_type, lake_root)

    return {
        "status": "success",
        "file_path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "year": int(year),
        "round_number": int(round_number),
        "event_name": identity["event_name"],
        "lap_rows": table_stats["laps"]["rows"],
        "telemetry_rows": table_stats["car_telemetry"]["rows"] + table_stats["pos_telemetry"]["rows"],
        "file_size_mb": round(total_bytes / 1024 / 1024, 2),
        "driver_count": len(session.drivers),
        "warnings": warnings,
        "elapsed_sec": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    print(extract_session(2024, 12, "R"))
