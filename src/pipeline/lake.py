"""
Data lake layout and Parquet I/O (src/pipeline/lake.py)

Single source of truth for WHERE every table lives and HOW it is written.
No other module builds a path by hand.

Layout (one folder per session, same shape in every layer):

    data/
      bronze/season=2024/round=12/session=R/
          laps.parquet  car_telemetry.parquet  pos_telemetry.parquet
          weather.parquet  track_status.parquet  race_control.parquet
          results.parquet  session_status.parquet  _manifest.json
      silver/season=2024/round=12/session=R/
          laps.parquet  telemetry.parquet
      gold/season=2024/round=12/session=R/
          lap_features.parquet
      reports/
          backfill_2024.csv       one row per race: status, error, every quality count

Bronze telemetry (car_telemetry, pos_telemetry) is optional: a race ingested
laps only has no telemetry files, and its manifest says "telemetry": false.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_LAKE_ROOT = "data"

BRONZE, SILVER, GOLD = "bronze", "silver", "gold"
LAYERS = (BRONZE, SILVER, GOLD)

MANIFEST_NAME = "_manifest.json"
GOLD_TABLE = "lap_features"
BRONZE_TELEMETRY_TABLES = ("car_telemetry", "pos_telemetry")
BRONZE_PIT_STOPS_TABLE = "pit_stops"   # F1's own pit-stop record, only where F1 published one
REPORTS_DIR = "reports"
# UTF-8 with a byte-order mark: Excel on Windows reads a CSV without one as ANSI
# ("São Paulo" shows as "SÃ£o Paulo"). pandas reads it back with clean column names.
REPORT_ENCODING = "utf-8-sig"

PathLike = Union[str, Path]


def session_dir(layer: str, year: int, round_number: int, session: str = "R",
                root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Folder holding every table of one session in one layer."""
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer '{layer}'. Expected one of {LAYERS}.")
    return Path(root) / layer / f"season={int(year)}" / f"round={int(round_number)}" / f"session={session}"


def table_path(layer: str, table: str, year: int, round_number: int, session: str = "R",
               root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Full path of one Parquet table."""
    return session_dir(layer, year, round_number, session, root) / f"{table}.parquet"


def gold_features_path(year: int, round_number: int, session: str = "R",
                       root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Path of the Gold lap feature table (what analytics and the GUI open)."""
    return table_path(GOLD, GOLD_TABLE, year, round_number, session, root)


def write_table(df: pd.DataFrame, path: PathLike, metadata: Optional[Dict[str, object]] = None) -> Path:
    """
    Write a DataFrame to Parquet atomically.

    The file is written under a temporary name and renamed at the end, so a
    crash mid-write can never leave a half-written table that looks valid.
    `metadata` (session identity) is embedded in the Parquet schema so a file
    still describes itself after being copied out of its folder.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if metadata:
        existing = table.schema.metadata or {}
        extra = {str(k).encode(): str(v).encode() for k, v in metadata.items()}
        table = table.replace_schema_metadata({**existing, **extra})

    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def read_table(path: PathLike, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a Parquet table. Fails loudly if it does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lake table not found: {path}. Run the earlier pipeline stage first.")
    return pd.read_parquet(path, columns=columns)


def read_metadata(path: PathLike) -> Dict[str, str]:
    """Return the session metadata embedded in a table by write_table()."""
    raw = pq.read_schema(Path(path)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}


def write_manifest(manifest: dict, year: int, round_number: int, session: str = "R",
                   root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Write the Bronze manifest: what was ingested, when, and from which library version."""
    path = session_dir(BRONZE, year, round_number, session, root) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_manifest(year: int, round_number: int, session: str = "R",
                  root: PathLike = DEFAULT_LAKE_ROOT) -> dict:
    path = session_dir(BRONZE, year, round_number, session, root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Bronze manifest not found: {path}. Run Bronze extraction first.")
    return json.loads(path.read_text(encoding="utf-8"))


def has_bronze(year: int, round_number: int, session: str = "R", root: PathLike = DEFAULT_LAKE_ROOT) -> bool:
    """
    True once a session's Bronze is complete.

    The manifest is written after every table, so it marks completion: tables left by a
    crash before it are not Bronze, and the next extraction overwrites them.
    """
    return (session_dir(BRONZE, year, round_number, session, root) / MANIFEST_NAME).exists()


def bronze_has_telemetry(year: int, round_number: int, session: str = "R",
                         root: PathLike = DEFAULT_LAKE_ROOT) -> bool:
    """True if Bronze holds car and position telemetry (False for a laps-only ingestion)."""
    return all(table_path(BRONZE, t, year, round_number, session, root).exists() for t in BRONZE_TELEMETRY_TABLES)


def bronze_has_pit_stops(year: int, round_number: int, session: str = "R",
                         root: PathLike = DEFAULT_LAKE_ROOT) -> bool:
    """True if Bronze holds F1's pit-stop record (the manifest's "pit_stops" says why not, if not)."""
    return table_path(BRONZE, BRONZE_PIT_STOPS_TABLE, year, round_number, session, root).exists()


def report_path(name: str, root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """A report about the lake (CSV, for people), e.g. report_path('backfill_2024')."""
    return Path(root) / REPORTS_DIR / f"{name}.csv"


def write_report(df: pd.DataFrame, name: str, root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Write a report atomically (temporary file + rename), like every table."""
    path = report_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding=REPORT_ENCODING)
    os.replace(tmp, path)
    return path


def read_report(name: str, root: PathLike = DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """Read a report written by write_report."""
    return pd.read_csv(report_path(name, root), encoding=REPORT_ENCODING)


def list_sessions(layer: str, table: str, root: PathLike = DEFAULT_LAKE_ROOT) -> List[Path]:
    """Every file of `table` across all sessions in a layer, sorted."""
    return sorted((Path(root) / layer).glob(f"season=*/round=*/session=*/{table}.parquet"))


def read_layer(layer: str, table: str, root: PathLike = DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """Concatenate one table across every session in a layer (e.g. all Gold lap features)."""
    files = list_sessions(layer, table, root)
    if not files:
        raise FileNotFoundError(f"No '{table}' tables found under {Path(root) / layer}.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
