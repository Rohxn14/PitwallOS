"""
Data lake layout and Parquet I/O (src/pipeline/lake.py)

Single source of truth for WHERE every table lives and HOW it is written.
No other module builds a path by hand.

Layout (one folder per session, same shape in every layer):

    data/
      bronze/season=2024/round=12/session=R/
          openf1_laps.parquet  openf1_stints.parquet  openf1_pit.parquet  openf1_race_control.parquet  ...
          openf1_car_data.parquet  openf1_location.parquet  _manifest.json
      silver/season=2024/round=12/session=R/
          laps.parquet  telemetry.parquet  results.parquet
      gold/season=2024/round=12/session=R/
          lap_features.parquet
      reports/
          backfill_2024.csv       one row per race: status, error, every quality count
      reference/
          calendar_2024.json      the season's race weekends (src/lib/race_calendar.py)
          circuit_150_2024.json   a circuit as MultiViewer describes it: rotation, corners

Bronze holds OpenF1's tables as delivered, one per endpoint, named openf1_<endpoint> (openf1_table();
src/pipeline/openf1_bronze.py; manifest "source": "openf1"). session=R is a race, S a sprint, Q qualifying, SQ sprint
qualifying; only races and sprints go on to Silver and Gold.

Until November 2026 Bronze came from FastF1: laps, weather, track_status, results ... as FastF1's own tables. That
shape is still what Silver's cleaner reads (openf1_translate.py produces it from OpenF1's tables), and the test
fixtures in tests/fixtures/lake are Bronze of that kind: a manifest without "source" marks one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# "data" under the working folder, unless PITWALLOS_LAKE says where the lake is: a session in a git worktree
# points it at the one real lake. Do NOT link data/ into a worktree instead: on Windows `git worktree remove`
# follows a junction and deletes what it points to (Git 2.52, tested on a dummy folder, 2026-11-05).
DEFAULT_LAKE_ROOT = os.environ.get("PITWALLOS_LAKE", "data")

BRONZE, SILVER, GOLD = "bronze", "silver", "gold"
LAYERS = (BRONZE, SILVER, GOLD)

MANIFEST_NAME = "_manifest.json"
GOLD_TABLE = "lap_features"
BRONZE_OPENF1_TABLES = ("openf1_car_data", "openf1_location", "openf1_laps")   # what Silver telemetry is built from
OPENF1 = "openf1"                      # the manifest's "source" of a Bronze built from OpenF1, and its telemetry's
SILVER_RESULTS_TABLE = "results"       # the classification, translated from OpenF1's session_result
BRONZE_PIT_STOPS_TABLE = "pit_stops"   # F1's own pit-stop record, in a FastF1-shaped Bronze (the test fixtures)
REPORTS_DIR = "reports"
REFERENCE_DIR = "reference"      # small facts fetched once and kept: the calendar, the circuits (JSON)
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


def openf1_table(endpoint: str) -> str:
    """Bronze's table for one OpenF1 endpoint, as delivered: 'race_control' -> 'openf1_race_control'."""
    return f"{OPENF1}_{endpoint}"


def results_path(year: int, round_number: int, session: str = "R", root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """The classification in FastF1's shape: Silver's, translated from OpenF1; for a FastF1-shaped Bronze (the test
    fixtures), that Bronze's own."""
    silver = table_path(SILVER, SILVER_RESULTS_TABLE, year, round_number, session, root)
    return silver if silver.exists() else table_path(BRONZE, "results", year, round_number, session, root)


def gold_features_path(year: int, round_number: int, session: str = "R",
                       root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """Path of the Gold lap feature table (what analytics and the GUI open)."""
    return table_path(GOLD, GOLD_TABLE, year, round_number, session, root)


def remove_table(path: PathLike) -> bool:
    """Remove one derived table, if it is there (True if it was). For a table that a newer build refused to write:
    an older version must not outlive the refusal and be read as if it were current."""
    path = Path(path)
    if not path.exists():
        return False
    path.unlink()
    return True


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


def bronze_telemetry_source(year: int, round_number: int, session: str = "R",
                            root: PathLike = DEFAULT_LAKE_ROOT) -> Optional[str]:
    """Where Bronze's car and position telemetry came from: OPENF1, or None (laps only)."""
    def has(tables: Tuple[str, ...]) -> bool:
        return all(table_path(BRONZE, t, year, round_number, session, root).exists() for t in tables)
    # The manifest is written last: tables without its "openf1" entry are a crash's leftovers.
    if (has(BRONZE_OPENF1_TABLES) and has_bronze(year, round_number, session, root)
            and OPENF1 in read_manifest(year, round_number, session, root)):
        return OPENF1
    return None


def bronze_has_telemetry(year: int, round_number: int, session: str = "R",
                         root: PathLike = DEFAULT_LAKE_ROOT) -> bool:
    """True if Bronze holds car and position telemetry (False for a laps-only ingestion)."""
    return bronze_telemetry_source(year, round_number, session, root) is not None


def bronze_has_pit_stops(year: int, round_number: int, session: str = "R",
                         root: PathLike = DEFAULT_LAKE_ROOT) -> bool:
    """True if a FastF1-shaped Bronze (the test fixtures) holds F1's pit-stop record."""
    return table_path(BRONZE, BRONZE_PIT_STOPS_TABLE, year, round_number, session, root).exists()


def reference_path(name: str, root: PathLike = DEFAULT_LAKE_ROOT) -> Path:
    """A piece of reference data kept in the lake (JSON), e.g. reference_path('calendar_2024')."""
    return Path(root) / REFERENCE_DIR / f"{name}.json"


def write_json(content: object, path: PathLike) -> Path:
    """Write JSON atomically (temporary file + rename), like every table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: PathLike) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def list_sessions(layer: str, table: str, root: PathLike = DEFAULT_LAKE_ROOT,
                  session: Optional[str] = "R") -> List[Path]:
    """Every file of `table` across the races in a layer, sorted. `session`: which kind ("R" races, the default: the
    analytics and the models are about races; "S" sprints), or None for every session."""
    kind = "*" if session is None else session
    return sorted((Path(root) / layer).glob(f"season=*/round=*/session={kind}/{table}.parquet"))


def list_races(layer: str, table: str, root: PathLike = DEFAULT_LAKE_ROOT,
               session: str = "R") -> List[Tuple[int, int]]:
    """(season, round) of every `session` holding `table` in a layer, in calendar order (round 2 before 10)."""
    races = []
    for path in list_sessions(layer, table, root, session):
        season_part, round_part = path.parts[-4], path.parts[-3]
        races.append((int(season_part.split("=")[1]), int(round_part.split("=")[1])))
    return sorted(races)


def read_layer(layer: str, table: str, root: PathLike = DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """Concatenate one table across every session in a layer (e.g. all Gold lap features)."""
    files = list_sessions(layer, table, root)
    if not files:
        raise FileNotFoundError(f"No '{table}' tables found under {Path(root) / layer}.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
