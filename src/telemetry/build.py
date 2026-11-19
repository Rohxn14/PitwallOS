"""
Phase 5's Gold for one race (src/telemetry/build.py): the track model (step 5.1), then the corner features (5.2)

The pipeline (src/pipeline/runner.py, backfill.py) calls this after Gold whenever a race has Silver telemetry, so a
race added to the lake gets its corner tables without anyone running the two commands by hand. The command-line tools
(`python -m src.telemetry.track_model`, `... corner_features`) remain for rebuilding many races at once.

It never raises: a race whose positions are faulty (Hungary 2026) still has its laps, its Gold features and every
analysis; it only has no corner tables, and the returned summary says why.

One rule kept here: Gold's corner tables are either built from the race's CURRENT Silver telemetry or absent. If the
track model fails for any reason, the race's old track tables are removed, and the corner features then refuse the race
and remove their own old tables (an old map read as if it were current is how Hungary 2026 got corner features from a
line its contract had refused, found in step 5.2).
"""

from __future__ import annotations

from typing import Any, Dict

from src.pipeline import lake
from src.telemetry import corner_features, track_model


def build_race_telemetry(year: int, round_number: int, session: str = "R",
                         root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> Dict[str, Any]:
    """The track model and the corner features of one race. Returns a summary; never raises."""
    out: Dict[str, Any] = {"track_built": False, "corner_rows": 0, "corner_source": None,
                           "sector_error_median_ms": None, "telemetry_error": "", "no_official_corners": ""}
    circuit, why = track_model.circuit_for(year, round_number, session, root)
    out["no_official_corners"] = why
    try:
        track_model.build_track(year, round_number, session, root, circuit)
        out["track_built"] = True
    except Exception as err:
        out["telemetry_error"] = f"track model: {type(err).__name__}: {str(err)[:200]}"
        for table in (track_model.LINE_TABLE, track_model.CORNERS_TABLE):
            lake.remove_table(lake.table_path(lake.GOLD, table, year, round_number, session, root))
    try:
        row = corner_features.build_corner_features(year, round_number, session, root)
        out.update(corner_rows=row["rows"], corner_source=row["corner_source"],
                   sector_error_median_ms=row["sector_error_median_ms"])
    except Exception as err:
        if not out["telemetry_error"]:
            out["telemetry_error"] = f"corner features: {type(err).__name__}: {str(err)[:200]}"
    return out
