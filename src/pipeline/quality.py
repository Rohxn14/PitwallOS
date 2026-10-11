"""
Data quality per race (src/pipeline/quality.py)

What the pipeline flagged and dropped for one race, counted from the lake. One definition,
used by the backfill report (one row per race) and by the Analyze window's Data quality tab,
so the two can never disagree.

These are counts, not judgements: a race with 300 safety-car laps is not "bad data". The
report shows them side by side so an odd race stands out against the other races.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from src.pipeline import lake


def data_quality(laps: pd.DataFrame) -> Dict[str, int]:
    """What the pipeline flagged for this race, counted from Silver's lap flags (in driver-laps)."""
    flag = lambda name: int(laps[name].sum()) if name in laps.columns else 0
    kinds = laps["pit_kind"].value_counts() if "pit_kind" in laps.columns else pd.Series(dtype=int)
    return {
        "laps": int(len(laps)),
        "drivers": int(laps["driver_code"].nunique()) if len(laps) else 0,
        "clean_laps": flag("is_clean_lap"),
        "no_lap_time": int(laps["lap_time_sec"].isna().sum()) if "lap_time_sec" in laps.columns else 0,
        "lap_1": flag("is_lap1"),
        "pit_laps": flag("is_pit_lap"),
        "not_green": int((~laps["is_green"]).sum()) if "is_green" in laps.columns else 0,
        "safety_car_laps": flag("is_sc"),
        "vsc_laps": flag("is_vsc"),
        "red_flag_laps": flag("is_red"),
        "inaccurate": int((~laps["is_accurate"]).sum()) if "is_accurate" in laps.columns else 0,
        "wet_tyre_laps": flag("is_wet_tyre"),
        "deleted_laps": flag("is_deleted"),
        "tyre_changes": int(kinds.get("tyre_change", 0)),
        "pass_throughs": int(kinds.get("pass_through", 0)),
        "ended_in_pits": int(kinds.get("ended", 0)),
        "unknown_passages": int(kinds.get("unknown", 0)),
    }


def lap_ranges(lap_numbers: Iterable[int]) -> str:
    """Race lap numbers as compact ranges: [1, 2, 3, 4, 34, 35, 36, 40] -> '1-4, 34-36, 40'."""
    runs = []
    for n in sorted({int(n) for n in lap_numbers}):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def race_quality(year: int, round_number: int, session: str = "R",
                 root: str = lake.DEFAULT_LAKE_ROOT) -> Dict[str, Any]:
    """
    One row of the season report for a race whose Bronze and Silver laps exist.

    Columns, in groups:
      identity    event_name, event_date, total_laps (official), max_lap (in the data)
      rows        bronze_lap_rows -> laps (Silver) -> gold_rows; dropped_rows = Bronze - Silver
      flags       every data_quality count (driver-laps), clean_pct, weather_pct
      when        sc / vsc / red_flag_lap_numbers: race laps with that status, as ranges
      source      bronze_telemetry, bronze_warnings, fastf1_warnings
    """
    manifest = lake.read_manifest(year, round_number, session, root)
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, session, root))
    gold_path = lake.gold_features_path(year, round_number, session, root)
    gold_rows = len(pd.read_parquet(gold_path, columns=["lap_number"])) if gold_path.exists() else None

    counts = data_quality(laps)
    bronze_rows = int(manifest["tables"]["laps"]["rows"])
    laps_with = lambda flag: laps.loc[laps[flag], "lap_number"]
    return {
        "event_name": manifest.get("event_name"),
        "event_date": str(manifest.get("event_date", ""))[:10],
        "total_laps": manifest.get("total_laps"),
        "max_lap": int(laps["lap_number"].max()) if len(laps) else None,
        "bronze_lap_rows": bronze_rows,
        "dropped_rows": bronze_rows - counts["laps"],
        "gold_rows": gold_rows,
        **counts,
        "clean_pct": round(100.0 * counts["clean_laps"] / counts["laps"], 1) if counts["laps"] else None,
        "weather_pct": round(100.0 * float(laps["track_temp_c"].notna().mean()), 1) if len(laps) else None,
        "sc_lap_numbers": lap_ranges(laps_with("is_sc")),
        "vsc_lap_numbers": lap_ranges(laps_with("is_vsc")),
        "red_flag_lap_numbers": lap_ranges(laps_with("is_red")),
        "bronze_telemetry": lake.bronze_has_telemetry(year, round_number, session, root),
        "bronze_warnings": len(manifest.get("warnings", [])),
        "fastf1_warnings": len(manifest.get("fastf1_warnings", [])),
    }
