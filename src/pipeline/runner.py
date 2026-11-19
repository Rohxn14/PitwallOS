"""
Pipeline orchestrator and CLI (src/pipeline/runner.py)

Runs Bronze -> Silver -> Gold for one session and prints a summary.

    python -m src.pipeline.runner --year 2024 --round 12                  # downloads only what Bronze lacks
    python -m src.pipeline.runner --year 2024 --round 12 --skip-extract   # offline: from Bronze as it is
    python -m src.pipeline.runner --year 2024 --round 12 --no-telemetry   # laps only

Bronze is never downloaded twice: what is there is kept, and only what it lacks is fetched, from OpenF1
(openf1_bronze.py; docs/openf1_migration.md has why not FastF1 any more). Races and sprints go through the whole
pipeline; qualifying stops at Bronze (src/lib/lake_qualifying.py reads it). For many races at once, see backfill.py.
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import lake, openf1, openf1_bronze
from src.pipeline.cleaner import build_silver
from src.pipeline.feature_store import build_gold

# Windows terminals default to a legacy encoding that cannot print box characters.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

console = Console()

RACE = "R"


def _bronze_summary(year: int, round_number: int, session_type: str, lake_root: str) -> dict:
    """The Bronze summary, from the manifest."""
    manifest = lake.read_manifest(year, round_number, session_type, lake_root)
    tables = manifest["tables"]
    telemetry_tables = (openf1.CAR_TABLE, openf1.LOCATION_TABLE)
    lap_table = lake.openf1_table("laps") if manifest.get("source") == lake.OPENF1 else "laps"
    return {
        "status": "skipped",
        "file_path": str(lake.session_dir(lake.BRONZE, year, round_number, session_type, lake_root)),
        "event_name": manifest["event_name"],
        "lap_rows": tables[lap_table]["rows"],
        # a laps-only Bronze has no telemetry tables in its manifest
        "telemetry_rows": sum(tables.get(t, {}).get("rows", 0) for t in telemetry_tables),
        "file_size_mb": round(sum(t["bytes"] for t in tables.values()) / 1024 / 1024, 2),
        "warnings": manifest.get("warnings", []),
        "elapsed_sec": 0.0,
    }


def bronze_step(year: int, round_number: int, session_type: str, lake_root: str,
                skip_extract: bool, telemetry: bool) -> dict:
    """
    Bronze for one session, downloading only what it lacks (openf1_bronze.ingest_session decides what that is).
    A Bronze that did not come from OpenF1 (the FastF1-shaped test fixtures) is used as it is.
    """
    if skip_extract:
        return _bronze_summary(year, round_number, session_type, lake_root)
    if session_type not in openf1_bronze.RACE_LIKE:
        raise ValueError(f"session '{session_type}': only races and sprints ({', '.join(openf1_bronze.RACE_LIKE)}) go "
                         "through the pipeline. Qualifying is ingested by openf1_bronze and read from Bronze.")
    start = time.time()
    there = lake.has_bronze(year, round_number, session_type, lake_root)
    if there and lake.read_manifest(year, round_number, session_type, lake_root).get("source") != lake.OPENF1:
        return _bronze_summary(year, round_number, session_type, lake_root)
    out = openf1_bronze.ingest_session(year, round_number, session_type, lake_root, telemetry=telemetry)
    summary = _bronze_summary(year, round_number, session_type, lake_root)
    summary.update(status=out["status"], elapsed_sec=round(time.time() - start, 2))
    return summary


def run_pipeline(year: int = 2024, round_number: int = 12, session_type: str = RACE,
                 lake_root: str = lake.DEFAULT_LAKE_ROOT, skip_extract: bool = False,
                 telemetry: bool = True) -> dict:
    """
    Run the full pipeline for one session.

    Args:
        skip_extract: use Bronze as it is, never the network (works offline).
        telemetry: set False for laps only: no telemetry downloaded into Bronze, no Silver
            telemetry table. Much faster and ~20x smaller.

    Returns:
        {'bronze': {...}, 'silver': {...}, 'gold': {...}, 'telemetry_gold': {...} or None} summary dicts
        (telemetry_gold: the track model and corner features, src/telemetry/build.py; None for laps only).
    """
    if skip_extract and telemetry and not lake.bronze_has_telemetry(year, round_number, session_type, lake_root):
        raise ValueError(f"Bronze for {year} round {round_number} was ingested laps only, so Silver telemetry "
                         "cannot be built from it. Run again without --skip-extract to download telemetry, "
                         "or add --no-telemetry.")

    console.print(Panel("[bold cyan]PITWALL OS - DATA PIPELINE[/bold cyan]", border_style="cyan"))
    console.print(f"[bold yellow]Target:[/bold yellow] {year} round {round_number}, session '{session_type}'"
                  + ("" if telemetry else " (laps only)"))

    with console.status("[yellow][1/3] Bronze: raw tables, downloading only what is missing..."):
        bronze = bronze_step(year, round_number, session_type, lake_root, skip_extract, telemetry)
    with console.status("[green][2/3] Silver: cleaning laps" + (" and resampling telemetry..." if telemetry else "...")):
        silver = build_silver(year, round_number, session_type, lake_root, telemetry=telemetry)
    with console.status("[magenta][3/3] Gold: building lap features..."):
        gold = build_gold(year, round_number, session_type, lake_root)
    telemetry_gold = None
    if telemetry:
        # Phase 5's Gold: the track model and the corner features. Never stops the pipeline: a race whose positions
        # are faulty keeps its laps and features, and the summary says why it has no corner tables.
        from src.telemetry.build import build_race_telemetry
        with console.status("[cyan]Gold: track model and corner features..."):
            telemetry_gold = build_race_telemetry(year, round_number, session_type, lake_root)

    table = Table(title=f"{bronze['event_name']} {year}", border_style="green", header_style="bold gold1")
    for col, justify in (("Layer", "left"), ("Rows", "right"), ("Size", "right"), ("Time", "right")):
        table.add_column(col, justify=justify)
    table.add_row("Bronze (raw)", f"{bronze['lap_rows']:,} laps / {bronze['telemetry_rows']:,} telemetry",
                  f"{bronze['file_size_mb']:.2f} MB", f"{bronze['elapsed_sec']:.2f} s")
    tel_rows = silver["telemetry_report"].get("output_rows")
    silver_rows = f"{silver['clean_rows']:,} laps" + (f" / {tel_rows:,} telemetry" if tel_rows else "")
    table.add_row("Silver (clean)", silver_rows, f"{silver['file_size_mb']:.2f} MB", f"{silver['elapsed_sec']:.2f} s")
    table.add_row("Gold (features)", f"{gold['feature_rows']:,} rows x {gold['feature_count']} columns",
                  f"{gold['file_size_kb']:.1f} KB", f"{gold['elapsed_sec']:.2f} s")
    if telemetry_gold is not None:
        corners = (f"{telemetry_gold['corner_rows']:,} corner rows ({telemetry_gold['corner_source']} corners)"
                   if telemetry_gold["corner_rows"] else "not built")
        table.add_row("Gold (corners)", corners, "", "")
    console.print(table)

    report = silver["lap_report"]
    console.print(f"Laps dropped: {report['dropped_null_key']} unidentifiable, {report['dropped_duplicates']} duplicate. "
                  f"Clean laps: {report['clean_laps']} of {report['output_rows']}.")
    console.print(f"Pit-lane passages: {report['pit_tyre_change']} tyre changes, "
                  f"{report['pit_pass_through']} pass-throughs (no new tyres), "
                  f"{report['pit_ended']} ended in the pits, {report['pit_unknown']} unknown.")
    for warning in bronze.get("warnings", []):
        console.print(f"[yellow]Bronze warning:[/yellow] {warning}")
    if telemetry_gold is not None and telemetry_gold["telemetry_error"]:
        console.print(f"[yellow]No corner tables:[/yellow] {telemetry_gold['telemetry_error']}")
    if telemetry_gold is not None and telemetry_gold["no_official_corners"]:
        console.print(f"[yellow]No official corner list[/yellow] (corners numbered from the map): "
                      f"{telemetry_gold['no_official_corners']}")
    console.print(f"[bold green]Done.[/bold green] Gold features: {gold['file_path']}")

    return {"bronze": bronze, "silver": silver, "gold": gold, "telemetry_gold": telemetry_gold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PitwallOS data pipeline for one session")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--round", type=int, default=12)
    parser.add_argument("--session", type=str, default=RACE, help="'R' race, 'S' sprint")
    parser.add_argument("--skip-extract", action="store_true", help="Bronze as it is, never the network (offline)")
    parser.add_argument("--no-telemetry", action="store_true", help="laps only: no telemetry in Bronze or Silver")
    args = parser.parse_args()
    run_pipeline(args.year, args.round, args.session, skip_extract=args.skip_extract, telemetry=not args.no_telemetry)
