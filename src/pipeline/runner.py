"""
Pipeline orchestrator and CLI (src/pipeline/runner.py)

Runs Bronze -> Silver -> Gold for one session and prints a summary.

    python -m src.pipeline.runner --year 2024 --round 12
    python -m src.pipeline.runner --year 2024 --round 12 --skip-extract   # rebuild offline from Bronze
    python -m src.pipeline.runner --year 2024 --round 12 --no-telemetry   # laps only: no telemetry downloaded

For many races at once, see backfill.py.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import lake
from src.pipeline.cleaner import build_silver
from src.pipeline.extractor import extract_session
from src.pipeline.feature_store import build_gold

# Windows terminals default to a legacy encoding that cannot print box characters.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

console = Console()


def _bronze_summary(year: int, round_number: int, session_type: str, lake_root: str) -> dict:
    """Rebuild the Bronze summary from the manifest when extraction is skipped."""
    manifest = lake.read_manifest(year, round_number, session_type, lake_root)
    tables = manifest["tables"]
    return {
        "status": "skipped",
        "file_path": str(lake.session_dir(lake.BRONZE, year, round_number, session_type, lake_root)),
        "event_name": manifest["event_name"],
        "lap_rows": tables["laps"]["rows"],
        # a laps-only Bronze has no telemetry tables in its manifest
        "telemetry_rows": sum(tables.get(t, {}).get("rows", 0) for t in lake.BRONZE_TELEMETRY_TABLES),
        "file_size_mb": round(sum(t["bytes"] for t in tables.values()) / 1024 / 1024, 2),
        "warnings": manifest.get("warnings", []),
        "elapsed_sec": 0.0,
    }


def run_pipeline(year: int = 2024, round_number: int = 12, session_type: str = "R",
                 lake_root: str = lake.DEFAULT_LAKE_ROOT, skip_extract: bool = False,
                 telemetry: bool = True) -> dict:
    """
    Run the full pipeline for one session.

    Args:
        skip_extract: reuse existing Bronze instead of calling FastF1 (works offline).
        telemetry: set False for laps only: no telemetry downloaded into Bronze, no Silver
            telemetry table. Much faster and ~20x smaller.

    Returns:
        {'bronze': {...}, 'silver': {...}, 'gold': {...}} summary dicts.
    """
    if skip_extract and telemetry and not lake.bronze_has_telemetry(year, round_number, session_type, lake_root):
        raise ValueError(f"Bronze for {year} round {round_number} was ingested laps only, so Silver telemetry "
                         "cannot be built from it. Run again without --skip-extract to download telemetry, "
                         "or add --no-telemetry.")

    console.print(Panel("[bold cyan]PITWALL OS - DATA PIPELINE[/bold cyan]", border_style="cyan"))
    console.print(f"[bold yellow]Target:[/bold yellow] {year} round {round_number}, session '{session_type}'"
                  + ("" if telemetry else " (laps only)"))

    with console.status("[yellow][1/3] Bronze: ingesting raw FastF1 tables..."):
        bronze = (_bronze_summary(year, round_number, session_type, lake_root) if skip_extract
                  else extract_session(year, round_number, session_type, lake_root, telemetry=telemetry))
    with console.status("[green][2/3] Silver: cleaning laps" + (" and resampling telemetry..." if telemetry else "...")):
        silver = build_silver(year, round_number, session_type, lake_root, telemetry=telemetry)
    with console.status("[magenta][3/3] Gold: building lap features..."):
        gold = build_gold(year, round_number, session_type, lake_root)

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
    console.print(table)

    report = silver["lap_report"]
    console.print(f"Laps dropped: {report['dropped_null_key']} unidentifiable, {report['dropped_duplicates']} duplicate. "
                  f"Clean laps: {report['clean_laps']} of {report['output_rows']}.")
    console.print(f"Pit-lane passages: {report['pit_tyre_change']} tyre changes, "
                  f"{report['pit_pass_through']} pass-throughs (no new tyres), "
                  f"{report['pit_ended']} ended in the pits, {report['pit_unknown']} unknown.")
    for warning in bronze.get("warnings", []):
        console.print(f"[yellow]Bronze warning:[/yellow] {warning}")
    for warning in bronze.get("fastf1_warnings", []):
        console.print(f"[yellow]FastF1 warning:[/yellow] {warning}")
    console.print(f"[bold green]Done.[/bold green] Gold features: {gold['file_path']}")

    return {"bronze": bronze, "silver": silver, "gold": gold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PitwallOS data pipeline for one session")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--round", type=int, default=12)
    parser.add_argument("--session", type=str, default="R", help="'R' race, 'Q' qualifying, 'S' sprint")
    parser.add_argument("--skip-extract", action="store_true", help="reuse existing Bronze (offline)")
    parser.add_argument("--no-telemetry", action="store_true", help="laps only: no telemetry in Bronze or Silver")
    args = parser.parse_args()
    run_pipeline(args.year, args.round, args.session, skip_extract=args.skip_extract, telemetry=not args.no_telemetry)
