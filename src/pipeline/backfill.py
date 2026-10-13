"""
Phase 1 backfill: many races into the lake with one command (src/pipeline/backfill.py)

    python -m src.pipeline.backfill --season 2024                   # every 2024 race run so far, laps only
    python -m src.pipeline.backfill --season 2022 2023 2024 2025
    python -m src.pipeline.backfill --season 2024 --rounds 12 23    # only these rounds
    python -m src.pipeline.backfill --season 2024 --rebuild         # redo Silver + Gold from Bronze (offline)
    python -m src.pipeline.backfill --season 2024 --telemetry       # also telemetry (~100 MB download per race)
    python -m src.pipeline.backfill --season 2024 --pit-stops       # add F1's pit-stop record to older Bronze

The same Bronze -> Silver -> Gold steps as runner.py, in a loop, with four properties that a
single run does not need:

  Resumable  A race whose Gold exists is skipped. A race whose Bronze exists is rebuilt from it
             without downloading. Stopping (Ctrl+C, a crash, a lost connection) and running the
             same command again never repeats finished work.
  Isolated   One race failing never stops the others. It is reported with the stage that failed
             (bronze / silver / gold) and the error, and is tried again on the next run.
  Retried    Only the download is retried: a contract failure is deterministic, retrying it would
             fail the same way. FastF1 caps itself at 500 requests per hour and raises when the
             cap is hit; that gets a long wait instead of a short one.
  Reported   data/reports/backfill_<season>.csv, one row per race: what happened in the latest
             run, and every quality count (quality.py). Rewritten after each race, so an
             interrupted run still leaves a report. Races not in this run keep their row.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from src.pipeline import extractor, lake, quality
from src.pipeline.cleaner import build_silver
from src.pipeline.feature_store import build_gold

if sys.platform == "win32":   # legacy Windows consoles cannot print box characters
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

console = Console()

SESSION = "R"
# A race's timing data is published some time after the flag. Before this delay it is not "run yet".
RACE_DATA_DELAY = pd.Timedelta(hours=6)

OK, FAILED = "ok", "failed"
# Where a race's Bronze came from in this run.
FROM_FASTF1 = "fastf1"      # extracted now (FastF1's cache if it had it, else downloaded)
FROM_BRONZE = "bronze"      # Bronze already in the lake: Silver + Gold rebuilt from it, offline
ADDED_PIT_STOPS = "bronze+stops"   # Bronze in the lake; F1's pit-stop record downloaded; Silver + Gold rebuilt
ALREADY_BUILT = "existing"  # Gold already in the lake: nothing done

REPORT_LEAD = ["season", "round", "event_name", "status", "stage", "error", "source", "seconds"]


@dataclass(frozen=True)
class RetryPolicy:
    """How a failed download is retried. Silver and Gold are never retried."""
    waits_sec: Tuple[float, ...] = (30.0, 120.0)   # ordinary failures: two more tries
    rate_limit_wait_sec: float = 600.0             # FastF1's hourly cap: wait 10 minutes ...
    rate_limit_max_waits: int = 6                  # ... up to an hour in total


def _one_line(err: BaseException) -> str:
    return f"{type(err).__name__}: " + " | ".join(line.strip() for line in str(err).splitlines() if line.strip())


def _is_rate_limit(err: BaseException) -> bool:
    # Raised directly, or caught by FastF1 inside a loader and reported through ExtractionError.
    return "RateLimitExceeded" in _one_line(err)


def is_built(year: int, round_number: int, telemetry: bool, root: str) -> bool:
    """Gold exists (and Silver telemetry too, if telemetry is wanted)."""
    if not lake.gold_features_path(year, round_number, SESSION, root).exists():
        return False
    return not telemetry or lake.table_path(lake.SILVER, "telemetry", year, round_number, SESSION, root).exists()


def has_usable_bronze(year: int, round_number: int, telemetry: bool, root: str) -> bool:
    """Bronze is complete, and holds telemetry if telemetry is wanted."""
    if not lake.has_bronze(year, round_number, SESSION, root):
        return False
    return not telemetry or lake.bronze_has_telemetry(year, round_number, SESSION, root)


def needs_pit_stops(year: int, round_number: int, root: str) -> bool:
    """Bronze is there, but was ingested before the extractor fetched F1's pit-stop record."""
    return (lake.has_bronze(year, round_number, SESSION, root)
            and "pit_stops" not in lake.read_manifest(year, round_number, SESSION, root))


def extract_with_retry(year: int, round_number: int, *, root: str, telemetry: bool,
                       policy: RetryPolicy = RetryPolicy(), sleep: Callable[[float], None] = time.sleep,
                       say: Callable[[str], None] = console.print) -> Dict[str, Any]:
    """extractor.extract_session, retried according to `policy`. Raises the last error if it never succeeds."""
    return with_retry(lambda: extractor.extract_session(year, round_number, SESSION, root, telemetry=telemetry),
                      round_number, policy=policy, sleep=sleep, say=say)


def with_retry(download: Callable[[], Dict[str, Any]], round_number: int, *, policy: RetryPolicy = RetryPolicy(),
               sleep: Callable[[float], None] = time.sleep,
               say: Callable[[str], None] = console.print) -> Dict[str, Any]:
    """Run one download, retried according to `policy`. Raises the last error if it never succeeds."""
    tries = rate_limit_waits = 0
    while True:
        try:
            return download()
        except Exception as err:
            if _is_rate_limit(err) and rate_limit_waits < policy.rate_limit_max_waits:
                rate_limit_waits += 1
                wait = policy.rate_limit_wait_sec
            elif tries < len(policy.waits_sec):
                wait = policy.waits_sec[tries]
                tries += 1
            else:
                raise
            say(f"   [yellow]round {round_number}: {escape(_one_line(err)[:200])}; retrying in {wait:.0f} s[/yellow]")
            sleep(wait)


def build_race(year: int, round_number: int, *, root: str = lake.DEFAULT_LAKE_ROOT, telemetry: bool = False,
               rebuild: bool = False, pit_stops: bool = False, event_name: Optional[str] = None,
               policy: RetryPolicy = RetryPolicy(), sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """
    Bring one race up to Gold, doing only the work that is missing. Never raises for a race
    failure: the returned row says what happened. pit_stops=True also adds F1's pit-stop record
    to a Bronze ingested without it (the only download for that race) and rebuilds from it.
    """
    start = time.time()
    row: Dict[str, Any] = {"season": int(year), "round": int(round_number), "event_name": event_name,
                           "status": OK, "stage": "", "error": "", "source": ALREADY_BUILT}

    add_stops = pit_stops and needs_pit_stops(year, round_number, root)
    if rebuild or add_stops or not is_built(year, round_number, telemetry, root):
        stage = "bronze"
        try:
            if has_usable_bronze(year, round_number, telemetry, root):
                row["source"] = FROM_BRONZE
                if add_stops:
                    row["source"] = ADDED_PIT_STOPS
                    with_retry(lambda: extractor.add_pit_stops(year, round_number, SESSION, root), round_number,
                               policy=policy, sleep=sleep)
            else:
                row["source"] = FROM_FASTF1
                extract_with_retry(year, round_number, root=root, telemetry=telemetry, policy=policy, sleep=sleep)
            stage = "silver"
            build_silver(year, round_number, SESSION, root, telemetry=telemetry)
            stage = "gold"
            build_gold(year, round_number, SESSION, root)
        except Exception as err:
            row.update(status=FAILED, stage=stage, error=_one_line(err))

    if row["status"] == OK:
        try:
            row.update(quality.race_quality(year, round_number, SESSION, root))
        except Exception as err:
            row.update(status=FAILED, stage="report", error=_one_line(err))
    elif lake.has_bronze(year, round_number, SESSION, root):
        row["event_name"] = lake.read_manifest(year, round_number, SESSION, root).get("event_name", event_name)

    row["seconds"] = round(time.time() - start, 1)
    return row


def races_to_run(season: int, rounds: Optional[Sequence[int]] = None,
                 now: Optional[pd.Timestamp] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    (races to build, races left out because they have not been run yet).

    With explicit `rounds` the schedule is not read at all (works offline). Otherwise every race
    of the season from FastF1's schedule whose data should be published by `now`.
    """
    if rounds:
        return [{"round": int(r), "event_name": None} for r in rounds], []
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    run, not_yet = [], []
    for race in extractor.race_rounds(season):
        started = race["race_start_utc"]
        (run if started is not None and started + RACE_DATA_DELAY <= now else not_yet).append(race)
    return run, not_yet


def _report_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    lead = [c for c in REPORT_LEAD if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def save_report(season: int, rows: List[Dict[str, Any]], root: str) -> Path:
    """Write this run's rows into the season report, keeping earlier rows for rounds not in this run."""
    name = f"backfill_{season}"
    new = _report_frame(rows)
    if lake.report_path(name, root).exists():
        old = lake.read_report(name, root)
        new = pd.concat([old[~old["round"].isin(new["round"])], new], ignore_index=True)
    new = _report_frame(new.sort_values("round").to_dict("records"))
    return lake.write_report(new, name, root)


def _describe(row: Dict[str, Any]) -> str:
    if row["status"] == FAILED:
        return f"[bold red]FAILED at {row['stage']}[/bold red]  {escape(str(row['error'])[:240])}"
    sc = row.get("sc_lap_numbers") or "-"
    times = f"  stop times {row['pit_stop_times']:>2}" if row.get("pit_stop_times") else ""
    return (f"[green]ok[/green]  {row['source']:<12} {row['laps']:>5,} laps  clean {row['clean_pct']:>5.1f}%  "
            f"stops {row['tyre_changes']:>3}{times}  SC {escape(sc)}")


def backfill(season: int, rounds: Optional[Sequence[int]] = None, *, root: str = lake.DEFAULT_LAKE_ROOT,
             telemetry: bool = False, rebuild: bool = False, pit_stops: bool = False,
             policy: RetryPolicy = RetryPolicy(), sleep: Callable[[float], None] = time.sleep,
             now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Build every race of a season (or the given rounds). Returns this run's report rows."""
    races, not_yet = races_to_run(season, rounds, now)
    mode = "laps + telemetry" if telemetry else "laps only"
    extras = (", rebuild" if rebuild else "") + (", pit stops" if pit_stops else "")
    console.rule(f"[bold cyan]Backfill {season}[/bold cyan]: {len(races)} races, {mode}{extras}")
    if not_yet:
        console.print(f"Not run yet, left out: {', '.join(r['event_name'] for r in not_yet)}")

    rows: List[Dict[str, Any]] = []
    for i, race in enumerate(races, 1):
        row = build_race(season, race["round"], root=root, telemetry=telemetry, rebuild=rebuild, pit_stops=pit_stops,
                         event_name=race["event_name"], policy=policy, sleep=sleep)
        rows.append(row)
        save_report(season, rows, root)
        name = escape(str(row.get("event_name") or f"Round {race['round']}"))
        console.print(f"[{i:>2}/{len(races)}] R{race['round']:02d} {name:<28} {_describe(row)}  ({row['seconds']:.1f} s)")

    report = _report_frame(rows) if rows else pd.DataFrame(columns=REPORT_LEAD)
    _print_summary(season, report, root)
    return report


def _print_summary(season: int, report: pd.DataFrame, root: str) -> None:
    if report.empty:
        console.print("Nothing to do.")
        return
    ok = report[report["status"] == OK]
    failed = report[report["status"] == FAILED]

    if len(ok):
        table = Table(title=f"{season}: data quality per race", header_style="bold gold1", border_style="green")
        for col in ("Rd", "Event", "Laps", "Clean %", "Stops", "Pass-thr", "Unknown", "SC laps",
                    "VSC laps", "Red", "Wet laps", "Dropped", "Warn"):
            table.add_column(col, justify="left" if col in ("Event", "SC laps", "VSC laps", "Red") else "right")
        for r in ok.to_dict("records"):
            table.add_row(str(r["round"]), str(r["event_name"]), f"{r['laps']:,}", f"{r['clean_pct']:.1f}",
                          str(r["tyre_changes"]), str(r["pass_throughs"]), str(r["unknown_passages"]),
                          escape(r["sc_lap_numbers"] or "-"), escape(r["vsc_lap_numbers"] or "-"),
                          escape(r["red_flag_lap_numbers"] or "-"), str(r["wet_tyre_laps"]),
                          str(r["dropped_rows"]), str(r["bronze_warnings"] + r["fastf1_warnings"]))
        console.print(table)

    sources = ", ".join(f"{n} {s}" for s, n in report["source"].value_counts().items())
    console.print(f"[bold]{len(ok)} ok, {len(failed)} failed[/bold] ({sources}). "
                  f"Report: {lake.report_path(f'backfill_{season}', root)}")
    for r in failed.to_dict("records"):
        console.print(f"[red]  R{r['round']:02d} {escape(str(r['event_name']))}: failed at {r['stage']}: "
                      f"{escape(str(r['error']))}[/red]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build many races into the PitwallOS lake (Phase 1 backfill)")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2024")
    parser.add_argument("--rounds", type=int, nargs="+", help="only these rounds (default: every race run so far)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild Silver + Gold even where Gold exists")
    parser.add_argument("--telemetry", action="store_true", help="also ingest telemetry (default: laps only)")
    parser.add_argument("--pit-stops", action="store_true",
                        help="add F1's pit-stop record to Bronze ingested without it (one small download per race)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    parser.add_argument("--verbose", action="store_true", help="show FastF1's own progress messages")
    args = parser.parse_args(argv)

    if not args.verbose:
        extractor.quiet_fastf1_console()
    failed = 0
    try:
        for season in args.season:
            report = backfill(season, args.rounds, root=args.lake, telemetry=args.telemetry, rebuild=args.rebuild,
                              pit_stops=args.pit_stops)
            failed += int((report["status"] == FAILED).sum()) if len(report) else 0
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Finished races are saved; run the same command again to resume.[/yellow]")
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
