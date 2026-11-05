"""
OpenF1, where the car telemetry comes from now (src/pipeline/openf1.py) - Phase 5 step 5.0

Since 14-15 October 2026 F1's timing archive (livetiming.formula1.com/static) answers 401 without an F1 TV login, so
FastF1 can no longer download telemetry. OpenF1 (api.openf1.org, a community service, free for past races since 2023,
no login) keeps a copy of the same live feed. Checked on Baku 2026, driver 63: every one of FastF1's 36,103 car and
37,092 position samples has an OpenF1 twin within 1 ms, with identical speed, rpm, gear, throttle, x, y and z.

This module adds three tables to a race already in Bronze (its laps came from FastF1), AS DELIVERED (OpenF1's own
column names, the time stamps as text):

    openf1_car_data   per driver, ~4 samples a second: date, speed, rpm, n_gear, throttle, brake, drs
    openf1_location   per driver, ~4 samples a second: date, x, y, z (tenths of a metre)
    openf1_laps       OpenF1's lap table (Silver takes the start of lap 1 from it to line up the two clocks)

Nothing is converted here: Silver lines up the clocks (telemetry_clock.py) and resamples (cleaner.py).

Manners: OpenF1's free tier allows 3 requests a second and 30 a minute. One driver's car data for a race is ~5 MB and
takes ~10 s, so 3 requests run at once and their starts are spaced to at most 25 a minute; an answer 429 ("too many")
waits a minute, a failed connection or a server error is retried after 10, 30, 60 and 120 s. A race is ~41 requests
(2 per driver + the lap table), ~3 minutes.

Rules kept: the session is found by its DATE (2023 lists two Italian "Race" sessions: Imola, cancelled, and Monza);
a driver OpenF1 has no samples for is written down in the manifest ("missing"), never invented; everything is
downloaded before anything is written and the manifest is written last, so a failed download leaves Bronze as it was.

    python -m src.pipeline.openf1 --season 2024 --rounds 23     adds OpenF1 telemetry to Qatar 2024's Bronze
    (the backfill does it for whole seasons: python -m src.pipeline.backfill --season 2024 --telemetry)
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.pipeline import lake

API_URL = "https://api.openf1.org/v1/"
CAR_TABLE, LOCATION_TABLE, LAPS_TABLE = lake.BRONZE_OPENF1_TABLES
PER_DRIVER = (("car_data", CAR_TABLE), ("location", LOCATION_TABLE))   # endpoint, Bronze table
REQUESTS_PER_MINUTE = 25        # OpenF1's free tier: 30 a minute and 3 a second; stay under both
WORKERS = 3
TIMEOUT_SEC = 180
RETRY_WAITS_SEC = (10, 30, 60, 120)
RATE_LIMIT_WAIT_SEC = 60
RATE_LIMIT_MAX_WAITS = 10
SESSION_DAYS = 1                # a race's UTC start can fall on the day before or after its local date (Las Vegas)
NO_RESULTS = "No results found."

Fetch = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]   # (endpoint, query) -> rows


class OpenF1Error(RuntimeError):
    """OpenF1 could not deliver (after the retries), or did not deliver what was asked for."""


class Pace:
    """Spaces the starts of requests, shared by every thread: at most `per_minute` starts a minute."""

    def __init__(self, per_minute: float = REQUESTS_PER_MINUTE, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.step = 60.0 / per_minute
        self.clock, self.sleep = clock, sleep
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:                     # take the next free slot...
            now = self.clock()
            start = max(now, self.next_start)
            self.next_start = start + self.step
        if start > now:                     # ...and wait for it outside the lock
            self.sleep(start - now)


def http_fetch(pace: Optional[Pace] = None, get: Callable[..., Any] = requests.get,
               sleep: Callable[[float], None] = time.sleep) -> Fetch:
    """The real fetch: OpenF1's JSON as a list of rows; "No results found" = no rows; retried as described above."""
    pace = pace or Pace()

    def fetch(endpoint: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        waits, rate_waits = list(RETRY_WAITS_SEC), 0
        while True:
            pace.wait()
            try:
                response = get(API_URL + endpoint, params=query, timeout=TIMEOUT_SEC)
            except requests.RequestException as err:
                if not waits:
                    raise OpenF1Error(f"{endpoint} {query}: {type(err).__name__}: {err}") from err
                sleep(waits.pop(0))
                continue
            status = response.status_code
            if status == 200:
                rows = response.json()
                if not isinstance(rows, list):
                    raise OpenF1Error(f"{endpoint} {query}: expected a list of rows, got {str(rows)[:200]}")
                return rows
            if status == 404 and _detail(response) == NO_RESULTS:
                return []
            if status == 429 and rate_waits < RATE_LIMIT_MAX_WAITS:
                rate_waits += 1
                sleep(RATE_LIMIT_WAIT_SEC)
                continue
            if status >= 500 and waits:
                sleep(waits.pop(0))
                continue
            raise OpenF1Error(f"{endpoint} {query}: HTTP {status} {response.text[:200]}")

    return fetch


def _detail(response: Any) -> Optional[str]:
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("detail") if isinstance(body, dict) else None


def find_session(fetch: Fetch, year: int, event_date: str, location: Optional[str] = None) -> Dict[str, Any]:
    """OpenF1's race session for the race held on `event_date` (the manifest's local date): a 'Race' session of that
    year, not cancelled, starting (UTC) within a day of it; if two qualify, the one at `location`."""
    day = pd.Timestamp(event_date).normalize()
    candidates = []
    for s in fetch("sessions", {"year": int(year), "session_name": "Race"}):
        start = pd.Timestamp(s["date_start"]).tz_convert(None).normalize()
        if not s.get("is_cancelled") and abs(start - day) <= pd.Timedelta(days=SESSION_DAYS):
            candidates.append(s)
    if len(candidates) > 1 and location:
        candidates = [s for s in candidates if s.get("location") == location] or candidates
    if len(candidates) != 1:
        raise OpenF1Error(f"{year} race on {day.date()}: {len(candidates)} OpenF1 sessions match, need exactly one")
    return candidates[0]


def drivers_of(year: int, round_number: int, session: str = "R", root: str = lake.DEFAULT_LAKE_ROOT) -> List[int]:
    """The car numbers in Bronze's laps: the drivers whose telemetry we want."""
    laps = lake.read_table(lake.table_path(lake.BRONZE, "laps", year, round_number, session, root),
                           columns=["DriverNumber"])
    return sorted({int(d) for d in laps["DriverNumber"].dropna()})


def download(fetch: Fetch, session_key: int, drivers: Sequence[int],
             workers: int = WORKERS) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    """Every driver's car data and location + the lap table, as delivered. Each answer becomes a table in its worker
    (a race is ~1.4 million rows: kept as Python dicts it would take a gigabyte). Raises if any request fails."""
    jobs = [(endpoint, table, int(d)) for d in drivers for endpoint, table in PER_DRIVER]

    def one(job: Tuple[str, str, int]) -> pd.DataFrame:
        endpoint, _, driver = job
        return pd.DataFrame(fetch(endpoint, {"session_key": int(session_key), "driver_number": driver}))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(one, jobs))
    tables: Dict[str, pd.DataFrame] = {}
    missing: Dict[str, List[int]] = {}
    for endpoint, table in PER_DRIVER:
        parts = [(job[2], f) for job, f in zip(jobs, frames) if job[1] == table]
        missing[endpoint] = [d for d, f in parts if f.empty]
        tables[table] = pd.concat([f for _, f in parts if not f.empty], ignore_index=True) if any(
            not f.empty for _, f in parts) else pd.DataFrame()
    tables[LAPS_TABLE] = pd.DataFrame(fetch("laps", {"session_key": int(session_key)}))
    return tables, missing


def add_telemetry(year: int, round_number: int, session: str = "R", root: str = lake.DEFAULT_LAKE_ROOT,
                  fetch: Optional[Fetch] = None, workers: int = WORKERS) -> Dict[str, Any]:
    """
    Add OpenF1's telemetry to a race already in Bronze. The manifest gains an "openf1" entry (session key, drivers,
    who is missing, when) and the three tables; it is written last.
    Returns {'session_key', 'rows': {table: rows}, 'missing': {...}, 'seconds'}.
    """
    start = time.time()
    fetch = fetch or http_fetch()
    manifest = lake.read_manifest(year, round_number, session, root)
    found = find_session(fetch, year, manifest["event_date"], manifest.get("location"))
    key = int(found["session_key"])
    drivers = drivers_of(year, round_number, session, root)
    tables, missing = download(fetch, key, drivers, workers)
    for name in (CAR_TABLE, LOCATION_TABLE, LAPS_TABLE):
        if tables[name].empty:
            raise OpenF1Error(f"{year} round {round_number}: OpenF1 session {key} has no {name} at all")

    identity = {k: manifest[k] for k in ("season", "round", "session", "event_name", "location", "country")}
    identity["openf1_session_key"] = key
    entries = dict(manifest.get("tables", {}))
    for name, table in tables.items():
        path = lake.table_path(lake.BRONZE, name, year, round_number, session, root)
        lake.write_table(table, path, metadata=identity)
        entries[name] = {"rows": int(len(table)), "bytes": int(path.stat().st_size), "columns": list(table.columns)}
    manifest["tables"] = entries
    manifest[lake.OPENF1] = {
        "session_key": key,
        "meeting_key": found.get("meeting_key"),
        "session_start_utc": found.get("date_start"),
        "drivers": drivers,
        "missing": missing,
        "api": API_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    lake.write_manifest(manifest, year, round_number, session, root)
    return {"session_key": key, "rows": {n: int(len(t)) for n, t in tables.items()}, "missing": missing,
            "seconds": round(time.time() - start, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Add OpenF1's car telemetry to races already in Bronze")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    fetch = http_fetch()
    failed = 0
    for rnd in args.rounds:
        try:
            out = add_telemetry(args.season, rnd, root=args.lake, fetch=fetch)
            print(f"{args.season} round {rnd}: session {out['session_key']}, {out['rows']}, "
                  f"missing {out['missing']}, {out['seconds']:.0f} s")
        except Exception as err:
            failed += 1
            print(f"{args.season} round {rnd}: FAILED {type(err).__name__}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
