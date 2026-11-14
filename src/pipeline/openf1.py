"""
OpenF1's client (src/pipeline/openf1.py)

OpenF1 (api.openf1.org, a community service, free for past sessions since 2023, no login) keeps a copy of F1's live
timing feed, cut into one table per endpoint. This module is how the project talks to it: one request at a time,
with manners. What is asked, and where the answers go, is openf1_bronze.py's business.

Manners: OpenF1's free tier allows 3 requests a second and 30 a minute. Requests start at most 25 a minute (Pace,
shared by every thread); an answer 429 ("too many") waits a minute, a failed connection or a server error is retried
after 10, 30, 60 and 120 s. A 404 with "No results found." is not an error: it means no rows (and OpenF1 gives the
same answer to a question it does not understand, so an empty answer for a table that must exist is treated as a
failure by the caller).

Telemetry is asked per driver (download_per_driver): one driver's car data for a race is ~5 MB and takes ~10 s, and
a whole session in one request is refused as too large.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def download_per_driver(fetch: Fetch, session_key: int, drivers: Sequence[int],
                        workers: int = WORKERS) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    """Every driver's car data and location, as delivered: ({table: rows}, {endpoint: drivers OpenF1 has nothing for}).
    Each answer becomes a table in its worker (a race is ~1.4 million rows: kept as Python dicts it would take a
    gigabyte). Raises if any request fails."""
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
    return tables, missing
