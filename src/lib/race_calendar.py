"""
The season's calendar and the circuits, without FastF1 (src/lib/race_calendar.py) - the OpenF1 overhaul, piece 6

The session picker asked FastF1 for the list of race weekends, and the replay asked it for the map's rotation. Both
answers exist elsewhere:

    the calendar   OpenF1's sessions and meetings lists: every session of a season, with its scheduled start. A race's
                   round number is its place among the season's races (openf1_bronze.season_calendar).
    the circuit    MultiViewer (api.multiviewer.app), where FastF1 itself got it: the rotation that makes the map look
                   like the broadcast's, the corners and their numbers. It refuses a request that does not name its
                   client (HTTP 403), so this one says who it is.

Both are REFERENCE data: small, slow to change, needed offline. Each answer is kept in the lake as JSON
(<lake>/reference/, lake.reference_path) and read from there afterwards:

    a past season's calendar never changes        -> fetched once
    the current season's does (a race cancelled)  -> fetched again when the copy is older than CALENDAR_MAX_AGE
    a circuit of a given year never changes       -> fetched once
    no network                                    -> the copy, however old; no copy either: the error says so

OpenF1 starts in 2023, so the calendar does too (FIRST_SEASON).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

from src.pipeline import lake, openf1, openf1_bronze

FIRST_SEASON = 2023
CALENDAR_MAX_AGE = pd.Timedelta(hours=12)
MULTIVIEWER_URL = "https://api.multiviewer.app/api/v1/circuits/{circuit_key}/{year}"
USER_AGENT = "PitwallOS (+https://github.com/Rohxn14/PitwallOS)"
SPRINT, CONVENTIONAL = "sprint_qualifying", "conventional"        # the picker looks for "sprint" in a weekend's type
SESSION_ORDER = ("Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Sprint Shootout", "Sprint",
                 "Qualifying", "Race")


class CalendarError(RuntimeError):
    """The calendar or a circuit could not be fetched, and the lake has no copy."""


def build_weekends(sessions: List[Dict[str, Any]], meetings: List[Dict[str, Any]], year: int) -> List[Dict[str, Any]]:
    """
    OpenF1's sessions and meetings of one season -> the race weekends, in the picker's shape, in calendar order:
    round_number, event_name, date (the race's date at the circuit), country, type, session_dates {name: UTC start},
    + meeting_key, circuit_key, location. Testing and cancelled meetings have no race, so they are not weekends.
    """
    names = {m["meeting_key"]: m.get("meeting_name") for m in meetings}
    races = sorted((s for s in sessions if s.get("session_name") == "Race" and not s.get("is_cancelled")),
                   key=lambda s: pd.Timestamp(s["date_start"]))
    out = []
    for number, race in enumerate(races, start=1):
        mine = [s for s in sessions if s["meeting_key"] == race["meeting_key"] and not s.get("is_cancelled")]
        starts = {s["session_name"]: pd.Timestamp(s["date_start"]).isoformat() for s in mine}
        out.append({
            "round_number": number,
            "event_name": names.get(race["meeting_key"]) or f"Round {number}",
            "date": openf1_bronze.local_date(pd.Timestamp(race["date_start"]), race.get("gmt_offset")),
            "country": race.get("country_name"),
            "type": SPRINT if "Sprint" in starts else CONVENTIONAL,
            "session_dates": {name: starts[name] for name in SESSION_ORDER if name in starts},
            "year": int(year),
            "meeting_key": int(race["meeting_key"]),
            "circuit_key": race.get("circuit_key"),
            "location": race.get("location"),
        })
    return out


def weekends(year: int, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT, fetch: Optional[openf1.Fetch] = None,
             now: Optional[pd.Timestamp] = None) -> List[Dict[str, Any]]:
    """The race weekends of a season: from the lake's copy when it is fresh enough, else from OpenF1 (and kept)."""
    year = int(year)
    if year < FIRST_SEASON:
        return []
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    path = lake.reference_path(f"calendar_{year}", root)
    kept = lake.read_json(path) if path.exists() else None
    over = year < now.year                                           # a finished season's calendar is final
    if kept and (over or now - pd.Timestamp(kept["fetched_at_utc"]) <= CALENDAR_MAX_AGE):
        return kept["weekends"]
    try:
        fetch = fetch or openf1.http_fetch()
        built = build_weekends(fetch("sessions", {"year": year}), fetch("meetings", {"year": year}), year)
    except Exception as err:
        if kept:                                                     # no network: yesterday's calendar beats none
            return kept["weekends"]
        raise CalendarError(f"The {year} calendar could not be fetched from OpenF1 and the lake has no copy "
                            f"({type(err).__name__}: {err})") from err
    if not built:
        raise CalendarError(f"OpenF1 lists no race for {year}")
    lake.write_json({"year": year, "fetched_at_utc": now.isoformat(), "source": openf1.API_URL, "weekends": built}, path)
    return built


def seasons(today: Optional[date] = None) -> List[int]:
    """The seasons the calendar covers: FIRST_SEASON to this year."""
    return list(range(FIRST_SEASON, (today or date.today()).year + 1))


def weekends_by_place(place: str, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT,
                      fetch: Optional[openf1.Fetch] = None) -> List[Dict[str, Any]]:
    """Every weekend of every season with this event name ("Italian Grand Prix"), oldest first."""
    wanted = place.lower().strip()
    out = []
    for year in seasons():
        try:
            out += [w for w in weekends(year, root, fetch) if str(w["event_name"]).strip().lower() == wanted]
        except CalendarError:
            continue
    return out


def race_names(root: lake.PathLike = lake.DEFAULT_LAKE_ROOT, fetch: Optional[openf1.Fetch] = None) -> List[str]:
    """Every event name in the calendar, sorted."""
    names = set()
    for year in seasons():
        try:
            names |= {str(w["event_name"]).strip() for w in weekends(year, root, fetch)}
        except CalendarError:
            continue
    return sorted(names)


def has_started(weekend: Dict[str, Any], session_name: str, now: Optional[datetime] = None) -> bool:
    """True if the weekend has this session and its scheduled start has passed."""
    start = weekend.get("session_dates", {}).get(session_name)
    return bool(start) and datetime.fromisoformat(start) <= (now or datetime.now(timezone.utc))


# --- circuits ---------------------------------------------------------------------------------------------------

def circuit(year: int, circuit_key: int, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT,
            get: Callable[..., Any] = requests.get) -> Dict[str, Any]:
    """MultiViewer's description of a circuit in a given year (rotation, corners, marshal sectors ...), as delivered:
    the lake's copy, else fetched and kept."""
    path = lake.reference_path(f"circuit_{int(circuit_key)}_{int(year)}", root)
    if path.exists():
        return lake.read_json(path)
    url = MULTIVIEWER_URL.format(circuit_key=int(circuit_key), year=int(year))
    try:
        response = get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    except requests.RequestException as err:
        raise CalendarError(f"MultiViewer could not be reached for circuit {circuit_key} ({type(err).__name__})") from err
    if response.status_code != 200:
        raise CalendarError(f"MultiViewer answered HTTP {response.status_code} for circuit {circuit_key} in {year}")
    info = response.json()
    lake.write_json(info, path)
    return info


def rotation(year: int, circuit_key: int, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT,
             get: Callable[..., Any] = requests.get) -> float:
    """Degrees to turn the map so that it looks like the broadcast's."""
    return float(circuit(year, circuit_key, root, get)["rotation"])
