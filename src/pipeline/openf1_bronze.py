"""
Bronze from OpenF1 alone (src/pipeline/openf1_bronze.py) - the OpenF1 overhaul, piece 1

Since October 2026 F1's own timing archive needs a login, so FastF1 can no longer fill Bronze. OpenF1 (api.openf1.org)
serves the same live feed, cut into one table per "endpoint" (an address that answers one kind of question: /laps,
/pit, /weather, ...). This module copies those tables for one race into Bronze AS DELIVERED: OpenF1's column names,
its time stamps as text (UTC), nothing renamed, converted, filtered or joined. Turning them into the lap table the
cleaner reads is Silver's job, so a mistake there never costs a download.

    openf1_sessions, openf1_meetings   what the race is: OpenF1's keys, the place, the scheduled start
    openf1_drivers                     who drove: car number, three-letter code, team
    openf1_laps                        one row per driver and lap: start (UTC), duration, sectors, speed traps
    openf1_stints                      tyre sets: compound, first and last lap, age when fitted
    openf1_race_control                Race Control's messages: flags, safety car, deleted lap times, penalties
    openf1_weather                     one reading a minute
    openf1_position                    every change of position
    openf1_session_result              the classification
    openf1_pit                         pit-lane passages: when, how long in the lane, how long stationary
    openf1_intervals                   gap to the leader and to the car ahead, every ~4 s (for the replay)
    openf1_overtakes                   who passed whom, when
    openf1_car_data, openf1_location   telemetry, ~4 samples a second per driver (openf1.py downloads them)

SPRINTS too (session "S"): a sprint is a short race, with the same tables; it is the 'Sprint' session of the race's
meeting. Qualifying ("Q") and sprint qualifying ("SQ") are ingested the same way, for the qualifying replay
(src/lib/lake_qualifying.py reads their Bronze directly: they get no Silver).

WHICH RACE: OpenF1 has no round numbers. A season's rounds are its 'Race' sessions that were not cancelled, in date
order (season_calendar). Checked on 2023-2025: all 70 races get the round number the lake already used.

RULES KEPT
  * As delivered. The one change Parquet forces: a column mixing numbers and text (session_result's gap_to_leader:
    6.031 and "+1 LAP") is stored as text, each value written as OpenF1 sent it (as_table).
  * Never an empty table standing in for missing data. The first nine tables above and the telemetry are REQUIRED:
    if OpenF1 has no rows for one, nothing is written and the race fails loudly. pit, intervals and overtakes may
    truly be empty (a race without a stop): then no table is written and the manifest lists them under "empty".
  * A driver OpenF1 has no telemetry for is named in the manifest ("missing"), never invented.
  * Everything is downloaded before anything is written, and the manifest is written last: a failed download leaves
    Bronze as it was.
  * Only what Bronze lacks is downloaded: run it again after adding an endpoint and it fetches that one table.
  * It refuses a Bronze that came from FastF1 (no manifest "source": "openf1"), and a round whose Bronze holds
    another OpenF1 session: OpenF1's Bronze is built in its own lake, never on top of something else.

A race is 12 small requests + 2 per driver (telemetry, ~95 % of the bytes): ~3 minutes at OpenF1's free speed.
Telemetry already downloaded into another lake is copied from there instead (reuse_from): 12 requests, ~30 s.

    python -m src.pipeline.openf1_bronze --season 2024 --rounds 23 --lake data_openf1
    python -m src.pipeline.openf1_bronze --season 2023 2024 2025 --lake data_openf1 --reuse-telemetry-from data
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline import lake
from src.pipeline.openf1 import (API_URL, CAR_TABLE, LOCATION_TABLE, PER_DRIVER, WORKERS, Fetch, OpenF1Error,
                                 download_per_driver, http_fetch)

RACE_SESSION = "R"
# our session code -> the names OpenF1 has used for it (2023 called sprint qualifying the "Sprint Shootout")
SESSION_NAMES = {RACE_SESSION: ("Race",), "S": ("Sprint",), "Q": ("Qualifying",),
                 "SQ": ("Sprint Qualifying", "Sprint Shootout")}
RACE_LIKE = (RACE_SESSION, "S")        # sessions with a start and laps to the flag: these go on to Silver and Gold

# Endpoints asked once per race, by session key (meetings: by meeting key).
REQUIRED = ("sessions", "meetings", "drivers", "laps", "stints", "race_control", "weather", "position",
            "session_result")
OPTIONAL = ("pit", "intervals", "overtakes")                 # may truly have no rows
TELEMETRY = tuple(endpoint for endpoint, _ in PER_DRIVER)    # car_data, location: asked per driver
TELEMETRY_TABLES = (CAR_TABLE, LOCATION_TABLE)

DATA_DELAY = pd.Timedelta(hours=6)     # a race counts as run this long after its scheduled end


def as_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """OpenF1's rows as a table Parquet can hold, values untouched. Parquet wants one type per column, so a column
    mixing text with numbers becomes text: 6.031 -> "6.031", "+1 LAP" stays, nothing stays nothing."""
    table = pd.DataFrame(rows)
    for col in table.columns:
        if table[col].dtype != object:
            continue
        values = [v for v in table[col] if not _is_nothing(v)]
        if any(isinstance(v, str) for v in values) and not all(isinstance(v, str) for v in values):
            table[col] = pd.array([None if _is_nothing(v) else v if isinstance(v, str) else json.dumps(v)
                                   for v in table[col]], dtype="string")
    return table


def _is_nothing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)      # None, or NaN


# --- which race ---------------------------------------------------------------------------------------

def season_calendar(fetch: Fetch, year: int) -> pd.DataFrame:
    """
    Every championship race of a season, in the order they were run, one row each: round, session_key, meeting_key,
    event_name, location, country, circuit, start_utc, end_utc (scheduled, UTC), gmt_offset.

    The round is the race's place in date order among the 'Race' sessions that were not cancelled (2023 lists Imola,
    cancelled, so Monaco is round 6). Sprints are left out by their name: OpenF1 calls them 'Sprint'.
    """
    sessions = [s for s in fetch("sessions", {"year": int(year), "session_name": SESSION_NAMES[RACE_SESSION][0]})
                if not s.get("is_cancelled")]
    if not sessions:
        raise OpenF1Error(f"OpenF1 lists no race for {year}")
    names = {m["meeting_key"]: m.get("meeting_name") for m in fetch("meetings", {"year": int(year)})}
    sessions.sort(key=lambda s: pd.Timestamp(s["date_start"]))
    return pd.DataFrame([{
        "season": int(year), "round": number, "session_key": int(s["session_key"]),
        "meeting_key": int(s["meeting_key"]), "event_name": names.get(s["meeting_key"]),
        "location": s.get("location"), "country": s.get("country_name"), "circuit": s.get("circuit_short_name"),
        "start_utc": pd.Timestamp(s["date_start"]), "end_utc": pd.Timestamp(s["date_end"]),
        "gmt_offset": s.get("gmt_offset"),
    } for number, s in enumerate(sessions, start=1)])


def race_of(calendar: pd.DataFrame, year: int, round_number: int) -> pd.Series:
    """The calendar's row for one round. Raises if the season has no such round."""
    match = calendar[(calendar["season"] == int(year)) & (calendar["round"] == int(round_number))]
    if len(match) != 1:
        raise OpenF1Error(f"{year} has no round {round_number}: OpenF1 lists rounds 1-{int(calendar['round'].max())}")
    return match.iloc[0]


def session_of(fetch: Fetch, race: pd.Series, session: str) -> Dict[str, Any]:
    """
    OpenF1's keys and scheduled times for one session of a race weekend: {'session_key', 'start_utc', 'end_utc'}.
    The race's own are in the calendar; a sprint is the 'Sprint' session of the same meeting (one more request).
    Raises if the weekend has no such session.
    """
    if session == RACE_SESSION:
        return {"session_key": int(race["session_key"]), "start_utc": race["start_utc"], "end_utc": race["end_utc"]}
    names = SESSION_NAMES[session]
    found = []
    for name in names:
        found += [s for s in fetch("sessions", {"meeting_key": int(race["meeting_key"]), "session_name": name})
                  if not s.get("is_cancelled")]
        if found:
            break
    if len(found) != 1:
        raise OpenF1Error(f"{int(race['season'])} round {int(race['round'])} ({race['event_name']}): {len(found)} "
                          f"'{names[0]}' sessions in OpenF1, need exactly one")
    return {"session_key": int(found[0]["session_key"]), "start_utc": pd.Timestamp(found[0]["date_start"]),
            "end_utc": pd.Timestamp(found[0]["date_end"])}


def has_run(race: pd.Series, now: Optional[pd.Timestamp] = None) -> bool:
    """True once the race's data should be there: DATA_DELAY after its scheduled end."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return race["end_utc"] + DATA_DELAY <= now


def local_date(start_utc: pd.Timestamp, gmt_offset: Optional[str]) -> str:
    """The race's date at the circuit (Las Vegas starts on Sunday morning UTC, Saturday night there)."""
    offset = pd.Timedelta(gmt_offset) if gmt_offset else pd.Timedelta(0)
    return str((start_utc.tz_convert(None) + offset).date())


# --- what Bronze already holds -----------------------------------------------------------------------------

def held_endpoints(year: int, round_number: int, session: str, root: lake.PathLike, session_key: int) -> List[str]:
    """The endpoints this Bronze already answers (a table, or a recorded 'empty'). [] if there is no Bronze yet.
    Raises if the Bronze there is not OpenF1's, or is another session's."""
    if not lake.has_bronze(year, round_number, session, root):
        return []
    manifest = lake.read_manifest(year, round_number, session, root)
    if manifest.get("source") != lake.OPENF1:
        raise OpenF1Error(f"{year} round {round_number}: the Bronze in '{root}' did not come from OpenF1 alone. "
                          "OpenF1's Bronze is built in its own lake (--lake), never on top of FastF1's.")
    there = manifest[lake.OPENF1]["session_key"]
    if int(there) != int(session_key):
        raise OpenF1Error(f"{year} round {round_number}: Bronze holds OpenF1 session {there}, the calendar says "
                          f"{session_key}. The season's rounds have moved: this race needs a fresh lake.")
    tables = manifest.get("tables", {})
    held = [e for e in REQUIRED + OPTIONAL + TELEMETRY if lake.openf1_table(e) in tables]
    return held + [e for e in manifest[lake.OPENF1].get("empty", []) if e not in held]


def telemetry_in(other_root: lake.PathLike, year: int, round_number: int, session: str,
                 session_key: int) -> Optional[Tuple[Dict[str, pd.DataFrame], str]]:
    """OpenF1's telemetry for this very session, already downloaded into another lake: ({table: rows}, where it is
    from), or None if that lake has none. Raises if what it holds is another session's."""
    if lake.bronze_telemetry_source(year, round_number, session, other_root) != lake.OPENF1:
        return None
    tables = {name: lake.read_table(lake.table_path(lake.BRONZE, name, year, round_number, session, other_root))
              for name in TELEMETRY_TABLES}
    for name, table in tables.items():
        keys = {int(k) for k in table["session_key"].unique()}
        if keys != {int(session_key)}:
            raise OpenF1Error(f"{year} round {round_number}: {name} in '{other_root}' is session {sorted(keys)}, "
                              f"not {session_key}")
    when = lake.read_manifest(year, round_number, session, other_root)[lake.OPENF1].get("downloaded_at_utc")
    return tables, f"copied from the lake '{other_root}', downloaded {when}"


# --- one race ----------------------------------------------------------------------------------------------

def ingest_session(year: int, round_number: int, session: str = RACE_SESSION,
                   root: lake.PathLike = lake.DEFAULT_LAKE_ROOT, fetch: Optional[Fetch] = None,
                   calendar: Optional[pd.DataFrame] = None, telemetry: bool = True,
                   reuse_from: Optional[lake.PathLike] = None, workers: int = WORKERS) -> Dict[str, Any]:
    """
    One race into Bronze from OpenF1, as delivered; only what Bronze lacks is fetched.

    calendar:   season_calendar(fetch, year), if the caller already has it (saves two requests a race).
    telemetry:  False = the small tables only (12 requests); the telemetry can be added by a later run.
    reuse_from: another lake's root. If its Bronze holds OpenF1's telemetry for this session, it is copied from
                there instead of downloaded again.

    Returns {'status': 'ingested' or 'complete' (nothing was missing), 'session_key', 'event_name', 'fetched',
    'reused', 'empty', 'missing', 'rows', 'seconds'}.
    """
    start = time.time()
    if session not in SESSION_NAMES:
        raise OpenF1Error(f"session '{session}': OpenF1's Bronze knows {', '.join(SESSION_NAMES)}")
    fetch = fetch or http_fetch()
    race = race_of(calendar if calendar is not None else season_calendar(fetch, year), year, round_number)
    this = session_of(fetch, race, session)
    key, meeting = this["session_key"], int(race["meeting_key"])

    held = held_endpoints(year, round_number, session, root, key)
    wanted = REQUIRED + OPTIONAL + (TELEMETRY if telemetry else ())
    todo = [e for e in wanted if e not in held]
    if not todo:
        return {"status": "complete", "session_key": key, "event_name": race["event_name"], "fetched": [],
                "reused": [], "empty": [], "missing": {}, "rows": {}, "seconds": round(time.time() - start, 1)}

    # 1. Everything into memory first. Nothing is written until every answer is in.
    new: Dict[str, pd.DataFrame] = {}                 # Bronze table -> rows
    origin: Dict[str, str] = {}                       # Bronze table -> where a table that was not downloaded is from
    empty: List[str] = []
    for endpoint in (e for e in todo if e not in TELEMETRY):
        rows = fetch(endpoint, {"meeting_key": meeting} if endpoint == "meetings" else {"session_key": key})
        if rows:
            new[lake.openf1_table(endpoint)] = as_table(rows)
        elif endpoint in REQUIRED:
            raise OpenF1Error(f"{year} round {round_number}: OpenF1 session {key} has no '{endpoint}' at all. "
                              "Nothing written.")
        else:
            empty.append(endpoint)

    missing: Dict[str, List[int]] = {}
    if any(e in todo for e in TELEMETRY):
        drivers_table = new.get(lake.openf1_table("drivers"))
        if drivers_table is None:                     # a top-up: the drivers are already in Bronze
            drivers_table = lake.read_table(lake.table_path(lake.BRONZE, lake.openf1_table("drivers"), year,
                                                            round_number, session, root))
        drivers = sorted(int(d) for d in drivers_table["driver_number"])
        in_hand = telemetry_in(reuse_from, year, round_number, session, key) if reuse_from is not None else None
        if in_hand is not None:
            tables, note = in_hand
            origin.update({name: note for name in tables})
            missing = {endpoint: [d for d in drivers if d not in set(tables[table]["driver_number"].unique())]
                       for endpoint, table in PER_DRIVER}
        else:
            tables, missing = download_per_driver(fetch, key, drivers, workers)
        for name in TELEMETRY_TABLES:
            if tables[name].empty:
                raise OpenF1Error(f"{year} round {round_number}: OpenF1 session {key} has no {name} at all. "
                                  "Nothing written.")
        new.update(tables)

    # 2. Write the tables, then the manifest: the manifest is what marks this Bronze as complete.
    old = lake.read_manifest(year, round_number, session, root) if held else {}
    identity = {"season": int(year), "round": int(round_number), "session": session,
                "event_name": str(race["event_name"]), "location": str(race["location"]),
                "country": str(race["country"])}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = dict(old.get("tables", {}))
    for name, table in new.items():
        path = lake.table_path(lake.BRONZE, name, year, round_number, session, root)
        lake.write_table(table, path, metadata={**identity, "openf1_session_key": key})
        entries[name] = {"rows": int(len(table)), "bytes": int(path.stat().st_size), "columns": list(table.columns),
                         "origin": origin.get(name, f"downloaded {now}")}

    before = old.get(lake.OPENF1, {})
    manifest = {
        **identity,
        "circuit": str(race["circuit"]),
        "event_date": local_date(this["start_utc"], race["gmt_offset"]),
        "source": lake.OPENF1,
        "driver_count": int(entries[lake.openf1_table("drivers")]["rows"]),
        "telemetry": all(name in entries for name in TELEMETRY_TABLES),
        "warnings": list(old.get("warnings", [])),
        lake.OPENF1: {
            "session_key": key,
            "meeting_key": meeting,
            "session_start_utc": this["start_utc"].isoformat(),
            "session_end_utc": this["end_utc"].isoformat(),
            "gmt_offset": race["gmt_offset"],
            "empty": sorted(set(before.get("empty", [])) | set(empty)),
            "missing": {**before.get("missing", {}), **missing},
            "api": API_URL,
            "updated_at_utc": now,
        },
        "tables": entries,
    }
    lake.write_manifest(manifest, year, round_number, session, root)
    return {"status": "ingested", "session_key": key, "event_name": race["event_name"],
            "fetched": [n for n in new if n not in origin], "reused": sorted(origin), "empty": empty,
            "missing": missing, "rows": {n: int(len(t)) for n, t in new.items()},
            "seconds": round(time.time() - start, 1)}


# --- many races ----------------------------------------------------------------------------------------------

def ingest_season(year: int, rounds: Optional[Sequence[int]] = None, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT,
                  fetch: Optional[Fetch] = None, telemetry: bool = True, reuse_from: Optional[lake.PathLike] = None,
                  now: Optional[pd.Timestamp] = None, say=print) -> List[Dict[str, Any]]:
    """Every race of a season that has been run (or just `rounds`), one after the other. A race that fails is
    reported and the next one is tried: [{'season', 'round', 'status': 'ingested' / 'complete' / 'failed', ...}]."""
    fetch = fetch or http_fetch()
    calendar = season_calendar(fetch, year)
    chosen = [int(r) for r in rounds] if rounds else [int(r["round"]) for _, r in calendar.iterrows()
                                                       if has_run(r, now)]
    results = []
    for rnd in chosen:
        try:
            out = ingest_session(year, rnd, root=root, fetch=fetch, calendar=calendar, telemetry=telemetry,
                                 reuse_from=reuse_from)
            say(f"{year} round {rnd:>2}  {out['status']:<9} {out['event_name']}: {len(out['fetched'])} tables "
                f"fetched, {len(out['reused'])} reused, empty {out['empty']}, missing {out['missing']}, "
                f"{out['seconds']:.0f} s")
        except Exception as err:                       # one race failing never stops the others
            out = {"status": "failed", "error": f"{type(err).__name__}: {err}"}
            say(f"{year} round {rnd:>2}  FAILED    {out['error']}")
        results.append({"season": int(year), "round": rnd, **out})
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bronze from OpenF1 alone: every table of a race, as delivered")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, nargs="+", help="only these rounds (default: every race run so far)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--no-telemetry", action="store_true", help="the small tables only")
    parser.add_argument("--reuse-telemetry-from", metavar="LAKE",
                        help="copy OpenF1 telemetry already downloaded into another lake instead of fetching it again")
    args = parser.parse_args(argv)
    fetch = http_fetch()
    results = []
    for year in args.season:
        results += ingest_season(year, args.rounds, root=args.lake, fetch=fetch, telemetry=not args.no_telemetry,
                                 reuse_from=args.reuse_telemetry_from)
    failed = [r for r in results if r["status"] == "failed"]
    print(f"{len(results) - len(failed)} of {len(results)} races in Bronze ({args.lake}); {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
