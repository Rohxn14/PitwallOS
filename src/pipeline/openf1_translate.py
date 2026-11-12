"""
OpenF1's tables -> the tables the cleaner reads (src/pipeline/openf1_translate.py) - the OpenF1 overhaul, pieces 2-6

Bronze now holds OpenF1's tables as delivered (openf1_bronze.py). The Silver cleaner was written for FastF1's lap
table: one row per driver and lap with LapTime, Time, LapStartTime, PitInTime, Compound, TyreLife, TrackStatus, ...
OpenF1 spreads the same facts over eight tables and leaves some out. This module puts them back together, in memory,
each time Silver is built, so the cleaner, Gold, the analytics and the models keep their code.

It is PURE: tables in, tables out, no file and no network. Every rule below was measured against the FastF1 lake
(70 races, docs/openf1_migration.md).

THE CLOCK. Session time 0 = LIGHTS OUT, so that a lap's end is the official race time at that moment. It is found
from the first "SESSION STARTED" among Race Control's messages (OpenF1 merges F1's session-status feed into them, to
the millisecond). In 67 of 70 races that message equals the start OpenF1 gives lap 1; in the other three (Japan and
Monaco 2024: lap 1 dated from the restart after a red flag; Miami 2025: no lap-1 start) only the message has the real
start. The message is logged 0.27 s BEFORE the lights go out: with zero at the message, the winner's last lap ends
0.27 s after the official race time in every one of the 70 races (sd 0.04 s), and the cars would leave the grid
0.66 s after "lights out", too slow for a racing start (0.39 s with the correction). So zero = message + 0.27 s.
Everything OpenF1 stamps in UTC becomes seconds since that instant.

LAP STARTS AND ENDS. OpenF1 gives each lap a start (date_start) and a duration. Its starts of laps 2+ are stamped
0.173 s late and jitter by ~40 ms; its durations are F1's official lap times, exact. So a driver's laps are CHAINED:
within an unbroken run of laps, each start = the previous start + the previous duration, and the whole run is placed
where the median of (date_start - 0.173 s - chain) says. Lap 1 starts at lights out and ends where lap 2 starts.
A lap without a duration ends where the next lap starts.

TRACK STATUS. OpenF1 has no status feed. It is rebuilt from Race Control's messages, FastF1's codes:
    SAFETY CAR DEPLOYED 4 | VIRTUAL SAFETY CAR DEPLOYED 6 | VIRTUAL SAFETY CAR ENDING 7 | RED FLAG 5
    TRACK CLEAR -> 1 (not under a red flag: that ends at the next SESSION STARTED)
    YELLOW / DOUBLE YELLOW in a sector -> 2 until that sector's CLEAR, unless a safety car or red flag covers it
A lap's status is every code in force during it, each once, in order ("124").

PITS. A pit row is one passage through the pit lane: `lap_number` is the in-lap, `date` the pit EXIT (measured: FastF1's
PitOutTime + 0.03-0.12 s), `lane_duration` entry to exit. So entry = date - lane_duration, on the in-lap; exit = date,
on the next lap. OpenF1 has no pit table for 2023 rounds 1-6: there the in- and out-laps are still known (the lap
table marks out-laps), their TIMES are not, and stay null: PitInLap / PitOutLap carry the flag without a time.

TYRES. A stint row is one set of tyres on laps lap_start..lap_end. Tyre life on lap n = tyre_age_at_start + (n -
lap_start) + 1, after two repairs explained in tyres_by_lap. If two stints claim a lap, the later one wins.

WHAT OPENF1 CANNOT SAY: IsAccurate (FastF1's own judgement; here: a lap time, and no three sectors that fail to add up to it),
FastF1Generated (always False), SpeedFL, the scheduled race distance (total_laps = the winner's laps).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline import lake
from src.pipeline.telemetry_clock import LATER_LAP_LAG_SEC, utc

SESSION_STARTED = "SESSION STARTED"
LIGHTS_AFTER_MESSAGE_SEC = 0.27         # the lights go out this long after the message (70 races, sd 0.04 s)
GREEN, YELLOW, SC, RED, VSC, VSC_ENDING = "1", "2", "4", "5", "6", "7"
SECTORS_TOLERANCE_SEC = 0.0035          # sectors and lap time are published to the millisecond
POSITION_SLACK_SEC = 1.0                # a position change is stamped up to a second after the line
DELETED = re.compile(r"CAR (\d+) \(\w+\) (?:LAP )?TIME [\d:.]+ DELETED .* LAP (\d+)")
REINSTATED = re.compile(r"CAR (\d+) \(\w+\) (?:LAP )?TIME [\d:.]+ REINSTATED .* LAP (\d+)")
LAPS_DOWN = re.compile(r"\+\s*(\d+)\s*LAP")

SMALL_TABLES = ("drivers", "laps", "stints", "race_control", "weather", "position", "session_result", "pit")


class TranslationError(ValueError):
    """OpenF1's tables cannot be turned into a race (no start, no laps)."""


@dataclass
class Translation:
    laps: pd.DataFrame                      # FastF1's lap table, as the cleaner reads it
    weather: pd.DataFrame
    results: pd.DataFrame                   # FastF1's results table, as race_overview reads it
    pit_stops: Optional[pd.DataFrame]       # F1's pit-stop record shape, or None without a pit table
    lights_out_utc: pd.Timestamp
    total_laps: Optional[int]
    report: Dict[str, Any] = field(default_factory=dict)


# --- the clock ---------------------------------------------------------------------------------------------

def lights_out(race_control: pd.DataFrame, laps: pd.DataFrame) -> Tuple[pd.Timestamp, Dict[str, Any]]:
    """The UTC instant of lights out, and how it was found. Raises if neither source has it."""
    started = utc(race_control.loc[race_control["message"] == SESSION_STARTED, "date"]).sort_values()
    lap_one = utc(laps.loc[laps["lap_number"] == 1, "date_start"].dropna())
    shared = lap_one.mode().iloc[0] if len(lap_one) else pd.NaT       # the start most drivers' lap 1 carries
    if len(started):
        message, method = started.iloc[0], "session started"
        gap = None if pd.isna(shared) else round((shared - message).total_seconds(), 3)
    elif pd.notna(shared):
        message, method, gap = shared, "lap 1", 0.0
    else:
        raise TranslationError("no SESSION STARTED message and no lap-1 start: lights out is unknown")
    t0 = message + pd.Timedelta(seconds=LIGHTS_AFTER_MESSAGE_SEC)
    return t0, {"clock_method": method, "clock_lap1_gap_sec": gap, "lights_out_utc": str(t0)}


def seconds_since(stamps: pd.Series, t0: pd.Timestamp) -> pd.Series:
    return (utc(stamps) - t0).dt.total_seconds()


# --- lap starts and ends -------------------------------------------------------------------------------------

def lap_times(laps: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Each lap's start and end in session seconds (see THE CLOCK and LAP STARTS AND ENDS above).
    One row per (driver_number, lap_number), columns start_sec, end_sec."""
    out = []
    stamped = laps.assign(stamp=seconds_since(laps["date_start"], t0))
    for driver, d in stamped.sort_values("lap_number").groupby("driver_number", sort=False):
        number = d["lap_number"].to_numpy()
        stamp = d["stamp"].to_numpy(dtype=float)
        duration = d["lap_duration"].to_numpy(dtype=float)
        start = np.full(len(d), np.nan)
        i = 0
        while i < len(d):
            if number[i] == 1:                                   # lap 1 starts when the lights go out
                start[i] = 0.0
                i += 1
                continue
            j = i                                                # the run i..j: consecutive laps, each timed
            while j + 1 < len(d) and number[j + 1] == number[j] + 1 and np.isfinite(duration[j]):
                j += 1
            chain = np.concatenate([[0.0], np.cumsum(duration[i:j])])
            anchors = stamp[i:j + 1] - LATER_LAP_LAG_SEC - chain
            if np.isfinite(anchors).any():
                start[i:j + 1] = np.nanmedian(anchors) + chain
            i = j + 1
        end = start + duration                                   # nan without a duration...
        follows = np.append(number[1:] == number[:-1] + 1, False)
        next_start = np.append(start[1:], np.nan)
        is_lap_one = number == 1
        use_next = follows & np.isfinite(next_start) & (is_lap_one | ~np.isfinite(end))
        end = np.where(use_next, next_start, end)                # ...then it ends where the next lap starts
        out.append(pd.DataFrame({"driver_number": driver, "lap_number": number, "start_sec": start, "end_sec": end}))
    return pd.concat(out, ignore_index=True)


# --- track status --------------------------------------------------------------------------------------------

def status_timeline(race_control: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Every change of track status, rebuilt from Race Control: columns t_sec, code (FastF1's codes)."""
    rc = race_control.assign(t_sec=seconds_since(race_control["date"], t0)).sort_values("t_sec", kind="stable")
    neutral: Optional[str] = None             # a safety car, virtual safety car or red flag covers the whole track
    yellow: set = set()                       # sectors under yellow
    changes: List[Tuple[float, str]] = []
    last = None
    for row in rc.itertuples():
        message, flag, scope = str(row.message or ""), str(row.flag or ""), str(row.scope or "")
        if message.startswith("VIRTUAL SAFETY CAR DEPLOYED"):
            neutral = VSC
        elif message.startswith("VIRTUAL SAFETY CAR ENDING"):
            neutral = VSC_ENDING
        elif message.startswith("SAFETY CAR DEPLOYED"):
            neutral = SC
        elif flag == "RED":
            neutral = RED
        elif message.startswith("TRACK CLEAR"):
            yellow = set()
            if neutral != RED:                # a red flag ends when the race starts again, not when the track is clear
                neutral = None
        elif message == SESSION_STARTED:      # the lights go out on a clear track: yellows shown on the way to the
            yellow = set()                    # grid are over (Italy 2025: one never cleared, 30 minutes of false yellow)
            if neutral == RED:
                neutral = None
        elif flag in ("YELLOW", "DOUBLE YELLOW") and scope == "Sector":
            yellow.add(row.sector)
        elif flag == "CLEAR" and scope == "Sector":
            yellow.discard(row.sector)
        code = neutral or (YELLOW if yellow else GREEN)
        if code != last:
            changes.append((row.t_sec, code))
            last = code
    return pd.DataFrame(changes, columns=["t_sec", "code"])


def lap_status(start: np.ndarray, end: np.ndarray, timeline: pd.DataFrame) -> List[Optional[str]]:
    """For each lap: every code in force between its start and its end, each once, in order. None without both."""
    times, codes = timeline["t_sec"].to_numpy(dtype=float), timeline["code"].tolist()
    out: List[Optional[str]] = []
    for s, e in zip(start, end):
        if not (np.isfinite(s) and np.isfinite(e)):
            out.append(None)
            continue
        first = int(np.searchsorted(times, s, side="right")) - 1          # the code in force at the start
        last = int(np.searchsorted(times, e, side="left"))               # changes strictly before the end
        seen = [codes[first]] if first >= 0 else [GREEN]
        for code in codes[max(first + 1, 0):last]:
            if code not in seen:
                seen.append(code)
        out.append("".join(seen))
    return out


# --- the pieces of a lap row -----------------------------------------------------------------------------------

def tyres_by_lap(stints: pd.DataFrame, out_laps: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (driver_number, lap_number) a stint covers: Compound, TyreLife, FreshTyre, Stint.

    out_laps: (driver_number, lap_number) of every out-lap. Tyres only change in the pit lane, so a new set's first
    lap is an out-lap. OpenF1 sometimes starts a stint ONE LAP EARLY, on the in-lap (2023's first races; every stint
    after a red flag on lap 1, Monaco 2024, where the first stint is even left empty, laps 1..0). So a stint that
    starts on a lap that is not an out-lap, while the next lap is one, is moved a lap later, and the stint before it
    is stretched to meet it.

    A car driven through the pit lane keeps its tyres, yet OpenF1 opens a new stint (Australia 2025, laps 2-4 behind
    the safety car). Such a stint has the same compound and an age that simply carries on; it is the same set, so
    the tyre life and the stint number continue.
    """
    is_out = set(zip(out_laps["driver_number"], out_laps["lap_number"]))
    rows = []
    for driver, d in stints.dropna(subset=["lap_start", "lap_end"]).sort_values("stint_number").groupby("driver_number"):
        starts = [int(v) for v in d["lap_start"]]
        late = [int(k > 0 and (driver, s) not in is_out and (driver, s + 1) in is_out) for k, s in enumerate(starts)]
        before = None                                     # the stint before: (compound, tyre life on its last lap, its number)
        for k, s in enumerate(d.itertuples()):
            first = starts[k] + late[k]
            last = int(s.lap_end) + (late[k + 1] if k + 1 < len(starts) else late[k])
            age, number = s.tyre_age_at_start, s.stint_number
            if pd.isna(age):
                life_first = np.nan
            elif before is not None and before[0] == s.compound and before[1] + 1 == age:
                life_first, number = age, before[2]       # the same set carried on: OpenF1 gives its age ON this lap
            else:
                life_first = age + 1                      # a set fitted now: laps done before, so this lap is one more
            for lap in range(first, last + 1):
                rows.append((driver, lap, s.compound, life_first + lap - first,
                             None if pd.isna(age) else bool(life_first == 1), number))
            if last >= first:
                before = (s.compound, life_first + last - first, number)
    out = pd.DataFrame(rows, columns=["driver_number", "lap_number", "Compound", "TyreLife", "FreshTyre", "Stint"])
    return out.drop_duplicates(["driver_number", "lap_number"], keep="last")      # the later stint wins


def pit_by_lap(pit: Optional[pd.DataFrame], t0: pd.Timestamp) -> pd.DataFrame:
    """Per pit passage: the in-lap, the entry and the exit in session seconds (in_lap, pit_in_sec, pit_out_sec)."""
    columns = ["driver_number", "in_lap", "pit_in_sec", "pit_out_sec"]
    if pit is None or pit.empty:
        return pd.DataFrame(columns=columns)
    exit_sec = seconds_since(pit["date"], t0)
    out = pd.DataFrame({"driver_number": pit["driver_number"], "in_lap": pit["lap_number"],
                        "pit_in_sec": exit_sec - pit["lane_duration"].astype(float), "pit_out_sec": exit_sec})
    return out.dropna(subset=["in_lap"]).drop_duplicates(["driver_number", "in_lap"], keep="first")[columns]


def deleted_laps(race_control: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    """{(car, lap): Race Control's message} for every lap time deleted and not given back."""
    out: Dict[Tuple[int, int], str] = {}
    for message in race_control.sort_values("date", kind="stable")["message"].dropna():
        hit = DELETED.match(message)
        if hit:
            out[(int(hit.group(1)), int(hit.group(2)))] = message
        back = REINSTATED.match(message)
        if back:
            out.pop((int(back.group(1)), int(back.group(2))), None)
    return out


def positions_at(position: pd.DataFrame, t0: pd.Timestamp, drivers: np.ndarray, end_sec: np.ndarray) -> np.ndarray:
    """Each lap's position: the driver's last position record up to a second after the lap ends (nan if none)."""
    pos = position.assign(t_sec=seconds_since(position["date"], t0)).sort_values("t_sec", kind="stable")
    by_driver = {d: (g["t_sec"].to_numpy(dtype=float), g["position"].to_numpy(dtype=float))
                 for d, g in pos.groupby("driver_number")}
    out = np.full(len(drivers), np.nan)
    for i, (driver, end) in enumerate(zip(drivers, end_sec)):
        if driver in by_driver and np.isfinite(end):
            times, places = by_driver[driver]
            k = int(np.searchsorted(times, end + POSITION_SLACK_SEC, side="right")) - 1
            if k >= 0:
                out[i] = places[k]
    return out


def _timedelta(seconds: pd.Series) -> pd.Series:
    return pd.to_timedelta(seconds.astype(float), unit="s")


# --- the tables ---------------------------------------------------------------------------------------------------

def lap_table(tables: Dict[str, pd.DataFrame], t0: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """FastF1's lap table from OpenF1's laps, drivers, stints, pit, position and race_control."""
    laps = tables["laps"]
    duplicates = laps.duplicated(["driver_number", "lap_number"], keep="first")
    laps = laps[~duplicates].sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    df = laps.merge(lap_times(laps, t0), on=["driver_number", "lap_number"], how="left")
    df = df.merge(tables["drivers"][["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number"),
                  on="driver_number", how="left")
    out_lap = df["is_pit_out_lap"].fillna(False).astype(bool) if "is_pit_out_lap" in df else pd.Series(False, index=df.index)
    df = df.merge(tyres_by_lap(tables["stints"], df.loc[out_lap, ["driver_number", "lap_number"]]),
                  on=["driver_number", "lap_number"], how="left")

    passages = pit_by_lap(tables.get("pit"), t0)
    df = df.merge(passages.rename(columns={"in_lap": "lap_number"})[["driver_number", "lap_number", "pit_in_sec"]],
                  on=["driver_number", "lap_number"], how="left", indicator="_in")
    exits = passages.assign(lap_number=passages["in_lap"] + 1)[["driver_number", "lap_number", "pit_out_sec"]]
    df = df.merge(exits, on=["driver_number", "lap_number"], how="left", indicator="_out")
    same_driver_next = df["driver_number"].eq(df["driver_number"].shift(-1)) & df["lap_number"].shift(-1).eq(df["lap_number"] + 1)
    pit_in_lap = (df["_in"] == "both") | (same_driver_next & out_lap.shift(-1, fill_value=False))
    pit_out_lap = (df["_out"] == "both") | out_lap

    timeline = status_timeline(tables["race_control"], t0)
    start, end = df["start_sec"].to_numpy(dtype=float), df["end_sec"].to_numpy(dtype=float)
    deleted = deleted_laps(tables["race_control"])
    keys = list(zip(df["driver_number"].astype(int), df["lap_number"].astype(int)))
    sectors = df[["duration_sector_1", "duration_sector_2", "duration_sector_3"]].astype(float)
    # Inaccurate only on evidence: no lap time, or three sectors that do not add up to it (a missing sector proves nothing).
    off = (sectors.sum(axis=1, skipna=False) - df["lap_duration"].astype(float)).abs() > SECTORS_TOLERANCE_SEC
    accurate = df["lap_duration"].notna() & ~off.fillna(False)

    out = pd.DataFrame({
        "Time": _timedelta(df["end_sec"]),
        "Driver": df["name_acronym"].astype("string"),
        "DriverNumber": df["driver_number"].astype(int).astype(str).astype("string"),
        "LapTime": _timedelta(df["lap_duration"]),
        "LapNumber": df["lap_number"].astype(float),
        "Stint": df["Stint"].astype(float),
        "PitOutTime": _timedelta(df["pit_out_sec"]),
        "PitInTime": _timedelta(df["pit_in_sec"]),
        "Sector1Time": _timedelta(df["duration_sector_1"]),
        "Sector2Time": _timedelta(df["duration_sector_2"]),
        "Sector3Time": _timedelta(df["duration_sector_3"]),
        "SpeedI1": df["i1_speed"].astype(float),
        "SpeedI2": df["i2_speed"].astype(float),
        "SpeedST": df["st_speed"].astype(float),
        "Compound": df["Compound"].astype("string"),
        "TyreLife": df["TyreLife"].astype(float),
        "FreshTyre": df["FreshTyre"].astype("boolean"),
        "Team": df["team_name"].astype("string"),
        "LapStartTime": _timedelta(df["start_sec"]),
        "TrackStatus": pd.array(lap_status(start, end, timeline), dtype="string"),
        "Position": positions_at(tables["position"], t0, df["driver_number"].to_numpy(), end),
        "Deleted": [k in deleted for k in keys],
        "DeletedReason": pd.array([deleted.get(k) for k in keys], dtype="string"),
        "FastF1Generated": False,
        "IsAccurate": accurate.astype(bool),
        "PitInLap": pit_in_lap.astype(bool),
        "PitOutLap": pit_out_lap.astype(bool),
    })
    report = {"openf1_laps": int(len(laps)), "openf1_duplicate_laps": int(duplicates.sum()),
              "laps_without_end": int(out["Time"].isna().sum()), "pit_passages": int(len(passages)),
              "pit_laps_without_time": int((out["PitInLap"] & out["PitInTime"].isna()).sum()),
              "laps_deleted": int(out["Deleted"].sum()), "status_changes": int(len(timeline))}
    return out, report


def weather_table(weather: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """FastF1's weather table. Time is negative before lights out."""
    return pd.DataFrame({
        "Time": _timedelta(seconds_since(weather["date"], t0)),
        "AirTemp": weather["air_temperature"].astype(float),
        "Humidity": weather["humidity"].astype(float),
        "Pressure": weather["pressure"].astype(float),
        "Rainfall": weather["rainfall"].astype(float) > 0,
        "TrackTemp": weather["track_temperature"].astype(float),
        "WindDirection": weather["wind_direction"].astype(float),
        "WindSpeed": weather["wind_speed"].astype(float),
    })


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def results_table(session_result: pd.DataFrame, drivers: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    """FastF1's results table: one row per driver, classified first. `Time` is the winner's race time and the others'
    gap, as FastF1 has it. The grid is each driver's first position record (OpenF1's starting_grid is empty)."""
    grid = position.assign(t=utc(position["date"])).sort_values("t", kind="stable").drop_duplicates("driver_number")
    grid = grid.set_index("driver_number")["position"]
    who = drivers.drop_duplicates("driver_number").set_index("driver_number")
    rows = []
    for r in session_result.itertuples():
        d = who.loc[r.driver_number] if r.driver_number in who.index else {}
        gap_text = "" if pd.isna(r.gap_to_leader) else str(r.gap_to_leader)
        laps_down = LAPS_DOWN.search(gap_text.upper())
        if bool(getattr(r, "dsq", False)):
            status, classified = "Disqualified", "D"
        elif bool(getattr(r, "dns", False)):
            status, classified = "Did not start", "W"
        elif bool(getattr(r, "dnf", False)):
            status, classified = "Retired", "R"
        elif laps_down:
            n = int(laps_down.group(1))
            status, classified = f"+{n} Lap" + ("s" if n > 1 else ""), None
        else:
            status, classified = "Finished", None
        place = _number(r.position)
        seconds = _number(r.duration) if place == 1 else _number(r.gap_to_leader)
        first, last = d.get("first_name"), d.get("last_name")
        rows.append({
            "DriverNumber": str(int(r.driver_number)), "Abbreviation": d.get("name_acronym"),
            "FullName": f"{first} {last}" if first and last else d.get("full_name"),
            "TeamName": d.get("team_name"), "TeamColor": d.get("team_colour"),
            "Position": place, "ClassifiedPosition": classified or ("" if np.isnan(place) else str(int(place))),
            "GridPosition": _number(grid.get(r.driver_number)),
            "Time": pd.to_timedelta(seconds, unit="s") if status in ("Finished",) and np.isfinite(seconds) else pd.NaT,
            "Status": status, "Points": _number(getattr(r, "points", np.nan)), "Laps": _number(r.number_of_laps),
        })
    out = pd.DataFrame(rows).sort_values(["Position", "Laps"], ascending=[True, False], na_position="last")
    for col in ("DriverNumber", "Abbreviation", "FullName", "TeamName", "TeamColor", "ClassifiedPosition", "Status"):
        out[col] = out[col].astype("string")
    out["Time"] = pd.to_timedelta(out["Time"])
    return out.reset_index(drop=True)


def pit_stop_record(pit: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """OpenF1's pit table in the shape of F1's pit-stop record (cleaner.attach_pit_stops): the time stationary and
    the time in the pit lane, on the in-lap. None without a pit table."""
    if pit is None or pit.empty:
        return None
    stationary = pit["stop_duration"] if "stop_duration" in pit else np.nan
    return pd.DataFrame({"RacingNumber": pit["driver_number"].astype(int).astype(str), "Lap": pit["lap_number"],
                         "PitStopTime": stationary, "PitLaneTime": pit["lane_duration"]})


def translate(tables: Dict[str, pd.DataFrame]) -> Translation:
    """OpenF1's small tables ({endpoint: rows}, see SMALL_TABLES; 'pit' may be missing) -> what Silver needs."""
    if tables["laps"].empty:
        raise TranslationError("OpenF1's lap table is empty")
    t0, clock = lights_out(tables["race_control"], tables["laps"])
    laps, report = lap_table(tables, t0)
    results = results_table(tables["session_result"], tables["drivers"], tables["position"])
    winner_laps = results["Laps"].max()
    return Translation(laps=laps, weather=weather_table(tables["weather"], t0), results=results,
                       pit_stops=pit_stop_record(tables.get("pit")), lights_out_utc=t0,
                       total_laps=None if pd.isna(winner_laps) else int(winner_laps), report={**clock, **report})


def read_bronze(year: int, round_number: int, session: str = "R",
                root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> Dict[str, pd.DataFrame]:
    """The small OpenF1 tables of one race from Bronze, by endpoint; an endpoint Bronze has no table for is left out."""
    tables = {}
    for endpoint in SMALL_TABLES:
        path = lake.table_path(lake.BRONZE, lake.openf1_table(endpoint), year, round_number, session, root)
        if path.exists():
            tables[endpoint] = lake.read_table(path)
    return tables
