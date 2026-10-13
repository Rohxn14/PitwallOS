"""
Silver layer: cleaned, conformed tables (src/pipeline/cleaner.py)

Reads ONLY from Bronze. Produces two tables per session:

  laps       one row per driver per lap, times in seconds, plus boolean flags
             that say what kind of lap it was (pit lap, safety car, clean, ...)
  telemetry  car + position data merged onto one true 10 Hz clock per driver,
             with a lap number and distance-in-lap on every row

Silver rules:
  * Never invent a value. Unknown stays null.
  * Never delete an inconvenient lap. Flag it and let the consumer decide.
  * Every drop is counted and reported.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline import contracts, lake

# --- constants -------------------------------------------------------------

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
# Compound labels that do not identify a tyre, so they cannot show a tyre change.
UNIDENTIFIED_COMPOUNDS = ("UNKNOWN", "TEST_UNKNOWN")
# What FastF1 gives (as text) when the timing feed has no compound at all for a stint, e.g.
# 2023 Canadian GP, Tsunoda laps 36-70. That is a missing value, not a tyre: Silver makes it null.
NO_COMPOUND_MARKERS = ("NONE", "NAN", "")

# What happened on a pit-lane passage, recorded on its in-lap (null on every other lap).
# FastF1's own `Stint` goes up on EVERY passage, including a safety car led through the pit lane
# and drive-through / stop-go penalties, so it cannot be used to count stops.
PIT_TYRE_CHANGE = "tyre_change"    # compound changed, or tyre age did not simply continue
PIT_PASS_THROUGH = "pass_through"  # same compound and tyre age continues: no new tyres
PIT_ENDED = "ended"                # no next lap: retired or finished in the pit lane
PIT_UNKNOWN = "unknown"            # tyre data missing, cannot tell
PIT_KINDS = (PIT_TYRE_CHANGE, PIT_PASS_THROUGH, PIT_ENDED, PIT_UNKNOWN)

# FastF1 track status codes. A lap's status is every code seen during that lap,
# concatenated, e.g. "124" = green, then yellow, then safety car.
STATUS_GREEN, STATUS_YELLOW, STATUS_SC, STATUS_RED = "1", "2", "4", "5"
STATUS_VSC = ("6", "7")  # 6 = VSC deployed, 7 = VSC ending

WEATHER_MAX_AGE_SEC = 180.0   # weather is sampled once a minute; older than this = unknown
TELEMETRY_HZ = 10
TELEMETRY_MAX_GAP_SEC = 2.0   # never interpolate across a hole longer than this
THROTTLE_NO_DATA = 100.0      # the feed sends 104 to mean "no reading"
SPEED_MAX_KMH = 400.0
DRS_OPEN_CODES = (10, 12, 14)

REQUIRED_BRONZE_LAP_COLUMNS = (
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Time", "LapStartTime",
    "PitInTime", "PitOutTime", "Compound", "TyreLife", "FreshTyre", "Position",
    "TrackStatus", "IsAccurate", "Deleted", "FastF1Generated",
    "Sector1Time", "Sector2Time", "Sector3Time",
)


def _seconds(series: pd.Series) -> pd.Series:
    """Timedelta column -> float seconds (NaT -> NaN)."""
    return series.dt.total_seconds()


def _compound(raw: pd.Series) -> pd.Series:
    """Bronze Compound -> upper-case label, with FastF1's 'no compound' text turned into null."""
    compound = raw.astype("string").str.upper()
    no_compound = compound.str.strip().isin(NO_COMPOUND_MARKERS).fillna(False).astype(bool)
    return compound.mask(no_compound)


# --- laps ------------------------------------------------------------------

def clean_laps(laps: pd.DataFrame, weather: pd.DataFrame,
               season: int, round_number: int, session: str,
               pit_stops: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Bronze laps (+ weather, + F1's pit-stop record if published) -> Silver lap table. Pure
    function: no file access.

    Returns (silver_laps, report) where report counts every row dropped.
    """
    missing = [c for c in REQUIRED_BRONZE_LAP_COLUMNS if c not in laps.columns]
    if missing:
        raise ValueError(f"Bronze laps table is missing required columns: {missing}")

    report: Dict[str, int] = {"input_rows": int(len(laps))}
    df = laps.copy()

    # A lap with no driver or no lap number cannot be identified: drop and count.
    null_key = df["Driver"].isna() | df["LapNumber"].isna()
    report["dropped_null_key"] = int(null_key.sum())
    df = df[~null_key]

    # (driver, lap) must be unique. Keep the first occurrence, count the rest.
    dup = df.duplicated(["Driver", "LapNumber"], keep="first")
    report["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    status = df["TrackStatus"].astype("string").fillna("")
    compound = _compound(df["Compound"])
    lap_number = df["LapNumber"].astype("int64")

    out = pd.DataFrame({
        "season": int(season),
        "round": int(round_number),
        "session": session,
        "driver_code": df["Driver"].astype("string"),
        "driver_number": df["DriverNumber"].astype("string"),
        "team": df["Team"].astype("string"),
        "lap_number": lap_number,
        "position": df["Position"].round().astype("Int64"),
        "compound": compound,
        "tyre_life": df["TyreLife"].round().astype("Int64"),
        "fresh_tyre": df["FreshTyre"].astype("boolean"),
        "lap_time_sec": _seconds(df["LapTime"]),
        "sector1_sec": _seconds(df["Sector1Time"]),
        "sector2_sec": _seconds(df["Sector2Time"]),
        "sector3_sec": _seconds(df["Sector3Time"]),
        "lap_start_sec": _seconds(df["LapStartTime"]),
        "lap_end_sec": _seconds(df["Time"]),
        "pit_in_sec": _seconds(df["PitInTime"]),
        "pit_out_sec": _seconds(df["PitOutTime"]),
        "track_status": status,
    })

    # --- flags: what kind of lap was this? ---
    out["is_in_lap"] = out["pit_in_sec"].notna()
    out["is_out_lap"] = out["pit_out_sec"].notna()
    out["is_pit_lap"] = out["is_in_lap"] | out["is_out_lap"]
    out["is_lap1"] = out["lap_number"] == 1
    out["is_green"] = status == STATUS_GREEN                 # green for the WHOLE lap
    out["is_yellow"] = status.str.contains(STATUS_YELLOW, regex=False)
    out["is_sc"] = status.str.contains(STATUS_SC, regex=False)
    out["is_vsc"] = status.str.contains(STATUS_VSC[0], regex=False) | status.str.contains(STATUS_VSC[1], regex=False)
    out["is_red"] = status.str.contains(STATUS_RED, regex=False)
    out["is_accurate"] = df["IsAccurate"].astype(bool)
    out["is_deleted"] = df["Deleted"].astype(bool)
    out["is_generated"] = df["FastF1Generated"].astype(bool)
    out["is_wet_tyre"] = compound.isin(WET_COMPOUNDS).fillna(False).astype(bool)

    # A "clean" lap is a representative racing lap: timed, trustworthy, no pit
    # entry/exit, not the standing-start lap, and green from start to finish.
    out["is_clean_lap"] = (
        out["lap_time_sec"].notna()
        & out["is_accurate"]
        & ~out["is_generated"]
        & ~out["is_pit_lap"]
        & ~out["is_lap1"]
        & out["is_green"]
    )

    # Flags are plain True/False (never null): "unknown" is not a useful answer for a filter.
    flag_cols = [c for c in out.columns if c.startswith("is_")]
    out[flag_cols] = out[flag_cols].fillna(False).astype(bool)

    out = _attach_weather(out, weather)
    out = out.sort_values(["driver_code", "lap_number"]).reset_index(drop=True)
    out = _add_pit_kind_and_stint(out)
    out, stop_report = attach_pit_stops(out, pit_stops)
    report.update(stop_report)

    report["output_rows"] = int(len(out))
    report["clean_laps"] = int(out["is_clean_lap"].sum())
    for kind in PIT_KINDS:
        report[f"pit_{kind}"] = int((out["pit_kind"] == kind).sum())
    return out, report


def attach_pit_stops(laps: pd.DataFrame, pit_stops: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    F1's own times for each pit stop, on the stop's in-lap (F1 numbers a stop by its in-lap):

        pit_stop_time_sec  the car stationary in its box (F1's PitStopTime, to 0.1 s)
        pit_lane_time_sec  pit entry to pit exit (F1's PitLaneTime)

    Null on every other lap, and on every lap of a race F1 published no record for (before 2024
    round 19): unknown, never 0. A published stop that matches no in-lap, a second record of the
    same stop and a value that is not a number are counted in the report, never silently dropped.
    """
    out = laps.copy()
    out["pit_stop_time_sec"] = np.nan
    out["pit_lane_time_sec"] = np.nan
    report = {"pit_stops_published": 0, "pit_stops_matched": 0, "dropped_pit_stop_duplicates": 0,
              "dropped_pit_stops_unmatched": 0, "pit_stop_values_not_numbers": 0}
    if pit_stops is None or pit_stops.empty:
        return out, report

    stops = pd.DataFrame({
        "driver_number": pit_stops["RacingNumber"].astype("string").str.strip(),
        "lap_number": pd.to_numeric(pit_stops["Lap"], errors="coerce"),
        "pit_stop_time_sec": pd.to_numeric(pit_stops["PitStopTime"], errors="coerce"),
        "pit_lane_time_sec": pd.to_numeric(pit_stops["PitLaneTime"], errors="coerce"),
    })
    report["pit_stops_published"] = int(len(stops))
    for col, raw in (("pit_stop_time_sec", "PitStopTime"), ("pit_lane_time_sec", "PitLaneTime")):
        report["pit_stop_values_not_numbers"] += int((stops[col].isna() & pit_stops[raw].notna()).sum())
    stops = stops.dropna(subset=["driver_number", "lap_number"])
    stops["lap_number"] = stops["lap_number"].astype("int64")
    duplicate = stops.duplicated(["driver_number", "lap_number"], keep="first")
    report["dropped_pit_stop_duplicates"] = int(duplicate.sum())
    stops = stops[~duplicate]

    in_laps = out.loc[out["is_in_lap"], ["driver_number", "lap_number"]].reset_index()
    matched = in_laps.merge(stops, on=["driver_number", "lap_number"], how="inner")
    out.loc[matched["index"], "pit_stop_time_sec"] = matched["pit_stop_time_sec"].to_numpy()
    out.loc[matched["index"], "pit_lane_time_sec"] = matched["pit_lane_time_sec"].to_numpy()
    report["pit_stops_matched"] = int(len(matched))
    report["dropped_pit_stops_unmatched"] = int(len(stops) - len(matched))
    return out, report


def classify_pit_passage(compound, tyre_life, next_compound, next_tyre_life, has_next_lap: bool) -> str:
    """
    What happened on one pit-lane passage, judged from the tyres on the in-lap and the next lap.

    The official timing feed keeps counting a tyre set's age across a passage when the tyres
    stay on the car (e.g. 2 -> 3), and restarts it, or changes the compound, when they come off.
    """
    if not has_next_lap:
        return PIT_ENDED
    known = [c for c in (compound, next_compound) if not pd.isna(c) and c not in UNIDENTIFIED_COMPOUNDS]
    if len(known) == 2 and known[0] != known[1]:
        return PIT_TYRE_CHANGE
    if pd.isna(tyre_life) or pd.isna(next_tyre_life):
        return PIT_UNKNOWN
    return PIT_PASS_THROUGH if int(next_tyre_life) == int(tyre_life) + 1 else PIT_TYRE_CHANGE


def tyre_record_inconsistent(laps: pd.DataFrame) -> pd.Series:
    """
    True on every lap of a driver whose tyre record changes where no tyre change can happen.

    Tyres only come off in the pit lane (or under a red flag), so the record may change (compound
    changes, or the age does not count up by one) only on the out-lap or the lap after it (the
    feed can lag a lap), or on / right after a red-flag lap. A change anywhere else means the feed
    has the tyre stints in the wrong place: 2025 Miami shows tyre age restarting up to four laps
    before the stops, for 12 drivers. Then neither the stops nor the tyre ages of that driver can
    be trusted, and every lap of theirs is flagged (rule 3: flag, never delete).

    Expects laps sorted by driver then lap. Missing values are never a change: unknown is not
    evidence.
    """
    by_driver = laps.groupby("driver_code", sort=False)
    prev_lap = by_driver["lap_number"].shift(1)
    prev_compound = by_driver["compound"].shift(1)
    age = laps["tyre_life"].astype("float64")
    prev_age = by_driver["tyre_life"].shift(1).astype("float64")
    consecutive = prev_lap.eq(laps["lap_number"] - 1).fillna(False).astype(bool)

    both_compounds = laps["compound"].notna() & prev_compound.notna()
    compound_changed = both_compounds & (laps["compound"].astype(object) != prev_compound.astype(object))
    age_jumped = age.notna() & prev_age.notna() & (age != prev_age + 1)
    changed = consecutive & (compound_changed | age_jumped).fillna(False).astype(bool)

    flag = lambda col, k: by_driver[col].shift(k, fill_value=False).astype(bool)
    explained = flag("is_in_lap", 1) | flag("is_in_lap", 2) | laps["is_red"] | flag("is_red", 1)
    unexplained = changed & ~explained
    return unexplained.groupby(laps["driver_code"]).transform("any").astype(bool)


def _add_pit_kind_and_stint(laps: pd.DataFrame) -> pd.DataFrame:
    """
    pit_kind on every in-lap, is_tyre_change, and `stint` counted from tyre changes only.

    Expects laps sorted by driver then lap. The stint after a passage that cannot be classified
    is unknown (null) for the rest of that driver's race, rather than guessed.

    The feed can lag a lap: the out-lap still shows the old tyres and the change appears on the
    lap after it (2023 Canada, TSU lap 34: the out-lap says HARD 34, the next lap has no tyre
    data at all; the only case in 2023-2025). A tyre change needs a pit passage, so a change seen
    there belongs to this passage. A pass-through must therefore also continue from the out-lap
    to the next lap; otherwise it is a tyre change (change seen) or unknown (tyre data missing).
    The check is skipped when the out-lap is itself an in-lap (back-to-back passages: a penalty
    then a stop, 2024 Las Vegas OCO laps 11-12; the field led through the pit lane on laps 2-4
    behind the safety car, 2025 Australia): that next passage is judged on its own.

    A driver whose tyre record is inconsistent (see tyre_record_inconsistent) gets `unknown` on
    every passage: the record that would classify them is the thing that is wrong.
    """
    laps = laps.copy()
    laps["is_tyre_inconsistent"] = tyre_record_inconsistent(laps)
    driver, lap = laps["driver_code"], laps["lap_number"]
    # driver_code is a nullable string column: comparing with the shifted last row gives NA, not False
    same_driver = driver.eq(driver.shift(-1)).fillna(False).astype(bool)
    next_is_consecutive = same_driver & lap.shift(-1).eq(lap + 1)
    after_next_is_consecutive = next_is_consecutive & driver.eq(driver.shift(-2)).fillna(False).astype(bool) \
        & lap.shift(-2).eq(lap + 2)
    next_is_in_lap = laps["is_in_lap"].shift(-1, fill_value=False).astype(bool)
    compound, tyre_life = laps["compound"], laps["tyre_life"]

    kinds = pd.Series(pd.NA, index=laps.index, dtype="string")
    for i in laps.index[laps["is_in_lap"]]:
        if same_driver.loc[i] and not next_is_consecutive.loc[i]:
            kinds.loc[i] = PIT_UNKNOWN         # the next lap exists but a lap is missing in between
            continue
        pos = laps.index.get_loc(i)
        kind = classify_pit_passage(compound.iloc[pos], tyre_life.iloc[pos],
                                    compound.iloc[pos + 1] if same_driver.loc[i] else pd.NA,
                                    tyre_life.iloc[pos + 1] if same_driver.loc[i] else pd.NA,
                                    has_next_lap=bool(same_driver.loc[i]))
        if kind == PIT_PASS_THROUGH and after_next_is_consecutive.loc[i] and not next_is_in_lap.loc[i]:
            kind = classify_pit_passage(compound.iloc[pos + 1], tyre_life.iloc[pos + 1],
                                        compound.iloc[pos + 2], tyre_life.iloc[pos + 2], has_next_lap=True)
        if laps.at[i, "is_tyre_inconsistent"]:
            kind = PIT_UNKNOWN                 # this driver's tyre record cannot be trusted
        kinds.loc[i] = kind
    laps["pit_kind"] = kinds
    laps["is_tyre_change"] = (kinds == PIT_TYRE_CHANGE).fillna(False).astype(bool)

    # A stint starts on the lap AFTER a tyre change.
    by_driver = laps.groupby("driver_code", sort=False)
    changes_before = by_driver["is_tyre_change"].transform(lambda s: s.shift(1, fill_value=False).cumsum())
    unknown_before = by_driver["pit_kind"].transform(
        lambda s: s.eq(PIT_UNKNOWN).fillna(False).shift(1, fill_value=False).cumsum())
    stint = (changes_before + 1).astype("Int64")
    stint[unknown_before > 0] = pd.NA
    laps["stint"] = stint
    return laps


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most recent weather reading taken at or before each lap's end.

    Uses an as-of join: for each lap, look BACKWARD in time for the latest
    reading, and give up (null) if it is older than WEATHER_MAX_AGE_SEC.
    """
    cols = {"AirTemp": "air_temp_c", "TrackTemp": "track_temp_c", "Humidity": "humidity_pct",
            "WindSpeed": "wind_speed_ms", "Rainfall": "rainfall"}
    laps = laps.copy()

    usable = (weather is not None and not weather.empty and "Time" in weather.columns)
    if usable:
        w = weather[["Time"] + [c for c in cols if c in weather.columns]].copy()
        w["weather_time_sec"] = _seconds(w["Time"])
        w = w.drop(columns="Time").dropna(subset=["weather_time_sec"]).sort_values("weather_time_sec")
        w = w.rename(columns=cols)

        has_end = laps["lap_end_sec"].notna()
        left = laps[has_end].sort_values("lap_end_sec")
        merged = pd.merge_asof(left, w, left_on="lap_end_sec", right_on="weather_time_sec",
                               direction="backward", tolerance=WEATHER_MAX_AGE_SEC)
        merged = merged.drop(columns="weather_time_sec")
        laps = pd.concat([merged, laps[~has_end]], ignore_index=True)

    for name in cols.values():
        if name not in laps.columns:
            laps[name] = np.nan
    laps["rainfall"] = laps["rainfall"].astype("boolean")
    return laps


# --- telemetry -------------------------------------------------------------

def _interp(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linear interpolation of a continuous signal onto the grid, ignoring missing samples."""
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, t[ok], v[ok], left=np.nan, right=np.nan)


def _hold_index(grid: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the last source sample at or before each grid time (-1 if none)."""
    return np.searchsorted(t, grid, side="right") - 1


def _source_gap(grid: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Seconds between the two real samples that surround each grid time (inf at the edges)."""
    nxt = np.searchsorted(t, grid, side="left")
    prev = np.searchsorted(t, grid, side="right") - 1
    gap = np.full(grid.shape, np.inf)
    ok = (prev >= 0) & (nxt < len(t))
    gap[ok] = t[nxt[ok]] - t[prev[ok]]
    return gap


def resample_driver(car: pd.DataFrame, pos: pd.DataFrame, lap_start_sec: np.ndarray,
                    lap_numbers: np.ndarray, end_sec: float,
                    hz: int = TELEMETRY_HZ, max_gap_sec: float = TELEMETRY_MAX_GAP_SEC) -> pd.DataFrame:
    """
    Merge one driver's car and position streams onto a uniform clock.

    The source streams tick at ~4 Hz on two different clocks. We build a grid of
    exact 1/hz-second ticks (shared by all drivers, so cars can be compared at
    the same instant) and compute every signal at those ticks:
      * continuous signals (speed, rpm, throttle, x, y, z): linear interpolation
      * discrete signals (gear, brake, DRS): hold the last real value
    Where the surrounding real samples are more than max_gap_sec apart, values
    are left null and is_data_gap is True.
    """
    car = car.copy()
    car["t"] = _seconds(car["SessionTime"])
    car = car.dropna(subset=["t"]).drop_duplicates("t").sort_values("t")
    car = car[(car["Speed"] >= 0) & (car["Speed"] <= SPEED_MAX_KMH)]
    car.loc[car["Throttle"] > THROTTLE_NO_DATA, "Throttle"] = np.nan

    pos = pos.copy()
    pos["t"] = _seconds(pos["SessionTime"])
    pos = pos.dropna(subset=["t"]).drop_duplicates("t").sort_values("t")
    pos = pos[~((pos["X"] == 0) & (pos["Y"] == 0))]   # (0, 0) means "no position fix"

    start_sec = float(lap_start_sec[0])
    ticks = np.arange(int(np.ceil(start_sec * hz)), int(np.floor(end_sec * hz)) + 1)
    grid = ticks / hz
    if len(grid) == 0 or len(car) < 2 or len(pos) < 2:
        return pd.DataFrame()

    ct, pt = car["t"].to_numpy(), pos["t"].to_numpy()
    car_gap, pos_gap = _source_gap(grid, ct), _source_gap(grid, pt)
    car_bad, pos_bad = car_gap > max_gap_sec, pos_gap > max_gap_sec

    out = pd.DataFrame({"session_time_sec": grid})

    lap_idx = np.clip(np.searchsorted(lap_start_sec, grid, side="right") - 1, 0, len(lap_numbers) - 1)
    out["lap_number"] = lap_numbers[lap_idx].astype("int32")
    out["lap_elapsed_sec"] = (grid - lap_start_sec[lap_idx]).astype("float32")

    for src, name in (("Speed", "speed_kmh"), ("RPM", "rpm"), ("Throttle", "throttle_pct")):
        values = _interp(grid, ct, car[src].to_numpy(dtype=float))
        values[car_bad] = np.nan
        out[name] = values.astype("float32")

    hold = _hold_index(grid, ct)
    safe = np.clip(hold, 0, None)
    known = (hold >= 0) & ~car_bad
    gear = pd.array(car["nGear"].to_numpy()[safe], dtype="Int8")
    drs = pd.array(car["DRS"].to_numpy()[safe], dtype="Int8")
    brake = pd.array(car["Brake"].to_numpy(dtype=bool)[safe], dtype="boolean")
    gear[~known], drs[~known], brake[~known] = pd.NA, pd.NA, pd.NA
    out["gear"], out["brake"], out["drs"] = gear, brake, drs
    out["drs_open"] = out["drs"].isin(DRS_OPEN_CODES)

    for src, name in (("X", "x"), ("Y", "y"), ("Z", "z")):
        values = _interp(grid, pt, pos[src].to_numpy(dtype=float))
        values[pos_bad] = np.nan
        out[name] = values.astype("float32")

    out["is_data_gap"] = car_bad | pos_bad

    # Distance travelled in the current lap: integrate speed over time, restart each lap.
    step_m = (out["speed_kmh"].fillna(0.0).to_numpy(dtype=float) / 3.6) / hz
    out["lap_distance_m"] = pd.Series(step_m).groupby(out["lap_number"].to_numpy()).cumsum().astype("float32")
    return out


def clean_telemetry(car: pd.DataFrame, pos: pd.DataFrame, silver_laps: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resample every driver and stack the results. Lap boundaries come from Silver laps."""
    report: Dict[str, Any] = {"car_rows_in": int(len(car)), "pos_rows_in": int(len(pos)), "drivers_skipped": []}
    parts = []
    ident = silver_laps.iloc[0][["season", "round", "session"]]

    for drv_num, drv_laps in silver_laps.groupby("driver_number", sort=True):
        drv_laps = drv_laps.dropna(subset=["lap_start_sec"]).sort_values("lap_number")
        ends = drv_laps["lap_end_sec"].dropna()
        d_car, d_pos = car[car["DriverNumber"] == drv_num], pos[pos["DriverNumber"] == drv_num]
        if drv_laps.empty or ends.empty or d_car.empty or d_pos.empty:
            report["drivers_skipped"].append(str(drv_num))
            continue
        frame = resample_driver(d_car, d_pos, drv_laps["lap_start_sec"].to_numpy(dtype=float),
                                drv_laps["lap_number"].to_numpy(), float(ends.max()))
        if frame.empty:
            report["drivers_skipped"].append(str(drv_num))
            continue
        frame.insert(0, "driver_code", str(drv_laps["driver_code"].iloc[0]))
        frame.insert(0, "driver_number", str(drv_num))
        parts.append(frame)

    if not parts:
        raise ValueError("No driver produced any telemetry. Check Bronze car_telemetry / pos_telemetry.")

    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "session", ident["session"])
    out.insert(0, "round", int(ident["round"]))
    out.insert(0, "season", int(ident["season"]))
    report["output_rows"] = int(len(out))
    report["gap_rows"] = int(out["is_data_gap"].sum())
    return out, report


# --- orchestration ---------------------------------------------------------

def build_silver(year: int, round_number: int, session: str = "R",
                 lake_root: str = lake.DEFAULT_LAKE_ROOT, telemetry: bool = True) -> Dict[str, Any]:
    """Bronze -> Silver for one session. Set telemetry=False to build only the lap table."""
    start = time.time()
    manifest = lake.read_manifest(year, round_number, session, lake_root)
    identity = {k: manifest[k] for k in ("season", "round", "session", "event_name", "location", "country")}
    identity["total_laps"] = manifest.get("total_laps")

    def bronze(table: str) -> pd.DataFrame:
        return lake.read_table(lake.table_path(lake.BRONZE, table, year, round_number, session, lake_root))

    # F1's pit-stop record: only for races it was published for, and only in Bronze ingested since the
    # extractor fetches it (`backfill --pit-stops` adds it to older Bronze). Otherwise the times are unknown.
    has_stops = lake.bronze_has_pit_stops(year, round_number, session, lake_root)
    laps, lap_report = clean_laps(bronze("laps"), bronze("weather"), year, round_number, session,
                                  pit_stops=bronze(lake.BRONZE_PIT_STOPS_TABLE) if has_stops else None)
    contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")
    laps_path = lake.table_path(lake.SILVER, "laps", year, round_number, session, lake_root)
    lake.write_table(laps, laps_path, metadata=identity)
    total_bytes = laps_path.stat().st_size

    tel_report: Dict[str, Any] = {}
    tel_path = None
    if telemetry:
        tel, tel_report = clean_telemetry(bronze("car_telemetry"), bronze("pos_telemetry"), laps)
        contracts.validate(tel, contracts.SILVER_TELEMETRY, "Silver telemetry")
        tel_path = lake.table_path(lake.SILVER, "telemetry", year, round_number, session, lake_root)
        lake.write_table(tel, tel_path, metadata=identity)
        total_bytes += tel_path.stat().st_size

    return {
        "status": "success",
        "file_path": str(laps_path),
        "telemetry_path": str(tel_path) if tel_path else None,
        "year": int(year),
        "round_number": int(round_number),
        "event_name": identity["event_name"],
        "initial_rows": lap_report["input_rows"],
        "clean_rows": lap_report["output_rows"],
        "lap_report": lap_report,
        "telemetry_report": tel_report,
        "file_size_mb": round(total_bytes / 1024 / 1024, 2),
        "elapsed_sec": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    print(build_silver(2024, 12, "R"))
