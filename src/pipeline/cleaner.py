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
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.pipeline import contracts, lake

# --- constants -------------------------------------------------------------

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
# Compound labels that do not identify a tyre, so they cannot show a tyre change.
UNIDENTIFIED_COMPOUNDS = ("UNKNOWN", "TEST_UNKNOWN")

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


# --- laps ------------------------------------------------------------------

def clean_laps(laps: pd.DataFrame, weather: pd.DataFrame,
               season: int, round_number: int, session: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Bronze laps (+ weather) -> Silver lap table. Pure function: no file access.

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
    compound = df["Compound"].astype("string").str.upper()
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

    report["output_rows"] = int(len(out))
    report["clean_laps"] = int(out["is_clean_lap"].sum())
    for kind in PIT_KINDS:
        report[f"pit_{kind}"] = int((out["pit_kind"] == kind).sum())
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


def _add_pit_kind_and_stint(laps: pd.DataFrame) -> pd.DataFrame:
    """
    pit_kind on every in-lap, is_tyre_change, and `stint` counted from tyre changes only.

    Expects laps sorted by driver then lap. The stint after a passage that cannot be classified
    is unknown (null) for the rest of that driver's race, rather than guessed.
    """
    laps = laps.copy()
    # driver_code is a nullable string column: comparing with the shifted last row gives NA, not False
    same_driver = laps["driver_code"].eq(laps["driver_code"].shift(-1)).fillna(False).astype(bool)
    next_is_consecutive = same_driver & laps["lap_number"].shift(-1).eq(laps["lap_number"] + 1)
    nxt_compound = laps["compound"].shift(-1)
    nxt_tyre_life = laps["tyre_life"].shift(-1)

    kinds = pd.Series(pd.NA, index=laps.index, dtype="string")
    for i in laps.index[laps["is_in_lap"]]:
        if same_driver.loc[i] and not next_is_consecutive.loc[i]:
            kinds.loc[i] = PIT_UNKNOWN         # the next lap exists but a lap is missing in between
            continue
        kinds.loc[i] = classify_pit_passage(laps.at[i, "compound"], laps.at[i, "tyre_life"],
                                            nxt_compound.loc[i], nxt_tyre_life.loc[i],
                                            has_next_lap=bool(same_driver.loc[i]))
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

    laps, lap_report = clean_laps(bronze("laps"), bronze("weather"), year, round_number, session)
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
