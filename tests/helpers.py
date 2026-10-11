"""Builders for small, readable test tables shaped like FastF1 / Bronze data."""

import numpy as np
import pandas as pd


def td(seconds):
    """None -> NaN, so a column of times converts cleanly to Timedelta (NaN -> NaT)."""
    return np.nan if seconds is None else float(seconds)


def make_bronze_laps(rows):
    """
    Build a Bronze-shaped laps table from short dicts.

    Each row needs: driver, lap, start, end. Everything else has a sensible default,
    so a test only states the fields it is about.
    """
    records = []
    for r in rows:
        start, end = r["start"], r["end"]
        lap_time = r.get("lap_time", (end - start) if end is not None else None)
        records.append({
            "Driver": r["driver"],
            "DriverNumber": r.get("number", {"AAA": "1", "BBB": "2", "CCC": "3"}.get(r["driver"], "9")),
            "Team": r.get("team", "Team " + str(r["driver"])),
            "LapNumber": float(r["lap"]) if r["lap"] is not None else np.nan,
            "LapTime": td(lap_time),
            "Time": td(end),
            "LapStartTime": td(start),
            "PitInTime": td(r.get("pit_in")),
            "PitOutTime": td(r.get("pit_out")),
            "Stint": r.get("stint", 1.0),
            "Compound": r.get("compound", "MEDIUM"),
            "TyreLife": r.get("tyre_life", float(r["lap"]) if r["lap"] is not None else np.nan),
            "FreshTyre": True,
            "Position": r.get("position", 1.0),
            "TrackStatus": r.get("status", "1"),
            "IsAccurate": r.get("accurate", True),
            "Deleted": False,
            "FastF1Generated": False,
            "Sector1Time": np.nan, "Sector2Time": np.nan, "Sector3Time": np.nan,
        })
    df = pd.DataFrame(records)
    for col in ("LapTime", "Time", "LapStartTime", "PitInTime", "PitOutTime",
                "Sector1Time", "Sector2Time", "Sector3Time"):
        df[col] = pd.to_timedelta(df[col].astype(float), unit="s")   # FastF1 stores times as Timedelta
    for col in ("LapNumber", "Stint", "TyreLife", "Position"):
        df[col] = df[col].astype(float)                               # None -> NaN, as in real Bronze
    return df


def make_weather(rows):
    """rows: list of (time_sec, track_temp, rainfall)."""
    return pd.DataFrame({
        "Time": pd.to_timedelta([r[0] for r in rows], unit="s"),
        "AirTemp": 20.0, "Humidity": 50.0, "WindSpeed": 1.0,
        "TrackTemp": [r[1] for r in rows],
        "Rainfall": [r[2] for r in rows],
    })
