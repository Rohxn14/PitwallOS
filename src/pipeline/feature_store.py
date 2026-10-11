"""
Gold layer: lap feature table (src/pipeline/feature_store.py)

Reads ONLY Silver laps. Produces one row per driver per lap with model-ready
features and the prediction target.

Time convention (this is what prevents leakage):
    Every feature on a row describes what is known when that lap ENDS.
    The target is the NEXT lap's time. So no feature may look at lap N+1.

Gold keeps every lap (pit laps, safety car laps, ...) and carries the Silver
flags forward. Which laps to train or analyse on is the consumer's decision.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.pipeline import contracts, lake

# --- modelling assumptions (named so they can be found and challenged) ------

FUEL_START_KG = 110.0          # maximum race fuel load allowed by the regulations
FUEL_END_KG = 5.0              # assumed left in the tank at the flag
DIRTY_AIR_THRESHOLD_SEC = 1.5  # following closer than this counts as dirty air
ROLLING_WINDOW_LAPS = 3

FLAG_COLUMNS = [
    "is_pit_lap", "is_in_lap", "is_out_lap", "is_lap1", "is_green", "is_yellow", "is_sc",
    "is_vsc", "is_red", "is_accurate", "is_deleted", "is_wet_tyre", "is_clean_lap",
    "is_tyre_change",
]

GOLD_COLUMNS = [
    # identity
    "season", "round", "session", "driver_code", "team_code", "lap_number", "total_race_laps",
    # race state at the end of this lap
    "lap_time_sec", "position", "laps_remaining", "race_progress",
    # tyres and fuel
    "compound", "tyre_age", "stint_number", "stint_lap", "fresh_tyre", "fuel_mass_est",
    # conditions
    "track_temp", "air_temp", "rainfall",
    # traffic
    "gap_ahead_sec", "gap_to_leader_sec", "gap_on_track_start_sec", "gap_on_track_end_sec", "in_dirty_air",
    # recent pace (this lap and earlier only)
    "prev_lap_time_sec", "last_clean_lap_time_sec", "roll3_clean_mean_sec", "roll3_clean_std_sec",
    # flags
    *FLAG_COLUMNS,
    # target
    "target_next_lap_time_sec", "target_next_is_clean",
]


def _add_fuel(df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Estimated fuel on board at the START of each lap, assuming a constant burn per lap."""
    burn_per_lap = (FUEL_START_KG - FUEL_END_KG) / total_laps
    df["fuel_mass_est"] = (FUEL_START_KG - (df["lap_number"] - 1) * burn_per_lap).clip(lower=0.0).round(2)
    return df


def _add_race_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gaps in the race classification, measured at the finish line at the end of each lap.

    gap_ahead_sec     = time between the car one position ahead completing lap N
                        and this car completing lap N.
    gap_to_leader_sec = the same, against the car in P1.
    """
    pos = df["position"].astype("float64")
    ends = df.assign(_pos=pos)[["lap_number", "_pos", "lap_end_sec"]].dropna()
    ends = ends.drop_duplicates(["lap_number", "_pos"])

    ahead = ends.rename(columns={"lap_end_sec": "_ahead_end"}).assign(_pos=lambda d: d["_pos"] + 1)
    leader = ends[ends["_pos"] == 1][["lap_number", "lap_end_sec"]].rename(columns={"lap_end_sec": "_leader_end"})

    df = df.assign(_pos=pos).merge(ahead, on=["lap_number", "_pos"], how="left").merge(leader, on="lap_number", how="left")
    df["gap_ahead_sec"] = (df["lap_end_sec"] - df["_ahead_end"]).round(3)
    df["gap_to_leader_sec"] = (df["lap_end_sec"] - df["_leader_end"]).round(3)
    # A negative gap means the position data and the timing disagree: unknown, not zero.
    df.loc[df["gap_ahead_sec"] < 0, "gap_ahead_sec"] = np.nan
    df.loc[df["gap_to_leader_sec"] < 0, "gap_to_leader_sec"] = np.nan
    return df.drop(columns=["_pos", "_ahead_end", "_leader_end"])


def _add_track_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gap to the car physically ahead ON THE ROAD, which is what causes dirty air.

    This differs from gap_ahead_sec when lapped cars are involved: the car in
    front of you on track may be a lap down and nowhere near you in the race.

    Method: collect every finish-line crossing by every car, sorted by time.
    For each crossing, the latest earlier crossing by a DIFFERENT car is the car
    ahead on the road. Crossings made in the pit lane (in-laps) are left out,
    because a car in the pit lane does not disturb the air on track.
    """
    on_track = (df["lap_end_sec"].notna() & ~df["is_in_lap"]).to_numpy()
    order = np.argsort(df.loc[on_track, "lap_end_sec"].to_numpy(dtype=float), kind="stable")
    crossings = df.loc[on_track, "lap_end_sec"].to_numpy(dtype=float)[order]
    crossing_driver = df.loc[on_track, "driver_code"].to_numpy(dtype=object)[order]

    t = df["lap_end_sec"].to_numpy(dtype=float)
    driver = df["driver_code"].to_numpy(dtype=object)
    prev_idx = np.searchsorted(crossings, t, side="left") - 1

    # Step back past the car's own earlier crossings (only happens when it is alone on track).
    while True:
        own = on_track & (prev_idx >= 0) & (crossing_driver[np.clip(prev_idx, 0, None)] == driver)
        if not own.any():
            break
        prev_idx[own] -= 1

    gap = np.full(len(df), np.nan)
    ok = on_track & (prev_idx >= 0)
    gap[ok] = t[ok] - crossings[prev_idx[ok]]
    df["gap_on_track_end_sec"] = np.round(gap, 3)

    # The gap at the start of lap N is the gap at the end of lap N-1.
    prev = df.groupby("driver_code")[["gap_on_track_end_sec", "lap_number"]].shift(1)
    consecutive = prev["lap_number"] == df["lap_number"] - 1
    df["gap_on_track_start_sec"] = prev["gap_on_track_end_sec"].where(consecutive)

    # In dirty air if the average of the start and end gaps is under the threshold.
    # With no gap information at all the flag is False (the gap columns show null).
    mean_gap = df[["gap_on_track_start_sec", "gap_on_track_end_sec"]].mean(axis=1, skipna=True)
    df["in_dirty_air"] = (mean_gap < DIRTY_AIR_THRESHOLD_SEC).fillna(False).astype(bool)
    return df


def _add_recent_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Pace history per driver, using this lap and earlier laps only."""
    by_driver = df.groupby("driver_code", sort=False)

    prev = by_driver[["lap_time_sec", "lap_number"]].shift(1)
    df["prev_lap_time_sec"] = prev["lap_time_sec"].where(prev["lap_number"] == df["lap_number"] - 1)

    clean_time = df["lap_time_sec"].where(df["is_clean_lap"])
    grouped_clean = clean_time.groupby(df["driver_code"], sort=False)

    # Most recent clean lap strictly BEFORE this one.
    df["last_clean_lap_time_sec"] = grouped_clean.transform(lambda s: s.shift(1).ffill())
    # Mean / spread of the clean laps among the last 3 laps (including this one).
    df["roll3_clean_mean_sec"] = grouped_clean.transform(
        lambda s: s.rolling(ROLLING_WINDOW_LAPS, min_periods=1).mean()).round(3)
    df["roll3_clean_std_sec"] = grouped_clean.transform(
        lambda s: s.rolling(ROLLING_WINDOW_LAPS, min_periods=2).std()).round(3)
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = the same driver's next lap time. Null on a driver's final lap."""
    by_driver = df.groupby("driver_code", sort=False)
    nxt = by_driver[["lap_time_sec", "lap_number"]].shift(-1)
    next_is_clean = by_driver["is_clean_lap"].shift(-1, fill_value=False)
    consecutive = nxt["lap_number"] == df["lap_number"] + 1
    df["target_next_lap_time_sec"] = nxt["lap_time_sec"].where(consecutive)
    df["target_next_is_clean"] = consecutive & next_is_clean
    return df


def build_lap_features(silver_laps: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Silver laps -> Gold lap features. Pure function: no file access."""
    if total_laps is None or total_laps < 1:
        raise ValueError(f"total_laps must be a positive integer, got {total_laps!r}")

    df = silver_laps.sort_values(["driver_code", "lap_number"]).reset_index(drop=True).copy()

    df["total_race_laps"] = int(total_laps)
    df["laps_remaining"] = int(total_laps) - df["lap_number"]
    df["race_progress"] = (df["lap_number"] / int(total_laps)).round(4)
    df["stint_lap"] = df.groupby(["driver_code", "stint"], dropna=False).cumcount() + 1

    df = _add_fuel(df, int(total_laps))
    df = _add_race_gaps(df)
    df = _add_track_gaps(df)
    df = _add_recent_pace(df)
    df = _add_target(df)

    # Model-facing names and types: plain floats, with NaN as the one missing marker.
    df = df.rename(columns={"team": "team_code", "tyre_life": "tyre_age", "stint": "stint_number",
                            "track_temp_c": "track_temp", "air_temp_c": "air_temp"})
    for col in ("position", "tyre_age", "stint_number"):
        df[col] = df[col].astype("float64")
    df["fresh_tyre"] = df["fresh_tyre"].astype("float64")
    df["rainfall"] = df["rainfall"].astype("float64")
    for col in ("driver_code", "team_code", "compound", "session"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)

    return df[GOLD_COLUMNS]


def build_gold(year: int, round_number: int, session: str = "R",
               lake_root: str = lake.DEFAULT_LAKE_ROOT) -> Dict[str, Any]:
    """Silver -> Gold for one session."""
    start = time.time()
    silver_path = lake.table_path(lake.SILVER, "laps", year, round_number, session, lake_root)
    silver_laps = lake.read_table(silver_path)
    meta = lake.read_metadata(silver_path)

    total_laps_meta = meta.get("total_laps", "")
    total_laps = int(total_laps_meta) if total_laps_meta.isdigit() else int(silver_laps["lap_number"].max())

    features = build_lap_features(silver_laps, total_laps)
    contracts.validate(features, contracts.GOLD_LAP_FEATURES, "Gold lap features")

    gold_path = lake.gold_features_path(year, round_number, session, lake_root)
    metadata = {
        **meta,
        "total_laps": total_laps,
        # keys the analytics engine reads
        "year": int(year),
        "round_number": int(round_number),
        "session_type": session,
    }
    lake.write_table(features, gold_path, metadata=metadata)

    return {
        "status": "success",
        "file_path": str(gold_path),
        "year": int(year),
        "round_number": int(round_number),
        "event_name": meta.get("event_name", ""),
        "feature_count": len(features.columns),
        "feature_rows": len(features),
        "file_size_kb": round(gold_path.stat().st_size / 1024, 1),
        "elapsed_sec": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    print(build_gold(2024, 12, "R"))
