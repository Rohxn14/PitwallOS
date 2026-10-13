"""
Data contracts (src/pipeline/contracts.py)

A contract is a written promise about a table: which columns exist, what
values are allowed, and which rules hold between columns. Silver and Gold are
checked against their contract BEFORE being written, so a bad table never
reaches disk and never reaches a model.

Checks are deliberately about things that must ALWAYS be true (a lap number is
positive, a key is unique), not things that are usually true.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

KNOWN_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN", "TEST_UNKNOWN",
                   "HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "TEST"]

# A representative racing lap on any current circuit sits well inside this range.
CLEAN_LAP_MIN_SEC, CLEAN_LAP_MAX_SEC = 50.0, 200.0

LAP_FLAGS = ["is_pit_lap", "is_in_lap", "is_out_lap", "is_lap1", "is_green", "is_yellow", "is_sc",
             "is_vsc", "is_red", "is_accurate", "is_deleted", "is_wet_tyre", "is_clean_lap",
             "is_tyre_change"]

# What a pit-lane passage was (src/pipeline/cleaner.py); written on in-laps only.
PIT_KINDS = ["tyre_change", "pass_through", "ended", "unknown"]


class ContractError(ValueError):
    """A table broke its contract. The message lists every failed rule."""


def _clean_laps_are_plausible(df: pd.DataFrame) -> bool:
    clean = df.loc[df["is_clean_lap"], "lap_time_sec"]
    return bool(clean.between(CLEAN_LAP_MIN_SEC, CLEAN_LAP_MAX_SEC).all())


def _clean_excludes_pit_and_lap1(df: pd.DataFrame) -> bool:
    return not bool((df["is_clean_lap"] & (df["is_pit_lap"] | df["is_lap1"] | ~df["is_green"])).any())


def _pit_flag_is_consistent(df: pd.DataFrame) -> bool:
    return bool((df["is_pit_lap"] == (df["is_in_lap"] | df["is_out_lap"])).all())


def _tyre_change_happens_on_an_in_lap(df: pd.DataFrame) -> bool:
    return not bool((df["is_tyre_change"] & ~df["is_in_lap"]).any())


def _pit_kind_only_on_in_laps(df: pd.DataFrame) -> bool:
    has_kind = df["pit_kind"].notna()
    agrees = (df["pit_kind"] == "tyre_change").fillna(False) == df["is_tyre_change"]
    return bool((has_kind == df["is_in_lap"]).all() and agrees.all())


def _pit_stop_times_only_on_in_laps(df: pd.DataFrame) -> bool:
    timed = df["pit_stop_time_sec"].notna() | df["pit_lane_time_sec"].notna()
    return not bool((timed & ~df["is_in_lap"]).any())


_SHARED_FRAME_CHECKS = [
    pa.Check(_clean_laps_are_plausible, error=f"clean lap time outside {CLEAN_LAP_MIN_SEC}-{CLEAN_LAP_MAX_SEC}s"),
    pa.Check(_clean_excludes_pit_and_lap1, error="a lap is flagged clean but is a pit lap, lap 1 or not green"),
    pa.Check(_pit_flag_is_consistent, error="is_pit_lap disagrees with is_in_lap / is_out_lap"),
    pa.Check(_tyre_change_happens_on_an_in_lap, error="a tyre change is flagged on a lap that is not an in-lap"),
]

_FLAG_COLUMNS = {name: pa.Column(bool) for name in LAP_FLAGS}

SILVER_LAPS = pa.DataFrameSchema(
    {
        "season": pa.Column(int, pa.Check.in_range(1950, 2100)),
        "round": pa.Column(int, pa.Check.in_range(1, 30)),
        "driver_code": pa.Column(nullable=False, checks=pa.Check.str_length(2, 4)),
        "driver_number": pa.Column(nullable=False),
        "lap_number": pa.Column(int, pa.Check.in_range(1, 120)),
        "position": pa.Column("Int64", pa.Check.in_range(1, 30), nullable=True),
        "stint": pa.Column("Int64", pa.Check.ge(1), nullable=True),
        "tyre_life": pa.Column("Int64", pa.Check.in_range(0, 120), nullable=True),
        "compound": pa.Column(nullable=True, checks=pa.Check.isin(KNOWN_COMPOUNDS)),
        "lap_time_sec": pa.Column(float, pa.Check.gt(0), nullable=True),
        "lap_start_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "lap_end_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "track_temp_c": pa.Column(float, pa.Check.in_range(-10, 80), nullable=True),
        "pit_kind": pa.Column(nullable=True, checks=pa.Check.isin(PIT_KINDS)),
        "pit_stop_time_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "pit_lane_time_sec": pa.Column(float, pa.Check.gt(0), nullable=True),
        **_FLAG_COLUMNS,
    },
    checks=[
        *_SHARED_FRAME_CHECKS,
        pa.Check(_pit_kind_only_on_in_laps,
                 error="pit_kind must be set on exactly the in-laps and agree with is_tyre_change"),
        pa.Check(_pit_stop_times_only_on_in_laps, error="F1's pit-stop times on a lap that is not an in-lap"),
        pa.Check(lambda df: bool((df["lap_end_sec"].isna() | df["lap_start_sec"].isna()
                                  | (df["lap_end_sec"] > df["lap_start_sec"])).all()),
                 error="a lap ends before it starts"),
    ],
    unique=["driver_code", "lap_number"],
    strict=False,   # extra columns are allowed; the listed ones are guaranteed
)

SILVER_TELEMETRY = pa.DataFrameSchema(
    {
        "driver_number": pa.Column(nullable=False),
        "session_time_sec": pa.Column(float, pa.Check.ge(0)),
        "lap_number": pa.Column(checks=pa.Check.ge(1)),
        "speed_kmh": pa.Column(checks=pa.Check.in_range(0, 400), nullable=True),
        "throttle_pct": pa.Column(checks=pa.Check.in_range(0, 100), nullable=True),
        "gear": pa.Column(checks=pa.Check.in_range(0, 8), nullable=True),
        "lap_distance_m": pa.Column(checks=pa.Check.ge(0)),
        "is_data_gap": pa.Column(bool),
    },
    unique=["driver_number", "session_time_sec"],
    strict=False,
)

GOLD_LAP_FEATURES = pa.DataFrameSchema(
    {
        "season": pa.Column(int),
        "round": pa.Column(int),
        "driver_code": pa.Column(nullable=False),
        "lap_number": pa.Column(int, pa.Check.ge(1)),
        "total_race_laps": pa.Column(int, pa.Check.ge(1)),
        "laps_remaining": pa.Column(int),
        "lap_time_sec": pa.Column(float, pa.Check.gt(0), nullable=True),
        "position": pa.Column(float, pa.Check.in_range(1, 30), nullable=True),
        "tyre_age": pa.Column(float, pa.Check.in_range(0, 120), nullable=True),
        "stint_number": pa.Column(float, pa.Check.ge(1), nullable=True),
        "stint_lap": pa.Column(int, pa.Check.ge(1)),
        "compound": pa.Column(nullable=True, checks=pa.Check.isin(KNOWN_COMPOUNDS)),
        "fuel_mass_est": pa.Column(float, pa.Check.in_range(0, 110)),
        "gap_ahead_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "gap_to_leader_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "gap_on_track_start_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "gap_on_track_end_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "in_dirty_air": pa.Column(bool),
        "roll3_clean_mean_sec": pa.Column(float, pa.Check.gt(0), nullable=True),
        "roll3_clean_std_sec": pa.Column(float, pa.Check.ge(0), nullable=True),
        "target_next_lap_time_sec": pa.Column(float, pa.Check.gt(0), nullable=True),
        "target_next_is_clean": pa.Column(bool),
        **_FLAG_COLUMNS,
    },
    checks=[
        *_SHARED_FRAME_CHECKS,
        pa.Check(lambda df: bool(df.loc[df["target_next_is_clean"], "target_next_lap_time_sec"].notna().all()),
                 error="target is flagged clean but has no value"),
    ],
    unique=["driver_code", "lap_number"],
    strict=False,
)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, table_name: str) -> pd.DataFrame:
    """
    Check a table against its contract. Returns the table unchanged if it passes.

    lazy=True collects EVERY failure before raising, so one run shows all problems.
    """
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases
        summary = (cases.groupby(["column", "check"], dropna=False).size()
                   .reset_index(name="failures").to_string(index=False))
        raise ContractError(f"{table_name} broke its data contract:\n{summary}") from err
    return df
