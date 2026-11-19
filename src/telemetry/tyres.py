"""
Tyres from telemetry (src/telemetry/tyres.py) - Phase 5 step 5.4

Two questions about a set of tyres, from step 5.2's corner tables:

1. WHERE does tyre wear cost time? Each corner's segment time gets slower with tyre age; with the fuel effect shared out
   in proportion to segment time, the corners' wear slopes add up EXACTLY to the lap's (a lap's segments add up to the
   lap). So "slow corners carry X % of the wear" is a sum, not a guess. (`corner_wear`)

2. Does the telemetry SEE a worn tyre before the lap time does? Rohan's idea, on record before step 5.4: "corner speeds
   warn of a tyre going off 3+ laps before the lap time does; fresh tyres show no early fade." Tested as he chose after
   the first measurement: in each stint, after how many laps is the wear trend clear beyond noise (`detection_lap`), in
   the slow corners' speeds (his choice of signal) and in the lap time? Whichever is clear first saw the tyre first.
   Both signals have their fuel effect estimated from the race by ONE method (src/analytics/degradation.fuel_check: tyre
   age resets at a stop, fuel does not), so neither is favoured by a correction the other lacks. "Fresh tyres": the
   slope over each stint's first EARLY_LAPS laps (`early_slope`).

   His first definition, a tyre "going off" = the lap time 0.5 s above the stint's trend 2 laps in a row (`going_off`),
   was measured first and kept for the record: it fires in 40 % of dry stints (2023-2024), a quarter of them on the
   same lap as 3+ other cars, and teams pit no sooner after it than after a random lap (`pit_reaction`). So it marks
   traffic, restarts and tyre management, not tyres; the detection test replaced it (his decision).

Which laps: step 2.3's (src/analytics/degradation.py): pace laps on slicks, dry track, consistent tyre record, in stints
of at least 6 such laps; races with any car on inters or wets are left out (the track changes under the tyre). The test
has no threshold to tune (2 standard errors, fixed before measuring); results are given for 2023-2024 and 2025-2026.

    python -m src.telemetry.tyres --season 2023 2024 2025 2026     # data/reports/tyre_wear_corners.csv, tyre_stints.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.degradation import (
    FUEL_EFFECT_SEC_PER_KG, USED, degradation_laps, fuel_check, mixed_conditions, within_stint_slope,
)
from src.analytics.laps import classify_laps
from src.pipeline import lake
from src.telemetry import corner_features as cf

SLOW_KMH = 150.0              # a slow corner (the race's median speed at it): grip and traction, as in step 5.3
MIN_TREND_LAPS = 5            # a stint's trend is judged from this many laps on
CLEAR_T = 2.0                 # "clear beyond noise": the slope at least this many standard errors from zero
EARLY_LAPS = 5                # "fresh tyres": a stint's first laps
JUMP_SEC, JUMP_LAPS = 0.5, 2  # Rohan's first definition of going off (kept for the record)
REACTION_LAPS = 3             # a team "reacts" to an event by stopping within this many laps
DEV_SEASONS, TEST_SEASONS = (2023, 2024), (2025, 2026)
TIME, SPEED = "time", "speed"


# --- one race: its stint laps with the slow corners' speed ------------------------------------------------------------

def slow_corners(features: pd.DataFrame, clean_keys: pd.DataFrame) -> List[int]:
    """The race's slow corners (corner_order): median speed at the corner, over clean complete laps, under SLOW_KMH."""
    rows = features[features["is_complete"]].merge(clean_keys, on=["driver_code", "lap_number"])
    speed = rows.groupby("corner_order")["corner_speed_kmh"].median()
    return sorted(int(k) for k in speed.index[speed < SLOW_KMH])


def slow_speed(features: pd.DataFrame, corners: Sequence[int]) -> pd.DataFrame:
    """Per car and lap, the mean over the slow corners of the speed at the corner and the exit speed (km/h). Only laps
    where every slow corner is complete: a mean over some of them would move with which ones are missing."""
    rows = features[features["corner_order"].isin(corners)]
    rows = rows.assign(speed=(rows["corner_speed_kmh"] + rows["exit_speed_kmh"]) / 2.0)
    per_lap = rows.groupby(["driver_code", "lap_number"]).agg(slow_speed_kmh=("speed", "mean"),
                                                              complete=("is_complete", "all"))
    return per_lap[per_lap["complete"]].drop(columns="complete").reset_index()


def race_stint_laps(year: int, round_number: int, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """(stint laps, corner segments, mixed conditions) for one race. Stint laps: step 2.3's used laps (stint, compound,
    age, fuel_kg, corrected_sec, deg_status) with the slow corners' speed. Corner segments: one row per used lap and
    corner, the segment time fuel-corrected in proportion to its share of the lap."""
    silver = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, "R", root))
    classified = classify_laps(silver)
    used = degradation_laps(classified)
    used = used[used["deg_status"] == USED].copy()
    features = lake.read_table(lake.table_path(lake.GOLD, cf.CORNER_TABLE, year, round_number, "R", root))
    clean = classified.loc[classified["is_clean_lap"].astype(bool), ["driver_code", "lap_number"]]
    corners = slow_corners(features, clean)
    laps = used.merge(slow_speed(features, corners), on=["driver_code", "lap_number"], how="left")
    laps = laps.assign(season=int(year), round=int(round_number), slow_corners=len(corners))

    segments = features[features["is_complete"]].merge(
        laps[["driver_code", "lap_number", "stint", "compound", "age", "fuel_kg", "lap_time_sec"]],
        on=["driver_code", "lap_number"])
    share = segments["segment_time_sec"] / segments["lap_time_sec"]
    segments = segments.assign(corrected_sec=segments["segment_time_sec"]
                               - FUEL_EFFECT_SEC_PER_KG * segments["fuel_kg"] * share)
    return laps, segments, mixed_conditions(classified)


# --- 1. where tyre wear costs time ------------------------------------------------------------------------------------

def corner_wear(segments: pd.DataFrame) -> pd.DataFrame:
    """Per compound and corner: the wear slope of the segment time (s per lap of tyre age, within each stint, fuel
    removed) and the corner's speed. Only laps with every corner complete, so the corners' slopes add up exactly to the
    lap's slope over the same laps (`lap_slope`)."""
    corners_per_lap = segments.groupby(["driver_code", "lap_number"])["corner_order"].transform("nunique")
    whole = segments[corners_per_lap == segments["corner_order"].nunique()]
    speed = whole.groupby("corner_order")["corner_speed_kmh"].median()
    rows = []
    for compound, by_compound in whole.groupby("compound"):
        lap_level = by_compound.groupby(["driver_code", "lap_number"]).agg(
            corrected_sec=("corrected_sec", "sum"), age=("age", "first"), stint=("stint", "first"))
        lap_slope, lap_se, stints = within_stint_slope(lap_level["age"], lap_level["corrected_sec"], lap_level["stint"])
        for order, g in by_compound.groupby("corner_order"):
            slope, se, _ = within_stint_slope(g["age"], g["corrected_sec"], g["stint"])
            rows.append({"compound": compound, "corner_order": int(order), "corner": str(g["corner"].iloc[0]),
                         "corner_speed_kmh": float(speed[order]), "slope": slope, "se": se,
                         "segment_time_sec": float(g["segment_time_sec"].median()),
                         "laps": int(len(g)), "stints": stints, "lap_slope": lap_slope, "lap_se": lap_se})
    return pd.DataFrame(rows)


def slow_share(wear: pd.DataFrame) -> pd.DataFrame:
    """Per compound: the lap's wear slope, the part of it made in slow corners, and the slow corners' share of the lap
    time (their segments' median times), for a race's corner_wear rows."""
    rows = []
    for compound, g in wear.groupby("compound"):
        slow = g["corner_speed_kmh"] < SLOW_KMH
        rows.append({"compound": compound, "lap_slope": float(g["lap_slope"].iloc[0]),
                     "lap_se": float(g["lap_se"].iloc[0]), "slow_slope": float(g.loc[slow, "slope"].sum()),
                     "slow_time_share": float(g.loc[slow, "segment_time_sec"].sum() / g["segment_time_sec"].sum()),
                     "stints": int(g["stints"].iloc[0])})
    return pd.DataFrame(rows)


# --- 2. which signal sees the wear first ------------------------------------------------------------------------------

def fuel_corrected(laps: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """The stint laps with time_fc and speed_fc: the lap time and the slow corners' speed with their own fuel effect,
    estimated from the race (degradation.fuel_check, one method for both), taken out ("as if the tank were empty").
    Returns NaN columns for a signal whose fuel effect the race cannot identify."""
    coefficients = {}
    out = laps.copy()
    for name, column in ((TIME, "lap_time_sec"), (SPEED, "slow_speed_kmh")):
        coefficient = fuel_check(laps, column)["fuel_sec_per_kg"]
        coefficients[name] = coefficient
        out[f"{name}_fc"] = laps[column] - coefficient * laps["fuel_kg"]
    return out, coefficients


def running_t(age: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For k = 1 .. n: the t-statistic of the least-squares slope of y on age over the stint's first k laps (NaN under
    MIN_TREND_LAPS laps or where the ages do not vary). Only the laps so far: a pit wall could compute it live."""
    out = np.full(len(y), np.nan)
    for k in range(MIN_TREND_LAPS, len(y) + 1):
        x, v = age[:k], y[:k]
        ok = np.isfinite(x) & np.isfinite(v)
        n = int(ok.sum())
        if n < MIN_TREND_LAPS:
            continue
        x, v = x[ok], v[ok]
        xd = x - x.mean()
        sxx = float((xd ** 2).sum())
        if sxx == 0.0:
            continue
        slope = float((xd * (v - v.mean())).sum() / sxx)
        resid = (v - v.mean()) - slope * xd
        se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
        out[k - 1] = slope / se if se > 0 else np.sign(slope) * np.inf
    return out


def detection_lap(age: np.ndarray, y: np.ndarray, worse: int, t: float = CLEAR_T) -> Tuple[Optional[int], bool]:
    """(how many laps into the stint the wear trend is clear, or None; whether a trend the WRONG way, the tyre getting
    better, was ever clear first). `worse`: +1 when larger is worse (lap time), -1 when smaller is (speed)."""
    stat = running_t(age, y) * worse
    clear = np.flatnonzero(np.nan_to_num(stat, nan=0.0) > t)
    wrong = np.flatnonzero(np.nan_to_num(stat, nan=0.0) < -t)
    first = int(clear[0]) + 1 if len(clear) else None
    wrong_first = bool(len(wrong) and (first is None or wrong[0] + 1 < first))
    return first, wrong_first


def early_slope(age: np.ndarray, y: np.ndarray, laps: int = EARLY_LAPS) -> float:
    """The least-squares slope of y on age over the stint's first `laps` laps (NaN with fewer than 4 of them)."""
    x, v = age[:laps], y[:laps]
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], v[ok], 1)[0])


def stint_detection(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint (at least MIN_TREND_LAPS laps): when each signal saw the wear, whether it first saw the wrong
    way, and the early slopes. Expects fuel_corrected laps."""
    rows = []
    last_stint = laps.groupby("driver_code")["lap_number"].transform("max")
    for stint, g in laps.sort_values("lap_number").groupby("stint", sort=False):
        if len(g) < MIN_TREND_LAPS:
            continue
        age = g["age"].to_numpy(dtype=float)
        row = {"season": int(g["season"].iloc[0]), "round": int(g["round"].iloc[0]), "stint": stint,
               "driver_code": g["driver_code"].iloc[0], "compound": g["compound"].iloc[0], "laps": int(len(g)),
               "first_lap": int(g["lap_number"].min()), "last_lap": int(g["lap_number"].max()),
               "final": bool(g["lap_number"].max() == last_stint.loc[g.index].max()),
               "event_lap": going_off(g)}
        for name, worse in ((TIME, +1), (SPEED, -1)):
            y = g[f"{name}_fc"].to_numpy(dtype=float)
            first, wrong = detection_lap(age, y, worse)
            row[f"{name}_clear_lap"] = first
            row[f"{name}_wrong_first"] = wrong
            row[f"{name}_early_slope"] = early_slope(age, y)
            row[f"{name}_slope"] = early_slope(age, y, laps=len(y))      # the whole stint, for comparison
            row[f"{name}_laps"] = int(np.isfinite(y).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def who_first(stints: pd.DataFrame) -> Dict[str, Any]:
    """Over stints where both signals were measured on every lap: which saw the wear first, and by how many laps."""
    both = stints[(stints[f"{TIME}_laps"] == stints["laps"]) & (stints[f"{SPEED}_laps"] == stints["laps"])]
    t, s = both[f"{TIME}_clear_lap"], both[f"{SPEED}_clear_lap"]
    seen = t.notna() & s.notna()
    lead = (t - s)[seen]                        # > 0: the speed saw it first, by this many laps
    return {"stints": int(len(both)), "time_saw": float(t.notna().mean()), "speed_saw": float(s.notna().mean()),
            "both_saw": int(seen.sum()), "speed_first": float((lead > 0).mean()) if seen.any() else np.nan,
            "same_lap": float((lead == 0).mean()) if seen.any() else np.nan,
            "time_first": float((lead < 0).mean()) if seen.any() else np.nan,
            "speed_3_earlier": float((lead >= 3).mean()) if seen.any() else np.nan,
            "median_lead": float(lead.median()) if seen.any() else np.nan,
            "only_speed": int((s.notna() & t.isna()).sum()), "only_time": int((t.notna() & s.isna()).sum()),
            "time_wrong_first": float(both[f"{TIME}_wrong_first"].mean()),
            "speed_wrong_first": float(both[f"{SPEED}_wrong_first"].mean())}


# --- Rohan's first definition, kept for the record ---------------------------------------------------------------------

def going_off(stint: pd.DataFrame, jump_sec: float = JUMP_SEC, run: int = JUMP_LAPS) -> Optional[int]:
    """Rohan's first definition: the first lap of `run` laps in a row with the lap time (fuel removed) `jump_sec`
    above the stint's straight-line trend fitted on the laps BEFORE the run (at least MIN_TREND_LAPS of them); None if
    it never happens. Every lap of the run is judged against the same trend: refitting it after the first jumped lap
    would pull the trend up and hide the second (a test caught exactly that)."""
    age = stint["age"].to_numpy(dtype=float)
    y = stint["corrected_sec"].to_numpy(dtype=float)
    for k in range(MIN_TREND_LAPS, len(y) - run + 1):
        x, v = age[:k], y[:k]
        ok = np.isfinite(x) & np.isfinite(v)
        if ok.sum() < MIN_TREND_LAPS or np.ptp(x[ok]) == 0:
            continue
        slope, intercept = np.polyfit(x[ok], v[ok], 1)
        above = y[k:k + run] - (intercept + slope * age[k:k + run])
        if np.all(np.isfinite(above)) and np.all(above > jump_sec):
            return int(stint["lap_number"].iloc[k])
    return None


def pit_reaction(stints: pd.DataFrame, laps: int = REACTION_LAPS) -> Dict[str, Any]:
    """Do teams react to the events? In stints that ended with a stop: the share of events followed by the stop within
    `laps` laps, against chance (the same for every lap where an event could have started)."""
    stopped = stints[~stints["final"]]
    events = stopped[stopped["event_lap"].notna()]
    soon = float((events["last_lap"] - events["event_lap"] <= laps).mean()) if len(events) else np.nan
    chance = []
    for r in stopped.itertuples():
        candidates = np.arange(r.first_lap + MIN_TREND_LAPS, r.last_lap - JUMP_LAPS + 2)
        if len(candidates):
            chance.append(float(np.mean(r.last_lap - candidates <= laps)))
    return {"stints": int(len(stints)), "went_off": float(stints["event_lap"].notna().mean()),
            "events_in_stopped_stints": int(len(events)), "stop_soon": soon,
            "chance": float(np.mean(chance)) if chance else np.nan}


# --- every race ----------------------------------------------------------------------------------------------------

def season(seasons: Sequence[int], root: lake.PathLike = lake.DEFAULT_LAKE_ROOT
           ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """(per race and compound: where the wear is; per stint: who saw it first; races left out with why)."""
    wear, stints, skipped = [], [], []
    for year, round_number in lake.list_races(lake.GOLD, cf.CORNER_TABLE, root):
        if year not in seasons:
            continue
        try:
            laps, segments, mixed = race_stint_laps(year, round_number, root)
        except Exception as err:
            skipped.append(f"{year} round {round_number}: {type(err).__name__}: {str(err)[:120]}")
            continue
        if mixed:
            skipped.append(f"{year} round {round_number}: mixed conditions (a car on inters or wets)")
            continue
        share = slow_share(corner_wear(segments))
        if len(share):
            wear.append(share.assign(season=year, round=round_number))
        laps, coefficients = fuel_corrected(laps)
        if not all(np.isfinite(list(coefficients.values()))):
            skipped.append(f"{year} round {round_number}: the race cannot separate fuel from tyre age")
            continue
        s = stint_detection(laps)
        if len(s):
            stints.append(s.assign(fuel_time=coefficients[TIME], fuel_speed=coefficients[SPEED]))
    return (pd.concat(wear, ignore_index=True) if wear else pd.DataFrame(),
            pd.concat(stints, ignore_index=True) if stints else pd.DataFrame(), skipped)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Where tyre wear costs time, and which signal sees it first")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    wear, stints, skipped = season(args.season, args.lake)
    for line in skipped:
        print("left out:", line)
    if stints.empty:
        print("No dry race with corner tables in these seasons.")
        return 1
    lake.write_report(wear, "tyre_wear_corners", args.lake)
    path = lake.write_report(stints, "tyre_stints", args.lake)
    for part, years in (("dev", DEV_SEASONS), ("test", TEST_SEASONS)):
        w, s = wear[wear["season"].isin(years)], stints[stints["season"].isin(years)]
        if s.empty:
            continue
        clear = w[w["lap_slope"] > 2 * w["lap_se"]]
        print(f"--- {part} {years}: {s[['season', 'round']].drop_duplicates().shape[0]} dry races, {len(s)} stints")
        print(f"where the wear is ({len(clear)} race-compounds with clear wear): slow corners carry "
              f"{100 * (clear['slow_slope'] / clear['lap_slope']).median():.0f} % of it in "
              f"{100 * clear['slow_time_share'].median():.0f} % of the lap time")
        r = who_first(s)
        print(f"who saw the wear first ({r['stints']} stints): lap time saw it in {100 * r['time_saw']:.0f} %, slow-corner "
              f"speed in {100 * r['speed_saw']:.0f} %; of the {r['both_saw']} both saw: speed first {100 * r['speed_first']:.0f} %, "
              f"same lap {100 * r['same_lap']:.0f} %, time first {100 * r['time_first']:.0f} %; speed 3+ laps earlier "
              f"{100 * r['speed_3_earlier']:.0f} %; median lead {r['median_lead']:+.0f} laps (+ = speed first)")
        print(f"a trend the wrong way seen first: time {100 * r['time_wrong_first']:.0f} %, speed {100 * r['speed_wrong_first']:.0f} %")
        print(f"fresh tyres, first {EARLY_LAPS} laps: lap time slope median {s[f'{TIME}_early_slope'].median():+.3f} s/lap, "
              f"slow-corner speed slope median {s[f'{SPEED}_early_slope'].median():+.3f} km/h per lap")
        reaction = pit_reaction(s)
        print(f"going off (0.5 s, 2 laps): {100 * reaction['went_off']:.0f} % of stints; stop within {REACTION_LAPS} laps after "
              f"{100 * reaction['stop_soon']:.0f} % of events, chance {100 * reaction['chance']:.0f} %")
    print(f"Reports: {path} and tyre_wear_corners.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
