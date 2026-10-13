"""
Fuel-corrected tyre degradation (src/analytics/degradation.py) — Phase 2 step 3

How much slower a tyre gets per lap of age, with the fuel effect taken out.

Why fuel has to come out: the car burns ~1.8 kg of fuel per lap and every kg costs time, so a car
gets faster lap after lap while its tyres wear. Measured raw, the two cancel: over 2023-2025 the
raw medium slope is ~0.000 s per lap of tyre age, as if tyres never wore.

    fuel_kg     = FUEL_START_KG - (lap - 1) * burn,  burn = (FUEL_START_KG - FUEL_END_KG) / race laps
                  (the Gold feature store's fuel model and constants, one definition)
    corrected   = lap time - FUEL_EFFECT_SEC_PER_KG * fuel_kg   (the lap as if the tank were empty)

Three estimates, simplest first (each reports n and a standard error):

  stint slopes     ordinary least squares of corrected time on tyre age, one stint at a time.
                   Few laps each, so noisy; the SE says how noisy (and is optimistic: laps in a
                   stint are not independent).
  compound slopes  one slope per compound for the race, from every stint on it. Each stint keeps
                   its own level (driver, car, fuel at the start: "stint fixed effects"), and only
                   the change WITHIN each stint counts. SE clustered by stint (honest about laps of
                   one stint moving together).
  fuel check       the fuel constant estimated from the data instead of assumed: driver levels,
                   compound offsets, a slope per compound, and one fuel coefficient, all fitted
                   together on raw times. Possible only because tyre age resets at a stop and fuel
                   does not. It also absorbs track evolution (the track gets faster as fuel goes
                   down), so it is "fuel + track" per kg. 2023-2025 dry races: median ~0.031 s/kg,
                   against the assumed 0.03.

Which laps: step 1's pace laps, on slick tyres (on inters and wets the track changes under the
tyre, so a slope is not wear), in stints on a dry track (no rain reported and no car on inters or
wets during the stint's laps: a drying track makes slick laps faster), from drivers whose tyre
record is consistent (2025 Miami), with a tyre age, in stints of at least MIN_STINT_LAPS such laps.
Every exclusion is counted (degradation_ledger). Restart laps stay (Rohan's step 1 decision);
measured: they move the pooled medium slope by less than 0.001 s/lap.

A race in which any car ran inters or wets is MIXED CONDITIONS: the track keeps drying after the
last wet tyre comes off (2024 Canada and 2025 British slick stints still "improve" by ~0.2 s a
lap). Its slopes are reported but marked, and season-wide figures use dry races only.

A stint is a tyre run of step 1 (laps between two pit passages) on one compound.

Run it over every race in the lake:

    python -m src.analytics.degradation --season 2023 2024 2025
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analytics.laps import classify_laps, compound_rank
from src.pipeline import lake
from src.pipeline.feature_store import FUEL_END_KG, FUEL_START_KG

FUEL_EFFECT_SEC_PER_KG = 0.03   # named assumption: 10 kg of fuel costs ~0.3 s a lap (checked: data ~0.031)
MIN_STINT_LAPS = 6              # a stint needs this many usable laps to give a slope
SLICKS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

USED = "used"
EXCLUDED_WET_TYRE = "wet tyre (the track changes under it)"
EXCLUDED_NO_COMPOUND = "no compound"
EXCLUDED_WET_TRACK = "wet or drying track"
EXCLUDED_TYRE_RECORD = "tyre record inconsistent"
EXCLUDED_NO_AGE = "no tyre age"
EXCLUDED_SHORT_STINT = f"stint shorter than {MIN_STINT_LAPS} laps"
LEDGER_ORDER = (EXCLUDED_WET_TYRE, EXCLUDED_NO_COMPOUND, EXCLUDED_WET_TRACK, EXCLUDED_TYRE_RECORD, EXCLUDED_NO_AGE,
                EXCLUDED_SHORT_STINT)


# --- which laps, and their fuel -----------------------------------------------------------------

def fuel_on_board(lap_number: pd.Series, race_laps: int) -> pd.Series:
    """Estimated fuel at the start of each lap (kg): the Gold feature store's model."""
    burn = (FUEL_START_KG - FUEL_END_KG) / race_laps
    return (FUEL_START_KG - (lap_number - 1) * burn).clip(lower=0.0)


def _on_wet_tyres(laps: pd.DataFrame) -> pd.Series:
    return laps["compound"].isin(WET_COMPOUNDS).fillna(False).astype(bool)


def mixed_conditions(classified: pd.DataFrame) -> bool:
    """True if any car ran inters or wets at any point of the race."""
    return bool(_on_wet_tyres(classified).any())


def degradation_laps(classified: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_SEC_PER_KG) -> pd.DataFrame:
    """
    The race's pace laps with: stint (driver|tyre run|compound), age, fuel_kg, corrected_sec, and
    deg_status: 'used' or the first reason in LEDGER_ORDER it is left out. Pace laps only.
    """
    race_laps = int(classified["lap_number"].max())
    p = classified[classified["is_pace_lap"]].copy()
    p["stint"] = (p["driver_code"].astype(str) + "|" + p["tyre_run"].astype(str) + "|"
                  + p["compound"].astype("string").fillna("-").astype(str))
    p["age"] = p["tyre_life"].astype("float64")
    p["fuel_kg"] = fuel_on_board(p["lap_number"], race_laps)
    p["corrected_sec"] = p["lap_time_sec"] - fuel_effect * p["fuel_kg"]

    # A slick stint is on a wet or drying track if the weather station reports rain during it, or if any
    # car is still on inters or wets during its laps (the station stops reporting rain long before the
    # track is dry).
    rain = p["rainfall"].fillna(False).astype(bool) if "rainfall" in p.columns else pd.Series(False, index=p.index)
    wet_tyre_laps = set(classified.loc[_on_wet_tyres(classified), "lap_number"].astype(int))
    inters_in_use = p["lap_number"].astype(int).isin(wet_tyre_laps)
    wet_track = (rain | inters_in_use).groupby(p["stint"]).transform("any").astype(bool)
    inconsistent = (p["is_tyre_inconsistent"].astype(bool) if "is_tyre_inconsistent" in p.columns
                    else pd.Series(False, index=p.index))

    status = pd.Series(USED, index=p.index, dtype="object")
    checks = [
        (_on_wet_tyres(p), EXCLUDED_WET_TYRE),
        (~p["compound"].isin(SLICKS).fillna(False).astype(bool), EXCLUDED_NO_COMPOUND),
        (wet_track, EXCLUDED_WET_TRACK),
        (inconsistent, EXCLUDED_TYRE_RECORD),
        (p["age"].isna(), EXCLUDED_NO_AGE),
    ]
    for mask, reason in checks:
        status[(status == USED) & mask.to_numpy()] = reason
    usable = status == USED
    stint_size = usable.groupby(p["stint"]).transform("sum")
    status[usable & (stint_size < MIN_STINT_LAPS)] = EXCLUDED_SHORT_STINT
    p["deg_status"] = status
    return p


def degradation_ledger(deg_laps: pd.DataFrame) -> pd.DataFrame:
    """Pace laps -> laps used for slopes, one row per reason (listed even at 0)."""
    counts = deg_laps["deg_status"].value_counts()
    rows = [{"reason": "pace laps", "laps": int(len(deg_laps))}]
    rows += [{"reason": r, "laps": int(counts.get(r, 0))} for r in LEDGER_ORDER]
    rows.append({"reason": "used for slopes", "laps": int(counts.get(USED, 0))})
    return pd.DataFrame(rows, columns=["reason", "laps"])


# --- the three estimates ------------------------------------------------------------------------

def _ols_slope(x: np.ndarray, y: np.ndarray):
    """Slope and its ordinary standard error (NaN when it cannot be computed)."""
    n = len(x)
    xd = x - x.mean()
    sxx = float((xd ** 2).sum())
    if n < 3 or sxx == 0.0:
        return np.nan, np.nan
    b = float((xd * (y - y.mean())).sum() / sxx)
    resid = (y - y.mean()) - b * xd
    return b, float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))


def stint_slopes(deg_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint used: its raw and fuel-corrected slope (s per lap of tyre age) with SE."""
    used = deg_laps[deg_laps["deg_status"] == USED]
    rows = []
    for _, g in used.groupby("stint", sort=False):
        g = g.sort_values("lap_number")
        x = g["age"].to_numpy(dtype=float)
        raw, _ = _ols_slope(x, g["lap_time_sec"].to_numpy(dtype=float))
        slope, se = _ols_slope(x, g["corrected_sec"].to_numpy(dtype=float))
        rows.append({"driver_code": g["driver_code"].iloc[0], "compound": g["compound"].iloc[0],
                     "tyre_run": int(g["tyre_run"].iloc[0]), "first_lap": int(g["lap_number"].min()),
                     "last_lap": int(g["lap_number"].max()), "laps": int(len(g)),
                     "age_first": int(x.min()), "age_last": int(x.max()),
                     "raw_slope": raw, "slope": slope, "se": se})
    out = pd.DataFrame(rows, columns=["driver_code", "compound", "tyre_run", "first_lap", "last_lap", "laps",
                                      "age_first", "age_last", "raw_slope", "slope", "se"])
    out["_rank"] = out["compound"].map(compound_rank)
    return out.sort_values(["_rank", "driver_code", "first_lap"]).drop(columns="_rank").reset_index(drop=True)


def within_stint_slope(age: pd.Series, y: pd.Series, stint: pd.Series):
    """
    Slope of y on age using only changes within each stint (stint fixed effects), with a standard
    error clustered by stint. Returns (slope, se, stints).
    """
    x = age - age.groupby(stint).transform("mean")
    t = y - y.groupby(stint).transform("mean")
    sxx = float((x * x).sum())
    stints = int(stint.nunique())
    if sxx == 0.0 or stints < 2:
        return np.nan, np.nan, stints
    b = float((x * t).sum() / sxx)
    score = (x * (t - b * x)).groupby(stint).sum()
    se = float(np.sqrt(stints / (stints - 1) * (score ** 2).sum()) / sxx)
    return b, se, stints


def compound_slopes(deg_laps: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_SEC_PER_KG) -> pd.DataFrame:
    """
    One row per slick compound with used stints: stints, laps, raw_slope, slope (fuel-corrected),
    se, and fuel_per_lap = how much the correction adds to every slope here (fuel effect x burn).
    """
    used = deg_laps[deg_laps["deg_status"] == USED]
    race_laps = int(deg_laps["lap_number"].max()) if len(deg_laps) else 0
    fuel_per_lap = fuel_effect * (FUEL_START_KG - FUEL_END_KG) / race_laps if race_laps else np.nan
    rows = []
    for compound in sorted(used["compound"].dropna().unique(), key=compound_rank):
        g = used[used["compound"] == compound]
        raw, _, stints = within_stint_slope(g["age"], g["lap_time_sec"], g["stint"])
        slope, se, _ = within_stint_slope(g["age"], g["corrected_sec"], g["stint"])
        rows.append({"compound": compound, "stints": stints, "laps": int(len(g)), "raw_slope": raw,
                     "slope": slope, "se": se, "fuel_per_lap": fuel_per_lap})
    return pd.DataFrame(rows, columns=["compound", "stints", "laps", "raw_slope", "slope", "se", "fuel_per_lap"])


def fuel_check(deg_laps: pd.DataFrame) -> Dict[str, Any]:
    """
    The fuel effect estimated from the race itself (s per kg, "fuel + track evolution"), with an SE
    clustered by driver: raw lap time = driver level + compound offset + slope per compound x age
    + coefficient x fuel. NaN if the race cannot identify it (too few drivers, nobody changed tyres).
    """
    used = deg_laps[deg_laps["deg_status"] == USED]
    drivers = int(used["driver_code"].nunique())
    result = {"fuel_sec_per_kg": np.nan, "fuel_se": np.nan, "drivers": drivers, "laps": int(len(used))}
    if drivers < 3 or used["stint"].nunique() <= drivers:
        return result                           # nobody has a second stint: age and fuel move together
    compounds = sorted(used["compound"].unique(), key=compound_rank)
    parts = [pd.get_dummies(used["driver_code"].astype(str), dtype=float)]
    for c in compounds[1:]:
        parts.append((used["compound"] == c).astype(float).rename(f"is_{c}"))
    for c in compounds:
        parts.append(used["age"].where(used["compound"] == c, 0.0).rename(f"age_{c}"))
    parts.append(used["fuel_kg"].astype(float).rename("fuel"))
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)
    y = used["lap_time_sec"].to_numpy(dtype=float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return result
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    groups = used["driver_code"].astype(str).to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for code in np.unique(groups):
        score = X[groups == code].T @ resid[groups == code]
        meat += np.outer(score, score)
    g = len(np.unique(groups))
    cov = g / (g - 1) * xtx_inv @ meat @ xtx_inv
    result.update(fuel_sec_per_kg=float(coef[-1]), fuel_se=float(np.sqrt(cov[-1, -1])))
    return result


# --- one race, every race -----------------------------------------------------------------------

@dataclass
class RaceDegradation:
    mixed: bool                 # mixed_conditions: slopes reported, but not for season-wide figures
    laps: pd.DataFrame          # degradation_laps (pace laps with deg_status, fuel, corrected time)
    ledger: pd.DataFrame        # degradation_ledger
    stints: pd.DataFrame        # stint_slopes
    compounds: pd.DataFrame     # compound_slopes
    fuel: Dict[str, Any]        # fuel_check


def race_degradation(classified: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_SEC_PER_KG) -> RaceDegradation:
    """Everything step 3 says about one race, from its classified laps (laps.classify_laps). Pure."""
    d = degradation_laps(classified, fuel_effect)
    return RaceDegradation(mixed=mixed_conditions(classified), laps=d, ledger=degradation_ledger(d),
                           stints=stint_slopes(d), compounds=compound_slopes(d, fuel_effect), fuel=fuel_check(d))


def load_race_degradation(year: int, round_number: int, session: str = "R",
                          root: str = lake.DEFAULT_LAKE_ROOT) -> RaceDegradation:
    path = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    if not path.exists():
        raise FileNotFoundError(f"{year} round {round_number}: no Silver laps in the lake ({path}).")
    return race_degradation(classify_laps(lake.read_table(path)))


def season_summary(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """One row per race and slick compound: slopes with SE, whether the race was mixed, and its fuel check."""
    wanted = {int(s) for s in seasons}
    rows: List[Dict[str, Any]] = []
    for path in lake.list_sessions(lake.SILVER, "laps", root):
        if "session=R" not in path.parts:
            continue
        laps = lake.read_table(path)
        if laps.empty or int(laps["season"].iloc[0]) not in wanted:
            continue
        season, round_number = int(laps["season"].iloc[0]), int(laps["round"].iloc[0])
        result = race_degradation(classify_laps(laps))
        try:
            event = lake.read_manifest(season, round_number, "R", root).get("event_name")
        except FileNotFoundError:
            event = None
        for r in result.compounds.to_dict("records"):
            rows.append({"season": season, "round": round_number, "event_name": event, "mixed": result.mixed, **r,
                         "fuel_check_sec_per_kg": result.fuel["fuel_sec_per_kg"], "fuel_check_se": result.fuel["fuel_se"]})
    return pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True) if rows else pd.DataFrame()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 step 3 over every race in the lake: fuel-corrected degradation")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2023 2024")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    table = season_summary(args.season, args.lake)
    if table.empty:
        print("No Silver laps in the lake for these seasons.")
        return 1
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    races = lambda t: t[["season", "round"]].drop_duplicates().shape[0]
    dry = table[~table["mixed"]]
    print(f"\n{races(dry)} dry races ({races(table[table['mixed']])} mixed-conditions races left out of these figures). "
          f"Median over races of the fuel-corrected slope (s per lap of tyre age), stints of {MIN_STINT_LAPS}+ laps:")
    enough = dry[dry["stints"] >= 3]
    for compound in SLICKS:
        c = enough[enough["compound"] == compound]
        if len(c):
            print(f"  {compound:6s} {c['slope'].median():+.3f}  (raw {c['raw_slope'].median():+.3f}; {len(c)} races "
                  f"with 3+ stints; p10 {c['slope'].quantile(0.1):+.3f}, p90 {c['slope'].quantile(0.9):+.3f})")
    fuel = dry.drop_duplicates(["season", "round"])["fuel_check_sec_per_kg"].dropna()
    if len(fuel):
        print(f"Fuel check: median {fuel.median():.4f} s/kg over {len(fuel)} races "
              f"(IQR {fuel.quantile(0.25):.4f}-{fuel.quantile(0.75):.4f}); assumed {FUEL_EFFECT_SEC_PER_KG} s/kg")
    print(f"Report: {lake.write_report(table, 'degradation', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
