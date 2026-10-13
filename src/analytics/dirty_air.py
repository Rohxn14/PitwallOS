"""
Dirty air (src/analytics/dirty_air.py) — Phase 2 step 4

How much slower a car laps when it starts the lap close behind another car on the road: the cost of
following. It is the traffic term of the lap-time model; Phase 3 learns it, the Phase 4 simulator
applies it, and step 5 (undercuts) needs it when a car rejoins behind another.

What it measures, and what it cannot: lap times cannot tell losing downforce in the wake of the car
ahead from being held up by it (stuck behind a slower car, a car does that car's lap time), nor from
DRS (within 1 s it helps on the straights). The number is the net cost of following, all three
together. In 2023-2025 it is highest at Monaco 2025 (+1.37 s/lap): tactics, not aerodynamics, in a
race with a two-stop rule where teams backed the field up on purpose.

Which laps: step 3's (pace laps on slick tyres, on a dry track, in stints of MIN_STINT_LAPS or more
such laps, fuel-corrected with the same constant), minus
  - laps whose start gap is unknown (nobody crossed the line before this car), and
  - laps that start within CLEAR_AIR_SEC of a car on ANOTHER lap: a backmarker about to be lapped
    (Rohan's note: it moves aside under blue flags), or a car that has just lapped this one (this
    car is the one under blue flags). Neither is racing the car behind.
Every exclusion is counted (dirty_air_ledger), step 3's included.

The gap is the one at the START of the lap: the gap to the car ahead on the road when this car began
the lap (Gold's gap_on_track_start_sec; one definition: feature_store.car_ahead_on_track). It is
fixed before the lap is driven, which is what a simulator knows; the end gap is partly a result of
the lap itself (measured: the two give the same cost to 0.01 s/lap over 2023-2025).

The model, one race (stint fixed effects, as in step 3):

    corrected lap time = stint level + wear slope per compound x tyre age + cost of following + noise

  following   cost = beta x (start gap < FOLLOW_THRESHOLD_SEC), against every other lap. 1.5 s is
              Gold's dirty-air convention. Laps 1.5-3 s behind still lose ~0.1-0.2 s, so beta
              understates the cost against truly clear air.
  gap bins    one cost per bin of start gap (GAP_EDGES) against laps CLEAR_AIR_SEC or more behind
              the car ahead: the shape of the effect, and what the simulator should use.
  wear check  step 3's wear slopes on the same laps, fitted again with the cost of following in the
              model: does traffic bias them?

Only changes within a stint count (same driver, car, tyres), so a slow car that spends the race in
traffic cannot fake the effect. Standard errors: clustered by stint within a race (laps of one stint
move together; measured: clustering by stint and lap together, for trains, is ~10 % wider); across
races, clustered by race (a race nests its stints and its laps).

Run it over every race in the lake:

    python -m src.analytics.dirty_air --season 2023 2024 2025
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.degradation import (
    FUEL_EFFECT_SEC_PER_KG, LEDGER_ORDER as STEP3_LEDGER_ORDER, SLICKS, USED as STEP3_USED, degradation_laps,
    mixed_conditions,
)
from src.analytics.laps import classify_laps, compound_rank
from src.pipeline import lake
from src.pipeline.feature_store import DIRTY_AIR_THRESHOLD_SEC, car_ahead_on_track

FOLLOW_THRESHOLD_SEC = DIRTY_AIR_THRESHOLD_SEC   # 1.5 s: Gold's convention, the line for beta
CLEAR_AIR_SEC = 3.0                              # the gap bins' reference: this far behind or more
GAP_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, CLEAR_AIR_SEC)
MIN_LAPS = 10                                    # a cost needs this many laps in its group (and in its reference)

USED = "used"
EXCLUDED_NO_GAP = "no gap at the lap's start"
EXCLUDED_BACKMARKER = f"behind a backmarker (within {CLEAR_AIR_SEC:g} s)"
EXCLUDED_BEING_LAPPED = f"being lapped (within {CLEAR_AIR_SEC:g} s)"
AIR_LEDGER_ORDER = (EXCLUDED_NO_GAP, EXCLUDED_BACKMARKER, EXCLUDED_BEING_LAPPED)


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} s"


BIN_LABELS = tuple(_bin_label(lo, hi) for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]))
CLEAR_LABEL = f"{CLEAR_AIR_SEC:g} s +"


# --- which laps, and the gap at their start ------------------------------------------------------

def start_of_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """
    For every lap, the car ahead on the road when the lap STARTED (= when this car ended its lap
    before), on the same index: gap_start_sec, car_ahead, and car_ahead_laps_up (0: the same lap;
    negative: that car is a lap or more down, a backmarker; positive: it is a lap or more up).
    Null when the lap before is missing, ended in the pit lane, or nobody had crossed before it.
    """
    end = car_ahead_on_track(laps)
    end["driver_code"] = laps["driver_code"]
    end["lap_number"] = laps["lap_number"].astype("float64")
    end = end.sort_values(["driver_code", "lap_number"], kind="stable")
    prev = end.groupby("driver_code", sort=False)[["gap_sec", "car_ahead", "car_ahead_lap", "lap_number"]].shift(1)
    consecutive = prev["lap_number"] == end["lap_number"] - 1
    out = pd.DataFrame(index=end.index)
    out["gap_start_sec"] = prev["gap_sec"].where(consecutive)
    out["car_ahead"] = prev["car_ahead"].where(consecutive)
    out["car_ahead_laps_up"] = (prev["car_ahead_lap"] - prev["lap_number"]).where(consecutive)
    return out.reindex(laps.index)


def gap_bin(gap: pd.Series) -> pd.Series:
    """The start gap's bin label (BIN_LABELS, or CLEAR_LABEL at CLEAR_AIR_SEC and beyond); null if unknown."""
    labels = pd.Series(pd.NA, index=gap.index, dtype="object")
    for lo, hi, label in zip(GAP_EDGES[:-1], GAP_EDGES[1:], BIN_LABELS):
        labels[(gap >= lo) & (gap < hi)] = label
    labels[gap >= CLEAR_AIR_SEC] = CLEAR_LABEL
    return labels


def dirty_air_laps(classified: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_SEC_PER_KG) -> pd.DataFrame:
    """
    The race's pace laps (step 3's degradation_laps: stint, age, fuel_kg, corrected_sec, deg_status)
    with the start of each lap (start_of_lap), following (start gap < FOLLOW_THRESHOLD_SEC; null if the
    gap is unknown), gap_bin, and air_status: 'used' or the first reason it is left out (step 3's
    reasons first, then AIR_LEDGER_ORDER).
    """
    p = degradation_laps(classified, fuel_effect)
    p = p.join(start_of_lap(classified))
    gap = p["gap_start_sec"]
    following = pd.Series(pd.NA, index=p.index, dtype="boolean")
    following[gap.notna()] = (gap[gap.notna()] < FOLLOW_THRESHOLD_SEC).to_numpy()
    p["following"] = following
    p["gap_bin"] = gap_bin(gap)

    near = gap < CLEAR_AIR_SEC
    status = p["deg_status"].astype("object").copy()
    checks = [
        (gap.isna(), EXCLUDED_NO_GAP),
        (near & (p["car_ahead_laps_up"] < 0), EXCLUDED_BACKMARKER),
        (near & (p["car_ahead_laps_up"] > 0), EXCLUDED_BEING_LAPPED),
    ]
    for mask, reason in checks:
        status[(status == STEP3_USED) & mask.to_numpy()] = reason
    status[status == STEP3_USED] = USED
    p["air_status"] = status
    return p


def dirty_air_ledger(air_laps: pd.DataFrame) -> pd.DataFrame:
    """Pace laps -> laps used here, one row per reason (step 3's, then this step's; listed even at 0)."""
    counts = air_laps["air_status"].value_counts()
    rows = [{"reason": "pace laps", "laps": int(len(air_laps))}]
    rows += [{"reason": r, "laps": int(counts.get(r, 0))} for r in STEP3_LEDGER_ORDER + AIR_LEDGER_ORDER]
    rows.append({"reason": "used for the cost of following", "laps": int(counts.get(USED, 0))})
    return pd.DataFrame(rows, columns=["reason", "laps"])


# --- the fit ------------------------------------------------------------------------------------

def fixed_effects_fit(y: pd.Series, X: pd.DataFrame, groups: pd.Series, cluster: pd.Series
                      ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Least squares of y on the columns of X with one free level per group (fixed effects: y and each
    column are measured from their group's mean, so only changes within a group count), standard
    errors clustered by `cluster`.

    Returns (coefficients: one row per column of X with coef and se, residuals on y's index).
    A column that never changes within a group has no coefficient (NaN); if the remaining columns
    are collinear nothing is estimated; with fewer than 2 clusters there is no SE.
    """
    out = pd.DataFrame({"coef": np.nan, "se": np.nan}, index=X.columns)
    residual = pd.Series(np.nan, index=y.index)
    groups = pd.Series(np.asarray(groups), index=y.index)
    Xd = X - X.groupby(groups).transform("mean")
    yd = y - y.groupby(groups).transform("mean")
    keep = [c for c in X.columns if (Xd[c].abs() > 1e-9).any()]
    if not keep:
        return out, residual
    A, b = Xd[keep].to_numpy(dtype=float), yd.to_numpy(dtype=float)
    xtx = A.T @ A
    if np.linalg.matrix_rank(xtx) < len(keep):
        return out, residual
    inv = np.linalg.inv(xtx)
    coef = inv @ (A.T @ b)
    e = b - A @ coef
    out.loc[keep, "coef"] = coef
    residual[:] = e
    labels = np.asarray(cluster)
    n_clusters = len(pd.unique(labels))
    if n_clusters >= 2:
        scores = pd.DataFrame(A * e[:, None]).groupby(labels).sum().to_numpy()
        cov = n_clusters / (n_clusters - 1) * inv @ (scores.T @ scores) @ inv
        out.loc[keep, "se"] = np.sqrt(np.diag(cov))
    return out, residual


def _wear_columns(laps: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Tyre age as one column per group (compound, or race and compound): a wear slope for each."""
    return pd.DataFrame({f"wear {g}": laps["age"].where(by == g, 0.0).astype(float) for g in pd.unique(by)},
                        index=laps.index)


def _bin_dummies(laps: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({label: (laps["gap_bin"] == label).astype(float) for label in BIN_LABELS}, index=laps.index)


def _fit(laps: pd.DataFrame, treatment: pd.DataFrame, wear_by: pd.Series, cluster: pd.Series):
    X = pd.concat([treatment, _wear_columns(laps, wear_by)], axis=1)
    return fixed_effects_fit(laps["corrected_sec"], X, laps["stint"], cluster)


def _used(air_laps: pd.DataFrame) -> pd.DataFrame:
    return air_laps[air_laps["air_status"] == USED]


def following_cost(used: pd.DataFrame, wear_by: Optional[pd.Series] = None,
                   cluster: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    beta: the cost of starting a lap within FOLLOW_THRESHOLD_SEC of the car ahead, against every other
    used lap, with its SE, the laps on each side and the stints that have both (only those inform
    beta). NaN when either side has fewer than MIN_LAPS laps. One race by default (wear per compound,
    SE by stint); pass wear_by and cluster to pool races.
    """
    follow = used["following"].astype(bool)
    stints_both = int((follow.groupby(used["stint"]).nunique() == 2).sum())
    result = {"beta": np.nan, "se": np.nan, "laps": int(len(used)), "following_laps": int(follow.sum()),
              "stints": int(used["stint"].nunique()), "stints_both": stints_both}
    if follow.sum() < MIN_LAPS or (~follow).sum() < MIN_LAPS:
        return result
    wear_by = used["compound"] if wear_by is None else wear_by
    cluster = used["stint"] if cluster is None else cluster
    coef, _ = _fit(used, pd.DataFrame({"following": follow.astype(float)}), wear_by, cluster)
    result.update(beta=float(coef.loc["following", "coef"]), se=float(coef.loc["following", "se"]))
    return result


def gap_bin_costs(used: pd.DataFrame, wear_by: Optional[pd.Series] = None,
                  cluster: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    One row per gap bin, then the reference: bin, lo, hi, laps, cost (s/lap against laps CLEAR_AIR_SEC
    or more behind), se. A bin with fewer than MIN_LAPS laps is still in the fit but gets no cost.

    Also returns each used lap's cost against its own clear-air pace: lap time minus what the model
    says this car does in clear air on this lap (stint level, tyre wear, fuel). Within a bin these
    average exactly to the bin's cost (and to 0 in the reference).
    """
    rows = [{"bin": label, "lo": lo, "hi": hi} for label, lo, hi in zip(BIN_LABELS, GAP_EDGES[:-1], GAP_EDGES[1:])]
    rows.append({"bin": CLEAR_LABEL, "lo": CLEAR_AIR_SEC, "hi": np.inf})
    table = pd.DataFrame(rows)
    table["laps"] = [int((used["gap_bin"] == r).sum()) for r in table["bin"]]
    table["cost"], table["se"] = np.nan, np.nan
    lap_cost = pd.Series(np.nan, index=used.index)
    reference_laps = int(table["laps"].iloc[-1])
    if reference_laps < MIN_LAPS or table["laps"].iloc[:-1].sum() == 0:
        return table, lap_cost
    dummies = _bin_dummies(used)
    wear_by = used["compound"] if wear_by is None else wear_by
    cluster = used["stint"] if cluster is None else cluster
    coef, residual = _fit(used, dummies, wear_by, cluster)
    est = coef.loc[list(BIN_LABELS)]
    enough = (table["laps"].iloc[:-1] >= MIN_LAPS).to_numpy()
    table.loc[table.index[:-1][enough], "cost"] = est["coef"].to_numpy()[enough]
    table.loc[table.index[:-1][enough], "se"] = est["se"].to_numpy()[enough]
    table.loc[table.index[-1], "cost"] = 0.0                    # the reference, by definition
    lap_cost = residual + dummies.mul(est["coef"].fillna(0.0), axis=1).sum(axis=1)
    return table, lap_cost


def wear_check(used: pd.DataFrame) -> pd.DataFrame:
    """
    Per slick compound: the wear slope on these laps as step 3 fits it (stint fixed effects), and again
    with the cost of following in the model. Traffic would bias step 3 if following were more common
    late (or early) in stints.
    """
    rows = []
    compounds = sorted(used["compound"].dropna().unique(), key=compound_rank)
    if not compounds:
        return pd.DataFrame(columns=["compound", "laps", "slope", "slope_with_following", "change"])
    without, _ = _fit(used, pd.DataFrame(index=used.index), used["compound"], used["stint"])
    follow = used["following"].astype(bool)
    with_follow = None
    if follow.sum() >= MIN_LAPS and (~follow).sum() >= MIN_LAPS:
        with_follow, _ = _fit(used, pd.DataFrame({"following": follow.astype(float)}), used["compound"], used["stint"])
    for c in compounds:
        a = without.loc[f"wear {c}", "coef"]
        b = with_follow.loc[f"wear {c}", "coef"] if with_follow is not None else np.nan
        rows.append({"compound": c, "laps": int((used["compound"] == c).sum()), "slope": a,
                     "slope_with_following": b, "change": b - a})
    return pd.DataFrame(rows, columns=["compound", "laps", "slope", "slope_with_following", "change"])


def driver_traffic(used: pd.DataFrame) -> pd.DataFrame:
    """Per driver: laps used, laps following, the share, and the car followed most (with its laps)."""
    rows = []
    for code, g in used.groupby("driver_code", sort=True):
        follow = g[g["following"].astype(bool)]
        top = follow["car_ahead"].value_counts()
        rows.append({"driver_code": code, "laps": int(len(g)), "following_laps": int(len(follow)),
                     "share": len(follow) / len(g), "most_followed": top.index[0] if len(top) else None,
                     "most_followed_laps": int(top.iloc[0]) if len(top) else 0})
    out = pd.DataFrame(rows, columns=["driver_code", "laps", "following_laps", "share", "most_followed",
                                      "most_followed_laps"])
    return out.sort_values(["share", "driver_code"], ascending=[False, True]).reset_index(drop=True)


# --- one race, every race -----------------------------------------------------------------------

@dataclass
class RaceDirtyAir:
    mixed: bool                 # mixed_conditions: reported, but not for season-wide figures
    laps: pd.DataFrame          # dirty_air_laps, plus lap_cost_sec on used laps (gap_bin_costs)
    ledger: pd.DataFrame        # dirty_air_ledger
    following: Dict[str, Any]   # following_cost: beta, se, laps, following_laps, stints, stints_both
    bins: pd.DataFrame          # gap_bin_costs
    wear: pd.DataFrame          # wear_check
    drivers: pd.DataFrame       # driver_traffic


def race_dirty_air(classified: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_SEC_PER_KG) -> RaceDirtyAir:
    """Everything step 4 says about one race, from its classified laps (laps.classify_laps). Pure."""
    a = dirty_air_laps(classified, fuel_effect)
    used = _used(a)
    bins, lap_cost = gap_bin_costs(used)
    a["lap_cost_sec"] = lap_cost.reindex(a.index)
    return RaceDirtyAir(mixed=mixed_conditions(classified), laps=a, ledger=dirty_air_ledger(a),
                        following=following_cost(used), bins=bins, wear=wear_check(used),
                        drivers=driver_traffic(used))


def load_race_dirty_air(year: int, round_number: int, session: str = "R",
                        root: str = lake.DEFAULT_LAKE_ROOT) -> RaceDirtyAir:
    path = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    if not path.exists():
        raise FileNotFoundError(f"{year} round {round_number}: no Silver laps in the lake ({path}).")
    return race_dirty_air(classify_laps(lake.read_table(path)))


def season_results(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT
                   ) -> List[Tuple[int, int, Optional[str], RaceDirtyAir]]:
    """(season, round, event name, RaceDirtyAir) for every race of these seasons in the lake."""
    wanted = {int(s) for s in seasons}
    out = []
    for path in lake.list_sessions(lake.SILVER, "laps", root):
        if "session=R" not in path.parts:
            continue
        laps = lake.read_table(path)
        if laps.empty or int(laps["season"].iloc[0]) not in wanted:
            continue
        season, round_number = int(laps["season"].iloc[0]), int(laps["round"].iloc[0])
        try:
            event = lake.read_manifest(season, round_number, "R", root).get("event_name")
        except FileNotFoundError:
            event = None
        out.append((season, round_number, event, race_dirty_air(classify_laps(laps))))
    return sorted(out, key=lambda r: (r[0], r[1]))


def season_table(results) -> pd.DataFrame:
    """One row per race: beta with SE and its laps, the cost per gap bin, the wear check, the ledger's exclusions."""
    rows = []
    for season, round_number, event, r in results:
        row = {"season": season, "round": round_number, "event_name": event, "mixed": r.mixed, **r.following}
        for b in r.bins.to_dict("records"):
            if b["bin"] != CLEAR_LABEL:
                row[f"cost {b['bin']}"], row[f"se {b['bin']}"], row[f"laps {b['bin']}"] = b["cost"], b["se"], b["laps"]
        row[f"laps {CLEAR_LABEL}"] = int(r.bins["laps"].iloc[-1])
        wear = r.wear.set_index("compound")
        for c in SLICKS:
            row[f"wear {c.lower()}"] = wear["slope"].get(c, np.nan)
            row[f"wear {c.lower()} with following"] = wear["slope_with_following"].get(c, np.nan)
        ledger = r.ledger.set_index("reason")["laps"]
        for reason in AIR_LEDGER_ORDER:
            row[reason] = int(ledger[reason])
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(results) -> Dict[str, Any]:
    """
    Every dry race's used laps in one fit: a wear slope per race and compound, stint levels, and SE
    clustered by race. Returns {'races', 'following': following_cost, 'bins': gap_bin_costs table}.
    """
    frames = []
    for season, round_number, _, r in results:
        if r.mixed:
            continue
        u = _used(r.laps).copy()
        race = f"{season}|{round_number}"
        u["race"], u["stint"] = race, race + "|" + u["stint"].astype(str)
        frames.append(u)
    if not frames:
        return {"races": 0, "following": None, "bins": None}
    used = pd.concat(frames)
    used.index = pd.RangeIndex(len(used))
    wear_by = used["race"] + "|" + used["compound"].astype(str)
    bins, _ = gap_bin_costs(used, wear_by, used["race"])
    return {"races": len(frames), "following": following_cost(used, wear_by, used["race"]), "bins": bins}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 step 4 over every race in the lake: the cost of following")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2023 2024")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    results = season_results(args.season, args.lake)
    if not results:
        print("No Silver laps in the lake for these seasons.")
        return 1
    table = season_table(results)
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.float_format", "{:.3f}".format):
        print(table[["season", "round", "event_name", "mixed", "laps", "following_laps", "stints_both", "beta",
                     "se"]].to_string(index=False))
    dry = table[~table["mixed"]]
    print(f"\n{len(dry)} dry races ({int(table['mixed'].sum())} mixed-conditions races left out of these figures).")
    p = pooled(results)
    f = p["following"]
    if f is not None and not np.isnan(f["beta"]):
        print(f"Cost of following (start gap < {FOLLOW_THRESHOLD_SEC:g} s, against every other lap), pooled: "
              f"{f['beta']:+.3f} ± {f['se']:.3f} s/lap (SE clustered by race); {f['following_laps']:,} following "
              f"laps of {f['laps']:,}, {f['stints_both']:,} stints with both")
        per_race = dry["beta"].dropna()
        lo, hi = dry["beta"] - 2 * dry["se"], dry["beta"] + 2 * dry["se"]
        print(f"  per race: median {per_race.median():+.3f} (IQR {per_race.quantile(0.25):+.3f} to "
              f"{per_race.quantile(0.75):+.3f}) over {len(per_race)} races; {int((lo > 0).sum())} clearly above 0, "
              f"{int((hi < 0).sum())} clearly below (2 SE)")
        print(f"Cost against clear air (start gap {CLEAR_AIR_SEC:g} s or more), pooled:")
        for b in p["bins"].to_dict("records")[:-1]:
            print(f"  {b['bin']:9s} {b['cost']:+.3f} ± {b['se']:.3f} s/lap  ({b['laps']:,} laps)")
        print(f"  reference {int(p['bins']['laps'].iloc[-1]):,} laps")
    for reason in AIR_LEDGER_ORDER:
        print(f"Left out, {reason}: {int(dry[reason].sum()):,} laps")
    change = (dry["wear medium with following"] - dry["wear medium"]).dropna()
    if len(change):
        print(f"Step 3 check: with the cost of following in the fit, the medium wear slope moves by median "
              f"{change.median():+.4f} s/lap (IQR {change.quantile(0.25):+.4f} to {change.quantile(0.75):+.4f}, "
              f"{len(change)} races)")
    print(f"Report: {lake.write_report(table, 'dirty_air', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
