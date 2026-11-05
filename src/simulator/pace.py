"""
Lap times for planned laps (src/simulator/pace.py) - Phase 4 step 1

The simulator needs, for every car and every lap after the start lap S, the time the car would run in
CLEAR AIR on the tyres the plan puts it on. That is the pace model's job (Phase 3), asked a question it
was never scored on: k laps ahead, possibly ACROSS a pit stop (new tyres, maybe a different compound).

    clear-air time of lap S+k = what the rung knows when lap S ends   (its base, level or state)
                              + the formula's terms for lap S+k AS PLANNED: compound offset, wear x tyre
                                age, fuel x kg on board; no traffic term (the simulator adds traffic
                                from the gaps it simulates)
                              + rung 4 only: the trees' correction for lap S+k

Each rung keeps its own method; only the conditions of lap S+k come from the plan instead of "lap S's
tyres, k laps older". So with no stop between S and S+k, the forecast is exactly the rung's
predict(laps, k) minus its traffic term (tested).

The trees' inputs for lap S+k after a stop are a combination they never saw in training (Phase 3 scored
k laps ahead only when no stop came in between, so a lap S+k always had at least k+1 laps on its tyres).
`after_stop` says what to do there: "trees" (use them anyway) or "rung 3" (no correction after the
first stop). Their traffic inputs: "held" at lap S (as Phase 3 asked) or "clear" air.

Persistence knows no terms: its forecast is the last clean lap, whatever the plan (that is rung 1).

Measured before choosing (2023 + 2024, every rung fitted out of fold, forecasts made at lap 5 for every
later lap of the real plan, against the real clear-air pace laps, dry races, MAE s per lap):
  * After a stop the trees still help: 0.948 against 1.015 with no correction after the first
    stop ("rung 3"). Default: trees.
  * Their traffic inputs held at lap S or set to clear air: 0.633 against 0.640 for laps 1-5 ahead, the
    same per driver (0.470 / 0.474). Default: clear air, because the simulator adds traffic itself.
  * Over a whole race, rung 2 is the best forecaster from lap 5 (1-5 laps ahead 0.521, 41+ laps 1.025;
    rung 4 0.640 / 1.229): rung 4's trees learned 1-10 laps ahead from mid-race laps, and rung 3's level
    lags the fast early-race improvement (+0.25 s bias 1-5 laps ahead). What decides the order is each
    driver's pace AGAINST the field: per driver it is off by 0.42 s a lap on average for rung 2 (median
    0.30), 0.47 for rung 4, 0.50 for rung 3; persistence 0.45 (its error is mostly the same for everyone).
    PacePrior attacks exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.boosted import AHEAD_FEATURES, NEXT_LAP_FEATURES, Boosted, features, rung3_inputs
from src.models.linear import LinearDecomposition, driver_race, lap_terms
from src.models.persistence import Persistence
from src.models.state_space import StateSpace
from src.pipeline.feature_store import FUEL_END_KG, FUEL_START_KG
from src.simulator import race as race_module

# The trees' traffic inputs on a planned lap: "held" = as they were when lap S ended (how Phase 3 asked),
# "clear" = clear air (CLEAR_AIR_INPUTS), matching the clear-air lap time the simulator wants.
HELD, CLEAR = "held", "clear"
CLEAR_AIR_INPUTS = {"gap_on_track_end_sec": 5.0, "gap_ahead_sec": 5.0}
TREES, RUNG3 = "trees", "rung 3"


def fuel_at_start(lap: pd.Series, total_laps: pd.Series) -> pd.Series:
    """Fuel on board when a lap starts, kg: Gold's model (FUEL_START_KG, burnt evenly down to FUEL_END_KG)."""
    burn_per_lap = (FUEL_START_KG - FUEL_END_KG) / total_laps
    return (FUEL_START_KG - (lap - 1) * burn_per_lap).clip(lower=0.0)


def at_start(hist: pd.DataFrame, values: pd.Series, planned: pd.DataFrame) -> pd.Series:
    """A per-row value of the history, read on each driver's last row (lap S) and set on their planned laps."""
    last = hist.groupby("driver_code", sort=False).tail(1)
    by_driver = pd.Series(values[last.index].to_numpy(), index=last["driver_code"].to_numpy())
    return pd.Series(planned["driver_code"].map(by_driver).to_numpy(dtype=float), index=planned.index)


def planned_terms(rung2: LinearDecomposition, planned: pd.DataFrame, total_laps: int) -> pd.Series:
    """Rung 2's terms of each planned lap (compound offset, wear, fuel), in clear air: s."""
    coef = rung2.coefficients()
    fuel = fuel_at_start(planned["lap_number"].astype(float), pd.Series(float(total_laps), index=planned.index))
    gap = pd.Series(np.nan, index=planned.index)               # no car ahead: no traffic term
    terms = lap_terms(planned["compound"], planned["tyre_age"].astype(float), fuel, gap)
    return terms[coef.index] @ coef


def crafted_rows(hist: pd.DataFrame, planned: pd.DataFrame, traffic: str) -> pd.DataFrame:
    """
    For the trees: each driver's row of lap S, once per planned lap, with the tyre columns set so that
    features(row, k) describes the PLANNED lap S+k (its age and stint lap are stored as minus k, because
    features() adds k back). Traffic inputs held at lap S or set to clear air.
    """
    last = hist.groupby("driver_code", sort=False).tail(1).set_index("driver_code")
    rows = last.loc[planned["driver_code"]].reset_index()
    rows.index = planned.index
    k = planned["lap_number"] - rows["lap_number"]
    rows["compound"] = planned["compound"].to_numpy()
    rows["tyre_age"] = planned["tyre_age"].astype(float) - k
    rows["stint_lap"] = planned["stint_lap"].astype(float) - k
    rows["fresh_tyre"] = planned["fresh_tyre"].astype(float)
    rows["is_wet_tyre"] = planned["is_wet_tyre"].astype(bool)
    if traffic == CLEAR:
        for column, value in CLEAR_AIR_INPUTS.items():
            rows[column] = value
    rows["laps_ahead"] = k
    return rows


def tree_correction(rung: Boosted, hist: pd.DataFrame, planned: pd.DataFrame, traffic: str) -> pd.Series:
    """The trees' correction for every planned lap (0 on wet tyres or without a model), s."""
    correction = pd.Series(0.0, index=planned.index)
    inputs = rung3_inputs(rung.rung3, hist)
    rows = crafted_rows(hist, planned, traffic)
    inputs_at_s = pd.DataFrame({c: at_start(hist, inputs[c], planned) for c in inputs.columns}, index=planned.index)
    slick = ~rows["is_wet_tyre"]
    for k, group in rows[slick].groupby("laps_ahead"):
        model, columns = (rung.next_lap, NEXT_LAP_FEATURES) if k == 1 else (rung.ahead, AHEAD_FEATURES)
        if model is None:
            continue
        X = features(group, inputs_at_s.loc[group.index], int(k))
        correction[group.index] = model.predict(X[columns])
    return correction


def first_stop_lap(planned: pd.DataFrame) -> pd.Series:
    """Each planned lap's driver's first out-lap after lap S (NaN if no stop is planned)."""
    out = planned["lap_number"].where(planned["is_out_lap"])
    return out.groupby(planned["driver_code"]).transform("min")


PRIOR_WINDOW_LAPS = 10            # a forecast's implied base: its median over the first 10 planned laps
PRIOR_CLIP_SEC = 3.0              # a realised relative pace further out than this is a broken race, clipped
PRIOR_S0_GRID = np.round(np.geomspace(0.5, 200.0, 60), 3)


def realised_relative_pace(rung2: LinearDecomposition, laps: pd.DataFrame, lap: int) -> pd.Series:
    """Each driver's real pace after `lap` against the field's (hindsight, for fitting only): the median of
    rung 2's readings after `lap`, minus the median driver's. s per lap, + = slower than the field."""
    reading = rung2.base_readings(laps)
    after = laps["lap_number"] > lap
    pace = reading[after].groupby(laps.loc[after, "driver_code"]).median().dropna()
    return (pace - pace.median()).clip(-PRIOR_CLIP_SEC, PRIOR_CLIP_SEC)


def implied_base(forecast: pd.Series, planned: pd.DataFrame, terms: pd.Series) -> pd.Series:
    """What a forecast says each driver's base is: the median of (forecast - planned terms) over its first
    PRIOR_WINDOW_LAPS planned laps, against the median driver's. s per lap."""
    first = planned["lap_number"] - planned.groupby("driver_code")["lap_number"].transform("min") < PRIOR_WINDOW_LAPS
    base = (forecast - terms)[first].groupby(planned.loc[first, "driver_code"]).median()
    return base - base.median()


def relative_grid(grid: dict, drivers) -> pd.Series:
    """Starting position against the median starting position of these drivers."""
    g = pd.Series({d: grid.get(d, np.nan) for d in drivers}, dtype=float)
    return g - g.median() if g.notna().any() else g


class PacePrior:
    """
    A driver's pace against the field, from what the race has shown so far (the rung's forecast) and from
    two things known before it started: the grid (qualifying) and the team's pace in its previous race
    (team_form). Early in a race the laps say little (traffic, heavy fuel, tyres coming in), so the prior
    counts most; later the laps take over:

        relative pace = w x (the rung's)  +  (1 - w) x (per_place x grid  +  per_team x team's last race)
        w = S / (S + s0)            S = the start lap; s0 = laps of racing worth as much as the prior
        grid                        grid position - the median grid position of the cars running
        team's last race            the team's realised relative pace in its previous race (0 if none)

    s0, per_place and per_team are fitted (fit) on training races: least squares between this and each
    driver's realised relative pace after S (realised_relative_pace), at several start laps per race.
    Measured on 2023 + 2024 (leave one race out, lap 5, s per lap): the lap-5 reading alone misses a
    driver's relative race pace by 0.367 on average, the grid alone 0.346, the team's last race alone
    0.346, reading + grid 0.327, all three 0.283. The forecast keeps its own shape (tyres, fuel, the
    trees); only each driver's level is moved.
    """

    def __init__(self, s0: float = 40.0, per_place: float = 0.075, per_team: float = 0.0) -> None:
        self.s0, self.per_place, self.per_team = s0, per_place, per_team

    def weight(self, lap: int) -> float:
        return float(lap / (lap + self.s0))

    def fit(self, examples: pd.DataFrame) -> "PacePrior":
        """examples: one row per driver per (race, start lap): lap, rel (the rung's), grid, team, realised.
        Fewer than 20 usable rows: the defaults stay (measured on 2023 + 2024)."""
        if examples.empty:
            return self
        x = examples.dropna(subset=["rel", "grid", "realised"]).fillna({"team": 0.0})
        if len(x) < 20:
            return self
        best = (np.inf, self.s0, self.per_place, self.per_team)
        for s0 in PRIOR_S0_GRID:
            w = (x["lap"] / (x["lap"] + s0)).to_numpy()
            X = (1.0 - w)[:, None] * x[["grid", "team"]].to_numpy(float)
            rest = x["realised"].to_numpy(float) - w * x["rel"].to_numpy(float)
            coef = np.linalg.lstsq(X, rest, rcond=None)[0]
            sse = float(((rest - X @ coef) ** 2).sum())
            if sse < best[0]:
                best = (sse, float(s0), float(coef[0]), float(coef[1]))
        _, self.s0, self.per_place, self.per_team = best
        return self

    def adjust(self, forecast: pd.Series, planned: pd.DataFrame, terms: pd.Series, grid: dict,
               team_last: dict, lap: int) -> pd.Series:
        """The forecast with each driver's level moved to the blended relative pace.
        team_last: driver -> the team's relative pace in its previous race (team_form)."""
        rel = implied_base(forecast, planned, terms)
        g = relative_grid(grid, rel.index).fillna(0.0)
        team = pd.Series({d: team_last.get(d, np.nan) for d in rel.index}, dtype=float).fillna(0.0)
        w = self.weight(lap)
        target = w * rel + (1.0 - w) * (self.per_place * g + self.per_team * team)
        shift = (target - rel).fillna(0.0)
        return (forecast + planned["driver_code"].map(shift).fillna(0.0).to_numpy()).rename("pace_sec")


def team_form(races, rung2: LinearDecomposition) -> pd.DataFrame:
    """
    Each team's relative race pace in each dry race (the mean of its drivers' realised_relative_pace after
    the race's start lap), in calendar order: season, round, team, pace. The prior reads a team's latest
    value BEFORE the race being simulated, so nothing of that race is in it.
    """
    rows = []
    for race in races:
        if race.mixed:
            continue
        lap = race_module.first_start_lap(race) or race_module.DEFAULT_START_LAP
        relative = realised_relative_pace(rung2, race.laps, lap)
        team = race.laps.groupby("driver_code")["team_code"].first()
        by_team = relative.groupby(team.reindex(relative.index)).mean()
        rows += [{"season": race.season, "round": race.round, "team": t, "pace": float(v)} for t, v in by_team.items()]
    return pd.DataFrame(rows, columns=["season", "round", "team", "pace"])


def team_last_race(race, form: pd.DataFrame) -> dict:
    """driver -> their team's relative pace in its latest race before this one (absent if none)."""
    earlier = form[(form["season"] < race.season) | ((form["season"] == race.season) & (form["round"] < race.round))]
    latest = earlier.sort_values(["season", "round"]).groupby("team")["pace"].last()
    team = race.laps.groupby("driver_code")["team_code"].first()
    return {d: float(latest[t]) for d, t in team.items() if t in latest.index}


def hindsight(rung2: LinearDecomposition, laps: pd.DataFrame, planned: pd.DataFrame) -> pd.Series:
    """
    NOT a forecast: each driver's REAL pace over the rest of the race, for measuring the simulator's rules
    on their own. Rung 2's reading of every clean lap after the start (lap time minus its own terms, so
    traffic, tyres and fuel taken out); its median over each stint of the plan (else over all the
    driver's laps after the start); plus the planned lap's terms. What the simulator would do with a
    perfect pace model, up to the lap-to-lap noise.

    laps: the WHOLE race (model inputs), future laps included on purpose.
    """
    total_laps = int(laps["total_race_laps"].iloc[0])
    reading = rung2.base_readings(laps)
    by_lap = pd.Series(reading.to_numpy(), index=pd.MultiIndex.from_frame(laps[["driver_code", "lap_number"]]))
    on_plan = by_lap.reindex(pd.MultiIndex.from_frame(planned[["driver_code", "lap_number"]])).to_numpy()
    stint = planned["is_out_lap"].astype(int).groupby(planned["driver_code"]).cumsum()
    on_plan = pd.Series(on_plan, index=planned.index)
    level = on_plan.groupby([planned["driver_code"], stint]).transform("median")
    level = level.fillna(on_plan.groupby(planned["driver_code"]).transform("median"))
    return (level + planned_terms(rung2, planned, total_laps)).rename("pace_sec")


def forecast(rung, hist: pd.DataFrame, planned: pd.DataFrame, traffic: str = CLEAR,
             after_stop: str = TREES) -> pd.Series:
    """
    The clear-air time of every planned lap (s), from a fitted rung and the race's rows up to lap S.

    hist      the race up to and including lap S (race.history): nothing after S is read
    planned   race.plan(...) rows (or a changed plan): driver_code, lap_number and the tyre columns
    """
    total_laps = int(hist["total_race_laps"].iloc[0])
    if isinstance(rung, Persistence):
        return at_start(hist, rung.predict(hist), planned).rename("pace_sec")
    if isinstance(rung, LinearDecomposition):
        base = rung.base_readings(hist).groupby(driver_race(hist)).ffill()
        return (at_start(hist, base, planned) + planned_terms(rung, planned, total_laps)).rename("pace_sec")
    rung3 = rung if isinstance(rung, StateSpace) else rung.rung3
    level, _ = rung3.filter(hist)
    pace = at_start(hist, level, planned) + planned_terms(rung3.rung2, planned, total_laps)
    if isinstance(rung, Boosted):
        correction = tree_correction(rung, hist, planned, traffic)
        if after_stop == RUNG3:
            correction = correction.where(~(planned["lap_number"] >= first_stop_lap(planned)), 0.0)
        pace = pace + correction
    return pace.rename("pace_sec")
