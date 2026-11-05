"""
The race as the simulator sees it (src/simulator/race.py) - Phase 4 step 1

A replay starts from a real race when lap S ends and simulates the rest. From the real race it takes
three things, and nothing else:

  the start    one moment (snapshot: when the last car on the leader's lap completes lap S), and every
               running car's laps done by then (S, fewer for a lapped car), its clock (the session time
               it last crossed the line) and its laps so far (the pace model reads those)
  the plan     what each team DECIDED for the laps after S: the laps it pitted on, the tyres it fitted
               (compound, new or used, their age), and the lap the car stopped for good
  the events   what HAPPENED to everyone: the safety cars, VSCs and red flags, in the leader's laps
               (Phase 2 step 1), and which cars took the chequered flag

Lap times after S are never taken from the real race: they are what the simulator has to predict.
Only decisions and events are given. A "what if" changes the plan.

How a replay is scored needs the same care (finish_ranks, hold_order_ranks): only the cars that took
the chequered flag are ranked, in the order they crossed the line (laps completed, then time), so a
car that retires ahead is not a place gained for anyone, and a penalty given after the race does not
count against the simulator (it simulates the race on the track, not the stewards).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR, neutralised_periods
from src.analytics.race_overview import classification
from src.models import evaluate
from src.pipeline import lake

DEFAULT_START_LAP = 5

# What the race was doing on a lap (the leader's lap, like Race Control counts), and which part of a
# neutralisation the lap is. A red flag outranks a safety car, which outranks a VSC (Qatar 2024 lap 40:
# a VSC that became a safety car 20 s later is a safety-car lap).
GREEN = "green"
KIND_RANK = {RED_FLAG: 3, SAFETY_CAR: 2, VIRTUAL_SAFETY_CAR: 1}
DEPLOY, FULL, END, SINGLE = "deploy", "full", "end", "single"     # first lap, laps between, last lap, one lap only

PLAN_COLUMNS = ["driver_code", "lap_number", "compound", "tyre_age", "stint_lap", "fresh_tyre", "is_wet_tyre",
                "is_in_lap", "is_out_lap", "is_tyre_change", "pit_kind"]


@dataclass
class Race:
    """One real race, as the simulator reads it."""
    season: int
    round: int
    event_name: str
    total_laps: int
    laps: pd.DataFrame              # Gold rows + answers (evaluate.load_race) + Silver's lap_start_sec,
                                    # lap_end_sec and pit_kind; one row per driver per lap
    finishers: Tuple[str, ...]      # took the chequered flag (official result)
    periods: pd.DataFrame           # neutralised_periods: kind, first_lap, last_lap (leader's laps)
    grid: Dict[str, float]          # starting position (official result; a pit-lane start = behind the last)

    @property
    def key(self) -> Tuple[int, int]:
        return self.season, self.round

    @property
    def mixed(self) -> bool:
        """Anyone on intermediates or wets: the pace model cannot see rain coming, so these are scored apart."""
        return bool(self.laps["is_wet_tyre"].astype(bool).any())

    def last_lap(self, driver: str) -> int:
        """The last lap the car completes: the race distance for a finisher (a lapped finisher is simulated
        over the full distance and ranked by its time), else its last lap with an end time."""
        if driver in self.finishers:
            return self.total_laps
        rows = self.laps[(self.laps["driver_code"] == driver) & self.laps["lap_end_sec"].notna()]
        return int(rows["lap_number"].max()) if len(rows) else 0


def load(season: int, round_number: int, root: str = lake.DEFAULT_LAKE_ROOT) -> Race:
    """Read one race from the lake (Bronze results, Silver laps, Gold features)."""
    gold = evaluate.load_race(season, round_number, root)
    silver = lake.read_table(lake.table_path(lake.SILVER, "laps", season, round_number, "R", root))
    results = lake.read_table(lake.table_path(lake.BRONZE, "results", season, round_number, "R", root))
    return build(gold, silver, results)


def build(gold: pd.DataFrame, silver: pd.DataFrame, results: pd.DataFrame) -> Race:
    """The Race from its three tables (Gold with answers attached, Silver laps, Bronze results). Pure."""
    extra = silver[["driver_code", "lap_number", "lap_start_sec", "lap_end_sec", "pit_kind"]].assign(
        driver_code=lambda d: d["driver_code"].astype(object))
    laps = gold.merge(extra, on=["driver_code", "lap_number"], how="left", validate="one_to_one")
    laps["pit_kind"] = laps["pit_kind"].astype(object).where(laps["pit_kind"].notna(), None)
    result = classification(results)
    finishers = tuple(d for d in result.loc[result["finished"], "driver_code"] if d in set(laps["driver_code"]))
    grid = result.set_index("driver_code")["grid"].astype(float)
    grid = grid.where(grid > 0, (grid.max() if grid.notna().any() else 0.0) + 1.0)     # 0 = from the pit lane
    return Race(season=int(laps["season"].iloc[0]), round=int(laps["round"].iloc[0]),
                event_name=str(laps["event_name"].iloc[0]), total_laps=int(laps["total_race_laps"].iloc[0]),
                laps=laps, finishers=finishers, periods=neutralised_periods(silver), grid=grid.to_dict())


# --- the race's events ----------------------------------------------------------------------------

def status_of(periods: Iterable[Tuple[str, int, int]], total_laps: int) -> Dict[int, Tuple[str, str]]:
    """
    {lap: (kind, phase)} for every neutralised lap of the leader's laps, from periods given as (kind, first
    lap, last lap); a green lap is not in it. Where periods overlap on a lap, the stronger kind wins (red
    flag > SC > VSC). Plain Python and tuples: the Monte Carlo builds one for every run (events.py).
    """
    status: Dict[int, Tuple[str, str]] = {}
    rank: Dict[int, int] = {}
    for kind, first, last in periods:
        first, last = int(first), int(last)
        for lap in range(first, min(last, total_laps) + 1):
            if KIND_RANK[kind] <= rank.get(lap, 0):
                continue
            rank[lap] = KIND_RANK[kind]
            status[lap] = (kind, SINGLE if first == last else DEPLOY if lap == first else END if lap == last else FULL)
    return status


def lap_conditions(race: Race) -> pd.DataFrame:
    """
    One row per lap number (1 .. race distance): `kind` (green, SC, VSC, red flag) and `phase` (deploy,
    full, end or single for a neutralised lap; empty for green). Where periods overlap on a lap, the
    stronger kind wins (red flag > SC > VSC).
    """
    status = status_of(zip(race.periods["kind"], race.periods["first_lap"], race.periods["last_lap"]), race.total_laps)
    laps = pd.RangeIndex(1, race.total_laps + 1, name="lap_number")
    return pd.DataFrame({"kind": pd.Series([status.get(lap, (GREEN, ""))[0] for lap in laps], index=laps, dtype=object),
                         "phase": pd.Series([status.get(lap, (GREEN, ""))[1] for lap in laps], index=laps, dtype=object)})


# --- the start ------------------------------------------------------------------------------------

def snapshot(race: Race, lap: int) -> pd.DataFrame:
    """
    The race at one moment: when the last car on the leader's lap completes `lap` (before the leader
    completes the next one). Every car that had crossed the line by then, with laps_done (its laps
    completed: `lap` on the leader's lap, fewer for a lapped car) and clock_sec (when it completed the
    last of them), in order on the road (index driver_code).

    One moment for everyone, not "each car's lap `lap`": a lapped car completes its lap `lap` a whole
    lap later, after the leaders have raced on. With one moment, every car is less than a lap from the
    others and the order on the road is the order of the clocks, lapped cars in their real places.
    """
    return snapshot_of(race.laps, lap)


def snapshot_of(laps: pd.DataFrame, lap: int) -> pd.DataFrame:
    """snapshot() of any table of line crossings (driver_code, lap_number, lap_end_sec): a real race's laps, or a
    simulated race's (step 4: the game stops the race it is playing at such a moment)."""
    ends = laps.dropna(subset=["lap_end_sec"])
    at_lap = ends[ends["lap_number"] == lap]
    following = ends.loc[ends["lap_number"] == lap + 1, "lap_end_sec"]
    leader_next = float(following.min()) if len(following) else np.inf
    on_lead_lap = at_lap[at_lap["lap_end_sec"] < leader_next]
    moment = float(on_lead_lap["lap_end_sec"].max()) if len(on_lead_lap) else -np.inf
    done = ends[ends["lap_end_sec"] <= moment].sort_values("lap_number").groupby("driver_code").tail(1)
    state = pd.DataFrame({"laps_done": done["lap_number"].astype(int).to_numpy(),
                          "clock_sec": done["lap_end_sec"].astype(float).to_numpy()},
                         index=pd.Index(done["driver_code"].to_numpy(), name="driver_code"))
    return state.sort_values("clock_sec")


def running_at(race: Race, lap: int) -> Tuple[str, ...]:
    """The cars on track at the moment of snapshot(race, lap): they race on after the laps they had done."""
    state = snapshot(race, lap)
    return tuple(d for d, n in state["laps_done"].items() if race.last_lap(d) > n)


def start_state(race: Race, lap: int) -> pd.DataFrame:
    """snapshot() of the running cars only: laps_done and clock_sec, in order on the road."""
    state = snapshot(race, lap)
    return state[[race.last_lap(d) > n for d, n in state["laps_done"].items()]]


def last_rows(race: Race, state: pd.DataFrame) -> pd.DataFrame:
    """Each car's row of the last lap it had completed in `state`."""
    key = pd.MultiIndex.from_arrays([race.laps["driver_code"], race.laps["lap_number"]])
    wanted = pd.MultiIndex.from_arrays([state.index, state["laps_done"]])
    return race.laps[key.isin(wanted)]


def first_start_lap(race: Race, lap: int = DEFAULT_START_LAP) -> Optional[int]:
    """
    The first lap from `lap` on at which every running car has driven a clean lap: before that, no
    pace model has seen every car's pace (Qatar 2024: safety car on laps 1-4, so lap 5 is the first
    green lap and nobody has a clean lap when it ends). None if that never happens.
    """
    for candidate in range(lap, race.total_laps):
        rows = last_rows(race, start_state(race, candidate))
        if len(rows) and bool(evaluate.pace_seen(rows).all()):
            return candidate
    return None


def history(race: Race, lap: int) -> pd.DataFrame:
    """Every car's rows up to the last lap it had completed at the moment of snapshot(race, lap), answers
    removed: all a forecast made then may read."""
    done = snapshot(race, lap)["laps_done"]
    cut = race.laps["driver_code"].map(done)
    return evaluate.model_inputs(race.laps[race.laps["lap_number"] <= cut])


# --- the plan -------------------------------------------------------------------------------------

def plan(race: Race, lap: int, run_on: bool = False) -> pd.DataFrame:
    """
    What each running car does on every lap after the moment of snapshot(race, lap), as decided in the
    real race (PLAN_COLUMNS): its tyres (compound, age, laps into the stint, new or used), its pit
    passages (in-lap, out-lap, and on the in-lap what the passage was: tyre_change, pass_through, ended)
    and its last lap.

    A lapped finisher completed fewer laps than the race distance; in the simulation it may not be
    lapped, so its plan runs on to the full distance on the tyres it finished on (the chequered flag
    decides how many it completes).

    run_on=True does the same for a car that really retired: nobody is known to retire (step 3b draws the
    retirements), so its plan runs on to the distance on the tyres it had. A car that retired in the pit lane
    (pit_kind "ended") is not in the pit lane any more.
    """
    parts = []
    for driver, done in start_state(race, lap)["laps_done"].items():
        last = race.total_laps if run_on else race.last_lap(driver)
        rows = race.laps[(race.laps["driver_code"] == driver) & race.laps["lap_number"].between(done + 1, last)]
        rows = rows[PLAN_COLUMNS].copy()
        if run_on and driver not in race.finishers:
            ended = rows["pit_kind"] == "ended"
            rows.loc[ended, "is_in_lap"] = False
            rows.loc[ended, "pit_kind"] = None
        missing = last - int(rows["lap_number"].max() if len(rows) else done)
        if missing > 0:                                  # a lapped finisher: carry the last tyres on
            tail = race.laps[(race.laps["driver_code"] == driver)].sort_values("lap_number").iloc[-1]
            extra = pd.DataFrame({
                "driver_code": driver,
                "lap_number": np.arange(1, missing + 1) + int(tail["lap_number"]),
                "compound": tail["compound"],
                "tyre_age": tail["tyre_age"] + np.arange(1, missing + 1),
                "stint_lap": tail["stint_lap"] + np.arange(1, missing + 1),
                "fresh_tyre": tail["fresh_tyre"], "is_wet_tyre": bool(tail["is_wet_tyre"]),
                "is_in_lap": False, "is_out_lap": False, "is_tyre_change": False, "pit_kind": None})
            rows = pd.concat([rows, extra], ignore_index=True)
        parts.append(rows)
    planned = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PLAN_COLUMNS)
    for col in ("is_wet_tyre", "is_in_lap", "is_out_lap", "is_tyre_change"):
        planned[col] = planned[col].astype(bool)
    planned["lap_number"] = planned["lap_number"].astype(int)
    return planned.reset_index(drop=True)


# --- scoring ----------------------------------------------------------------------------------------

def finish_ranks(race: Race, drivers: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """The real order at the line among the finishers (of `drivers`): laps completed, then crossing time.
    1 = first. Post-race penalties are not in it (the simulator races on the track, not in the stewards' room)."""
    wanted = set(race.finishers if drivers is None else drivers) & set(race.finishers)
    rows = race.laps[race.laps["driver_code"].isin(wanted) & race.laps["lap_end_sec"].notna()]
    last = rows.sort_values("lap_number").groupby("driver_code").tail(1)
    last = last.assign(minus_laps=-last["lap_number"]).sort_values(["minus_laps", "lap_end_sec"])
    return {d: i + 1 for i, d in enumerate(last["driver_code"])}


def hold_order_ranks(race: Race, lap: int, drivers: Iterable[str]) -> Dict[str, int]:
    """The do-nothing baseline: the order at the moment of snapshot(race, lap) (laps done, then who crossed
    the line first), kept to the flag; ranks among `drivers`."""
    state = snapshot(race, lap)
    state = state[state.index.isin(set(drivers))]
    state = state.assign(minus_laps=-state["laps_done"]).sort_values(["minus_laps", "clock_sec"])
    return {d: i + 1 for i, d in enumerate(state.index)}


def overtakes(race: Race, after_lap: int) -> int:
    """
    The real on-track overtakes after `after_lap` (step 3c), what the engine counts as a pass: on a green lap
    of the leader's, two cars on the same lap whose order at the line changed from the lap before, neither
    of them on an in- or out-lap. Measured on 2023 + 2024 from the first start lap: 36 a race (median 35, p10
    22, p90 56).
    """
    kind = lap_conditions(race)["kind"]
    ends = race.laps.dropna(subset=["lap_end_sec"]).pivot_table(index="driver_code", columns="lap_number", values="lap_end_sec")
    pit = race.laps.assign(pit=race.laps["is_pit_lap"].astype(bool).astype(float)).pivot_table(
        index="driver_code", columns="lap_number", values="pit", aggfunc="max")
    count = 0
    for lap in range(after_lap + 1, race.total_laps + 1):
        if kind.get(lap, GREEN) != GREEN or lap not in ends.columns or lap - 1 not in ends.columns:
            continue
        both = ends[[lap - 1, lap]].dropna()
        if lap in pit.columns:
            both = both[~(pit[lap].reindex(both.index).fillna(0.0) > 0)]
        before = both[lap - 1].rank(method="first").to_numpy()
        after = both[lap].rank(method="first").to_numpy()
        # every pair whose order changed: one car went from behind to in front
        count += int(((before[:, None] < before[None, :]) & (after[:, None] > after[None, :])).sum())
    return count


def finish_times(race: Race, drivers: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """The session time each finisher on the winner's lap crossed the line at the end of the race distance."""
    wanted = set(race.finishers if drivers is None else drivers) & set(race.finishers)
    rows = race.laps[race.laps["driver_code"].isin(wanted) & (race.laps["lap_number"] == race.total_laps)]
    return dict(zip(rows["driver_code"], rows["lap_end_sec"].astype(float)))


def development_races(root: str = lake.DEFAULT_LAKE_ROOT,
                      seasons: Iterable[int] = evaluate.DEVELOPMENT_SEASONS) -> Tuple[Tuple[int, int], ...]:
    """(season, round) of every development race in the lake with Gold features, in calendar order.
    The test season is not in here: evaluate.load_laps locks it."""
    wanted = {int(s) for s in seasons}
    locked = wanted & set(evaluate.TEST_SEASONS)
    if locked:
        raise evaluate.SeasonLocked(f"Season {sorted(locked)} holds the test races: locked until the simulator "
                                    f"is finished. Develop on {list(evaluate.DEVELOPMENT_SEASONS)}.")
    return tuple(r for r in lake.list_races(lake.GOLD, lake.GOLD_TABLE, root) if r[0] in wanted)
