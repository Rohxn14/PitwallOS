"""
Chance (src/simulator/montecarlo.py) - Phase 4 step 3

One simulated race (engine.py) uses the most likely value of everything: each driver's forecast pace,
the circuit's typical stop, a pass when the chances say it is more likely than not. The real race will
not be that one race. A Monte Carlo runs the race many times; in each run, every uncertain thing is drawn
at random, as often as it really happens. A driver's chance of a podium is then the share of runs in
which they finished in the top three.

What can be drawn (each from the misses measured on races the models never saw; calibration.py measures
them):

    offset   each driver's pace against the field, for the whole race: the forecast missed it by this
             much (s per lap). One draw per driver per run, from the misses at the same part of a race
             (early / middle / late start).
    stint    how much that miss changes from one tyre stint to the next (one draw per stint).
    lap      the lap-to-lap noise around a driver's pace (one draw per lap): mostly a tenth or two,
             sometimes a big loss. Real misses, not a bell curve: real laps have more big misses.
    stop     how much longer or shorter a stop is than the race's typical one (one draw per stop).
    pass     on every lap stuck behind a car, the pass is drawn with the rules' chance (rules.py),
             instead of passing once the chances add up to one half.

    pace of a run = forecast + offset (driver) + stint shift (driver, stint) + lap noise (lap)

What stays as it really was: the decisions (the plans: stops, tyres) and the events (safety cars, VSCs,
red flags, retirements). Step 3b will draw the events too.

Measured (calibration.py: the 41 dry races of 2023 + 2024, out of fold, from lap 5 and from halfway, 200
runs; 2025 stays locked):

    The offset is what matters. Against the one deterministic race (step 1), Brier score from lap 5:
    podium 0.095 -> 0.067, points 0.134 -> 0.088, both "better" (paired over races; the win: no clear
    difference). Each other draw was then added to the offset on its own: only the pass made the chances
    truer ("better" for the podium from halfway, never worse), so the Monte Carlo draws the offset and
    the passes (KEPT_DRAWS); the others stay so calibration.py can test them again. Their chances are as
    true as luck allows: calibration error (mean |happened - chance| over the bins) from lap 5 win 1.8,
    podium 1.9, points 1.9 points; from halfway 1.2, 1.7, 2.3. A perfect forecaster is at least that far
    off by luck alone in 27, 72, 91 % and 74, 70, 45 % of draws on the same 716 cars.

    Why lap noise hurts: the pass curve (rules.py) was fitted on smooth forecast advantages, and the real
    passes it was fitted on already include the laps' randomness. Noise added to the advantage counts it
    twice: two equal cars pass on 4.9 % of laps on the curve, 11.8 % with lap noise, so the order
    reshuffles more than in real races (12 races from lap 5: 2.87 places per car; real 2.49, offset
    alone 2.60, offset and passes 2.61).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.simulator import engine

ALL_DRAWS = ("offset", "stint", "lap", "stop", "pass")
KEPT_DRAWS = ("offset", "pass")    # the draws that made the chances truer (see Measured above)
PARTS = ("early", "middle", "late")

def part_of_race(lap: int, total_laps: int) -> str:
    """Early (the first third), middle or late: which offsets to draw from, by where the race starts."""
    share = lap / total_laps
    return "early" if share < 1 / 3 else "middle" if share < 2 / 3 else "late"

@dataclass
class Uncertainty:
    """What the simulator does not know, as measured misses to draw from (s)."""
    offsets: Dict[str, np.ndarray]      # by part of the race: drivers' whole-race misses, s per lap
    stint_shifts: np.ndarray            # a stint's miss minus the driver's whole-race miss, s per lap
    lap_noise: np.ndarray               # a clean lap minus its stint's typical lap, s
    stop_extra: np.ndarray              # a stop's pit-lane time minus the race's typical one, s
    draws: Tuple[str, ...] = KEPT_DRAWS  # which of them to draw

    def only(self, *draws: str) -> "Uncertainty":
        """The same misses, drawing only these."""
        return Uncertainty(self.offsets, self.stint_shifts, self.lap_noise, self.stop_extra, tuple(draws))


@dataclass
class Prepared:
    """What every run of one race shares: who each planned lap belongs to, and its stint."""
    driver_of_row: np.ndarray           # index into drivers, one per planned lap
    stint_of_row: np.ndarray            # index into the race's (driver, stint) pairs, one per planned lap
    drivers: np.ndarray
    stints: int
    stops: list                         # (driver, in-lap) of every planned tyre change


def prepare(planned: pd.DataFrame) -> Prepared:
    """Worked out once per start, not once per run: whose lap each planned row is, and in which stint."""
    drivers, driver_of_row = np.unique(planned["driver_code"].to_numpy(), return_inverse=True)
    # A stint starts after each tyre change: count the changes before each lap, per driver.
    change = planned["is_tyre_change"].astype(int)
    changes_before = change.groupby(planned["driver_code"]).cumsum() - change
    pairs = pd.Series(list(zip(driver_of_row, changes_before)))
    stint_of_row, uniques = pd.factorize(pairs)
    tyre_changes = planned[planned["is_tyre_change"]]
    stops = list(zip(tyre_changes["driver_code"], tyre_changes["lap_number"].astype(int)))
    return Prepared(driver_of_row=driver_of_row, stint_of_row=stint_of_row, drivers=drivers,
                    stints=len(uniques), stops=stops)


def draw_pace(pace: np.ndarray, prepared: Prepared, uncertainty: Uncertainty, part: str,
              rng: np.random.Generator) -> np.ndarray:
    """One run's pace for every planned lap: the forecast plus the drawn misses."""
    drawn = pace.copy()
    if "offset" in uncertainty.draws:
        offsets = rng.choice(uncertainty.offsets[part], size=len(prepared.drivers))
        drawn += offsets[prepared.driver_of_row]
    if "stint" in uncertainty.draws:
        shifts = rng.choice(uncertainty.stint_shifts, size=prepared.stints)
        drawn += shifts[prepared.stint_of_row]
    if "lap" in uncertainty.draws:
        drawn += rng.choice(uncertainty.lap_noise, size=len(drawn))
    return drawn


def draw_stops(prepared: Prepared, uncertainty: Uncertainty, rng: np.random.Generator) -> Dict[tuple, float]:
    """One run's extra seconds for every planned tyre change (none if stops are not drawn)."""
    if "stop" not in uncertainty.draws or not prepared.stops:
        return {}
    extra = rng.choice(uncertainty.stop_extra, size=len(prepared.stops))
    return dict(zip(prepared.stops, extra))


@dataclass
class Outcomes:
    """Every run's result: ranks has one row per run, one column per driver (NaN: did not finish)."""
    ranks: pd.DataFrame

    def chance(self, top: int) -> pd.Series:
        """Each driver's share of runs finishing in the first `top` places."""
        return (self.ranks <= top).mean()

    def table(self) -> pd.DataFrame:
        """Per driver: chances of a win, a podium, points (top 10), and the average rank."""
        return pd.DataFrame({"win": self.chance(1), "podium": self.chance(3), "points": self.chance(10),
                             "mean_rank": self.ranks.mean()}).sort_values("mean_rank")


def run(start: pd.DataFrame, planned: pd.DataFrame, pace: pd.Series, conditions: pd.DataFrame, rules,
        total_laps: int, uncertainty: Uncertainty, runs: int, seed: int, start_lap: int) -> Outcomes:
    """
    The race `runs` times from the same start, each with its own draws. The same seed gives the same
    runs (numpy's random generator), so a result can always be reproduced.
    """
    rng = np.random.default_rng(seed)
    prepared = prepare(planned)
    part = part_of_race(start_lap, total_laps)
    base = pace.to_numpy(dtype=float)
    ranks = []
    for _ in range(runs):
        drawn = pd.Series(draw_pace(base, prepared, uncertainty, part, rng), index=planned.index)
        # With a generator the engine draws every pass; with None it passes at the median waiting time.
        simulation = engine.simulate(start, planned, drawn, conditions, rules, total_laps,
                                     rng=rng if "pass" in uncertainty.draws else None,
                                     stop_extra=draw_stops(prepared, uncertainty, rng), record_laps=False)
        ranks.append(dict(zip(simulation.finish["driver_code"], simulation.finish["rank"])))
    return Outcomes(ranks=pd.DataFrame(ranks, columns=list(start.index), dtype=float))