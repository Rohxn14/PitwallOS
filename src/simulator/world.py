"""
The Monte Carlo with the events drawn (src/simulator/world.py) - Phase 4 step 3b

montecarlo.py draws what the pace model got wrong (offsets) and the passes, on a race whose safety cars,
VSCs and red flags were the real ones. This runs the same race with the events drawn too:

    each run   1. the pace misses and the passes, exactly as montecarlo.py draws them (generator 1)
               2. the neutralisations (events.EventModel), the retirements (events.RetirementModel) and the
                  teams' answers (teams.BoxModel): generator 2, so changing what the events do never changes
                  the pace misses (a paired comparison between configurations)
               3. the race, with the teams called on the lap each neutralisation comes out
                  (engine.simulate_plans, on_neutralisation)

The plan is the BASE plan (teams.decontaminate: the real decisions without the real events' influence), or
the real plan for the naive configuration that shows why. `events=False` is the clean race (nothing happens),
`react=False` is a race in which the teams keep their plans whatever comes out.

World is what a fold learns for this: the hazards, the box rule, the usual stopping ages, rung 2's tyre terms.

STEP 3c: the speed misses can be drawn BY PLACE AT THE START (PlaceOffsets): the cars at the front are known
better and the forecast squeezes the field. Measured on the whole-race misses of 2023 + 2024 (out of fold, against
the median driver), by place on the road at the start: P1-3 sd 0.295 s/lap and on average 0.052 FASTER than
forecast, P4-6 0.318 / 0.032 faster, P7-10 0.337 / 0.005 slower, P11-15 0.412 / 0.052 slower, P16+ 0.392 / 0.057
slower: a car is drawn from its own group's misses (by part of the race), so both the spread and the squeeze are
the measured ones. And every run counts its overtakes (Drawn.passes), to compare with the real ones.

STEP 4 (the game): run(common=True) gives the SAME FUTURES to two plans that differ in one car. Each run has its own
generators (seeded [seed, run] and [seed, run, EVENT_STREAM]), so what one run draws never shifts the next; inside a
run the passes are drawn by situation (engine.PassDraws) and the teams' answers one number per car (teams.Draw), so
a different plan for the player's car leaves the pace misses, the safety cars, the retirements and every other
car's luck where they were. Measured on the game's options: see game.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulator import engine, montecarlo, race as race_module
from src.simulator.events import EventModel, NO_EVENTS, RetirementModel
from src.simulator.race import Race
from src.simulator.teams import BOX_COLUMNS, BoxModel, GreenAges, Teams, TyreLife, TyreTerms, box_rows

EVENT_STREAM = 1                # the second generator of a run is seeded with [seed, EVENT_STREAM]
PLACE_GROUPS = (3, 6, 10, 15)   # places at the start: P1-3, P4-6, P7-10, P11-15, P16 and lower
GROUP_NAMES = ("P1-3", "P4-6", "P7-10", "P11-15", "P16+")


def place_group(place: float) -> int:
    """0 for P1-3, 1 for P4-6, 2 for P7-10, 3 for P11-15, 4 for P16 and lower."""
    return int(np.searchsorted(PLACE_GROUPS, place))


def start_places(start: pd.DataFrame) -> Dict[str, int]:
    """Each running car's place at the start: laps done, then who crossed the line first."""
    order = start.assign(minus_laps=-start["laps_done"]).sort_values(["minus_laps", "clock_sec"]).index
    return {d: i + 1 for i, d in enumerate(order)}


@dataclass
class PlaceOffsets:
    """The whole-race speed misses to draw from, by (part of the race, place group at the start)."""
    pools: Dict[Tuple[str, int], np.ndarray]

    def draw(self, rng: np.random.Generator, part: str, groups: np.ndarray) -> np.ndarray:
        """One miss per car, each from its own group's pool."""
        out = np.empty(len(groups))
        for i, g in enumerate(groups):
            pool = self.pools[(part, int(g))]
            out[i] = pool[rng.integers(len(pool))]
        return out


@dataclass
class Drawn(montecarlo.Outcomes):
    """montecarlo.Outcomes, and the number of on-track overtakes in every run."""
    passes: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class World:
    """What the drawn races need, learned from the training races of a fold."""
    events: EventModel
    box: BoxModel
    ages: GreenAges
    terms: TyreTerms
    retirements: RetirementModel = field(default_factory=RetirementModel)
    life: TyreLife = field(default_factory=lambda: TyreLife({}))     # step 4: the oldest a set may get

    def describe(self) -> str:
        return "\n".join([self.events.describe(), self.retirements.describe(), self.box.describe(),
                          "Usual tyre age at a green stop: " + ", ".join(
                              f"{c.lower()} {self.ages.median(c):.0f}" for c in ("SOFT", "MEDIUM", "HARD")) + " laps",
                          "Tyre life (teams almost never ran a set past): " + ", ".join(
                              f"{c.lower()} {self.life.limit(c):.0f}" for c in ("SOFT", "MEDIUM", "HARD")) + " laps"])


def fit_world(races: Iterable[Race], rung2) -> World:
    """The hazards, the box rule and the stopping ages of these (training) races; rung2: whose tyre terms re-price a
    changed plan (the one the pace forecast carries)."""
    races = list(races)
    ages = GreenAges.fit(races)
    rows = [box_rows(r) for r in races if not r.mixed]
    rows = [r for r in rows if len(r)]
    box = BoxModel.fit(pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=BOX_COLUMNS), ages)
    return World(events=EventModel.fit(races), box=box, ages=ages, terms=TyreTerms.from_rung2(rung2),
                 retirements=RetirementModel.fit(races), life=TyreLife.fit(races))


def run(start: pd.DataFrame, planned: pd.DataFrame, pace: pd.Series, rules, total_laps: int,
        uncertainty: montecarlo.Uncertainty, world: World, runs: int, seed: int, start_lap: int,
        events: bool = True, react: bool = True, retire: bool = False,
        place_offsets: Optional[PlaceOffsets] = None, common: bool = False,
        decided: Optional[Tuple[str, int]] = None) -> Drawn:
    """
    The race `runs` times from the same start, each run with its own pace misses, passes, neutralisations and
    answers. The same seed gives the same runs.

      planned, pace   the plan to run (the base plan, usually) and its clear-air forecast, same rows
      events          draw the neutralisations (False: a clean race)
      react           the teams answer a neutralisation (False: they keep their plans)
      retire          draw the retirements (False: a car whose plan ends early retires where it ends)
      place_offsets   draw each car's speed miss from its place group's (step 3c) instead of one pool for all
      common          the same futures for plans that differ in one car (step 4; see above): run i draws from
                      its own generators, the passes by situation, the answers one number per car
      decided         (driver, lap): that car's own call for a neutralisation coming out up to `lap` is made (the
                      game's player): it does not answer it
    """
    planned = planned.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    event_rng = np.random.default_rng([seed, EVENT_STREAM])
    prepared = montecarlo.prepare(planned)
    teams = Teams(planned, world.box, world.ages, world.terms, total_laps)
    model = world.events if events else NO_EVENTS
    part = montecarlo.part_of_race(start_lap, total_laps)
    base = np.asarray(pace, dtype=float)
    last_laps = teams.last_laps()
    by_place = place_offsets is not None and "offset" in uncertainty.draws
    if by_place:                         # the offsets come from the place pools, everything else as montecarlo.py draws it
        places = start_places(start)
        groups = np.array([place_group(places.get(d, len(places))) for d in prepared.drivers])
        uncertainty = uncertainty.only(*[d for d in uncertainty.draws if d != "offset"])
    passing = "pass" in uncertainty.draws
    ranks, passes = [], []
    for i in range(runs):
        if common:                       # this run's own numbers: nothing an earlier run drew moves them
            rng = np.random.default_rng([seed, i])
            event_rng = np.random.default_rng([seed, i, EVENT_STREAM])
        drawn = montecarlo.draw_pace(base, prepared, uncertainty, part, rng)
        if by_place:
            drawn = drawn + place_offsets.draw(rng, part, groups)[prepared.driver_of_row]
        status = race_module.status_of(model.draw(event_rng, start_lap, total_laps), total_laps)
        gone = world.retirements.draw(event_rng, last_laps, start_lap, total_laps) if retire else None
        draw = teams.begin(drawn, event_rng, gone, common=common, decided=decided)
        fixed = engine.PassDraws(start.index, total_laps, rng) if common and passing else None
        simulation = engine.simulate_plans(start, draw.plans, draw.passage, status, rules, total_laps,
                                           rng=rng if passing and not common else None, record_laps=False,
                                           on_neutralisation=draw.react if react else None, draws=fixed)
        ranks.append(dict(zip(simulation.finish["driver_code"], simulation.finish["rank"])))
        passes.append(simulation.passes)
    return Drawn(ranks=pd.DataFrame(ranks, columns=list(start.index), dtype=float), passes=np.array(passes))
