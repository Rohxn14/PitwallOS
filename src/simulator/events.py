"""
Events drawn (src/simulator/events.py) - Phase 4 step 3b

In step 3a the safety cars, VSCs and red flags were the real ones: a "40 % podium" quietly knew when the
safety car would come. A strategist at lap 5 does not know. Here the neutralisations are DRAWN, from how
often they happened in the races the model has not seen:

    every lap that is not inside a neutralisation has a chance h of a new one starting on it   (the HAZARD)
    h is the sum of three hazards, one per kind: safety car, VSC, red flag
    the kind is drawn in proportion to those hazards, the length (laps of the leader) from the lengths seen
    in the other races for that kind

A draw is a list of periods (kind, first lap, last lap), the same shape as Phase 2 step 1's, turned into
the engine's lap status by race.status_of.

MEASURED (2023 + 2024, the laps after each race's first start lap; "events" are the periods that start
there, one per starting lap: where a VSC and an SC start on the same lap, the stronger one):

    all 46 races, 2,428 green laps: SC 22 (0.0091 a green lap), VSC 16 (0.0066), red flag 6 (0.0025);
    the 41 dry races the model is fitted on, 2,167 laps at risk: SC 16 (0.0074 a lap), VSC 12 (0.0055), red
    flag 4 (0.0018): 0.74 neutralisations per 50 laps. 18 of the 41 had one after lap 5.
    SC lengths 1-6 laps (median 3), VSC 1-3 (mostly 1), red flag always 1 lap of the leader.
    They come at any part of the race (started in each fifth: 6, 9, 12, 8, 9).
    NOT a circuit effect we can see: 22 circuits ran in both seasons and their event counts correlate
    -0.03 between 2023 and 2024 (5 had one in both, 10 in one, 7 in neither). One hazard for every circuit.
    The hazard is not independent of retirements: 30 of the 44 periods have a car retiring within a lap
    of their start (a stopped car causes them). They are drawn apart from retirements (RetirementModel, below):
    that link is a known limit.

The hazards are fitted on DRY races only (the races the chances are judged on); the lengths too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator import race as race_module
from src.simulator.race import KIND_RANK, Race

KINDS = (SAFETY_CAR, VIRTUAL_SAFETY_CAR, RED_FLAG)
Period = Tuple[str, int, int]                  # kind, first lap, last lap (the leader's laps)


@dataclass(frozen=True)
class Observed:
    """One race's neutralisations after its first start lap."""
    total_laps: int
    start_lap: int
    events: Tuple[Tuple[str, int, int], ...]   # (kind, first lap, laps it lasted), after the start lap


def events_after(periods: pd.DataFrame, start_lap: int) -> Tuple[Tuple[str, int, int], ...]:
    """The periods that start after `start_lap`, one per starting lap (the stronger kind where two start
    together: Qatar 2024 lap 40 had a VSC and then a safety car), as (kind, first lap, laps)."""
    best: Dict[int, Tuple[str, int]] = {}
    for kind, first, last in zip(periods["kind"], periods["first_lap"], periods["last_lap"]):
        first, length = int(first), int(last) - int(first) + 1
        if first > start_lap and (first not in best or KIND_RANK[kind] > KIND_RANK[best[first][0]]):
            best[first] = (kind, length)
    return tuple((kind, first, length) for first, (kind, length) in sorted(best.items()))


@dataclass
class EventModel:
    """The hazards and lengths of neutralisations. Draw with `draw`; fit with `fit`."""
    hazard: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in KINDS})   # per lap at risk
    lengths: Dict[str, np.ndarray] = field(default_factory=lambda: {k: np.zeros(0, int) for k in KINDS})
    laps_at_risk: int = 0
    races: int = 0

    @property
    def total_hazard(self) -> float:
        return float(sum(self.hazard.values()))

    @classmethod
    def from_observed(cls, observed: Iterable[Observed]) -> "EventModel":
        """
        hazard of a kind = its events / laps at risk. A lap is at risk when it is not inside a neutralisation
        that began earlier: the laps after the start lap, minus the later laps of each event (the lap an event
        starts on counts: it was at risk). Lengths: every event's laps, by kind.
        """
        observed = list(observed)
        at_risk = sum(o.total_laps - o.start_lap - sum(length - 1 for _, _, length in o.events) for o in observed)
        counts = {k: sum(1 for o in observed for kind, _, _ in o.events if kind == k) for k in KINDS}
        lengths = {k: np.array([length for o in observed for kind, _, length in o.events if kind == k], dtype=int)
                   for k in KINDS}
        hazard = {k: counts[k] / at_risk if at_risk > 0 else 0.0 for k in KINDS}
        return cls(hazard=hazard, lengths=lengths, laps_at_risk=int(at_risk), races=len(observed))

    @classmethod
    def fit(cls, races: Iterable[Race]) -> "EventModel":
        """The hazards and lengths of these (training) races; dry races only, and only those with a start lap."""
        observed = []
        for race in races:
            start = race_module.first_start_lap(race) if not race.mixed else None
            if start is not None:
                observed.append(Observed(race.total_laps, start, events_after(race.periods, start)))
        return cls.from_observed(observed)

    def draw(self, rng: np.random.Generator, after_lap: int, total_laps: int) -> List[Period]:
        """
        One race's neutralisations from lap `after_lap` + 1 to the flag. The next event is the first lap at
        risk with a success at chance h (a geometric waiting time); once it has run its length the next
        waiting time starts after it.
        """
        h = min(self.total_hazard, 1.0)
        if h <= 0.0:
            return []
        shares = np.cumsum([self.hazard[k] for k in KINDS]) / self.total_hazard
        periods: List[Period] = []
        lap = after_lap
        while True:
            lap += int(rng.geometric(h))
            if lap > total_laps:
                return periods
            kind = KINDS[min(int(np.searchsorted(shares, rng.random())), len(KINDS) - 1)]
            pool = self.lengths[kind]
            length = int(pool[rng.integers(len(pool))]) if len(pool) else 1
            last = min(lap + length - 1, total_laps)
            periods.append((kind, lap, last))
            lap = last

    def describe(self) -> str:
        names = {SAFETY_CAR: "safety car", VIRTUAL_SAFETY_CAR: "VSC", RED_FLAG: "red flag"}
        parts = []
        for k in KINDS:
            n = len(self.lengths[k])
            median = f", median {int(np.median(self.lengths[k]))} laps" if n else ""
            parts.append(f"{names[k]} {self.hazard[k]:.4f} a lap ({n}{median})")
        return (f"Events drawn from {self.races} dry races, {self.laps_at_risk} laps at risk: " + "; ".join(parts)
                + f"; {50 * self.total_hazard:.2f} neutralisations per 50 laps")


NO_EVENTS = EventModel()          # nothing is ever drawn: the clean race


@dataclass
class FixedEvents(EventModel):
    """The same neutralisations in every run: a real race's own. With the real plans this is step 3a's
    Monte Carlo run through world.run (the same ranks, run for run), so 3a and 3b share one machine."""
    periods: Tuple[Period, ...] = ()

    @classmethod
    def of_race(cls, race: Race) -> "FixedEvents":
        return cls(periods=tuple((k, int(a), int(b)) for k, a, b in
                                 zip(race.periods["kind"], race.periods["first_lap"], race.periods["last_lap"])))

    def draw(self, rng: np.random.Generator, after_lap: int, total_laps: int) -> List[Period]:
        return [p for p in self.periods if p[2] > after_lap]


@dataclass
class Ongoing(EventModel):
    """
    Seen from the pit wall at the end of a lap (step 4, the game): the neutralisation the race is in on the NEXT lap
    is known (one coming out, or one still out), how long it will last is not. Its length is drawn from its kind's
    lengths, at least as long as it has already lasted; the ones after it are drawn by `base`.
    """
    base: EventModel = field(default_factory=EventModel)
    kind: str = SAFETY_CAR
    first: int = 1              # the leader's lap it came out on
    next_lap: int = 1           # the lap after the moment: it lasts at least until then

    def draw(self, rng: np.random.Generator, after_lap: int, total_laps: int) -> List[Period]:
        lengths = self.base.lengths.get(self.kind, np.zeros(0, int))
        at_least = self.next_lap - self.first + 1
        possible = lengths[lengths >= at_least]
        length = int(possible[rng.integers(len(possible))]) if len(possible) else at_least
        last = min(self.first + length - 1, total_laps)
        return [(self.kind, self.first, last)] + self.base.draw(rng, last, total_laps)


@dataclass
class RetirementModel:
    """
    Cars that do not take the flag. One hazard for every car and every lap after the start: the chance of
    retiring on a given lap. MEASURED (2023 + 2024, dry races, from each race's first start lap): 75 of 791 cars
    retired later, over 40,835 car-laps: 0.0018 a car-lap, i.e. 9 % of a car's next 50 laps; by thirds of the
    race 0.0019, 0.0012, 0.0023 (no trend we can use); 14 of the 75 in the last two laps (not classified).
    Not modelled: some teams retire more (a team's 2023 and 2024 rates correlate 0.69 over 8 teams, driven by
    one team), and retirements cause neutralisations (30 of 44 started within a lap of one): drawn apart.
    """
    hazard: float = 0.0             # per car-lap
    car_laps: int = 0
    retirements: int = 0

    @classmethod
    def fit(cls, races: Iterable[Race]) -> "RetirementModel":
        """Retirements over car-laps at risk of these (training) races: dry, with a start lap. A car at risk from
        the lap after the start; a retiring car's last lap counts (it was at risk on it)."""
        retired = car_laps = 0
        for race in races:
            start = race_module.first_start_lap(race) if not race.mixed else None
            if start is None:
                continue
            for driver in race_module.start_state(race, start).index:
                last = race.last_lap(driver)
                gone = driver not in race.finishers
                retired += int(gone)
                car_laps += max(0, (last if gone else race.total_laps) - start)
        return cls(hazard=retired / car_laps if car_laps else 0.0, car_laps=int(car_laps), retirements=int(retired))

    def draw(self, rng: np.random.Generator, last_laps: Dict[str, int], after_lap: int, total_laps: int) -> Dict[str, int]:
        """{driver: the last lap he drives} for the cars that retire before the flag. last_laps: each car's last
        planned lap. A car retires on lap after_lap + G, G geometric, if that is before the end of its plan."""
        if self.hazard <= 0.0:
            return {}
        gone = {}
        for driver, last in last_laps.items():
            lap = after_lap + int(rng.geometric(min(self.hazard, 1.0)))
            if lap < min(last, total_laps):
                gone[driver] = lap
        return gone

    def describe(self) -> str:
        return (f"Retirements: {self.retirements} in {self.car_laps} car-laps, {self.hazard:.5f} a car-lap "
                f"({50 * self.hazard:.1%} of a car's next 50 laps)")
