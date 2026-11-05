"""
The teams in a drawn race (src/simulator/teams.py) - Phase 4 step 3b

The plans a replay reads (race.plan) are the REAL decisions, and some of them were made BECAUSE of the real
events. In the dry development races, 180 of the 1,266 tyre changes after the first start lap (14 %) were made
under a safety car (117), a red flag (41) or a VSC (22), in 15 of the 41 races, and 129 of the 215 cars that
boxed under a safety car or VSC had no stop left afterwards: the safety car stop WAS their stop. If the events
are drawn, such a car would pay a full 22 s stop on the real lap in every run that has no safety car there.
So the drawn race has two parts, both measured on the training races:

1. THE BASE PLAN (`decontaminate`): what each team would have done without the events. A stop made under a
   neutralisation moves to its natural lap: the lap at which the tyre reaches the age at which that compound
   is usually stopped on under green (the median of the training races' green stops: hard 23 laps, medium 18,
   soft 15), never closer than MIN_STINT_LAPS to the next stop; later than the last LATE_LAPS laps it is dropped
   (kept at the last possible lap when dropping it would break the two-compounds rule). A drive-through under a
   neutralisation is dropped (the field was led through the pit lane). Everything else stays: the plans are
   still the real decisions.

2. THE ANSWER (`BoxModel`, `Draw.react`): when a drawn neutralisation comes out, every car on the leader's lap
   decides whether to box. A car that boxes moves its next planned stop to the neutralisation (the same set of
   tyres), or takes one more stop when it has none left (soft with 15 laps or fewer to go, else hard), on the
   first lap with the measured share, else the next. Its changed laps are re-priced with rung 2's tyre terms
   (compound offset + wear x tyre age), nothing else about its pace changes.

BoxModel, measured on the cars of the 32 safety cars / VSCs / red flags of the 18 dry development races that
had one (race-grouped 5-fold, out of fold; log loss per car, a first ladder on 615 cars):

    by kind only                           0.637   (safety car 44 % boxed, VSC 16 %, red flag 64 %)
    + young tyres (5 laps old or less)     0.511   (7 % of them boxed under a safety car; older ones 55-85 %)
    + age against the compound's usual     0.497
    + position on the road                 0.491   (first three 61 %, 16th and lower 28 %; at the same tyre age
                                                    85 % against 45 %)
    + a mood for each event                0.425   (sigma 1.3 on the log-odds scale in every fold)

and with this module's own rows (570 cars; age on the lap it comes out on, position by the road order) the
held-out log likelihood per car, events integrated over their mood, is -0.349 against -0.490 for the rate of
each kind; sigma 1.4 to 2.3 across the five folds, and fixing it at 1.35 scores the same (-0.349): the data
cannot say more than that.

The mood is the big one: the share of cars that box ranges from 0 % to 100 % between events (sd 0.32 against 0.24
if every car decided alone). Some safety cars find nearly everybody boxing, some nobody; the cars of one event
move together (a closed pit lane, a lap too late to gain, the front-runners' call). Drawing every car alone
would make every simulated safety car an average one, and the chances overconfident. Laps left, a flag for
a late race and "stops left in the plan" were tried and left out: one race (everybody boxes, five laps to go)
moved the first, and the last carries the real reaction in through the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator import engine, plans, race as race_module
from src.simulator.race import DEPLOY, GREEN, SINGLE, Race

SOFT, MEDIUM, HARD, OTHER = 0, 1, 2, 3
CODE = {"SOFT": SOFT, "MEDIUM": MEDIUM, "HARD": HARD}
NAME = {SOFT: "SOFT", MEDIUM: "MEDIUM", HARD: "HARD"}
DEFAULT_MEDIAN_AGE = {"SOFT": 15.0, "MEDIUM": 18.0, "HARD": 23.0}     # 2023 + 2024 green stops (laps of tyre age)
MIN_AGES = 5                    # green stops needed before a compound uses its own median
YOUNG_AGE = 5                   # laps: tyres this young are almost never changed
MIN_STINT_LAPS = 8              # a natural stop is at least this many laps before the next stop
LATE_LAPS = 4                   # no natural stop in the last laps
MIN_LAPS_TO_BOX = 3             # a stop with fewer laps left gains nothing
SOFT_LAPS_LEFT = 15             # an extra stop with this many laps or fewer left fits softs, else hards (measured)
RATIO_CAP, POSITION_CAP = 1.5, 20
MIN_PERIODS = 6                 # fewer events to learn from: the measured rates below, for every car
FALLBACK_RATES = {SAFETY_CAR: 0.44, VIRTUAL_SAFETY_CAR: 0.16, RED_FLAG: 0.64}
FALLBACK_SIGMA = 1.3
FIRST_LAP_SHARE = {SAFETY_CAR: 0.94, VIRTUAL_SAFETY_CAR: 0.81, RED_FLAG: 1.0}    # of the cars that box: on the first lap (109 of 116, 17 of 21)
SHARE_PRIOR = 2.0               # pseudo-counts pulling a kind's first-lap share to the measured one
PENALTY = 1.0                   # L2 on the standardised slopes
SIGMA_PRIOR = 1.0               # the mood's sd (log-odds) is pulled towards this, mildly
NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(24)                          # integrates over the mood
WEIGHTS = WEIGHTS / WEIGHTS.sum()
FEATURES = ("safety car", "VSC", "red flag", "young tyres", "age against usual", "position")
BOX_COLUMNS = ["season", "round", "lap", "kind", "length", "driver_code", "position", "compound", "age", "boxed", "offset"]


# --- how old a tyre usually is when it is changed ----------------------------------------------------------

@dataclass
class GreenAges:
    """The tyre age (laps, on the in-lap) of every tyre change made under green, by compound: when a team
    'normally' stops. Dry races only."""
    ages: Dict[str, np.ndarray]

    @classmethod
    def fit(cls, races: Iterable[Race]) -> "GreenAges":
        found: Dict[str, List[float]] = {}
        for race in races:
            if race.mixed:
                continue
            kind = race_module.lap_conditions(race)["kind"]
            rows = race.laps[race.laps["is_tyre_change"].astype(bool) & (race.laps["lap_number"] > 1)]
            for compound, lap, age in zip(rows["compound"], rows["lap_number"], rows["tyre_age"]):
                if isinstance(compound, str) and pd.notna(age) and kind.get(int(lap), GREEN) == GREEN:
                    found.setdefault(compound, []).append(float(age))
        return cls({c: np.sort(np.array(v)) for c, v in found.items()})

    def median(self, compound: Optional[str]) -> float:
        ages = self.ages.get(compound, np.zeros(0)) if isinstance(compound, str) else np.zeros(0)
        if len(ages) >= MIN_AGES:
            return float(np.median(ages))
        return DEFAULT_MEDIAN_AGE.get(compound, 20.0)


# --- how old a set can get (step 4: the game never plans past it) ----------------------------------------

LIFE_QUANTILE = 0.99            # a compound's life: the age teams almost never ran a set past
DEFAULT_LIFE = {"SOFT": 29.0, "MEDIUM": 41.0, "HARD": 57.0}     # 2023 + 2024 dry races (below)
MIN_SETS = 20                   # sets needed before a compound uses its own life


@dataclass
class TyreLife:
    """
    How old a set of each dry compound gets before it comes off: its tyre age at the end of every stint (to a stop,
    to the flag or to a retirement) in dry races. A compound's life is the age teams almost never ran a set past
    (LIFE_QUANTILE, rounded down).

    MEASURED (2023 + 2024, the 41 dry races; 370 soft, 878 medium, 932 hard sets): soft median 12.5 laps, 95 % off by
    27, 99 % by 29, the longest 31; medium 18 / 34 / 41, the longest 77; hard 25 / 46 / 58, the longest 78 (the two
    longest of both: Monaco 2024, a red flag on lap 1 and 77 laps on one set). Rung 2's straight-line wear was learned
    on these sets, so past a compound's life it is a guess: a soft set run 34 laps (Bahrain's 24 s stop made "softs
    to the flag" from lap 22 cheaper than a second stop, by the straight line) is a set no team ever ran. The last laps
    before a stop also run above the line (one lap before: soft +0.27 s, medium +0.32, hard +0.46 against the stint's
    typical; two laps before +0.16 / +0.19 / +0.33): teams stop as a set goes off, so what happens after is never seen.
    """
    ages: Dict[str, np.ndarray]

    @classmethod
    def fit(cls, races: Iterable[Race]) -> "TyreLife":
        found: Dict[str, List[float]] = {}
        for race in races:
            if race.mixed:
                continue
            laps = race.laps[race.laps["stint_number"].notna() & race.laps["compound"].isin(list(DEFAULT_LIFE))]
            ends = laps.groupby(["driver_code", "stint_number"]).agg(compound=("compound", "first"),
                                                                     age=("tyre_age", "max"))
            for compound, age in zip(ends["compound"], ends["age"]):
                if pd.notna(age):
                    found.setdefault(compound, []).append(float(age))
        return cls({c: np.sort(np.array(v)) for c, v in found.items()})

    def limit(self, compound: Optional[str]) -> float:
        """The oldest a set of this compound may get (laps of tyre age)."""
        ages = self.ages.get(compound, np.zeros(0)) if isinstance(compound, str) else np.zeros(0)
        if len(ages) >= MIN_SETS:
            return float(np.floor(np.quantile(ages, LIFE_QUANTILE)))
        return DEFAULT_LIFE.get(compound, 99.0)

    def limits(self) -> Dict[str, float]:
        return {c: self.limit(c) for c in DEFAULT_LIFE}


# --- the base plan: what the teams would have done without the real events -------------------------------

def decontaminate(race: Race, planned: pd.DataFrame, conditions: pd.DataFrame, ages: GreenAges) -> pd.DataFrame:
    """The plan with the real events taken out (see the module docstring): every tyre change made under a
    safety car, VSC or red flag moved to its natural lap, every drive-through under one dropped. conditions:
    the real race's lap_conditions."""
    kinds = conditions["kind"].to_dict()
    plan = planned
    end = min(race.total_laps, int(planned["lap_number"].max())) if len(planned) else race.total_laps
    for driver in sorted(set(planned["driver_code"])):
        last = int(planned.loc[planned["driver_code"] == driver, "lap_number"].max())
        seen = set()
        while True:
            mine = plan[(plan["driver_code"] == driver) & plan["is_in_lap"]]
            todo = [r for r in mine.itertuples(index=False)
                    if kinds.get(int(r.lap_number), GREEN) != GREEN and int(r.lap_number) not in seen]
            if not todo:
                break
            row = todo[0]
            n = int(row.lap_number)
            seen.add(n)
            if row.pit_kind == plans.PASS_THROUGH:
                plan = plans.stay_out(plan, driver, n)
                continue
            if row.pit_kind != plans.TYRE_CHANGE or not math.isfinite(float(row.tyre_age)):
                continue
            target = n + max(0, int(round(ages.median(row.compound) - float(row.tyre_age))))
            later = [lap for lap in plans.stops_of(plan, driver) if lap > n]
            if later:
                target = min(target, later[0] - MIN_STINT_LAPS)
            target = max(target, n)
            if target > min(end, last) - LATE_LAPS:                    # too late to be worth a stop
                dropped = plans.stay_out(plan, driver, n)
                if not plans.new_problems(race, planned, dropped, driver):
                    plan = dropped
                    continue
                target = min(end, last) - LATE_LAPS - 1                   # the two-compounds rule needs it
            if target > n:
                plan = plans.move_stop(plan, driver, n, target)
                seen.add(target)
    return plan.reset_index(drop=True)


def stops_for_run_on(race: Race, planned: pd.DataFrame, ages: GreenAges) -> pd.DataFrame:
    """
    The cars that really retired, run on to the flag (race.plan(run_on=True)) have their last tyres carried on for
    the laps they never drove: far past any tyre's life. Give each the stops a team makes: a new set when the
    tyre reaches the usual age of its compound (softs with 15 laps or fewer to go, else hards), not in the last
    LATE_LAPS laps. The laps they really drove are never touched.
    """
    plan = planned
    for driver in sorted(set(planned["driver_code"])):
        if driver in race.finishers:
            continue
        after = race.last_lap(driver)                       # only laps after the last stop made (or the last real lap)
        while True:
            mine = plan[(plan["driver_code"] == driver) & (plan["lap_number"] > after)
                        & (plan["lap_number"] < race.total_laps - LATE_LAPS)]
            usual = mine["compound"].map(ages.median)
            due = mine[(mine["tyre_age"] >= usual) & ~mine["is_wet_tyre"].astype(bool)]
            if due.empty:
                break
            after = int(due["lap_number"].min())            # the in-lap keeps the old age: look past it next time
            plan = plans.box(plan, driver, after, NAME[SOFT if race.total_laps - after <= SOFT_LAPS_LEFT else HARD])
    return plan.reset_index(drop=True)


# --- who boxes when a neutralisation comes out --------------------------------------------------------------

def box_rows(race: Race) -> pd.DataFrame:
    """
    One row per car on the leader's lap when each neutralisation of the race came out (the laps the engine
    calls the hook on): kind, length, position on the road, tyre and age on the lap it came out on, and
    whether the car made a tyre change during it (and on which lap of it).
    """
    status = race_module.status_of(zip(race.periods["kind"], race.periods["first_lap"], race.periods["last_lap"]),
                                   race.total_laps)
    by_driver = {d: g.sort_values("lap_number").set_index("lap_number") for d, g in race.laps.groupby("driver_code")}
    rows = []
    for lap, (kind, phase) in sorted(status.items()):
        if phase not in (DEPLOY, SINGLE) or lap < 3 or lap > race.total_laps - 2:
            continue
        length = engine.laps_neutralised(status, lap)
        snapshot = race_module.snapshot(race, lap - 1)
        lead = snapshot.index[snapshot["laps_done"] == lap - 1]
        for position, driver in enumerate(lead, start=1):
            mine = by_driver.get(driver)
            if mine is None or lap not in mine.index:
                continue
            here = mine.loc[lap]
            if bool(here["is_wet_tyre"]) or not isinstance(here["compound"], str) or pd.isna(here["tyre_age"]):
                continue
            during = mine.loc[lap:lap + length - 1]
            boxed = during.index[during["is_in_lap"].astype(bool) & (during["pit_kind"] == plans.TYRE_CHANGE)]
            rows.append({"season": race.season, "round": race.round, "lap": lap, "kind": kind, "length": length,
                         "driver_code": driver, "position": position, "compound": here["compound"],
                         "age": float(here["tyre_age"]), "boxed": len(boxed) > 0,
                         "offset": int(boxed[0]) - lap if len(boxed) else -1})
    return pd.DataFrame(rows, columns=BOX_COLUMNS)


def box_inputs(kind, age, compound, position, ages: GreenAges) -> np.ndarray:
    """The BoxModel's inputs, one row per car: the three kinds, young tyres, age against the compound's usual
    stopping age (capped), position (capped)."""
    kind, age, position = pd.Series(kind), np.asarray(age, float), np.asarray(position, float)
    usual = np.array([ages.median(c) for c in pd.Series(compound)])
    return np.column_stack([kind.eq(SAFETY_CAR), kind.eq(VIRTUAL_SAFETY_CAR), kind.eq(RED_FLAG), age <= YOUNG_AGE,
                            np.minimum(age / usual, RATIO_CAP), np.minimum(position, POSITION_CAP)]).astype(float)


@dataclass
class BoxModel:
    """
    P(a car on the leader's lap boxes when a neutralisation comes out):

        logistic( b0 + b . standardised inputs + sigma x z )         z ~ N(0, 1), one draw per event

    z is the event's mood, shared by every car in it. Fitted on the cars of the training races' events by
    maximum likelihood of whole events with the mood integrated out (24-point Gauss-Hermite), a mild
    penalty on the slopes and on log sigma. Fewer than MIN_PERIODS events: the measured rate of each kind.
    """
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray            # intercept first
    sigma: float
    first_lap_share: Dict[str, float]
    cars: int = 0
    periods: int = 0

    def __post_init__(self) -> None:
        # chance() is called for every car of every event of every run: fold the standardising into the weights.
        self._w = self.coef[1:] / self.scale
        self._b = float(self.coef[0] - self._w @ self.mean)

    @classmethod
    def fallback(cls, shares: Optional[Dict[str, float]] = None) -> "BoxModel":
        rates = [FALLBACK_RATES[k] for k in (SAFETY_CAR, VIRTUAL_SAFETY_CAR, RED_FLAG)]
        logit = [math.log(r / (1.0 - r)) for r in rates]
        return cls(mean=np.zeros(len(FEATURES)), scale=np.ones(len(FEATURES)),
                   coef=np.array([0.0] + logit + [0.0, 0.0, 0.0]), sigma=FALLBACK_SIGMA,
                   first_lap_share=dict(shares or FIRST_LAP_SHARE))

    @classmethod
    def fit(cls, rows: pd.DataFrame, ages: GreenAges) -> "BoxModel":
        """rows: box_rows() of the training races, stacked."""
        x = rows.dropna(subset=["age", "position"])
        groups, uniques = pd.factorize(pd.Series(list(zip(x["season"], x["round"], x["lap"])), dtype=object)) if len(x) else (np.zeros(0, int), [])
        shares = {k: (float(((x["kind"] == k) & x["boxed"] & (x["offset"] == 0)).sum()) + SHARE_PRIOR * FIRST_LAP_SHARE[k])
                  / (float(((x["kind"] == k) & x["boxed"]).sum()) + SHARE_PRIOR) for k in FIRST_LAP_SHARE}
        shares[RED_FLAG] = 1.0                                       # a red flag lasts one lap of the leader
        if len(uniques) < MIN_PERIODS:
            return cls.fallback(shares)
        X = box_inputs(x["kind"], x["age"], x["compound"], x["position"], ages)
        mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
        y = x["boxed"].astype(float).to_numpy()
        k = Z.shape[1]

        def minus_log_posterior(par):
            theta, log_sigma = par[:k], par[k]
            z = (Z @ theta)[:, None] + math.exp(log_sigma) * NODES[None, :]                  # cars x nodes
            log_p = -np.logaddexp(0.0, -z) * y[:, None] - np.logaddexp(0.0, z) * (1.0 - y)[:, None]
            per_event = np.zeros((len(uniques), len(NODES)))
            np.add.at(per_event, groups, log_p)                       # an event's cars share one mood
            top = per_event.max(axis=1, keepdims=True)
            log_lik = (top[:, 0] + np.log((np.exp(per_event - top) * WEIGHTS[None, :]).sum(axis=1))).sum()
            return -log_lik + PENALTY * (theta[1:] ** 2).sum() + 0.5 * (log_sigma - math.log(SIGMA_PRIOR)) ** 2

        start = np.zeros(k + 1)
        start[k] = math.log(0.5)
        best = minimize(minus_log_posterior, start, method="L-BFGS-B").x
        return cls(mean=mean, scale=scale, coef=best[:k], sigma=math.exp(best[k]), first_lap_share=shares,
                   cars=len(x), periods=len(uniques))

    def chance(self, kind: str, age: float, compound: Optional[str], position: float, mood: float,
               ages: GreenAges) -> float:
        """P(box) for one car in an event with this mood (a standard normal draw)."""
        usual = ages.median(compound)
        x = (kind == SAFETY_CAR, kind == VIRTUAL_SAFETY_CAR, kind == RED_FLAG, age <= YOUNG_AGE,
             min(age / usual, RATIO_CAP), min(position, POSITION_CAP))
        z = self._b + sum(w * v for w, v in zip(self._w, x)) + self.sigma * mood
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))

    def describe(self) -> str:
        slopes = ", ".join(f"{n} {c / s:+.2f}" for n, c, s in zip(FEATURES, self.coef[1:], self.scale))
        return (f"Box rule from {self.cars} cars in {self.periods} neutralisations: mood sd {self.sigma:.2f} "
                f"(log-odds); slopes per unit: {slopes}; boxed on the first lap: "
                + ", ".join(f"{k} {v:.0%}" for k, v in self.first_lap_share.items()))


# --- the tyres' share of a lap time, to re-price a changed plan ---------------------------------------------

@dataclass(frozen=True)
class TyreTerms:
    """Rung 2's tyre terms for the dry compounds: offset (against the medium) + wear x tyre age, in seconds,
    looked up by compound code. Wets, unknown tyres: nothing. The forecast for a changed plan is the
    forecast for the planned one plus the change in these (the trees' correction is not re-made)."""
    offset: np.ndarray
    wear: np.ndarray

    @classmethod
    def from_rung2(cls, rung2) -> "TyreTerms":
        coef = rung2.coefficients()
        return cls(offset=np.array([coef.get("offset SOFT", 0.0), 0.0, coef.get("offset HARD", 0.0), 0.0], dtype=float),
                   wear=np.array([coef.get(f"wear {c}", 0.0) for c in ("SOFT", "MEDIUM", "HARD")] + [0.0], dtype=float))

    def value(self, code: np.ndarray, age: np.ndarray) -> np.ndarray:
        return self.offset[code] + self.wear[code] * age


# --- one start's cars, and one run's teams ----------------------------------------------------------------

@dataclass(frozen=True)
class Stop:
    """One pit passage of a plan (an in-lap), and for a tyre change the set fitted on the out-lap."""
    in_lap: int
    kind: str                       # the plan's pit_kind: tyre_change, pass_through, ended, unknown
    code: int = OTHER               # the compound fitted (SOFT, MEDIUM, HARD), for a tyre change
    compound: Optional[str] = None
    age: float = 1.0                # that set's tyre age on the out-lap (1: new)


@dataclass
class Car:
    """A car's base plan as arrays (one entry per planned lap)."""
    driver: str
    rows: np.ndarray                # where its laps are in the planned table, in lap order
    laps: np.ndarray
    code: np.ndarray                # compound code of each lap
    compound: np.ndarray
    age: np.ndarray
    terms: np.ndarray               # the tyre terms of those laps (s)
    stops: Tuple[Stop, ...]
    first_is_out: bool              # the first planned lap is an out-lap (the car stopped on the start lap)


def build_cars(planned: pd.DataFrame, terms: TyreTerms) -> Dict[str, Car]:
    """Every car of the (base) plan as arrays."""
    cars = {}
    drivers = planned["driver_code"].to_numpy()
    lap_number = planned["lap_number"].to_numpy(int)
    for driver in sorted(set(drivers)):
        rows = np.flatnonzero(drivers == driver)
        rows = rows[np.argsort(lap_number[rows], kind="stable")]
        g = planned.iloc[rows]
        laps, compound, age = g["lap_number"].to_numpy(int), g["compound"].to_numpy(object), g["tyre_age"].to_numpy(float)
        code = np.array([CODE.get(c, OTHER) if isinstance(c, str) and not w else OTHER
                         for c, w in zip(compound, g["is_wet_tyre"].to_numpy(bool))], dtype=int)
        kinds, stops = g["pit_kind"].to_numpy(object), []
        for i in np.flatnonzero(g["is_in_lap"].to_numpy(bool)):
            if kinds[i] == plans.TYRE_CHANGE and i + 1 < len(laps):
                stops.append(Stop(int(laps[i]), plans.TYRE_CHANGE, int(code[i + 1]), compound[i + 1], float(age[i + 1])))
            else:
                stops.append(Stop(int(laps[i]), str(kinds[i]) if kinds[i] is not None else "unknown"))
        cars[driver] = Car(driver=driver, rows=rows, laps=laps, code=code, compound=compound, age=age,
                           terms=terms.value(code, age), stops=tuple(stops),
                           first_is_out=bool(g["is_out_lap"].iloc[0]))
    return cars


def tyres(car: Car, stops: Sequence[Stop]) -> Tuple[np.ndarray, np.ndarray]:
    """The compound codes and tyre ages of a car's laps with these stops. Laps up to the first stop that differs
    from the base plan keep the base plan's own (FastF1's tyre record is not perfectly regular); after it, a
    new set starts on the out-lap of each tyre change and every other lap adds one to the age."""
    changed = set(car.stops) ^ set(stops)
    code, age = car.code.copy(), car.age.copy()
    if not changed:
        return code, age
    first = min(s.in_lap for s in changed)
    by_in_lap = {s.in_lap: s for s in stops}
    for i in range(1, len(car.laps)):
        lap = int(car.laps[i])
        if lap <= first:
            continue
        before = by_in_lap.get(lap - 1)
        if before is not None and before.kind == plans.TYRE_CHANGE:
            code[i], age[i] = before.code, before.age
        else:
            code[i], age[i] = code[i - 1], age[i - 1] + 1.0
    return code, age


class Teams:
    """One start's teams: the base plan's cars, and how they answer a neutralisation."""

    def __init__(self, planned: pd.DataFrame, box: BoxModel, ages: GreenAges, terms: TyreTerms, total_laps: int) -> None:
        self.planned = planned.reset_index(drop=True)
        self.box, self.ages, self.terms, self.total_laps = box, ages, terms, total_laps
        self.cars = build_cars(self.planned, terms)
        self.index = {d: i for i, d in enumerate(sorted(self.cars))}     # each car's number in a common draw
        self.passage = engine._passage_of_out_lap(self.planned)

    def last_laps(self) -> Dict[str, int]:
        """Each car's last planned lap."""
        return {d: int(car.laps[-1]) for d, car in self.cars.items()}

    def begin(self, pace: np.ndarray, rng: np.random.Generator, retire: Optional[Dict[str, int]] = None,
              common: bool = False, decided: Optional[Tuple[str, int]] = None) -> "Draw":
        """One run: the engine's plans for this run's pace (aligned with the planned table), ready to be changed.
        retire: {driver: the last lap he drives} for the cars that retire in this run. common, decided: see Draw."""
        return Draw(self, pace, rng, retire, common, decided)


class Draw:
    """
    One run's teams: the plans the engine reads and the answers to the neutralisations that come out.

    common=True (step 4): each neutralisation draws one number per car of the start, used or not, so that two runs
    that differ in one car's plan draw the same answers for everyone else (the same futures for every option).
    decided=(driver, lap): that car has made its own call for a neutralisation coming out up to `lap` (the player's
    car in the game): it does not answer it; later ones it answers like any team.
    """

    def __init__(self, teams: Teams, pace: np.ndarray, rng: np.random.Generator,
                 retire: Optional[Dict[str, int]] = None, common: bool = False,
                 decided: Optional[Tuple[str, int]] = None) -> None:
        self.teams, self.rng, self.pace = teams, rng, np.asarray(pace, dtype=float)
        self.common, self.decided = common, decided
        self.plans = engine._plans(teams.planned, self.pace)
        self.passage = dict(teams.passage)
        self.stops = {d: car.stops for d, car in teams.cars.items()}
        self.last = teams.last_laps()                                    # a retiring car's plan ends on its last lap
        for driver, lap in (retire or {}).items():
            self.last[driver] = lap
            plan = self.plans[driver]
            for later in [n for n in plan if n > lap]:
                del plan[later]
        self.tyres: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}        # the cars whose plan has been changed
        self.boxed: List[Tuple[str, int]] = []                           # (driver, in-lap) of every answer, for tests

    def tyre_at(self, driver: str, index: int) -> Tuple[int, float]:
        code, age = self.tyres.get(driver, (self.teams.cars[driver].code, self.teams.cars[driver].age))
        return int(code[index]), float(age[index])

    def react(self, event: engine.Neutralisation) -> None:
        """The engine's on_neutralisation hook: every car on the leader's lap decides whether to box."""
        teams, rng, box = self.teams, self.rng, self.teams.box
        lap, total = event.lap, teams.total_laps
        if total - lap + 1 < MIN_LAPS_TO_BOX:
            return
        mood = float(rng.standard_normal())
        numbers = rng.random((2, len(teams.index))) if self.common else None    # [0]: box or not, [1]: on which lap
        lead_lap, position = event.done[event.leader], 0
        for driver in event.order:
            if event.done[driver] != lead_lap:
                continue
            position += 1
            car = teams.cars.get(driver)
            if car is None:
                continue
            index = lap - int(car.laps[0])
            if index < 0 or lap > self.last[driver]:
                continue                                                  # no lap to drive on: retired
            code, age = self.tyre_at(driver, index)
            stops = self.stops[driver]
            if code == OTHER or any(lap <= s.in_lap < lap + event.length for s in stops):
                continue                                                  # wets, or it stops under it anyway
            if self.decided is not None and driver == self.decided[0] and lap <= self.decided[1]:
                continue                                                  # its own call, made already
            p = box.chance(event.kind, age, NAME[code], position, mood, teams.ages)
            if numbers is not None:
                i = teams.index[driver]
                if numbers[0, i] >= p:
                    continue
                first = event.length == 1 or numbers[1, i] < box.first_lap_share[event.kind]
            else:
                if rng.random() >= p:
                    continue
                first = event.length == 1 or rng.random() < box.first_lap_share[event.kind]
            in_lap = lap if first else lap + 1
            if in_lap >= min(total, self.last[driver]):
                continue                                                  # cannot stop on its last lap
            self._box(driver, in_lap, lap, event)

    def _box(self, driver: str, in_lap: int, lap: int, event: engine.Neutralisation) -> None:
        """Move the car's next planned tyre change to `in_lap` (the same set), or add one when none is left."""
        stops = list(self.stops[driver])
        coming = [s for s in stops if s.kind == plans.TYRE_CHANGE and s.in_lap >= lap]
        if coming:
            stops.remove(coming[0])
            stops.append(replace(coming[0], in_lap=in_lap))
        else:
            left = self.teams.total_laps - in_lap
            code = SOFT if left <= SOFT_LAPS_LEFT else HARD
            stops.append(Stop(in_lap, plans.TYRE_CHANGE, code, NAME[code], 1.0))
        self.stops[driver] = tuple(sorted(stops, key=lambda s: s.in_lap))
        self.boxed.append((driver, in_lap))
        self._apply(driver, event)

    def _apply(self, driver: str, event: engine.Neutralisation) -> None:
        """Rebuild the engine's plan for the laps not yet driven: tyres, passages, and the pace re-priced."""
        teams, car = self.teams, self.teams.cars[driver]
        code, age = tyres(car, self.stops[driver])
        self.tyres[driver] = (code, age)
        pace = self.pace[car.rows] + teams.terms.value(code, age) - car.terms
        by_in_lap = {s.in_lap: s for s in self.stops[driver]}
        plan = event.plans[driver]
        for i, lap in enumerate(car.laps):
            lap = int(lap)
            if lap < event.lap or lap > self.last[driver]:
                continue                                                  # already driven, or never to be driven
            stop = by_in_lap.get(lap)
            before = by_in_lap.get(lap - 1)
            plan[lap] = (float(pace[i]), stop is not None, before is not None or (i == 0 and car.first_is_out),
                         stop.kind if stop is not None else None)
            if before is not None:
                event.out_lap_passage[(driver, lap)] = before.kind
            else:
                event.out_lap_passage.pop((driver, lap), None)
