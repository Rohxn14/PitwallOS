"""
The strategist game (src/simulator/game.py) - Phase 4 step 4, the text version

You sit on a team's pit wall. The game picks a dry race of 2023-2024 and plays it BLIND: no circuit, no year, no
names (the other cars are letters, in grid order), so what you remember of the real race cannot help; everything is
revealed at the flag. You take a car that finished the real race at the first lap the simulator can start from (lap 5
or so: it needs real laps to know each driver's pace), and the race runs on until a BIG MOMENT (Rohan's choice: "a
normal lap no, but when a tyre compound is old then yeah"):

    a safety car, VSC or red flag              the real ones, on the laps they really came (the race's own story)
    your tyres are getting old                 80 % of the age at which teams usually change that compound under
                                               green (2023-2024: soft 15, medium 18, hard 23 laps), once per set
    your planned stop is due                   the plan never boxes without asking you
    the car just ahead or just behind you      cover the undercut, or not (only your neighbours on the road, on your
    on the road, within 3 s, has just pitted   lap: Phase 2's undercut is the car directly ahead)

never with MIN_LAPS_TO_BOX laps or fewer left. Staying out means at least STAY_OUT_LAPS more laps before the next stop:
with "the next lap" instead, a planned stop was asked about again on every lap (in the bot's 123 games, 90 of the 184
planned-stop moments came the lap after another: Saudi 2023 Verstappen six laps running), the nagging Rohan ruled out. At each moment every option (stay out; box now for softs, mediums or
hards while a set is left) is shown as your chance to win, to finish on the podium, to score points, and your
average place (Rohan's choice: win / podium / points; the average place tells options apart when all three are
near zero), counted over FUTURES of the rest of the race: step 3c's world ('+ retirements': the safety cars, the
rivals' answers to them and the retirements drawn, DRS trains, the speed misses by place), from where YOUR race is,
with nothing of the real race's future in it. Every option is run on the SAME futures (world.run(common=True)): a
difference between two options is the options', not luck's.

What follows an option is the fastest way to the flag after it, in clear air (fastest_finish): which tyres and when
the next stop, by the simulator's own tyre model (rung 2's compound offsets and wear per lap of age, the circuit's pit
loss), keeping the rules (two dry compounds by the flag, no more stints on a compound than the allocation's sets) and
the TYRE LIFE (teams.TyreLife: no set older than teams almost ever ran one, 2023-2024: soft 29, medium 41, hard 57 laps;
a straight-line wear past it is a guess: Bahrain's 24 s stop once made 34 laps on softs "fastest"). "Box now for hards"
means: hards now, then the best finish from there. A set near the end of its life is a big moment of its own, with
only boxing on offer.

YOUR RACE (what really happens in the game) is the real race with your car changed:
    real      the safety cars, VSCs and red flags; every other team's plan; who retires; every driver's real pace
              per stint (pace.hindsight: rung 2's readings of the real laps). Yours: your driver's real level on
              each lap (the real stint's, tyres taken out) plus YOUR tyres' terms, so a call never changes a lap
              driven before it
    yours     your car's stops
    engine    every lap and every pass, drawn from one game's luck (engine.PassDraws)
At the flag: your place, the real team's plan run in the same race with the same luck, and what really happened.
The other teams do not answer your stops in your race (they keep their real plans): a known limit.

Why the odds can be trusted: they are step 3c's '+ retirements' chances, measured against what really happened in 41
races out of fold (calibration error 2-3 points; podium calls of 70-90 % came true 92 % of the time), from a pace
forecast that has seen the real laps up to the moment of the decision, as a strategist would, and nothing after it.

The models are fitted once on the development races (five race folds, every race played with the fold that never saw
it) and saved in the lake (lake.model_path(MODELS_NAME)); `--refit` fits them again. 2025 stays locked.

MEASURED (2026-10-21, the 41 dry development races, the saved models):
    A decision (500 futures x 4 options) takes 11-18 s; the forecast of the moment and the rivals' plans 0.6-0.8 s of it.
    The same futures make two options' points chances 6-9x steadier against each other (sd of the difference 0.25-0.34
    points against 1.7-2.3 with independent futures), the average place ~1.5-2x, the podium ~2x (once your car is in
    another place it meets other cars: that luck cannot all be shared).
    The bot on every race, three cars each (bot_games, 200 futures): 2-10 big moments a game, median 5; 20 s a game.
    Why the game stopped (123 games): old tyres 253, a neighbour pitted 191, planned stop 150, safety car 43, VSC 30,
    red flag 6; every game had old tyres, three in four a planned stop and a neighbour pitting, two in five a safety car.

    python -m src.simulator.game                       a blind race; choose your car from the order at the start
    python -m src.simulator.game --car 7               the car in 7th place at the start
    python -m src.simulator.game --season 2024 --round 13 --driver NOR     a race and a driver you name (not blind)
    python -m src.simulator.game --bot                 the computer plays: the option with the most points expected
    python -m src.simulator.game --evaluate            the bot on every dry race (3 cars each, 200 futures, ~45 min)
                                                       against the real teams' plans: data/reports/game_bot.csv
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.models import evaluate
from src.pipeline import lake
from src.simulator import calibration, engine, montecarlo, pace, plans, race as race_module, replay
from src.simulator import teams as teams_module, world as world_module
from src.simulator.events import Ongoing
from src.simulator.race import DEPLOY, GREEN, SINGLE, Race
from src.simulator.replay import FoldModels
from src.simulator.teams import CODE, MIN_LAPS_TO_BOX, MIN_STINT_LAPS, OTHER, TyreTerms

MODELS_NAME = "sim_game"        # lake.model_path(MODELS_NAME): the saved models
MODELS_VERSION = 2              # a saved file of another version is fitted again (2: the world has the tyre life)
FUTURES = 500                   # futures per option (the same ones for every option)
TYRE_WINDOW = 0.8               # tyres are "getting old" at this share of the compound's usual stopping age
NEAR_SEC = 3.0                  # a car this close on the road that pits: the game asks (step 4 of Phase 2: clear air)
STAY_OUT_LAPS = 3               # staying out: no stop for at least this many laps (else the game asks every lap)
LUCK_STREAM = 7                 # your race's passes are drawn from a generator seeded [seed, LUCK_STREAM]
BOT_FUTURES = 200               # futures per option when the bot plays every race (--evaluate)
BOT_REPORT = "game_bot"         # lake.write_report(..., BOT_REPORT): one row per game of the bot
POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
KIND_NAMES = {SAFETY_CAR: "safety car", VIRTUAL_SAFETY_CAR: "virtual safety car", RED_FLAG: "red flag"}
PLURAL = {"SOFT": "softs", "MEDIUM": "mediums", "HARD": "hards"}


# --- the models, fitted once and saved --------------------------------------------------------------------

@dataclass
class Models:
    """What the game needs: each fold's models (replay.fit_fold) and every race's misses (what the futures draw)."""
    folds: List[FoldModels]
    misses: calibration.Misses
    dry: Tuple[Tuple[int, int], ...]        # the races the game can play (no intermediates or wets)

    def fold_of(self, key: Tuple[int, int]) -> FoldModels:
        """The fold that never saw this race."""
        for fold in self.folds:
            if tuple(key) in fold.held_out:
                return fold
        raise KeyError(f"Race {key} is not one of the development races the models were fitted on.")


def fit_models(root: str = lake.DEFAULT_LAKE_ROOT, seasons: Sequence[int] = evaluate.DEVELOPMENT_SEASONS) -> Models:
    """Every fold of the development races (as every Phase 4 step) and the misses of every race, out of fold."""
    laps = evaluate.load_laps(seasons, root)
    races = {key: race_module.load(*key, root=root) for key in evaluate.races_of(laps)}
    folds = [replay.fit_fold(laps, held, races, root) for held in evaluate.race_folds(laps)]
    misses = calibration.Misses.combine(calibration.misses_of(races[k], f, root) for f in folds for k in f.held_out)
    return Models(folds=folds, misses=misses, dry=tuple(k for k, r in races.items() if not r.mixed))


def load_models(root: str = lake.DEFAULT_LAKE_ROOT, refit: bool = False,
                seasons: Sequence[int] = evaluate.DEVELOPMENT_SEASONS, log: Callable[[str], None] = print) -> Models:
    """The saved models, or fit and save them (a plain dictionary is saved, so the file does not depend on how this
    module was started)."""
    if lake.model_path(MODELS_NAME, root).exists() and not refit:
        saved = lake.read_model(MODELS_NAME, root)
        if isinstance(saved, dict) and saved.get("version") == MODELS_VERSION:
            return Models(folds=saved["folds"], misses=saved["misses"], dry=tuple(saved["dry"]))
        log("The saved models are from another version: fitting them again.")
    log("Fitting the models on the development races (once, about 10 minutes; saved for the next game) ...")
    models = fit_models(root, seasons)
    lake.write_model({"version": MODELS_VERSION, "folds": models.folds, "misses": models.misses, "dry": models.dry},
                     MODELS_NAME, root)
    return models


# --- the fastest way to the flag, by the tyre model ----------------------------------------------------------

def fastest_finish(compound: str, age: float, lap: int, total_laps: int, used: Sequence[str],
                   sets_left: Dict[str, int], terms: TyreTerms, pit_loss: float,
                   box_now: Optional[str] = None, stay_laps: int = 1,
                   life: Optional[Dict[str, float]] = None) -> Optional[List[Tuple[int, str]]]:
    """
    The fastest way to the flag in clear air from the end of `lap`, the car on `compound` tyres `age` laps old: at
    most two more stops, each at the end of a lap (its in-lap), on a new set.

        a stint of n laps on compound c, its first lap at age a:  n x offset(c) + wear(c) x (a + ... + (a + n - 1))
        a stop:                                                   the circuit's green pit loss

    Fuel and track evolution are the same for every plan, so they are left out. The rules: two different dry compounds
    by the flag (used: the race's stints so far, the current one included); no more stints on a compound than its
    sets (sets_left); no stop with MIN_LAPS_TO_BOX laps or fewer after it; at least MIN_STINT_LAPS between two stops;
    no set older than its compound's life (life: {compound: the oldest a set may get}, teams.TyreLife; none: no limit).
    The straight-line wear is only trusted where teams have run sets: within their life.

      box_now   a compound: the first stop is at the end of the NEXT lap, on it (an option of the game; boxing is
                allowed even on a set past its life, that is the way out);
                None: staying out: no stop on the next `stay_laps` laps; the first stop, if any, comes later.

    Returns [(in-lap, compound), ...] (empty: no more stops), or None when nothing keeps the rules.
    """
    code = {c: CODE[c] for c in plans.DRY_COMPOUNDS}
    offset = {c: float(terms.offset[code[c]]) for c in code}
    wear = {c: float(terms.wear[code[c]]) for c in code}
    now = CODE.get(compound, OTHER)
    now_offset, now_wear = float(terms.offset[now]), float(terms.wear[now])
    have = {c for c in used if c in code}
    T = total_laps
    last_in_lap = T - MIN_LAPS_TO_BOX
    oldest = dict(life or {})
    now_life = oldest.get(compound, np.inf) if box_now is None else np.inf     # boxing now is always allowed

    def stint(off: float, w: float, first_age: float, n: int) -> float:
        return n * off + w * (n * first_age + n * (n - 1) / 2.0)

    def head(s: int) -> float:                         # on the current tyres until the in-lap s
        return stint(now_offset, now_wear, age + 1.0, s - lap)

    def lasts(c: str, n: int) -> bool:                 # a new set of c can run n laps (ages 1 .. n)
        return n <= oldest.get(c, np.inf)

    best_cost, best = np.inf, None
    if box_now is None and len(have) >= 2 and age + (T - lap) <= now_life:     # no more stops
        best_cost, best = head(T), []
    firsts = [lap + 1] if box_now is not None else range(lap + 1 + stay_laps, last_in_lap + 1)
    for s1 in firsts:
        if s1 > last_in_lap or age + (s1 - lap) > now_life:     # the current set would be past its life
            break
        before = head(s1)
        for c1 in ([box_now] if box_now is not None else plans.DRY_COMPOUNDS):
            if sets_left.get(c1, 0) < 1:
                continue
            if len(have | {c1}) >= 2 and lasts(c1, T - s1):     # one stop
                cost = before + pit_loss + stint(offset[c1], wear[c1], 1.0, T - s1)
                if cost < best_cost:
                    best_cost, best = cost, [(s1, c1)]
            for s2 in range(s1 + MIN_STINT_LAPS, last_in_lap + 1):     # two stops
                if not lasts(c1, s2 - s1):
                    break
                middle = before + pit_loss + stint(offset[c1], wear[c1], 1.0, s2 - s1) + pit_loss
                for c2 in plans.DRY_COMPOUNDS:
                    if sets_left.get(c2, 0) - (c2 == c1) < 1 or len(have | {c1, c2}) < 2 or not lasts(c2, T - s2):
                        continue
                    cost = middle + stint(offset[c2], wear[c2], 1.0, T - s2)
                    if cost < best_cost:
                        best_cost, best = cost, [(s1, c1), (s2, c2)]
    return best


# --- one game ------------------------------------------------------------------------------------------------

@dataclass
class Moment:
    """A big moment of your race: the end of the leader's lap `lap` (race.snapshot_of's moment)."""
    lap: int
    state: pd.DataFrame                     # the running cars then: laps_done, clock_sec, in order on the road
    done: int                               # your laps done
    reasons: List[str]
    ongoing: Optional[Tuple[str, int]]      # (kind, first lap) of the neutralisation the next lap is in, if any


@dataclass
class Option:
    """One call at a moment: stay out (box None) or box now, and the fastest finish after it."""
    label: str
    box: Optional[str]
    stops: List[Tuple[int, str]]            # (in-lap, compound) of every stop from now on
    plan: pd.DataFrame                      # your plan from the start with them


@dataclass
class Decision:
    lap: int
    reasons: List[str]
    table: pd.DataFrame                     # every option's chances
    chosen: str


@dataclass
class Result:
    place: Optional[int]                    # your place at the flag, in your race
    real_team_place: Optional[int]          # the real team's plan in the same race, with the same luck
    real_place: Optional[int]               # what really happened
    your_stops: List[Tuple[int, str]]
    real_stops: List[Tuple[int, str]]


@dataclass
class Game:
    race: Race
    driver: str
    fold: FoldModels
    uncertainty: montecarlo.Uncertainty
    place_offsets: Optional[world_module.PlaceOffsets]
    start_lap: int
    start: pd.DataFrame                     # the real race at the start lap: laps_done, clock_sec
    conditions: pd.DataFrame                # the real neutralisations (race.lap_conditions)
    rules: object
    inputs: pd.DataFrame                    # the whole race's model inputs (what pace.hindsight reads)
    real_plan: pd.DataFrame                 # every car's real plan from the start: the other teams' race
    real_pace: pd.Series                    # their real pace (pace.hindsight), same rows
    my_level: pd.Series                     # your driver's real level on each lap (lap_number -> s), tyres taken out
    run_on: pd.DataFrame                    # every car run on to the flag: what the futures start from
    plan: pd.DataFrame                      # YOUR car's plan from the start (it changes with your calls)
    luck: engine.PassDraws                  # your race's passes
    futures: int = FUTURES
    seed: int = 0
    blind: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    asked: set = field(default_factory=set)     # the sets already flagged as old (by the lap they went on)
    decisions: List[Decision] = field(default_factory=list)

    @property
    def total_laps(self) -> int:
        return self.race.total_laps

    def name(self, driver: str) -> str:
        return self.labels.get(driver, driver)

    @cached_property
    def last_laps(self) -> Dict[str, int]:
        """Each car's last lap in your race: the other teams' real last laps (a retirement ends a plan), yours the flag."""
        last = self.real_plan.groupby("driver_code")["lap_number"].max().astype(int).to_dict()
        last[self.driver] = self.total_laps
        return last


def labels_of(race: Race, driver: str, blind: bool) -> Dict[str, str]:
    """Blind: the other cars are letters in grid order (pole is A), you are YOU. Not blind: the driver codes."""
    drivers = sorted(set(race.laps["driver_code"]), key=lambda d: (race.grid.get(d, 99.0), d))
    if not blind:
        return {d: ("YOU" if d == driver else d) for d in drivers}
    letters = [chr(ord("A") + i) if i < 26 else f"A{chr(ord('A') + i - 26)}" for i in range(len(drivers))]
    return {d: ("YOU" if d == driver else f"car {letters[i]}") for i, d in enumerate(drivers)}


def new_game(race: Race, driver: str, models: Models, seed: int = 0, futures: int = FUTURES,
             blind: bool = True) -> Game:
    """A game on this race with this car: it starts on its real tyres, with none of the real team's stops, and the
    fastest finish from the start as its plan."""
    if race.mixed:
        raise ValueError("The game plays dry races: the pace model cannot see rain coming.")
    if driver not in race.finishers:
        raise ValueError(f"{driver} did not finish this race: choose a car that took the flag (to compare with).")
    fold = models.fold_of(race.key)
    start_lap = race_module.first_start_lap(race)
    if start_lap is None:
        raise ValueError("No lap of this race has a clean lap for every car: the simulator cannot start it.")
    real_plan = race_module.plan(race, start_lap)
    inputs = evaluate.model_inputs(race.laps)
    rung2 = fold.rungs["linear"]
    real_pace = pace.hindsight(rung2, inputs, real_plan)
    mine = real_plan["driver_code"] == driver
    level = real_pace[mine] - pace.planned_terms(rung2, real_plan[mine], race.total_laps)
    game = Game(race=race, driver=driver, fold=fold,
                uncertainty=calibration.uncertainty_from(models.misses, fold.held_out, montecarlo.KEPT_DRAWS),
                place_offsets=calibration.place_offsets_from(models.misses, fold.held_out),
                start_lap=start_lap, start=race_module.start_state(race, start_lap),
                conditions=race_module.lap_conditions(race), rules=fold.book.rules_for(race), inputs=inputs,
                real_plan=real_plan, real_pace=real_pace,
                my_level=pd.Series(level.to_numpy(), index=real_plan.loc[mine, "lap_number"].to_numpy()),
                run_on=race_module.plan(race, start_lap, run_on=True),
                plan=real_plan[real_plan["driver_code"] == driver].reset_index(drop=True),
                luck=engine.PassDraws(race_module.start_state(race, start_lap).index, race.total_laps,
                                      np.random.default_rng([seed, LUCK_STREAM])),
                futures=futures, seed=seed, blind=blind, labels=labels_of(race, driver, blind))
    done = int(game.start.at[driver, "laps_done"])
    mine = game.plan
    for in_lap in plans.stops_of(mine, driver):        # the real team's stops are yours to make
        mine = plans.stay_out(mine, driver, in_lap)
    game.plan = mine
    game.plan = with_stops(game, done, fastest_after(game, done) or [])
    return game


def tyres_at(game: Game, done: int) -> Tuple[str, float, int]:
    """Your tyres when you have completed `done` laps: compound, age, and the lap the set went on (0: before the game)."""
    mine = game.plan.set_index("lap_number")
    if done in mine.index:
        compound, age = mine.at[done, "compound"], float(mine.at[done, "tyre_age"])
    else:                                              # the start: the real race's row
        laps = game.race.laps
        row = laps[(laps["driver_code"] == game.driver) & (laps["lap_number"] == done)].iloc[0]
        compound, age = row["compound"], float(row["tyre_age"])
    outs = mine.index[mine["is_out_lap"].astype(bool) & (mine.index <= done)]
    return compound, age, int(outs.max()) if len(outs) else 0


def stints_so_far(game: Game, done: int) -> List[str]:
    """The compound of each of your stints up to lap `done`, the real laps before the game included."""
    laps = game.race.laps
    history = laps[(laps["driver_code"] == game.driver) & (laps["lap_number"] <= game.start_lap)]
    rows = pd.concat([history[["lap_number", "compound", "is_tyre_change"]],
                      game.plan.loc[game.plan["lap_number"] <= done, ["lap_number", "compound", "is_tyre_change"]]])
    rows = rows.drop_duplicates("lap_number", keep="last").sort_values("lap_number")
    out, new_set = [], True
    for compound, change in zip(rows["compound"], rows["is_tyre_change"].astype(bool)):
        if new_set and isinstance(compound, str):
            out.append(compound)
        new_set = change
    return out


def fastest_after(game: Game, done: int, box_now: Optional[str] = None) -> Optional[List[Tuple[int, str]]]:
    """fastest_finish for your car after lap `done` (its own laps), with your race's tyres, sets and pit loss."""
    compound, age, _ = tyres_at(game, done)
    used = stints_so_far(game, done)
    sets_left = {c: plans.ALLOCATION[c] - used.count(c) for c in plans.DRY_COMPOUNDS}
    return fastest_finish(compound, age, done, game.total_laps, used, sets_left, game.fold.world.terms,
                          game.rules.pit_loss_sec, box_now, stay_laps=STAY_OUT_LAPS, life=game.fold.world.life.limits())


def with_stops(game: Game, done: int, stops: Sequence[Tuple[int, str]]) -> pd.DataFrame:
    """Your plan with every tyre change after lap `done` replaced by `stops` [(in-lap, compound)], on new sets."""
    plan = game.plan
    for in_lap in [lap for lap in plans.stops_of(plan, game.driver) if lap > done]:
        plan = plans.stay_out(plan, game.driver, in_lap)
    for in_lap, compound in stops:
        plan = plans.box(plan, game.driver, in_lap, compound)
    return plan.reset_index(drop=True)


# --- your race ----------------------------------------------------------------------------------------------

def my_pace(game: Game, plan: pd.DataFrame) -> pd.Series:
    """Your car's pace on a plan: your driver's real level on each lap plus the plan's terms (tyres, fuel). The level
    is the REAL stint's (it never depends on your plan), so changing a stop after a moment leaves every lap before it
    as it was (a level per stint of YOUR plan would re-average the laps already driven)."""
    terms = pace.planned_terms(game.fold.rungs["linear"], plan, game.total_laps)
    return (plan["lap_number"].map(game.my_level) + terms).rename("pace_sec")


def played(game: Game, plan: Optional[pd.DataFrame] = None) -> engine.Simulation:
    """
    Your race from the start to the flag with your plan (default: the current one): every other car on its real
    plan at its real pace, yours re-made for your tyres (my_pace), the real neutralisations, the game's luck. Run again
    from the start after every call: a change after the moment of lap S cannot move anything before it, so the race
    you have seen so far stays as it was.
    """
    plan = game.plan if plan is None else plan
    others = (game.real_plan["driver_code"] != game.driver).to_numpy()
    planned = pd.concat([game.real_plan[others], plan], ignore_index=True)
    real_pace = pd.concat([game.real_pace[others], my_pace(game, plan)], ignore_index=True)
    return engine.simulate(game.start, planned, real_pace, game.conditions, game.rules, game.total_laps,
                           draws=game.luck)


def crossings_of(start: pd.DataFrame, simulation: engine.Simulation) -> pd.DataFrame:
    """Every line crossing of a simulated race, the start's included (what race.snapshot_of reads)."""
    start = start.reset_index().rename(columns={"laps_done": "lap_number", "clock_sec": "lap_end_sec"})
    laps = simulation.laps.rename(columns={"clock_sec": "lap_end_sec"})
    return pd.concat([start[["driver_code", "lap_number", "lap_end_sec"]],
                      laps[["driver_code", "lap_number", "lap_end_sec"]]], ignore_index=True)


def state_at(game: Game, crossings: pd.DataFrame, lap: int) -> pd.DataFrame:
    """Your race at the moment of the leader's lap `lap`: the cars still running, in order on the road."""
    state = race_module.snapshot_of(crossings, lap)
    return state[[game.last_laps.get(d, 0) > n for d, n in state["laps_done"].items()]]


def order_of(state: pd.DataFrame) -> List[str]:
    """The running order: laps done, then who crossed the line first."""
    return list(state.assign(minus=-state["laps_done"]).sort_values(["minus", "clock_sec"]).index)


def ongoing_at(game: Game, lap: int) -> Optional[Tuple[str, int]]:
    """(kind, first lap) of the real neutralisation the leader's `lap` is in, the stronger kind where two overlap."""
    kind, _ = race_module.status_of(zip(game.race.periods["kind"], game.race.periods["first_lap"],
                                        game.race.periods["last_lap"]), game.total_laps).get(lap, (GREEN, ""))
    if kind == GREEN:
        return None
    periods = game.race.periods
    match = periods[(periods["kind"] == kind) & (periods["first_lap"] <= lap) & (periods["last_lap"] >= lap)]
    return kind, int(match["first_lap"].max())


def neighbours(state: pd.DataFrame, driver: str) -> List[Tuple[str, float]]:
    """The car just ahead of `driver` and the car just behind on the road, on the same lap: [(car, gap)], gap in
    seconds at the line (negative: ahead)."""
    same = state[(state["laps_done"] == state.at[driver, "laps_done"]) & (state.index != driver)]
    gaps = same["clock_sec"] - float(state.at[driver, "clock_sec"])
    out = []
    if (gaps < 0).any():
        out.append((gaps[gaps < 0].idxmax(), float(gaps[gaps < 0].max())))
    if (gaps > 0).any():
        out.append((gaps[gaps > 0].idxmin(), float(gaps[gaps > 0].min())))
    return out


def next_moment(game: Game, simulation: engine.Simulation, after: int) -> Optional[Moment]:
    """The first big moment of your race after the leader's lap `after` (see the module docstring), or None."""
    crossings = crossings_of(game.start, simulation)
    status = race_module.status_of(zip(game.race.periods["kind"], game.race.periods["first_lap"],
                                       game.race.periods["last_lap"]), game.total_laps)
    others = game.real_plan[(game.real_plan["driver_code"] != game.driver) & game.real_plan["is_tyre_change"]]
    their_stops = set(zip(others["driver_code"], others["lap_number"].astype(int)))
    mine = game.plan.set_index("lap_number")
    previous = state_at(game, crossings, after) if after >= game.start_lap else None
    for lap in range(after + 1, game.total_laps + 1):
        state = state_at(game, crossings, lap)
        if game.driver not in state.index:
            return None
        done = int(state.at[game.driver, "laps_done"])
        if game.total_laps - done <= MIN_LAPS_TO_BOX:
            return None
        coming = mine.loc[done + 1] if done + 1 in mine.index else None
        if coming is not None and bool(coming["is_out_lap"]):            # in the pit lane: nothing to decide
            previous = state
            continue
        reasons = []
        kind, phase = status.get(lap + 1, (GREEN, ""))
        if kind != GREEN and phase in (DEPLOY, SINGLE):
            reasons.append(f"a {KIND_NAMES[kind]} is coming out")
        if coming is not None and bool(coming["is_tyre_change"]):
            reasons.append(f"your planned stop is due (box at the end of lap {done + 1})")
        compound, age, since = tyres_at(game, done)
        usual = game.fold.world.ages.median(compound)
        old = isinstance(compound, str) and age >= TYRE_WINDOW * usual and since not in game.asked
        if old:
            reasons.append(f"your {PLURAL.get(compound, compound)} are {age:.0f} laps old "
                           f"(teams usually change them at {usual:.0f})")
        limit = game.fold.world.life.limit(compound)
        at_limit = isinstance(compound, str) and age + STAY_OUT_LAPS > limit and ("life", since) not in game.asked
        if at_limit:
            reasons.append(f"your {PLURAL.get(compound, compound)} are near the end of their life ({limit:.0f} laps: "
                           "teams almost never ran a set longer), so you must box")
        if previous is not None and game.driver in previous.index:
            for rival, gap in neighbours(previous, game.driver):
                if abs(gap) < NEAR_SEC and (rival, int(previous.at[rival, "laps_done"]) + 1) in their_stops:
                    where = f"{abs(gap):.1f} s {'behind' if gap > 0 else 'ahead of'} you"
                    reasons.append(f"{game.name(rival)} ({where}) has just pitted")
        if reasons:
            if old:
                game.asked.add(since)
            if at_limit:
                game.asked.add(("life", since))
            return Moment(lap=lap, state=state, done=done, reasons=reasons, ongoing=ongoing_at(game, lap + 1))
        previous = state
    return None


# --- the options and their chances ----------------------------------------------------------------------------

def options_at(game: Game, moment: Moment) -> List[Option]:
    """Stay out, and box now for each compound with a set left, each followed by the fastest finish (only the calls
    that keep the rules)."""
    out = []
    stay = fastest_after(game, moment.done)
    if stay is not None:
        then = ", ".join(f"box lap {s} for {PLURAL[c]}" for s, c in stay) or "no more stops"
        out.append(Option(f"stay out ({then})", None, stay, with_stops(game, moment.done, stay)))
    for compound in plans.DRY_COMPOUNDS:
        stops = fastest_after(game, moment.done, box_now=compound)
        if stops is not None:
            then = ", ".join(f"lap {s} for {PLURAL[c]}" for s, c in stops[1:])
            label = f"box now for {PLURAL[compound]}" + (f" (then {then})" if then else " (to the flag)")
            out.append(Option(label, compound, stops, with_stops(game, moment.done, stops)))
    return out


def forecast_at(game: Game, lap: int, planned: pd.DataFrame) -> pd.Series:
    """
    The pace a strategist expects at the end of the leader's `lap`: boosted + prior on the REAL laps up to then
    (replay.pace_of), for every planned row; nothing after the moment is read. A row the real race had already driven
    by then (the car is a lap behind where it really was) takes its driver's first forecast level plus its own terms.
    """
    real_done = race_module.snapshot(game.race, lap)["laps_done"]
    ahead = (planned["lap_number"] > planned["driver_code"].map(real_done).fillna(-1)).to_numpy()
    out = pd.Series(np.nan, index=planned.index)
    out[ahead] = replay.pace_of(calibration.BEST, game.fold, game.race, lap, planned[ahead]).to_numpy()
    if not ahead.all():
        terms = pace.planned_terms(replay.rung2_of(game.fold.rungs["boosted"]), planned, game.total_laps)
        level = (out - terms).groupby(planned["driver_code"]).transform("first")
        out[~ahead] = (level + terms)[~ahead]
    return out


def repriced(pace_: pd.Series, plan: pd.DataFrame, new_plan: pd.DataFrame, terms: TyreTerms) -> pd.Series:
    """The pace of new_plan's laps: plan's for the same laps plus the change in tyre terms (as teams.Draw re-prices an
    answer: 0.009 s a lap from a full forecast on average)."""
    def tyre_terms(p: pd.DataFrame) -> np.ndarray:
        code = np.array([CODE.get(c, OTHER) if isinstance(c, str) else OTHER for c in p["compound"]], dtype=int)
        return terms.value(code, p["tyre_age"].to_numpy(float))
    old, new = plan.sort_values("lap_number"), new_plan.sort_values("lap_number")
    if old["lap_number"].tolist() != new["lap_number"].tolist():
        raise ValueError("Two plans of different laps cannot be re-priced one on the other.")
    values = pace_.loc[old.index].to_numpy() + tyre_terms(new) - tyre_terms(old)
    return pd.Series(values, index=new.index).loc[new_plan.index]


def odds(game: Game, moment: Moment, options: Sequence[Option], futures: Optional[int] = None) -> pd.DataFrame:
    """
    Each option's chances for your car, over the same futures: the race from the moment with the other teams on their
    BASE plans (their real decisions without the neutralisations still to come: teams.decontaminate; every car run on
    to the flag, who retires drawn), the safety cars after the one you can see drawn, the teams' answers drawn (yours
    to the neutralisation coming out now is the option itself), the speed misses by place, the forecast of the moment.
    One row per option: win, podium, points (chances), place (average where you finish), points_expected, dnf.
    """
    futures = futures or game.futures
    race, state, world = game.race, moment.state, game.fold.world
    done = state["laps_done"]
    others = game.run_on[(game.run_on["driver_code"] != game.driver) & game.run_on["driver_code"].isin(state.index)]
    others = others[others["lap_number"] > others["driver_code"].map(done)]
    others = teams_module.decontaminate(race, others, game.conditions, world.ages)
    others = teams_module.stops_for_run_on(race, others, world.ages).reset_index(drop=True)
    mine = [o.plan[o.plan["lap_number"] > moment.done].reset_index(drop=True) for o in options]
    first = pd.concat([others, mine[0]], ignore_index=True)
    forecast = forecast_at(game, moment.lap, first)
    others_pace = forecast.iloc[:len(others)].reset_index(drop=True)
    stay_pace = forecast.iloc[len(others):].reset_index(drop=True)
    if moment.ongoing is not None:
        kind, first_lap = moment.ongoing
        world = replace(world, events=Ongoing(base=world.events, kind=kind, first=first_lap, next_lap=moment.lap + 1))
    seed = int(game.seed) * 1000 + moment.lap
    rows = []
    for plan_i in mine:
        planned = pd.concat([others, plan_i], ignore_index=True)
        pace_i = pd.concat([others_pace, repriced(stay_pace, mine[0], plan_i, world.terms)], ignore_index=True)
        outcomes = world_module.run(state, planned, pace_i, game.rules, game.total_laps, game.uncertainty, world,
                                    futures, seed, moment.lap, events=True, react=True, retire=True,
                                    place_offsets=game.place_offsets, common=True, decided=(game.driver, moment.lap + 1))
        ranks = outcomes.ranks[game.driver]
        points = ranks.map(lambda r: POINTS[int(r) - 1] if pd.notna(r) and r <= len(POINTS) else 0.0)
        rows.append({"win": float((ranks <= 1).mean()), "podium": float((ranks <= 3).mean()),
                     "points": float((ranks <= 10).mean()), "place": float(ranks.mean()),
                     "points_expected": float(points.mean()), "dnf": float(ranks.isna().mean())})
    return pd.DataFrame(rows, index=[o.label for o in options])


# --- playing ------------------------------------------------------------------------------------------------

Chooser = Callable[[Game, Moment, List[Option], pd.DataFrame], int]


def most_points(game: Game, moment: Moment, options: List[Option], table: pd.DataFrame) -> int:
    """The bot: the option with the most points expected (then the better average place)."""
    ranked = table.reset_index(drop=True).assign(i=range(len(table)))
    return int(ranked.sort_values(["points_expected", "place"], ascending=[False, True])["i"].iloc[0])


def ask(game: Game, moment: Moment, options: List[Option], table: pd.DataFrame) -> int:
    """You: type the number of an option."""
    while True:
        answer = input(f"Your call (1-{len(options)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"Type a number from 1 to {len(options)}.")


@dataclass
class Turn:
    """A big moment with calls to make: the moment, the calls, each call's chances (one row per option)."""
    moment: Moment
    options: List[Option]
    table: pd.DataFrame


def next_turn(game: Game, after: int,
              counting: Optional[Callable[[Moment, List[Option]], None]] = None) -> Optional[Turn]:
    """
    Run your race on from the leader's lap `after` to the next big moment with calls to make, and count each call's
    chances (the slow part: `counting` is told first, so a screen can say so); None at the flag. The terminal (play)
    and the window (src/gui/game_window.py) both play the game in these turns.
    """
    simulation = played(game)
    while True:
        moment = next_moment(game, simulation, after)
        if moment is None:
            return None
        options = options_at(game, moment)
        if options:
            if counting is not None:
                counting(moment, options)
            return Turn(moment, options, odds(game, moment, options))
        after = moment.lap


def take(game: Game, turn: Turn, pick: int) -> int:
    """Make call number `pick`: your plan becomes that option's. Returns the lap the race runs on from."""
    game.plan = turn.options[pick].plan
    game.decisions.append(Decision(turn.moment.lap, turn.moment.reasons, turn.table, turn.options[pick].label))
    return turn.moment.lap


def play(game: Game, choose: Chooser = ask, show: Callable[[str], None] = print) -> Result:
    """The game in the terminal: the race runs on to each big moment, you choose, and on to the flag."""
    show(intro(game))
    after = game.start_lap
    while True:
        turn = next_turn(game, after, counting=lambda moment, options: show(
            f"\n(lap {moment.lap}: counting {game.futures} futures for each of {len(options)} options ...)"))
        if turn is None:
            break
        show(describe_moment(game, turn.moment, turn.options, turn.table))
        pick = choose(game, turn.moment, turn.options, turn.table) if len(turn.options) > 1 else 0
        after = take(game, turn, pick)
        show(f"-> {turn.options[pick].label}")
    result = finish(game, played(game))
    show(describe_result(game, result))
    return result


def stops_in(plan: pd.DataFrame, driver: str) -> List[Tuple[int, str]]:
    """(in-lap, compound fitted) of every tyre change of a plan."""
    rows = plan.sort_values("lap_number").reset_index(drop=True)
    out = []
    for i in np.flatnonzero(rows["is_tyre_change"].astype(bool).to_numpy()):
        fitted = rows["compound"].iloc[i + 1] if i + 1 < len(rows) else None
        out.append((int(rows["lap_number"].iloc[i]), fitted))
    return out


def finish(game: Game, simulation: engine.Simulation) -> Result:
    """Your place, the real team's plan in the same race with the same luck, and the real result."""
    def place_in(sim: engine.Simulation) -> Optional[int]:
        ranks = dict(zip(sim.finish["driver_code"], sim.finish["rank"]))
        return int(ranks[game.driver]) if game.driver in ranks else None
    real_rows = game.real_plan[game.real_plan["driver_code"] == game.driver].reset_index(drop=True)
    real = race_module.finish_ranks(game.race)
    return Result(place=place_in(simulation), real_team_place=place_in(played(game, real_rows)),
                  real_place=real.get(game.driver), your_stops=stops_in(game.plan, game.driver),
                  real_stops=stops_in(real_rows, game.driver))


# --- what the game shows -----------------------------------------------------------------------------------

def _gap_words(state: pd.DataFrame, a: str, b: str) -> str:
    """How far car b is from car a on the road: seconds on the same lap, else laps."""
    laps = int(state.at[b, "laps_done"]) - int(state.at[a, "laps_done"])
    if laps == 0:
        return f"{abs(float(state.at[b, 'clock_sec'] - state.at[a, 'clock_sec'])):.1f} s"
    return f"{abs(laps)} lap{'s' if abs(laps) > 1 else ''}"


def _tyres_of(game: Game, driver: str, done: int) -> str:
    """A car's tyres when it has completed `done` laps, or "in the pit lane" if that lap was an in-lap: its crossing
    then holds only the in-lap's share of the stop, the out-lap still costs it most of the pit loss."""
    plan = game.plan if driver == game.driver else game.real_plan[game.real_plan["driver_code"] == driver]
    rows = plan[plan["lap_number"] == done]
    if len(rows) and bool(rows["is_in_lap"].iloc[0]):
        return "in the pit lane"
    if driver == game.driver:
        compound, age, _ = tyres_at(game, done)
    else:
        if rows.empty:
            laps = game.race.laps
            rows = laps[(laps["driver_code"] == driver) & (laps["lap_number"] == done)]
        if rows.empty:
            return "?"
        compound, age = rows["compound"].iloc[0], float(rows["tyre_age"].iloc[0])
    return f"{PLURAL.get(compound, str(compound).lower())}, {age:.0f} laps"


def where_you_are(game: Game, state: pd.DataFrame) -> str:
    order = order_of(state)
    me = order.index(game.driver)
    done = int(state.at[game.driver, "laps_done"])
    mine = _tyres_of(game, game.driver, done)
    parts = [f"You are P{me + 1} " + (f"{mine}." if mine == "in the pit lane" else f"on {mine}.")]
    if me > 0:
        ahead = order[me - 1]
        parts.append(f"Ahead: P{me} {game.name(ahead)}, {_gap_words(state, game.driver, ahead)} "
                     f"({_tyres_of(game, ahead, int(state.at[ahead, 'laps_done']))}).")
        if me > 1:
            parts.append(f"Leader: {game.name(order[0])}, {_gap_words(state, game.driver, order[0])} ahead.")
    if me + 1 < len(order):
        behind = order[me + 1]
        parts.append(f"Behind: P{me + 2} {game.name(behind)}, {_gap_words(state, game.driver, behind)} "
                     f"({_tyres_of(game, behind, int(state.at[behind, 'laps_done']))}).")
    return " ".join(parts)


def running_order(game: Game, state: pd.DataFrame) -> pd.DataFrame:
    """The running order at a moment, one row per running car: place, car (its label), the gap to the car ahead and
    to the leader (seconds on the same lap, else laps), its tyres, and whether it is yours."""
    order = order_of(state)
    rows = []
    for i, d in enumerate(order):
        rows.append({"place": i + 1, "car": game.name(d),
                     "gap ahead": _gap_words(state, order[i - 1], d) if i else "",
                     "to the leader": _gap_words(state, order[0], d) if i else "leader",
                     "tyres": _tyres_of(game, d, int(state.at[d, "laps_done"])), "yours": d == game.driver})
    return pd.DataFrame(rows)


def odds_table(options: Sequence[Option], table: pd.DataFrame) -> str:
    width = max(len(o.label) for o in options)
    lines = [f"     {'option':<{width}}    win  podium  points  avg place"]
    for i, (option, row) in enumerate(zip(options, table.itertuples(index=False)), start=1):
        lines.append(f"  {i}  {option.label:<{width}}  {row.win:5.0%}  {row.podium:6.0%}  {row.points:6.0%}"
                     f"  {row.place:9.1f}")
    return "\n".join(lines)


def intro(game: Game) -> str:
    race = game.race
    what = "A dry race of 2023-2024, played blind (all is revealed at the flag)" if game.blind else \
        f"The {race.season} {race.event_name}, as {game.driver}"
    plan = ", ".join(f"box lap {s} for {PLURAL[c]}" for s, c in stops_in(game.plan, game.driver)
                     if s > game.start_lap) or "no stop"
    return "\n".join([
        "=== PitwallOS: the strategist game ===",
        f"{what}. {game.total_laps} laps; a stop costs about {game.rules.pit_loss_sec:.0f} s under green.",
        f"You take over at the end of lap {game.start_lap}. " + where_you_are(game, game.start),
        f"The plan from here (the fastest finish by the tyre model): {plan}.",
        "The race runs on to each big moment: a safety car, VSC or red flag; your tyres getting old; your planned "
        f"stop; the car just ahead or behind you (within {NEAR_SEC:.0f} s) pitting. Staying out means at least "
        f"{STAY_OUT_LAPS} more laps. Chances are counted over {game.futures} futures per option, the same futures "
        "for every option."])


def describe_moment(game: Game, moment: Moment, options: Sequence[Option], table: pd.DataFrame) -> str:
    because = "; ".join(moment.reasons)
    return "\n".join([
        f"\n--- Lap {moment.lap} of {game.total_laps}: {because[0].upper() + because[1:]}.",
        where_you_are(game, moment.state),
        odds_table(options, table)])


def describe_result(game: Game, result: Result) -> str:
    race = game.race
    team = race.laps.loc[race.laps["driver_code"] == game.driver, "team_code"].iloc[0] \
        if "team_code" in race.laps.columns else ""
    words = lambda stops: ", ".join(f"lap {s} {PLURAL.get(c, c)}" for s, c in stops) or "none"
    mine = [s for s in result.your_stops if s[0] > game.start_lap]
    theirs = [s for s in result.real_stops if s[0] > game.start_lap]
    place = lambda p: f"P{p}" if p is not None else "did not finish"
    return "\n".join([
        "\n=== Chequered flag ===",
        f"You finished {place(result.place)}.",
        f"The real team's strategy in the same race, with the same luck: {place(result.real_team_place)}.",
        f"What really happened: {place(result.real_place)}.",
        f"It was the {race.season} {race.event_name}; you were {game.driver} ({team}).",
        f"Your stops after lap {game.start_lap}: {words(mine)}. The real team's: {words(theirs)}."])


# --- the bot on every race ------------------------------------------------------------------------------------

def bot_games(models: Models, root: str = lake.DEFAULT_LAKE_ROOT, futures: int = BOT_FUTURES,
              log: Callable[[str], None] = print) -> pd.DataFrame:
    """
    The bot (most_points) on every dry development race, three cars each: the 2nd, the middle and the last-but-one
    car that finished the real race, in the order at the start. Each game seeded season x 100 + round, so a run is
    reproduced exactly. One row per game: its big moments and calls, your place, the real team's plan in the same race
    with the same luck, and what really happened.
    """
    rows = []
    for key in sorted(models.dry):
        race = race_module.load(*key, root=root)
        start_lap = race_module.first_start_lap(race)
        order = [d for d in order_of(race_module.start_state(race, start_lap)) if d in race.finishers]
        for driver in dict.fromkeys([order[1], order[len(order) // 2], order[-2]]):
            began = time.time()
            game = new_game(race, driver, models, seed=key[0] * 100 + key[1], futures=futures)
            result = play(game, most_points, show=lambda text: None)
            rows.append({"season": key[0], "round": key[1], "event": race.event_name, "driver": driver,
                         "start_place": order.index(driver) + 1, "decisions": len(game.decisions),
                         "laps": " ".join(str(d.lap) for d in game.decisions),
                         "reasons": " | ".join("; ".join(d.reasons) for d in game.decisions),
                         "chosen": " | ".join(d.chosen for d in game.decisions),
                         "seconds": time.time() - began, "place": result.place,
                         "real_team_place": result.real_team_place, "real_place": result.real_place,
                         "your_stops": str(result.your_stops), "real_stops": str(result.real_stops)})
            log(f"{race.season} {race.event_name} {driver}: {len(game.decisions)} moments, place {result.place} "
                f"(the real team's plan {result.real_team_place}, really {result.real_place})")
    return pd.DataFrame(rows)


def bot_summary(games: pd.DataFrame) -> str:
    """How often the game stops, and the bot against the real team's own plan in the same race with the same luck."""
    both = games.dropna(subset=["place", "real_team_place"])
    gained = both["real_team_place"] - both["place"]                     # + = the bot finished ahead
    reasons = {"safety car": "a safety car is", "VSC": "a virtual safety car", "red flag": "a red flag",
               "planned stop": "planned stop", "old tyres": "laps old", "a neighbour pitted": "has just pitted"}
    counts = ", ".join(f"{name} {sum(text.count(key) for text in games['reasons'])}" for name, key in reasons.items())
    return "\n".join([
        f"{len(games)} games ({games[['season', 'round']].drop_duplicates().shape[0]} races).",
        f"Big moments per game: median {games['decisions'].median():.0f} (p10 {games['decisions'].quantile(0.1):.0f}, "
        f"p90 {games['decisions'].quantile(0.9):.0f}, max {games['decisions'].max():.0f}); "
        f"{games['seconds'].median():.0f} s a game.",
        f"Why the game stopped: {counts}.",
        f"The bot against the real team's own plan (same race, same luck), {len(both)} games: ahead in "
        f"{(gained > 0).mean():.0%}, level {(gained == 0).mean():.0%}, behind {(gained < 0).mean():.0%}; "
        f"places gained on average {gained.mean():+.2f}."])


# --- command line ------------------------------------------------------------------------------------------

def car_from_answer(answer: str, order: Sequence[str], labels: Dict[str, str]) -> Optional[str]:
    """The car a typed answer names: its place ("5", "P5") or its name ("F", "car F", or a driver code); None if none."""
    text = answer.strip().upper()
    if text.startswith("P") and text[1:].strip().isdigit():
        text = text[1:].strip()
    if text.isdigit():
        place = int(text)
        return order[place - 1] if 1 <= place <= len(order) else None
    text = text[4:].strip() if text.startswith("CAR ") else text
    for d in order:
        label = labels.get(d, d).upper()
        if text in (label, label[4:] if label.startswith("CAR ") else label, d.upper()):
            return d
    return None


def pick_car(game_race: Race, start_lap: int, blind: bool, rng: random.Random) -> str:
    """Show the order at the start and ask which car (the ones that finished the real race)."""
    state = race_module.start_state(game_race, start_lap)
    order = order_of(state)
    labels = labels_of(game_race, "", blind)
    print(f"The order at the end of lap {start_lap} (cars you can take: those that finished the real race):")
    for i, d in enumerate(order, start=1):
        mark = "" if d in game_race.finishers else "   (cannot be chosen)"
        print(f"  P{i:<3} {labels[d]}{mark}")
    while True:
        answer = input("Your car (its place, e.g. 5, or its letter, e.g. F; Enter: a random one): ").strip()
        if not answer:
            return rng.choice([d for d in order if d in game_race.finishers])
        driver = car_from_answer(answer, order, labels)
        if driver is not None and driver in game_race.finishers:
            return driver
        print("That car cannot be chosen (not in the list, or it did not finish the real race): try again.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 step 4: the strategist game (text version).")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--season", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--driver", help="the driver code of your car (with --season and --round: not blind)")
    parser.add_argument("--car", type=int, help="your car by its place at the start (blind)")
    parser.add_argument("--futures", type=int, help=f"futures per option (default {FUTURES}; --evaluate {BOT_FUTURES})")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bot", action="store_true", help="the computer plays: the option with the most points expected")
    parser.add_argument("--refit", action="store_true", help="fit the models again")
    parser.add_argument("--evaluate", action="store_true", help="the bot on every dry race, 3 cars each, against the "
                                                                 f"real teams' plans; writes the report {BOT_REPORT}")
    args = parser.parse_args(argv)

    models = load_models(args.lake, refit=args.refit)
    if args.evaluate:
        games = bot_games(models, args.lake, args.futures or BOT_FUTURES)
        print(bot_summary(games))
        print(f"Report: {lake.write_report(games, BOT_REPORT, args.lake)}")
        return 0
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    rng = random.Random(seed)
    blind = not (args.season and args.round)
    key = (args.season, args.round) if not blind else rng.choice(sorted(models.dry))
    race = race_module.load(*key, root=args.lake)
    start_lap = race_module.first_start_lap(race)
    everyone = order_of(race_module.start_state(race, start_lap))
    if args.driver:
        driver = args.driver
    elif args.car:
        if not 1 <= args.car <= len(everyone) or everyone[args.car - 1] not in race.finishers:
            parser.error(f"--car {args.car}: no car that finished the real race was in that place at the start.")
        driver = everyone[args.car - 1]
    elif args.bot:
        driver = rng.choice([d for d in everyone if d in race.finishers])
    else:
        driver = pick_car(race, start_lap, blind, rng)
    game = new_game(race, driver, models, seed=seed, futures=args.futures or FUTURES, blind=blind)
    play(game, most_points if args.bot else ask)
    again = f"--seed {seed}" + ("" if blind else f" --season {key[0]} --round {key[1]}") + f" --driver {driver}"
    print(f"(This game again: python -m src.simulator.game {again})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
