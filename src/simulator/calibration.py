"""
How true are the chances? (src/simulator/calibration.py) - Phase 4 step 3

The Monte Carlo (montecarlo.py) gives every driver a chance of a win, a podium, points. A chance is
"calibrated" when the things given 40 % happen 40 % of the time, those given 10 % happen 10 % of the time.
This module measures what the Monte Carlo draws from, and then whether its chances come true, on the
development races (2023 + 2024), out of fold; 2025 stays locked.

THE MISSES (what the Monte Carlo draws), measured on every development race with the models that never
saw it (the fold that held it out):
    offset   from starts every STEP_LAPS laps (the first start lap, +10, +20, ...): every driver's mean of
             (real pace - forecast) over the laps after the start, minus the median driver's (a miss the
             whole field shares changes no position). Real pace = pace.hindsight (rung 2's reading of each
             clean lap, the median of each stint, plus the planned terms); forecast = boosted+prior. Drivers
             with fewer than MIN_LAPS such laps are left out.
    stint    for the same starts: each stint's mean miss minus the driver's (stints of MIN_STINT_LAPS+).
    lap      each clean lap on slicks in clear air (CLEAR_AIR_SEC or more to the car ahead when it starts):
             its reading minus the median reading of its stint (no odd-lap guard: the big misses are real).
    stop     each green tyre change: its pit-lane time minus the race's median green one.
When a fold's races are simulated, the misses of those races are left out of what is drawn.

THE JUDGE, from the first start lap ("lap 5") and from half distance, every dry development race:
    the Monte Carlo's chance of each car's win, podium and points (RUNS runs), against what happened (the
    order at the line among the cars that took the flag, like step 1);
    reliability: the chances in bins (0-10 %, 10-20 %, ...): how often what was given that chance happened;
    calibration error: the mean |happened - chance| over the bins, against what LUCK alone gives a perfect
    forecaster on the same cars (each outcome drawn from its own chance, LUCK_DRAWS times);
    Brier score (mean of (chance - happened)^2: 0 is perfect) and log loss (chances clipped to
    [CLIP, 1 - CLIP] so a 0 % that happens is a large but finite penalty);
    against two baselines: the deterministic replay (every chance 0 or 1, step 1's simulator) and history
    (how often a car in that place at that part of the race finished in the top k, in the other races).
Each draw is added to the offset on its own (CONFIGS) and compared with the offset alone, paired over races;
a draw stays only if it makes the chances truer ("better": the whole 95 % interval below zero).

MEASURED (2026-10-19, `python -m src.simulator.calibration`, 28 min; 41 dry races, 716 cars per start;
misses from 202 starts, 15,748 clean laps, 1,016 green stops), Brier from lap 5 / halfway:
    offset vs the deterministic replay: podium 0.067 vs 0.095 / 0.044 vs 0.066, points 0.088 vs 0.134 /
    0.060 vs 0.078, "better"; the win 0.031 vs 0.039 / 0.018 vs 0.025, no clear difference. vs history:
    points "better" (0.088 vs 0.124 / 0.060 vs 0.096); podium and win no clear difference (history on the
    win 0.028 / 0.012: the leader usually won in 2023-2024).
    Added to the offset on its own: pass "better" for the podium from halfway (-0.0023, -0.0041 to -0.0005),
    no clear difference elsewhere; stint, lap, stop no clear difference anywhere (lap noise raised the Brier
    score in all three events from halfway: the pass curve already holds the laps' randomness, see
    montecarlo.py). Kept: offset + pass (montecarlo.KEPT_DRAWS): vs the deterministic replay podium and points
    "better", vs history points "better", both from lap 5 and halfway; calibration error win / podium /
    points from lap 5 1.8 / 1.9 / 1.9 points (a perfect forecaster at least that far off by luck alone 27 /
    72 / 91 %), from halfway 1.2 / 1.7 / 2.3 (74 / 70 / 45 %); podium chances of 70 %+ from lap 5 happened
    2.2 points more often than stated, 0-10 % calls within 1 point. Every draw at once (an earlier run, with
    the old rule book): underconfident, podium chances of 70-90 % from lap 5 happening 12 points MORE often.

STEP 3b, THE EVENTS DRAWN (`--drawn`, ~21 min): in the above the safety cars, VSCs, red flags, retirements and the teams'
answers to them were the real ones. Here six configurations are judged on EVERY car running at the start (one that
retires is a miss, so a simulator that knows who retires gets chance 0 for them; history counts them too):
    real events          step 3a's kept draws: the real events, plans and retirements (hindsight)
    plans as real        the events drawn, but the real plans (whose real answers to the real events are in them)
    clean                the base plans (teams.decontaminate), nothing ever happens
    events drawn         ... safety cars, VSCs, red flags drawn, every team keeps its plan
    events + reactions   ... and the teams answer a drawn one (teams.BoxModel); retirements still the real ones
    + retirements        ... and who retires is drawn (cars that retired run on): nothing of the real race is known
    history              how often a car in that place at that part of the race finished in the top k
MEASURED (2026-10-20; 41 dry races, 791 cars at lap 5 / 760 at halfway), Brier win / podium / points, lap 5 | halfway:
    real events          0.028 0.064 0.085 | 0.017 0.042 0.062       events + reactions   0.028 0.059 0.082 | 0.014 0.038 0.061
    plans as real        0.029 0.061 0.082 | 0.015 0.041 0.060       + retirements        0.029 0.062 0.107 | 0.016 0.043 0.078
    clean                0.027 0.060 0.084 | 0.014 0.038 0.061       history              0.026 0.069 0.143 | 0.011 0.048 0.112
    events drawn         0.029 0.061 0.083 | 0.015 0.040 0.061
Paired over races: not knowing the events costs nothing measurable, and knowing the real ones is no better than a
race where nothing happens (clean vs real events: podium -0.0043 "better" from lap 5); the teams' answers: podium
-0.0019 (-0.0038 to +0.0000) from lap 5, no clear difference; drawing the retirements costs points +0.0229 against
the real events and +0.0260 against the same draw with the real retirements (who retires cannot be known), and is
"better" than history on points (-0.0357 from lap 5, -0.0340 from halfway; win and podium no clear difference).
Calibration error of '+ retirements' lap 5 win / podium / points 2.4 / 3.2 / 4.6 points (luck 95 %: 2.2 / 3.3 / 4.3),
halfway 2.4 / 2.1 / 2.8. In the races that really had a safety car after the start (194 cars at lap 5) the real events
score podium 0.0743 against 0.0565 for a clean race: calibrated, but too little resolution (CLAUDE.md, step 3b: the
simulator moves cars ~25 % more places than reality).

STEP 3c, THE SPREAD (the same `--drawn` run, now 9 configurations, ~35 min): two changes, each a switch of Config, judged
against the same configuration without them ("(3b)"): DRS trains in the pass chance (rules.py: behind a car that is itself
within 1 s of the car in front, the most a car can hope for per lap is halved) and the speed misses drawn by place at the
start (world.PlaceOffsets). Two more judges: places moved per car from the start to the flag (places_moved, real against
simulated) and overtakes per race (race.overtakes against the runs' own count).
MEASURED (2026-10-20; 41 dry races; '+ retirements', nothing of the real race known; lap 5 | halfway):
    Brier win / podium / points: 3c 0.028 0.062 0.106 | 0.015 0.042 0.077, as 3b 0.029 0.062 0.107 | 0.016 0.043 0.078.
    Paired: from lap 5 no clear difference (points -0.0019, -0.0040 to +0.0003); from halfway win -0.0010 and podium
    -0.0017 "better". By place alone: points "better" from lap 5 (-0.0022), win from halfway (-0.0013); trains alone: no
    clear difference anywhere.
    Places moved per car (the real finishers) against reality: +19.6 % -> +13.8 % from lap 5, +33 % -> +28 % from halfway
    (by place alone +13.3 % / +28 %, trains alone +20.1 % / +33 %; with the real events +14.8 % -> +7.6 %, +29.7 % -> +25.1 %).
    Overtakes per race: 38.4 -> 35.8 (real 36.2); with the real safety cars, in those races, 52 -> 47 (real 37).
    Podium calls of 70-90 % from lap 5 came true 92 % of the time (as 3b: 100 %, 40 of 40).
    The real-events configuration got a little overconfident (podium calibration error 2.1 -> 3.0 points from lap 5,
    1.6 -> 2.7 from halfway): sharper misses suit the race where nothing is known, not the one that peeks.
    KEPT: both (never "worse"; the misses by place "better" twice; trains fit the passes better out of fold and give the
    real number of overtakes).

Run it (~30 min: the folds are fitted like replay.py's, the misses measured, then every configuration):

    python -m src.simulator.calibration
    python -m src.simulator.calibration --drawn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.laps import classify_laps
from src.analytics.stops import GREEN, stops
from src.models import evaluate
from src.models.linear import own_terms
from src.pipeline import lake
from src.simulator import events as events_module, montecarlo, pace, race as race_module, replay, teams as teams_module, \
    world as world_module
from src.simulator.race import Race
from src.simulator.replay import FoldModels

BEST = "boosted+prior"
STEP_LAPS = 10
MIN_LAPS = 10
MIN_STINT_LAPS = 5
CLEAR_AIR_SEC = 3.0
RUNS = 200
SEED = 2026
CLIP = 0.01
EVENTS = {"win": 1, "podium": 3, "points": 10}
BINS = np.linspace(0.0, 1.0, 11)
LUCK_DRAWS = 500
HISTORY_SMOOTHING = 2.0          # pseudo-counts pulling a (part, place) cell towards its part's overall rate
STARTS = (replay.LAP_5, replay.HALFWAY)
CONFIGS = {
    "deterministic": (),
    "offset": ("offset",),
    "offset + stint": ("offset", "stint"),
    "offset + lap": ("offset", "lap"),
    "offset + stop": ("offset", "stop"),
    "offset + pass": ("offset", "pass"),
}
BASE = "offset"                  # every other draw is judged by adding it to this one alone
REPORT = "sim_calibration"



@dataclass(frozen=True)
class Config:
    """One way of running the races many times (steps 3b and 3c). Every one draws what montecarlo.KEPT_DRAWS says
    (the speed misses and the passes); they differ in what happens on the track and in the pit lane."""
    plan: str               # "real" (the real plans), "base" (teams.decontaminate), "run_on" (+ the cars that retired run on)
    events: str             # "real" (the race's own safety cars, VSCs, red flags), "drawn" or "none"
    react: bool = False     # the teams answer a drawn neutralisation (teams.BoxModel)
    retire: bool = False    # who retires is drawn
    trains: bool = True     # step 3c: DRS trains in the pass chance (rules.py)
    by_place: bool = True   # step 3c: the speed misses drawn by place at the start (world.PlaceOffsets)


REAL = "real events"             # the real safety cars, VSCs, red flags, retirements, plans and answers (step 3a's world)
FULL = "+ retirements"           # nothing of the real race is known: what the game will use
AS_3B = " (3b)"                  # without the two step 3c changes, to compare with
DRAWN = {
    REAL: Config("real", "real"),
    "clean": Config("base", "none"),                                     # the base plans, nothing ever happens
    "events drawn": Config("base", "drawn"),                             # ... neutralisations drawn, plans kept
    "events + reactions": Config("base", "drawn", react=True),           # ... and the teams answer
    FULL: Config("run_on", "drawn", react=True, retire=True),            # ... and who retires is drawn
    REAL + AS_3B: Config("real", "real", trains=False, by_place=False),
    FULL + AS_3B: Config("run_on", "drawn", react=True, retire=True, trains=False, by_place=False),
    FULL + ", trains only": Config("run_on", "drawn", react=True, retire=True, by_place=False),
    FULL + ", by place only": Config("run_on", "drawn", react=True, retire=True, trains=False),
}
DRAWN_REPORT = "sim_calibration_drawn"
DRAWN_PAIRS = [(FULL, FULL + AS_3B), (FULL + ", trains only", FULL + AS_3B), (FULL + ", by place only", FULL + AS_3B),
               (REAL, REAL + AS_3B), (FULL, REAL), (FULL, "history"), (FULL, "events + reactions"),
               ("events + reactions", "events drawn"), ("events drawn", "clean"), ("clean", REAL)]
MIN_POOL = 30                    # place pools with fewer misses than this borrow from the other parts of the race


# --- the misses -------------------------------------------------------------------------------------

@dataclass
class Misses:
    offsets: pd.DataFrame       # season, round, start_lap, part, driver_code, miss
    stint_shifts: pd.DataFrame  # season, round, start_lap, driver_code, stint, shift
    lap_noise: pd.DataFrame     # season, round, driver_code, lap_number, noise
    stop_extra: pd.DataFrame    # season, round, driver_code, in_lap, extra
    places: pd.DataFrame        # season, round, start_lap, part, driver_code, start_place, final_place (history)
    places_all: pd.DataFrame    # the same for every car running at the start: a car that did not finish has final_place NaN

    @staticmethod
    def combine(parts: Iterable["Misses"]) -> "Misses":
        parts = list(parts)
        return Misses(*[pd.concat([getattr(p, f) for p in parts if len(getattr(p, f))], ignore_index=True)
                        for f in ("offsets", "stint_shifts", "lap_noise", "stop_extra", "places", "places_all")])


def stint_index(planned: pd.DataFrame) -> pd.Series:
    """Each planned lap's stint, counted per driver from the start (0 until the first tyre change)."""
    change = planned["is_tyre_change"].astype(int)
    return change.groupby(planned["driver_code"]).cumsum() - change


def empty_misses() -> Misses:
    return Misses(*[pd.DataFrame() for _ in range(6)])


def misses_of(race: Race, fold: FoldModels, root: str = lake.DEFAULT_LAKE_ROOT) -> Misses:
    """Every miss of one race (see THE MISSES), with the fold's models (which never saw this race). A race
    with intermediates or wets has none: the dry races are what is simulated and judged."""
    if race.mixed:
        return empty_misses()
    rung2 = fold.rungs["linear"]
    inputs = evaluate.model_inputs(race.laps)
    first = race_module.first_start_lap(race)
    offsets, shifts, places, places_all = [], [], [], []
    for lap in range(first or race.total_laps, race.total_laps - 10, STEP_LAPS):
        rows = race_module.last_rows(race, race_module.start_state(race, lap))
        if not len(rows) or not bool(evaluate.pace_seen(rows).all()):
            continue
        part = montecarlo.part_of_race(lap, race.total_laps)
        planned = race_module.plan(race, lap)
        miss = pace.hindsight(rung2, inputs, planned) - replay.pace_of(BEST, fold, race, lap, planned)
        frame = pd.DataFrame({"driver_code": planned["driver_code"], "stint": stint_index(planned), "miss": miss}).dropna()
        per_driver = frame.groupby("driver_code")["miss"].agg(["mean", "size"])
        per_driver = per_driver[per_driver["size"] >= MIN_LAPS]
        if len(per_driver) < 3:
            continue
        relative = per_driver["mean"] - per_driver["mean"].median()
        offsets += [{"season": race.season, "round": race.round, "start_lap": lap, "part": part,
                     "driver_code": d, "miss": m} for d, m in relative.items()]
        per_stint = frame[frame["driver_code"].isin(per_driver.index)].groupby(["driver_code", "stint"])["miss"].agg(["mean", "size"])
        per_stint = per_stint[per_stint["size"] >= MIN_STINT_LAPS]
        shifts += [{"season": race.season, "round": race.round, "start_lap": lap, "driver_code": d, "stint": s,
                    "shift": m - per_driver.at[d, "mean"]} for (d, s), m in per_stint["mean"].items()]
        running = race_module.running_at(race, lap)
        final = race_module.finish_ranks(race, running)
        held = race_module.hold_order_ranks(race, lap, final)
        places += [{"season": race.season, "round": race.round, "start_lap": lap, "part": part, "driver_code": d,
                    "start_place": held[d], "final_place": final[d]} for d in final if d in held]
        held_all = race_module.hold_order_ranks(race, lap, running)
        places_all += [{"season": race.season, "round": race.round, "start_lap": lap, "part": part, "driver_code": d,
                        "start_place": held_all[d], "final_place": final.get(d, np.nan)} for d in running if d in held_all]
    # Lap noise: clean laps on slicks in clear air, against their stint's median reading (no guard).
    coef = rung2.coefficients()
    laps = race.laps
    reading = (laps["lap_time_sec"] - own_terms(laps)[coef.index] @ coef).where(laps["is_clean_lap"].astype(bool))
    typical = reading.groupby([laps["driver_code"], laps["stint_number"]]).transform("median")
    clear = laps["gap_on_track_start_sec"].isna() | (laps["gap_on_track_start_sec"] >= CLEAR_AIR_SEC)
    # A lap after an `unknown` passage has no stint number, so no stint median: left out.
    keep = reading.notna() & typical.notna() & clear & ~laps["is_wet_tyre"].astype(bool)
    noise = pd.DataFrame({"season": race.season, "round": race.round, "driver_code": laps.loc[keep, "driver_code"],
                          "lap_number": laps.loc[keep, "lap_number"], "noise": (reading - typical)[keep]})
    # Stops: green tyre changes against the race's median green pit-lane time.
    classified = classify_laps(lake.read_table(lake.table_path(lake.SILVER, "laps", *race.key, "R", root)))
    green = stops(classified)
    green = green[(green["condition"] == GREEN) & green["pit_lane_sec"].notna()]
    stop = pd.DataFrame({"season": race.season, "round": race.round, "driver_code": green["driver_code"],
                         "in_lap": green["in_lap"], "extra": green["pit_lane_sec"] - green["pit_lane_sec"].median()})
    return Misses(offsets=pd.DataFrame(offsets), stint_shifts=pd.DataFrame(shifts), lap_noise=noise,
                  stop_extra=stop, places=pd.DataFrame(places), places_all=pd.DataFrame(places_all))


def _outside(table: pd.DataFrame, races: Iterable[Tuple[int, int]]) -> pd.DataFrame:
    if table.empty:
        return table
    return table[~evaluate.in_races(table, list(races))]


def uncertainty_from(misses: Misses, held_out: Iterable[Tuple[int, int]],
                     draws: Sequence[str] = montecarlo.ALL_DRAWS) -> montecarlo.Uncertainty:
    """What to draw when the races `held_out` are simulated: every other race's misses."""
    held_out = list(held_out)

    def pool(table: pd.DataFrame, column: str) -> np.ndarray:
        """The misses of every race but the held-out ones. Only real numbers: one NaN drawn for the last car
        on the road leaves a lap with no pace at all (the engine refuses it)."""
        table = _outside(table, held_out)
        return table[column].dropna().to_numpy(float) if len(table) else np.zeros(0)

    offsets = _outside(misses.offsets, held_out)
    every = pool(offsets, "miss")
    by_part = {p: pool(offsets[offsets["part"] == p], "miss") if len(offsets) else every for p in montecarlo.PARTS}
    by_part = {p: v if len(v) else every for p, v in by_part.items()}      # none at that part: every part's
    return montecarlo.Uncertainty(offsets=by_part, stint_shifts=pool(misses.stint_shifts, "shift"),
                                  lap_noise=pool(misses.lap_noise, "noise"), stop_extra=pool(misses.stop_extra, "extra"),
                                  draws=tuple(draws))


# --- the chances and the judge ----------------------------------------------------------------------

@dataclass
class Start:
    """One start of one race, ready to be run many times."""
    race: Race
    label: str
    lap: int
    state: pd.DataFrame
    planned: pd.DataFrame
    pace: pd.Series
    conditions: pd.DataFrame
    rules: object
    base_planned: Optional[pd.DataFrame] = None     # step 3b: the plans without the real events' influence ...
    base_pace: Optional[pd.Series] = None           # ... and their forecast
    run_on_planned: Optional[pd.DataFrame] = None   # ... and the same with every car that really retired run on to the flag
    run_on_pace: Optional[pd.Series] = None
    plain_rules: object = None                      # step 3c: the rules without DRS trains (as in 3b), to compare with
    real_passes: float = np.nan                     # the real on-track overtakes after the start (race.overtakes)


def start_of(race: Race, fold: FoldModels, label: str, lap: int, drawn: bool = False) -> Start:
    """One start of one race; drawn=True also makes the base plan (teams.decontaminate) and the plan with the cars
    that retired run on to the flag, and forecasts them."""
    planned = race_module.plan(race, lap)
    conditions = race_module.lap_conditions(race)
    start = Start(race=race, label=label, lap=lap, state=race_module.start_state(race, lap), planned=planned,
                  pace=replay.pace_of(BEST, fold, race, lap, planned), conditions=conditions,
                  rules=fold.book.rules_for(race), plain_rules=fold.book.rules_for(race, trains=False),
                  real_passes=float(race_module.overtakes(race, lap)))
    if drawn:
        start.base_planned = teams_module.decontaminate(race, planned, conditions, fold.world.ages)
        start.base_pace = replay.pace_of(BEST, fold, race, lap, start.base_planned)
        run_on = teams_module.decontaminate(race, race_module.plan(race, lap, run_on=True), conditions, fold.world.ages)
        start.run_on_planned = teams_module.stops_for_run_on(race, run_on, fold.world.ages)
        start.run_on_pace = replay.pace_of(BEST, fold, race, lap, start.run_on_planned)
    return start


def _rows_of(start: Start, outcomes: montecarlo.Outcomes, everyone: bool = False) -> pd.DataFrame:
    """
    One row per car that took the flag (in reality) and was running at the start: its chances, and what
    happened. everyone=True: one row per car running at the start; a car that did not take the flag has
    final_place NaN, so it did not win, make the podium or score (a simulator that draws the retirements is
    judged on them; one that knows them gives them chance 0).
    """
    race = start.race
    running = race_module.running_at(race, start.lap)
    real = race_module.finish_ranks(race, running)
    held = race_module.hold_order_ranks(race, start.lap, running if everyone else real)
    moved_sim, moved_real = places_moved(race, start.lap, real, outcomes)
    passes = getattr(outcomes, "passes", np.zeros(0))
    rows = []
    for d in (running if everyone else real):
        final = real.get(d, np.nan)
        row = {"season": race.season, "round": race.round, "start": start.label, "start_lap": start.lap,
               "part": montecarlo.part_of_race(start.lap, race.total_laps), "driver_code": d,
               "kind_after": replay.kind_after(race, start.lap), "start_place": held.get(d, np.nan),
               "final_place": final, "finished": d in real, "moved_sim": moved_sim.get(d, np.nan),
               "moved_real": moved_real.get(d, np.nan), "sim_passes": float(np.mean(passes)) if len(passes) else np.nan,
               "real_passes": start.real_passes}
        for event, top in EVENTS.items():
            row[f"p_{event}"] = float(outcomes.chance(top).get(d, 0.0))
            row[event] = bool(final <= top)                      # NaN <= top is False: a retirement is a miss
        rows.append(row)
    return pd.DataFrame(rows)


def places_moved(race: Race, lap: int, real: Dict[str, int], outcomes: montecarlo.Outcomes) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    How many places each car that really took the flag moved from its place at the start (step 3c): in reality,
    and on average over the runs (each run's order among the same cars). A simulator that reshuffles more than
    reality moves cars more places; it makes the chances too flat.
    """
    finishers = list(real)
    start_rank = pd.Series(race_module.hold_order_ranks(race, lap, finishers), dtype=float)
    sim = outcomes.ranks.reindex(columns=finishers).rank(axis=1, method="first")
    moved_sim = (sim - start_rank).abs().mean()                    # over the runs in which the car finished
    return moved_sim.to_dict(), {d: abs(real[d] - start_rank[d]) for d in finishers}


def chances(start: Start, uncertainty: montecarlo.Uncertainty, runs: int = RUNS, seed: int = SEED,
            everyone: bool = False) -> pd.DataFrame:
    """One row per car that took the flag (in reality): its chances, and what happened (everyone: see _rows_of)."""
    outcomes = montecarlo.run(start.state, start.planned, start.pace, start.conditions, start.rules,
                              start.race.total_laps, uncertainty, runs if uncertainty.draws else 1, seed, start.lap)
    return _rows_of(start, outcomes, everyone)


def drawn_chances(start: Start, uncertainty: montecarlo.Uncertainty, config: Config, world: world_module.World,
                  place_offsets: Optional[world_module.PlaceOffsets] = None, runs: int = RUNS, seed: int = SEED,
                  everyone: bool = True) -> pd.DataFrame:
    """chances() run by world.run for one Config: the real, base or run-on plans; the race's own neutralisations,
    drawn ones or none; the teams answering or not; retirements drawn or not; DRS trains; misses by place."""
    planned, pace_ = {"real": (start.planned, start.pace), "base": (start.base_planned, start.base_pace),
                      "run_on": (start.run_on_planned, start.run_on_pace)}[config.plan]
    if config.events == "real":
        world = replace(world, events=events_module.FixedEvents.of_race(start.race))
    outcomes = world_module.run(start.state, planned, pace_, start.rules if config.trains else start.plain_rules,
                                start.race.total_laps, uncertainty, world, runs, seed, start.lap,
                                events=config.events != "none", react=config.react, retire=config.retire,
                                place_offsets=place_offsets if config.by_place else None)
    return _rows_of(start, outcomes, everyone)


def place_offsets_from(misses: Misses, held_out: Iterable[Tuple[int, int]]) -> Optional[world_module.PlaceOffsets]:
    """The speed misses of every race but the held-out ones, by part of the race and place group at the start
    (step 3c). A (part, group) with fewer than MIN_POOL misses uses its group over every part, then its part over
    every group. None without the places (misses measured before 3c)."""
    held_out = list(held_out)
    places = getattr(misses, "places_all", pd.DataFrame())
    offsets = _outside(misses.offsets, held_out)
    if offsets.empty or places.empty:
        return None
    o = offsets.merge(_outside(places, held_out)[["season", "round", "start_lap", "driver_code", "start_place"]],
                      on=["season", "round", "start_lap", "driver_code"], how="inner").dropna(subset=["miss", "start_place"])
    if o.empty:
        return None
    o["group"] = [world_module.place_group(p) for p in o["start_place"]]
    pools = {}
    for part in montecarlo.PARTS:
        for group in range(len(world_module.GROUP_NAMES)):
            for pick in (o[(o["part"] == part) & (o["group"] == group)], o[o["group"] == group], o[o["part"] == part], o):
                if len(pick) >= MIN_POOL or pick is o:
                    break
            pools[(part, group)] = pick["miss"].to_numpy(float)
    return world_module.PlaceOffsets(pools)


def history_chances(rows: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    """The history baseline for these rows: in the other races (`places`), how often a car in the same place
    at the same part of the race finished in the top k (smoothed towards the part's overall rate)."""
    out = rows.copy()
    for event, top in EVENTS.items():
        hit = places.assign(hit=places["final_place"] <= top)
        overall = hit.groupby("part")["hit"].mean()
        cell = hit.groupby(["part", "start_place"])["hit"].agg(["sum", "size"])
        def chance(part, place):
            base = overall.get(part, hit["hit"].mean())
            k, n = cell.loc[(part, place)] if (part, place) in cell.index else (0.0, 0.0)
            return (k + HISTORY_SMOOTHING * base) / (n + HISTORY_SMOOTHING)
        out[f"p_{event}"] = [chance(p, s) for p, s in zip(out["part"], out["start_place"])]
    return out


def reliability(rows: pd.DataFrame, event: str) -> pd.DataFrame:
    """Chances in bins: how many, the mean chance, how often it happened."""
    p, hit = rows[f"p_{event}"], rows[event].astype(float)
    which = np.clip(np.digitize(p, BINS[1:-1]), 0, len(BINS) - 2)
    table = pd.DataFrame({"bin": which, "p": p, "hit": hit}).groupby("bin").agg(
        n=("p", "size"), chance=("p", "mean"), happened=("hit", "mean"))
    table.index = [f"{BINS[i]:.0%}-{BINS[i + 1]:.0%}" for i in table.index]
    return table


def calibration_error(p: np.ndarray, happened: np.ndarray) -> np.ndarray:
    """Mean |happened - chance| over the bins, weighted by their size; the same as: summed over the bins,
    |how many happened - how many the chances expected|, divided by the number of cars. `happened` may hold
    many sets of outcomes, one per row (one calibration error each)."""
    in_bin = np.eye(len(BINS) - 1)[np.clip(np.digitize(p, BINS[1:-1]), 0, len(BINS) - 2)]   # cars x bins, 0/1
    return np.abs(np.atleast_2d(happened) @ in_bin - p @ in_bin).sum(axis=1) / len(p)


def scores(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per event: Brier score, log loss, calibration error, and how overconfident the sure calls were
    (happened - chance, for chances of 70 %+ and of 10 % or less).

    Luck: even a PERFECT forecaster is off the diagonal on a few hundred cars (a bin of 25 cars given 40 %
    holds 10 +- 2.4 hits). So every car's outcome is drawn LUCK_DRAWS times from its own chance (what a
    perfect forecaster's world would do) and the calibration error measured each time: luck_p95 is how far
    off luck alone takes it 95 % of the time, off_by_luck the share of those draws at least as far off as the
    real outcomes. Under 5 %: the chances are off by more than luck explains.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for event in EVENTS:
        p, y = rows[f"p_{event}"].to_numpy(float), rows[event].astype(float).to_numpy()
        clipped = np.clip(p, CLIP, 1 - CLIP)
        error = float(calibration_error(p, y)[0])
        lucky = calibration_error(p, rng.random((LUCK_DRAWS, len(p))) < p)
        high, low = p >= 0.7, p <= 0.1
        out.append({"event": event, "n": len(p), "brier": float(((p - y) ** 2).mean()),
                    "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
                    "calibration_error": error, "luck_p95": float(np.percentile(lucky, 95)),
                    "off_by_luck": float((lucky >= error - 1e-12).mean()),
                    "sure_yes_gap": float(y[high].mean() - p[high].mean()) if high.any() else np.nan,
                    "sure_no_gap": float(y[low].mean() - p[low].mean()) if low.any() else np.nan})
    return pd.DataFrame(out)


def brier_difference(rows: pd.DataFrame, a: str, b: str, event: str) -> Dict[str, object]:
    """Brier score of configuration a minus b on the same cars (negative: a's chances are truer), with its 95 %
    interval from resampling whole races, the same races for both (paired; evaluate.draw_races, like every
    verdict in the project): "better" only when the whole interval is below zero."""
    per_race = []
    for name in (a, b):
        r = rows[rows["config"] == name]
        squared = (r[f"p_{event}"] - r[event].astype(float)) ** 2
        per_race.append(squared.groupby([r["season"], r["round"]]).agg(["sum", "size"]))
    first, second = per_race[0].align(per_race[1], join="inner", axis=0)
    total, cars = (first["sum"] - second["sum"]).to_numpy(float), first["size"].to_numpy(float)
    drawn = evaluate.draw_races(len(total))
    low, high = np.percentile(total[drawn].sum(axis=1) / cars[drawn].sum(axis=1), [2.5, 97.5])
    return {"compared": f"{a} vs {b}", "event": event, "difference": float(total.sum() / cars.sum()),
            "low": float(low), "high": float(high),
            "verdict": "better" if high < 0 else "worse" if low > 0 else "no clear difference"}


def kept_config() -> str:
    """The configuration that draws what the Monte Carlo draws (montecarlo.KEPT_DRAWS)."""
    return next((name for name, draws in CONFIGS.items() if draws == montecarlo.KEPT_DRAWS), BASE)


def comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdicts that decide what is drawn: the offset against the one replay and against history, each
    other draw added to the offset against the offset alone, and what is kept against both baselines (rows
    of one start)."""
    kept = kept_config()
    pairs = [(BASE, "deterministic"), (BASE, "history")] + \
        [(name, BASE) for name in CONFIGS if name not in ("deterministic", BASE)] + \
        ([(kept, "deterministic"), (kept, "history")] if kept != BASE else [])
    present = set(rows["config"])
    return pd.DataFrame([brier_difference(rows, a, b, event) for a, b in pairs if {a, b} <= present
                         for event in EVENTS])


def drawn_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    """Step 3b's verdicts (rows of one start): what knowing the real events was worth (the full draw against the
    real events), the full draw against history, and each stage against the one before it: do the teams' answers
    help, do the drawn events help, does taking the real events out of the plans help."""
    present = set(rows["config"])
    return pd.DataFrame([brier_difference(rows, a, b, event) for a, b in DRAWN_PAIRS if {a, b} <= present
                         for event in EVENTS])


# --- command line -----------------------------------------------------------------------------------

def evaluate_configs(folds: List[FoldModels], races: Dict[Tuple[int, int], Race], misses: Misses,
                     configs: Dict[str, Tuple[str, ...]] = CONFIGS, runs: int = RUNS) -> pd.DataFrame:
    """Every configuration's chances for every dry development race, from STARTS, out of fold."""
    rows = []
    for fold in folds:
        fold_rows = []
        for key in [k for k in fold.held_out if not races[k].mixed]:
            race = races[key]
            for label, lap in replay.starts(race):
                if label not in STARTS:
                    continue
                start = start_of(race, fold, label, lap)
                seed = SEED + 1000 * race.season + 10 * race.round + STARTS.index(label)
                for name, draws in configs.items():
                    uncertainty = uncertainty_from(misses, fold.held_out, draws)
                    fold_rows.append(chances(start, uncertainty, runs, seed).assign(config=name))
        if not fold_rows:
            continue
        cars = pd.concat(fold_rows, ignore_index=True)
        cars = cars[cars["config"] == cars["config"].iloc[0]]            # each car once, for the history baseline
        fold_rows.append(history_chances(cars, _outside(misses.places, fold.held_out)).assign(config="history"))
        rows += fold_rows
    return pd.concat(rows, ignore_index=True)


def evaluate_drawn(folds: List[FoldModels], races: Dict[Tuple[int, int], Race], misses: Misses,
                   runs: int = RUNS, configs: Dict[str, Config] = DRAWN) -> pd.DataFrame:
    """Steps 3b and 3c: every DRAWN configuration (REAL included: step 3a's world, the real events run through
    world.run), for every dry development race, from STARTS, out of fold, judged on every car running at the start
    (one that retires is a miss; the history baseline too). Seeded like evaluate_configs; the events, the
    retirements and the answers have their own random stream (world.run)."""
    rows = []
    for fold in folds:
        fold_rows = []
        uncertainty = uncertainty_from(misses, fold.held_out, montecarlo.KEPT_DRAWS)
        by_place = place_offsets_from(misses, fold.held_out)
        for key in [k for k in fold.held_out if not races[k].mixed]:
            race = races[key]
            for label, lap in replay.starts(race):
                if label not in STARTS:
                    continue
                start = start_of(race, fold, label, lap, drawn=True)
                seed = SEED + 1000 * race.season + 10 * race.round + STARTS.index(label)
                for name, config in configs.items():
                    fold_rows.append(drawn_chances(start, uncertainty, config, fold.world, by_place, runs, seed)
                                     .assign(config=name))
        if not fold_rows:
            continue
        cars = pd.concat(fold_rows, ignore_index=True)
        cars = cars[cars["config"] == next(iter(configs))]                   # each car once, for the history baseline
        history = history_chances(cars, _outside(misses.places_all, fold.held_out))
        history[["moved_sim", "sim_passes"]] = np.nan                        # history moves nobody
        fold_rows.append(history.assign(config="history"))
        rows += fold_rows
    return pd.concat(rows, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 step 3: how true the Monte Carlo's chances are (development "
                                                 "races, out of fold), one draw at a time; --drawn: step 3b, with "
                                                 "the events drawn too")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS))
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--drawn", action="store_true", help="step 3b: the safety cars, VSCs, red flags and the "
                                                              "teams' answers drawn, against the real ones")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    races = {key: race_module.load(*key, root=args.lake) for key in evaluate.races_of(laps)}
    folds = [replay.fit_fold(laps, held, races, args.lake) for held in evaluate.race_folds(laps)]
    misses = Misses.combine(misses_of(races[k], f, args.lake) for f in folds for k in f.held_out)
    if args.drawn:
        print(folds[0].book.describe())
        print(folds[0].world.describe())
        rows = evaluate_drawn(folds, races, misses, runs=args.runs)
        print(summary(rows, drawn=True))
        print(f"\nReport: {lake.write_report(rows, DRAWN_REPORT, args.lake)}")
        return 0
    rows = evaluate_configs(folds, races, misses, runs=args.runs)
    print(summary(rows))
    print(f"\nReport: {lake.write_report(rows, REPORT, args.lake)}")
    return 0


def shuffling(rows: pd.DataFrame) -> pd.DataFrame:
    """Per configuration (rows of one start): places moved per car, simulated and real, the extra share; overtakes
    per race simulated and real, all races and the races with a real safety car after the start."""
    out = {}
    for name, g in rows.groupby("config", sort=False):
        f = g.dropna(subset=["moved_sim"])
        per_start = g.drop_duplicates(["season", "round"])
        sc = per_start[per_start["kind_after"] == "SC"]
        if f.empty and per_start["sim_passes"].isna().all():
            continue
        out[name] = {"moved": f["moved_sim"].mean(), "real moved": f["moved_real"].mean(),
                     "extra": f["moved_sim"].mean() / f["moved_real"].mean() - 1.0,
                     "overtakes": per_start["sim_passes"].mean(), "real overtakes": per_start["real_passes"].mean(),
                     "SC races": sc["sim_passes"].mean(), "real SC races": sc["real_passes"].mean()}
    return pd.DataFrame(out).T


def summary(rows: pd.DataFrame, drawn: bool = False) -> str:
    lines = []
    kept = FULL if drawn else kept_config()
    for start in STARTS:
        picked = rows[rows["start"] == start]
        if picked.empty:
            continue
        cars = int((picked["config"] == picked["config"].iloc[0]).sum())
        races = picked[["season", "round"]].drop_duplicates().shape[0]
        lines.append(f"\nFrom {start} ({cars} cars in {races} dry races):")
        table = pd.concat([scores(group).assign(config=name) for name, group in picked.groupby("config", sort=False)])
        order = list(dict.fromkeys(picked["config"]))
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            for values in (["brier", "log_loss"], ["calibration_error", "luck_p95", "off_by_luck"],
                           ["sure_yes_gap", "sure_no_gap"]):
                columns = pd.MultiIndex.from_product([values, list(EVENTS)], names=[None, "event"])
                lines.append(table.pivot(index="config", columns="event", values=values)
                             .reindex(index=order, columns=columns).round(3).to_string())
            verdicts = (drawn_comparisons if drawn else comparisons)(picked)
            if len(verdicts):
                verdicts["brier"] = [f"{d:+.4f} ({lo:+.4f} to {hi:+.4f}) {v}" for d, lo, hi, v in
                                     verdicts[["difference", "low", "high", "verdict"]].itertuples(index=False)]
                lines.append("Paired over races, Brier score (negative: the first one's chances are truer):")
                lines.append(verdicts.pivot(index="compared", columns="event", values="brier")
                             .reindex(index=list(dict.fromkeys(verdicts["compared"])), columns=list(EVENTS)).to_string())
            if drawn and "moved_sim" in picked:
                lines.append("Shuffling and overtakes (places moved per car from the start to the flag, the real finishers; "
                             "overtakes per race on track after the start; real against simulated):")
                lines.append(shuffling(picked).reindex(order).dropna(how="all").round(3).to_string())
            if drawn and "kind_after" in picked:
                podium = picked.assign(squared=(picked["p_podium"] - picked["podium"].astype(float)) ** 2)
                lines.append("Podium Brier by what neutralised the race after the start (cars; the races' kinds):")
                by_kind = podium.pivot_table(index="kind_after", columns="config", values="squared", aggfunc="mean")
                by_kind["cars"] = podium[podium["config"] == REAL].groupby("kind_after").size()
                lines.append(by_kind.reindex(columns=order + ["cars"]).round(4).to_string())
        for event in EVENTS:
            for name in ([REAL, kept, FULL + AS_3B] if drawn else [kept]):
                lines.append(f"Reliability, {event}, '{name}'" + ("" if drawn else " (what the Monte Carlo draws)") + ":")
                lines.append(reliability(picked[picked["config"] == name], event).round(3).to_string())
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
