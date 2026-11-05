"""
Replaying real races (src/simulator/replay.py) - Phase 4 step 1

Can the simulator reproduce races that happened? Before any "what if" is trusted, it has to.

    For every development race (2023 + 2024; 2025 stays locked), out of fold:
      1. the pace models and the rule book are fitted on the OTHER races (the same 5 folds of whole
         races as Phase 3), so nothing is tuned on the race being replayed
      2. the real race is taken as it was when the start lap ended (race.py), with its real plan (stops,
         tyres, retirements) and its real events (safety cars, VSCs, red flags)
      3. the rest of the race is simulated (engine.py) with each pace source in turn
      4. the simulated order at the flag is compared with the real one

STARTS
    "lap 5"        the first lap from lap 5 on at which every car has driven a clean lap (the headline)
    "halfway"      the same from half distance
    "<kind> <lap>" the end of the lap before each safety car, VSC or red flag: the moment a team decides

SCORES (only the cars that took the chequered flag and were running at the start are ranked)
    places off     mean over those cars of |simulated rank - real rank|; real rank = the order at the line
    do nothing     the same for "the order at the start lap holds to the flag": the bar to beat
    exact          the share of cars in exactly the right place
    gap error      mean |simulated - real| gap to the leader at the flag, cars on the leader's lap
Races where anyone ran intermediates or wets are scored apart (the pace model cannot see rain coming).
Over races, the mean is weighted by cars ranked, with a 95 % interval from the bootstrap over races,
and every source is compared with the do-nothing baseline on the same draws (paired).

PACE SOURCES
    persistence, linear, state_space, boosted   Phase 3's rungs 1-4 (pace.forecast)
    <rung>+prior  the same forecast with each driver's level blended with what was known before the race:
                  the grid and the team's pace in its previous race (pace.PacePrior, fitted per fold)
    hindsight     each driver's REAL pace after the start, per stint (pace.hindsight): not a forecast, the
                  simulator with a perfect pace model; what is left is the rules' error

Result on the development races (2026-10-15; 41 dry races, 716 cars ranked from lap 5; places off per car):

                         from lap 5          halfway     before a SC / VSC / red flag
    do nothing           2.15                1.54        1.74
    persistence          2.25                1.85        1.95
    linear               2.29                1.75        1.93
    state_space          2.29                1.62        1.87
    boosted              2.25                1.44        1.83
    linear+prior         1.75                1.20        1.58
    state_space+prior    1.75                1.18        1.57
    boosted+prior        1.70 (1.49-1.93)    1.12        1.54
    hindsight            1.03                0.68        1.07

  * The rungs alone are no better than nothing moving: from lap 5 a driver's pace AGAINST THE FIELD is
    off by ~0.3-0.5 s a lap, which over 50 laps is more than the gaps between cars. With the prior (the
    grid and the team's last race) every rung beats doing nothing from lap 5 and halfway (paired
    bootstrap, "better"); boosted+prior is the best at every start: -0.45 places from lap 5 (-0.62 to
    -0.26), -0.42 from halfway; before a neutralisation -0.20 (-0.45 to +0.04: no clear difference).
  * boosted+prior beats doing nothing in 30 of the 41 races (2 ties); 27 % of cars exactly right from lap
    5, 41 % from halfway; the gap to the leader at the flag off by 13.4 s (median) from lap 5, 8.2 s from
    halfway. By what neutralised the race after lap 5: clean 1.62 (nothing 2.10), VSC 1.40 (1.98), SC 2.15
    (2.38), red flag 1.26 (2.15, 2 races). Mixed races (rain): 2.07 against 2.34.
  * With each driver's real pace (hindsight) the rules alone are 1.03 places off from lap 5: the rest of
    the error is the pace forecast. Worst races even so: Singapore 2023 (3.20), Monaco 2024 (cars on fresh
    tyres pass where nobody can), Austria 2024; the hardest race to replay is Las Vegas 2023 (4.59; nothing
    moving 4.47).

Run it on the development races (~25 min: every rung, rule book and prior is fitted 5 times):

    python -m src.simulator.replay
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.models import evaluate
from src.models.boosted import Boosted
from src.models.linear import LinearDecomposition
from src.models.persistence import Persistence
from src.models.state_space import StateSpace
from src.pipeline import lake
from src.simulator import engine, pace, race as race_module, reactions, world as world_module
from src.simulator.race import Race
from src.simulator.rules import RuleBook, Rules, race_facts

RUNGS = {"persistence": Persistence, "linear": LinearDecomposition, "state_space": StateSpace, "boosted": Boosted}
HINDSIGHT = "hindsight"
WITH_PRIOR = "+prior"
PRIOR_RUNGS = ("linear", "state_space", "boosted")
PRIOR_STEP_LAPS = 10                # the pace prior is fitted on starts every 10 laps of each training race
SOURCES = tuple(RUNGS) + tuple(r + WITH_PRIOR for r in PRIOR_RUNGS) + (HINDSIGHT,)
LAP_5, HALFWAY = "lap 5", "halfway"
CLEAN = "clean"
REPORT = "sim_replay"


@dataclass
class FoldModels:
    """What one fold learned from its training races."""
    held_out: Tuple[Tuple[int, int], ...]
    rungs: Dict[str, object]
    book: RuleBook
    priors: Dict[str, pace.PacePrior]
    form: pd.DataFrame              # pace.team_form over every race in the lake (read before each race only)
    cover: Optional[reactions.CoverModel] = None    # step 2: how the car ahead answers an undercut
    world: Optional[world_module.World] = None      # step 3b: the events drawn, the box rule, stopping ages, tyre terms


def rung2_of(rung) -> LinearDecomposition:
    """The rung 2 whose terms a rung's forecast carries."""
    return rung if isinstance(rung, LinearDecomposition) else rung.rung2 if isinstance(rung, StateSpace) \
        else rung.rung3.rung2


def prior_examples(rung, race: Race, form: pd.DataFrame) -> pd.DataFrame:
    """For fitting a PacePrior: at starts every PRIOR_STEP_LAPS laps of a (training) race, each driver's
    relative base by the rung's forecast, relative grid position, the team's last race, and the realised
    relative pace after the start."""
    rung2 = rung2_of(rung)
    inputs = evaluate.model_inputs(race.laps)
    first = race_module.first_start_lap(race)
    team_last = pace.team_last_race(race, form)
    rows = []
    for lap in range(first or race.total_laps, race.total_laps - 10, PRIOR_STEP_LAPS):
        at_lap = race_module.last_rows(race, race_module.start_state(race, lap))
        if not len(at_lap) or not bool(evaluate.pace_seen(at_lap).all()):
            continue
        planned = race_module.plan(race, lap)
        forecast = pace.forecast(rung, race_module.history(race, lap), planned)
        rel = pace.implied_base(forecast, planned, pace.planned_terms(rung2, planned, race.total_laps))
        team = pd.Series({d: team_last.get(d, np.nan) for d in rel.index}, dtype=float)
        rows.append(pd.DataFrame({"lap": lap, "rel": rel, "grid": pace.relative_grid(race.grid, rel.index),
                                  "team": team, "realised": pace.realised_relative_pace(rung2, inputs, lap).reindex(rel.index)}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["lap", "rel", "grid", "team", "realised"])


def fit_priors(rungs: Dict[str, object], races: Iterable[Race], form: pd.DataFrame) -> Dict[str, pace.PacePrior]:
    """One PacePrior per rung in PRIOR_RUNGS, from these (training) races; dry races only."""
    dry = [r for r in races if not r.mixed]
    priors = {}
    for name in PRIOR_RUNGS:
        examples = [prior_examples(rungs[name], r, form) for r in dry]
        priors[name] = pace.PacePrior().fit(pd.concat(examples) if examples else pd.DataFrame())
    return priors


def fit_fold(laps: pd.DataFrame, held_out: Sequence[Tuple[int, int]], races: Dict[Tuple[int, int], Race],
             root: str = lake.DEFAULT_LAKE_ROOT) -> FoldModels:
    """Every rung, the rule book, the pace priors, the cover model and the drawn world (events, box rule), from the
    races not held out (team form: every race, read before each race only)."""
    train = laps[~evaluate.in_races(laps, held_out)]
    rungs = {name: make().fit(train) for name, make in RUNGS.items()}
    training = [races[key] for key in evaluate.races_of(train)]
    facts = [race_facts(r, rungs["linear"], root) for r in training]
    form = pace.team_form(races.values(), rungs["linear"])
    threats = [t for t in (reactions.threats(r, root) for r in training) if len(t)]
    cover = reactions.CoverModel.fit(pd.concat(threats, ignore_index=True) if threats
                                     else pd.DataFrame(columns=reactions.INPUTS + ["covered"]))
    return FoldModels(held_out=tuple(held_out), rungs=rungs, book=RuleBook.measure(facts),
                      priors=fit_priors(rungs, training, form), form=form, cover=cover,
                      world=world_module.fit_world(training, rung2_of(rungs["boosted"])))


def kind_after(race: Race, lap: int) -> str:
    """What neutralised the race after `lap`: red flag, else SC, else VSC, else clean."""
    kinds = set(race.periods.loc[race.periods["first_lap"] > lap, "kind"])
    for kind in (RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR):
        if kind in kinds:
            return kind
    return CLEAN


def starts(race: Race) -> List[Tuple[str, int]]:
    """(label, start lap) of every replay of this race (see STARTS above)."""
    first = race_module.first_start_lap(race)
    if first is None:
        return []
    out = [(LAP_5, first)]
    half = race_module.first_start_lap(race, max(first + 1, race.total_laps // 2))
    if half is not None and half < race.total_laps - 3:
        out.append((HALFWAY, half))
    for kind, first_lap in zip(race.periods["kind"], race.periods["first_lap"]):
        lap = int(first_lap) - 1
        if lap <= first or lap >= race.total_laps - 1:
            continue
        rows = race_module.last_rows(race, race_module.start_state(race, lap))
        if len(rows) and bool(evaluate.pace_seen(rows).all()):
            out.append((f"{kind} lap {int(first_lap)}", lap))
    return out


def pace_of(source: str, fold: FoldModels, race: Race, lap: int, planned: pd.DataFrame) -> pd.Series:
    name = "linear" if source == HINDSIGHT else source.replace(WITH_PRIOR, "")
    forecast = pace.forecast(fold.rungs[name], race_module.history(race, lap), planned)
    if source == HINDSIGHT:          # a driver with no clean lap after the start keeps rung 2's forecast
        return pace.hindsight(fold.rungs["linear"], evaluate.model_inputs(race.laps), planned).fillna(forecast)
    if source.endswith(WITH_PRIOR):
        terms = pace.planned_terms(rung2_of(fold.rungs[name]), planned, race.total_laps)
        return fold.priors[name].adjust(forecast, planned, terms, race.grid, pace.team_last_race(race, fold.form), lap)
    return forecast


def score(race: Race, lap: int, simulation: engine.Simulation) -> Dict[str, float]:
    """places off, do nothing, exact and gap error for one simulated race (see SCORES above)."""
    running = set(race_module.running_at(race, lap))
    real = race_module.finish_ranks(race, running)
    cars = list(real)
    simulated_order = [d for d in simulation.finish["driver_code"] if d in real]
    simulated = {d: i + 1 for i, d in enumerate(simulated_order)}
    hold = race_module.hold_order_ranks(race, lap, cars)
    off = np.array([abs(simulated[d] - real[d]) for d in cars], dtype=float)
    hold_off = np.array([abs(hold[d] - real[d]) for d in cars], dtype=float)
    # Gaps at the flag to the leader, cars on the leader's lap in the real race.
    real_times = race_module.finish_times(race, cars)
    on_lead_lap = simulation.finish[simulation.finish["laps"] == race.total_laps]
    sim_times = dict(zip(on_lead_lap["driver_code"], on_lead_lap["clock_sec"]))
    both = [d for d in real_times if d in sim_times]
    gap_error = np.nan
    if len(both) >= 2:
        real_gap = np.array([real_times[d] for d in both]) - min(real_times[d] for d in both)
        sim_gap = np.array([sim_times[d] for d in both]) - min(sim_times[d] for d in both)
        gap_error = float(np.mean(np.abs(sim_gap - real_gap)))
    return {"cars": len(cars), "places_off": float(off.mean()) if len(off) else np.nan,
            "do_nothing": float(hold_off.mean()) if len(hold_off) else np.nan,
            "exact": float((off == 0).mean()) if len(off) else np.nan, "gap_error_sec": gap_error,
            "passes": simulation.passes, "no_pace": simulation.no_pace}


def replay_race(race: Race, fold: FoldModels, sources: Iterable[str] = SOURCES,
                rules_changes: Optional[Dict] = None) -> List[dict]:
    """Every start of one race, with every pace source: one score row each."""
    rules = fold.book.rules_for(race, **(rules_changes or {}))
    conditions = race_module.lap_conditions(race)
    rows = []
    for label, lap in starts(race):
        state = race_module.start_state(race, lap)
        planned = race_module.plan(race, lap)
        for source in sources:
            simulation = engine.simulate(state, planned, pace_of(source, fold, race, lap, planned), conditions, rules,
                                         race.total_laps)
            rows.append({"season": race.season, "round": race.round, "event_name": race.event_name,
                         "start": label, "start_lap": lap, "kind_after": kind_after(race, lap), "mixed": race.mixed,
                         "source": source, **score(race, lap, simulation)})
    return rows


def run(seasons: Sequence[int] = evaluate.DEVELOPMENT_SEASONS, root: str = lake.DEFAULT_LAKE_ROOT,
        sources: Iterable[str] = SOURCES, rules_changes: Optional[Dict] = None,
        folds: Optional[List[FoldModels]] = None, log: Callable[[str], None] = print) -> Tuple[pd.DataFrame, List[FoldModels]]:
    """Replay every race of these seasons out of fold. Returns the score rows and the fitted folds
    (pass them back in to replay again without refitting)."""
    laps = evaluate.load_laps(seasons, root)
    races = {key: race_module.load(*key, root=root) for key in evaluate.races_of(laps)}
    if folds is None:
        folds = []
        for i, held_out in enumerate(evaluate.race_folds(laps)):
            started = time.time()
            folds.append(fit_fold(laps, held_out, races, root))
            log(f"fold {i + 1}: fitted on {len(races) - len(held_out)} races in {time.time() - started:.0f} s")
    rows = []
    for fold in folds:
        for key in fold.held_out:
            rows += replay_race(races[key], fold, sources, rules_changes)
    return pd.DataFrame(rows), folds


# --- summaries ----------------------------------------------------------------------------------------

def weighted(rows: pd.DataFrame, column: str) -> float:
    return float((rows[column] * rows["cars"]).sum() / rows["cars"].sum()) if rows["cars"].sum() else np.nan


def bootstrap(rows: pd.DataFrame, column: str, against: Optional[str] = None,
              samples: int = evaluate.BOOTSTRAP_SAMPLES, seed: int = evaluate.BOOTSTRAP_SEED) -> Tuple[float, float]:
    """95 % interval of the car-weighted mean of `column` (minus `against`: paired), resampling races."""
    per_race = rows.assign(value=rows[column] - (rows[against] if against else 0.0))
    per_race = per_race.assign(total=per_race["value"] * per_race["cars"]).groupby(["season", "round"])[["total", "cars"]].sum()
    if per_race.empty:
        return np.nan, np.nan
    drawn = evaluate.draw_races(len(per_race), samples, seed)
    totals, cars = per_race["total"].to_numpy(), per_race["cars"].to_numpy()
    means = totals[drawn].sum(axis=1) / cars[drawn].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summary(rows: pd.DataFrame, start: str = LAP_5) -> pd.DataFrame:
    """Per pace source, over the dry races at one start: places off with its interval, against doing
    nothing (paired), exact, gap error."""
    out = []
    picked = rows[(~rows["mixed"]) & (rows["start"] == start)] if start != "neutralised" else \
        rows[(~rows["mixed"]) & ~rows["start"].isin([LAP_5, HALFWAY])]
    for source, group in picked.groupby("source", sort=False):
        low, high = bootstrap(group, "places_off")
        d_low, d_high = bootstrap(group, "places_off", against="do_nothing")
        difference = weighted(group, "places_off") - weighted(group, "do_nothing")
        out.append({"source": source, "replays": len(group), "cars": int(group["cars"].sum()),
                    "places_off": weighted(group, "places_off"), "low": low, "high": high,
                    "do_nothing": weighted(group, "do_nothing"), "vs_do_nothing": difference,
                    "vs_low": d_low, "vs_high": d_high,
                    "verdict": "better" if d_high < 0 else "worse" if d_low > 0 else "no clear difference",
                    "exact": weighted(group, "exact"), "gap_error_sec": float(group["gap_error_sec"].median())})
    return pd.DataFrame(out).round(3)


def by_kind(rows: pd.DataFrame, start: str = LAP_5) -> pd.DataFrame:
    """Places off per pace source by what neutralised the race after the start (dry races), with doing nothing."""
    picked = rows[(~rows["mixed"]) & (rows["start"] == start)]
    if picked.empty:
        return pd.DataFrame()
    table: Dict[str, Dict[str, float]] = {}
    for (kind, source), group in picked.groupby(["kind_after", "source"], sort=False):
        table.setdefault(kind, {})[source] = weighted(group, "places_off")
    out = pd.DataFrame(table).T
    once = picked[picked["source"] == picked["source"].iloc[0]]          # the baseline is the same for every source
    out["do_nothing"] = pd.Series({kind: weighted(group, "do_nothing") for kind, group in once.groupby("kind_after")})
    out["races"] = once.groupby("kind_after").size()
    return out.round(3)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 step 1: replay the development races out of fold and "
                                                 "score the simulated order at the flag")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to replay (default: the development seasons, 2023 2024)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    rows, folds = run(args.season, args.lake)
    print(f"\n{rows[['season', 'round']].drop_duplicates().shape[0]} races replayed, "
          f"{len(rows) // len(SOURCES)} starts.\n")
    print(folds[0].book.describe())
    with pd.option_context("display.width", 220):
        for start in (LAP_5, HALFWAY, "neutralised"):
            print(f"\nFrom {start} (dry races; places off per car at the flag, 95 % interval over races; "
                  f"against doing nothing, paired):")
            print(summary(rows, start).to_string(index=False))
        print("\nFrom lap 5, by what neutralised the race after it (dry races):")
        print(by_kind(rows).to_string())
        mixed = rows[rows["mixed"] & (rows["start"] == LAP_5)]
        if len(mixed):
            print(f"\nMixed races (intermediates or wets), from lap 5: "
                  + ", ".join(f"{s} {weighted(g, 'places_off'):.2f}" for s, g in mixed.groupby("source", sort=False))
                  + f"; do nothing {weighted(mixed[mixed['source'] == SOURCES[0]], 'do_nothing'):.2f}")
    print(f"\nReport: {lake.write_report(rows, REPORT, args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
