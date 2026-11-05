"""
Decisions (src/simulator/decisions.py) - Phase 4 step 2

What if a driver had done something else? The simulator (step 1) replays a race from any lap with the
real plans; here one driver's plan is changed (plans.py) and the race is simulated again. Rivals keep
their real plans, except one: the car ahead may answer the driver's first new stop (reactions.py, Rohan's
choice: rivals react from step 2 on). An answer is a chance p, so a "what if" is simulated twice, without
the answer and with it, and its outcome is weighted: expected rank = (1 - p) x rank + p x rank answered.

    what_if(race, fold, lap, driver, edit)   one changed plan, against the real one
    options(race, fold, lap, driver)         at the end of `lap`: carry on as planned, or box at the end of
                                             the next lap for each dry compound (instead of the next planned
                                             stop, or as an extra one), each with its expected rank

A "what if" cannot be checked against reality: it did not happen. What can be checked is the simulator at
the moments those decisions were really taken (out of fold, on 2023 + 2024; 2025 stays locked):

  UNDERCUT CALLS   every real undercut attempt (Phase 2 step 5): replayed from the lap before the stop with
                   the real plans, does the simulator say it worked? Against "it always fails" (52.4 %
                   right on 2023 + 2024) and a gap rule (it works if the gap was under 1 s: 76 % of those did).
  NEUTRALISATIONS  at every safety car and VSC: neighbouring cars on the road (the lap before it, same lap)
                   that chose differently, one boxed and one stayed out. Who finished ahead? Against "the
                   order holds" and "the one that stayed out stays ahead".
  ANSWERS          the cover model on the threats of races it was not fitted on (log loss, Brier, against
                   the base rate).
  UNDERCUT VALUE   for every real attempt, what waiting would have done: the driver stops on the same lap
                   as the car ahead instead. The simulator's value of undercutting, in seconds against the
                   car ahead after both stops (not checkable; it should agree with Phase 2's measured gains:
                   median +1.34 s, mostly the track time around the stops).

Result (2026-10-16, 2023 + 2024 out of fold, dry races):
  * Undercut calls, 102 real attempts (49.0 % worked): boosted+prior right 67.6 %, against 51.0 % for "it
    always fails" (Rohan expected under 55 %; Claude 65-75 %), but a one-line rule (it works if the gap was
    under 1 s) is right 69.6 %. Even the real pace (hindsight) only 68.6 %. With the real pace AND both cars'
    real pit-lane times: 77.5 %, gain off by 1.37 s (median; 1.96 with the forecast pace). What the
    simulator cannot know in advance is how the stops go: Phase 2 found the stop part of a gain follows the
    stationary-time difference (correlation 0.98). That is step 3's (slow stops drawn: a probability).
  * Neutralisations, 91 pairs of neighbours that chose differently: boosted+prior right 72.5 % (SC 75.0 %,
    VSC 66.7 %) against "the order holds" 58.2 % and "the one that stayed out stays ahead" 61.5 %; the real
    pace 90.1 %.
  * Answers: 354 threats in unseen races, covered 30.2 %; log loss 0.585 against 0.613, Brier 0.198 / 0.211.
  * Undercut value: against stopping on the same lap as the car ahead, undercutting was worth +0.75 s
    (median; mean +1.06), positive in 97 % of the 102 attempts: fresh tyres a lap or two earlier.

Run it on the development races (~30 min: the folds are fitted like replay.py's), or look at one moment
(models fitted on every other race, ~3 min):

    python -m src.simulator.decisions
    python -m src.simulator.decisions --options 2024 13 NOR 43
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics import undercuts
from src.analytics.laps import classify_laps
from src.analytics.neutralisations import SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.analytics.stops import stops
from src.models import evaluate
from src.pipeline import lake
from src.simulator import engine, plans, race as race_module, reactions, replay
from src.simulator.plans import DRY_COMPOUNDS
from src.simulator.race import Race
from src.simulator.replay import FoldModels

BEST = "boosted+prior"
REAL_STOPS = "hindsight+real stops"
CHECK_SOURCES = ("hindsight", "boosted", BEST)
CALL_SOURCES = CHECK_SOURCES + (REAL_STOPS,)
GAP_RULE_SEC = 1.0                     # the hand rule: an undercut from closer than this works
CALLS_REPORT, PAIRS_REPORT, VALUE_REPORT = "sim_undercut_calls", "sim_neutralisation_pairs", "sim_undercut_value"


def simulate(race: Race, fold: FoldModels, lap: int, planned: pd.DataFrame, source: str = BEST,
             stop_extra: Optional[Dict[Tuple[str, int], float]] = None) -> engine.Simulation:
    """The race from the moment of race.snapshot(race, lap) with these plans and this pace source."""
    state = race_module.start_state(race, lap)
    pace = replay.pace_of(source, fold, race, lap, planned)
    return engine.simulate(state, planned, pace, race_module.lap_conditions(race), fold.book.rules_for(race),
                           race.total_laps, stop_extra=stop_extra)


def rank_of(simulation: engine.Simulation, driver: str) -> float:
    row = simulation.finish[simulation.finish["driver_code"] == driver]
    return float(row["rank"].iloc[0]) if len(row) else np.nan


def finish_clock(simulation: engine.Simulation, driver: str) -> float:
    """When the driver took the flag (NaN if it did not); compare only at the same number of laps."""
    row = simulation.finish[simulation.finish["driver_code"] == driver]
    return float(row["clock_sec"].iloc[0]) if len(row) else np.nan


def gap_at(simulation: engine.Simulation, b: str, a: str, lap: int) -> float:
    """b's crossing minus a's at the end of `lap` (negative: b ahead)."""
    at = simulation.laps[simulation.laps["lap_number"] == lap].set_index("driver_code")["clock_sec"]
    return float(at[b] - at[a]) if b in at.index and a in at.index else np.nan


# --- a "what if" ------------------------------------------------------------------------------------

@dataclass
class WhatIf:
    """One changed plan for one driver, simulated against the real plans."""
    driver: str
    problems: List[str]                         # rules the change breaks (empty: legal)
    real: engine.Simulation
    changed: engine.Simulation
    answer: Optional[reactions.Answer]          # the car ahead's possible cover, if any
    answered: Optional[engine.Simulation]       # the race in which it covers

    @property
    def chance(self) -> float:
        return self.answer.chance if self.answer is not None else 0.0

    def expected_rank(self) -> float:
        rank = rank_of(self.changed, self.driver)
        if self.answered is None:
            return rank
        return (1.0 - self.chance) * rank + self.chance * rank_of(self.answered, self.driver)

    def expected_time_vs_real(self) -> float:
        """Seconds at the flag against the real plan (negative: earlier), weighted over the answer."""
        real = finish_clock(self.real, self.driver)
        changed = finish_clock(self.changed, self.driver) - real
        if self.answered is None:
            return changed
        return (1.0 - self.chance) * changed + self.chance * (finish_clock(self.answered, self.driver) - real)


def what_if(race: Race, fold: FoldModels, lap: int, driver: str, edit: Callable[[pd.DataFrame], pd.DataFrame],
            source: str = BEST, rivals_react: bool = True) -> WhatIf:
    """Simulate the real plans and `edit(real plans)` from the moment of snapshot(race, lap)."""
    real_plan = race_module.plan(race, lap)
    changed_plan = edit(real_plan)
    real = simulate(race, fold, lap, real_plan, source)
    changed = simulate(race, fold, lap, changed_plan, source)
    answer = answered = None
    if rivals_react and fold.cover is not None:
        # The crossings the answer looks at: the start's (a stop on the first planned lap) and the simulated ones.
        state = race_module.start_state(race, lap).reset_index()
        crossings = pd.concat([state.rename(columns={"laps_done": "lap_number"})[["driver_code", "lap_number", "clock_sec"]],
                               changed.laps[["driver_code", "lap_number", "clock_sec"]]], ignore_index=True)
        answer = reactions.answer(race, real_plan, changed_plan, driver, crossings,
                                  race_module.lap_conditions(race), fold.cover)
        if answer is not None:
            answered = simulate(race, fold, lap, answer.plan, source)
    return WhatIf(driver=driver, problems=plans.new_problems(race, real_plan, changed_plan, driver),
                  real=real, changed=changed, answer=answer, answered=answered)


def options(race: Race, fold: FoldModels, lap: int, driver: str, source: str = BEST,
            rivals_react: bool = True) -> pd.DataFrame:
    """
    At the end of `lap`: the driver carries on as planned, or boxes at the end of the next lap for each
    dry compound (a new set): instead of its next planned stop, or as an extra stop if none is left.
    One row per option: expected rank, seconds at the flag against carrying on as planned (weighted over
    the answer), rank without and with the rival's answer, its chance, the rules it breaks.
    """
    real_plan = race_module.plan(race, lap)
    upcoming = [s for s in plans.stops_of(real_plan, driver) if s > lap]
    rows = []
    real = simulate(race, fold, lap, real_plan, source)
    rows.append({"option": "as planned" + (f" (stop lap {upcoming[0]})" if upcoming else " (no stop left)"),
                 "expected_rank": rank_of(real, driver), "rank": rank_of(real, driver), "seconds_vs_planned": 0.0,
                 "answered_by": None, "chance": 0.0, "rank_if_answered": np.nan, "problems": ""})
    for compound in DRY_COMPOUNDS:
        def edit(plan, compound=compound):
            if upcoming:
                plan = plans.stay_out(plan, driver, upcoming[0])
            return plans.box(plan, driver, lap + 1, compound)
        result = what_if(race, fold, lap, driver, edit, source, rivals_react)
        rows.append({"option": f"box lap {lap + 1}, new {compound.lower()}" + (" (instead)" if upcoming else " (extra)"),
                     "expected_rank": result.expected_rank(), "rank": rank_of(result.changed, driver),
                     "seconds_vs_planned": result.expected_time_vs_real(),
                     "answered_by": result.answer.driver if result.answer else None, "chance": result.chance,
                     "rank_if_answered": rank_of(result.answered, driver) if result.answered is not None else np.nan,
                     "problems": "; ".join(result.problems)})
    return pd.DataFrame(rows)


# --- checks at real decision moments ----------------------------------------------------------------

def attempts(race: Race, root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """The race's real undercut attempts (Phase 2 step 5)."""
    classified = classify_laps(lake.read_table(lake.table_path(lake.SILVER, "laps", *race.key, "R", root)))
    events = undercuts.undercut_events(classified, stops(classified))
    return events[events["status"] == undercuts.ATTEMPT].reset_index(drop=True)


def undercut_calls(race: Race, fold: FoldModels, sources: Iterable[str] = CALL_SOURCES,
                   root: str = lake.DEFAULT_LAKE_ROOT) -> List[dict]:
    """Every real attempt replayed from the lap before the stop with the real plans: worked or not, and the
    gain (gap before - gap after the car ahead's out-lap), simulated and real. Source REAL_STOPS (a diagnosis,
    not a forecast): hindsight pace AND both cars' real pit-lane times (each stop's time against the race's
    typical green stop, added to its loss)."""
    rows = []
    classified = classify_laps(lake.read_table(lake.table_path(lake.SILVER, "laps", *race.key, "R", root)))
    stop_table = stops(classified)
    typical = stop_table.loc[stop_table["condition"] == "green", "pit_lane_sec"].median()
    for _, e in attempts(race, root).iterrows():
        b, a, n, m = e["driver_code"], e["car_ahead"], int(e["in_lap"]), int(e["car_ahead_in_lap"])
        lap = n - 1
        planned = race_module.plan(race, lap)
        if not {a, b} <= set(planned["driver_code"]):
            continue
        extra = {(b, n): float(e["pit_lane_sec"]) - typical, (a, m): float(e["car_ahead_pit_lane_sec"]) - typical}
        extra = {k: v for k, v in extra.items() if np.isfinite(v)}
        for source in sources:
            if source == REAL_STOPS:
                sim = simulate(race, fold, lap, planned, "hindsight", stop_extra=extra)
            else:
                sim = simulate(race, fold, lap, planned, source)
            after = gap_at(sim, b, a, m + 1)
            rows.append({"season": race.season, "round": race.round, "event_name": race.event_name, "driver": b,
                         "car_ahead": a, "in_lap": n, "car_ahead_in_lap": m, "source": source,
                         "gap_before_sec": float(e["gap_before_sec"]), "real_gain_sec": float(e["gain_sec"]),
                         "real_worked": bool(e["worked"]), "sim_gap_after_sec": after,
                         "sim_gain_sec": float(e["gap_before_sec"]) - after if np.isfinite(after) else np.nan,
                         "sim_worked": bool(after < 0) if np.isfinite(after) else np.nan})
    return rows


def undercut_value(race: Race, fold: FoldModels, source: str = BEST, root: str = lake.DEFAULT_LAKE_ROOT) -> List[dict]:
    """For every real attempt: the simulated gap to the car ahead after both stops as it was done, and if the
    driver had waited and stopped on the same lap as the car ahead. value = gap waiting - gap undercutting
    (positive: the undercut was worth that many seconds)."""
    rows = []
    for _, e in attempts(race, root).iterrows():
        b, a, n, m = e["driver_code"], e["car_ahead"], int(e["in_lap"]), int(e["car_ahead_in_lap"])
        lap = n - 1
        try:
            result = what_if(race, fold, lap, b, lambda plan: plans.move_stop(plan, b, n, m), source)
        except plans.PlanError:
            continue
        done, waited = gap_at(result.real, b, a, m + 1), gap_at(result.changed, b, a, m + 1)
        rows.append({"season": race.season, "round": race.round, "event_name": race.event_name, "driver": b,
                     "car_ahead": a, "in_lap": n, "car_ahead_in_lap": m, "real_gain_sec": float(e["gain_sec"]),
                     "real_track_gain_sec": float(e["track_gain_sec"]), "gap_undercut_sec": done,
                     "gap_waiting_sec": waited, "value_sec": waited - done})
    return rows


def neutralisation_pairs(race: Race, fold: FoldModels, sources: Iterable[str] = CHECK_SOURCES) -> List[dict]:
    """At every safety car and VSC after the first start lap: neighbours on the road (same lap) when the lap
    before it ended, one of which boxed during it and one of which stayed out, both finishers. Who finished
    ahead, really and simulated (real plans from the lap before)."""
    first = race_module.first_start_lap(race)
    if first is None:
        return []
    rows = []
    real_ranks = race_module.finish_ranks(race)
    for kind, first_lap, last_lap in zip(race.periods["kind"], race.periods["first_lap"], race.periods["last_lap"]):
        if kind not in (SAFETY_CAR, VIRTUAL_SAFETY_CAR) or int(first_lap) - 1 <= first:
            continue
        lap = int(first_lap) - 1
        state = race_module.start_state(race, lap)
        planned = race_module.plan(race, lap)
        boxed = set(planned.loc[planned["is_tyre_change"] & planned["lap_number"].between(int(first_lap), int(last_lap)),
                                "driver_code"])
        order = list(state.index)
        pairs = [(x, y) for x, y in zip(order, order[1:])
                 if state.at[x, "laps_done"] == state.at[y, "laps_done"] and (x in boxed) != (y in boxed)
                 and x in real_ranks and y in real_ranks]
        if not pairs:
            continue
        sims = {source: simulate(race, fold, lap, planned, source) for source in sources}
        for ahead, behind in pairs:
            row = {"season": race.season, "round": race.round, "event_name": race.event_name, "kind": kind,
                   "first_lap": int(first_lap), "ahead": ahead, "behind": behind, "boxed": ahead if ahead in boxed else behind,
                   "real_ahead_first": real_ranks[ahead] < real_ranks[behind]}
            for source, sim in sims.items():
                row[source] = rank_of(sim, ahead) < rank_of(sim, behind)
            rows.append(row)
    return rows


def answers_check(fold: FoldModels, races: Iterable[Race], root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """The cover model's chances on the threats of races it was not fitted on."""
    frames = [reactions.threats(r, root) for r in races]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["covered"] + reactions.INPUTS)
    return rows.assign(chance=fold.cover.chances(rows) if len(rows) else [])


# --- command line -----------------------------------------------------------------------------------

def share(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.astype(bool).mean()) if len(series) else np.nan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 step 2: the simulator at real decision moments (undercuts, "
                                                 "safety cars, VSCs), the rivals' answers, and the value of each undercut")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS))
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--options", nargs=4, metavar=("SEASON", "ROUND", "DRIVER", "LAP"),
                        help="only show one driver's options at the end of one lap (models fitted without that race)")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    races = {key: race_module.load(*key, root=args.lake) for key in evaluate.races_of(laps)}
    if args.options:
        key, driver, lap = (int(args.options[0]), int(args.options[1])), args.options[2], int(args.options[3])
        if key not in races:
            parser.error(f"{key} is not among the races of seasons {args.season} in the lake.")
        fold = replay.fit_fold(laps, [key], races, args.lake)
        table = options(races[key], fold, lap, driver)
        with pd.option_context("display.width", 200):
            print(f"{races[key].event_name} {key[0]}, {driver} at the end of lap {lap} ({BEST}; rivals answer):")
            print(table.round(2).to_string(index=False))
        return 0
    calls, pairs, values, answered = [], [], [], []
    for held_out in evaluate.race_folds(laps):
        fold = replay.fit_fold(laps, held_out, races, args.lake)
        for key in held_out:
            race = races[key]
            if race.mixed:
                continue
            calls += undercut_calls(race, fold, root=args.lake)
            pairs += neutralisation_pairs(race, fold)
            values += undercut_value(race, fold, root=args.lake)
        answered.append(answers_check(fold, [races[k] for k in held_out if not races[k].mixed], args.lake))
    calls, pairs, values = pd.DataFrame(calls), pd.DataFrame(pairs), pd.DataFrame(values)
    answered = pd.concat([a for a in answered if len(a)], ignore_index=True) if any(len(a) for a in answered)         else pd.DataFrame(columns=["covered", "chance"])
    print(summary(calls, pairs, values, answered))
    print(f"\nReports: {lake.write_report(calls, CALLS_REPORT, args.lake)}, {lake.write_report(pairs, PAIRS_REPORT, args.lake)}, "
          f"{lake.write_report(values, VALUE_REPORT, args.lake)}")
    return 0


def summary(calls: pd.DataFrame, pairs: pd.DataFrame, values: pd.DataFrame, answered: pd.DataFrame) -> str:
    lines = []
    if len(calls):
        one = calls[calls["source"] == calls["source"].iloc[0]]
        lines += [f"Undercut calls: {len(one)} real attempts in dry races, {share(one['real_worked']):.1%} worked.",
                  f"  'it always fails' right {1 - share(one['real_worked']):.1%}; gap rule (works under {GAP_RULE_SEC:g} s) "
                  f"right {share((one['gap_before_sec'] < GAP_RULE_SEC) == one['real_worked']):.1%}"]
        for source, group in calls.groupby("source", sort=False):
            right = (group["sim_worked"] == group["real_worked"])
            error = (group["sim_gain_sec"] - group["real_gain_sec"]).abs()
            both = group[["sim_gain_sec", "real_gain_sec"]].dropna()
            correlation = both["sim_gain_sec"].corr(both["real_gain_sec"]) if len(both) >= 3 else np.nan
            lines.append(f"  {source}: right {share(right):.1%}; gain off by {error.mean():.2f} s on average "
                         f"(median {error.median():.2f}); correlation {correlation:.2f}")
    if len(pairs):
        lines += ["", f"Neutralisations: {len(pairs)} pairs of neighbours that chose differently (one boxed, one stayed out)."]
        for kind, group in [("all", pairs)] + list(pairs.groupby("kind")):
            truth = group["real_ahead_first"]
            stayed_out_ahead = group["ahead"] != group["boxed"]
            lines.append(f"  {kind} ({len(group)}): order holds {share(truth):.1%}, stayed-out stays ahead "
                         f"{share(truth == stayed_out_ahead):.1%}; " +
                         ", ".join(f"{s} {share(group[s] == truth):.1%}" for s in CHECK_SOURCES if s in group))
    if len(answered):
        y, p = answered["covered"].astype(float).to_numpy(), np.clip(answered["chance"].to_numpy(float), 1e-9, 1 - 1e-9)
        base = np.clip(y.mean(), 1e-9, 1 - 1e-9)
        lines += ["", f"Answers: {len(answered)} threats in races the cover model never saw; covered {y.mean():.1%}. "
                      f"Log loss {-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean():.3f} (base rate "
                      f"{-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean():.3f}), Brier {((p - y) ** 2).mean():.3f} "
                      f"(base {((base - y) ** 2).mean():.3f})."]
    if len(values):
        v = values["value_sec"].dropna()
        lines += ["", f"Undercut value ({BEST}): {len(v)} attempts; undercutting instead of stopping with the car ahead "
                      f"was worth {v.median():+.2f} s (median; mean {v.mean():+.2f}), positive in {(v > 0).mean():.0%}. "
                      f"Measured real gains: median {values['real_gain_sec'].median():+.2f} s."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
