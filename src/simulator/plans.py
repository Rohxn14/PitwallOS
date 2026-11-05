"""
Changing a plan (src/simulator/plans.py) - Phase 4 step 2

A plan (race.plan) is every lap a car still has to drive: the tyres on each lap and its pit passages. A
"what if" changes ONE driver's plan; the edits:

    box(plan, driver, lap, compound, fresh)   a tyre change at the end of `lap` (its in-lap): a new set
                                              starts the out-lap at age 1, a used one at USED_AGE
    stay_out(plan, driver, in_lap)            no passage at the end of `in_lap`: the tyres stay on
    move_stop(plan, driver, in_lap, to_lap)   the same stop (the same set fitted), at the end of another lap

After an edit every lap of that driver is rebuilt from its passages: a stint runs from its out-lap, the
tyre age growing by one a lap (Gold's tyre_age is FastF1's TyreLife: the laps the set has done including
this one, so a new set's out-lap is 1), laps into the stint from 1 on the out-lap. A pass-through keeps
the tyres. The laps before the start are history: they are never edited.

USED_AGE: the median age of a used set on its first race lap, by compound (2023-2025, 1,241 dry
driver-races): soft 4 (qualifying sets), medium 3, hard 2.

`problems` lists what breaks the rules (empty: legal), over the whole race, history included:
  * a dry race must use two different dry compounds (soft, medium, hard), unless intermediates or wets
    were used;
  * no more stints on a compound than the sets of the weekend's allocation (ALLOCATION: 2 hard, 3 medium,
    8 soft; a sprint weekend has 4 medium and 6 soft). A stint counts as a set, so a set fitted twice is
    counted twice: the check is strict. Measured on 2023-2025: nobody ran 3 hard stints except 4 odd cases
    (red flags, a set refitted), 3 medium stints 16 times (10 of them Qatar 2023, a sprint weekend).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.models.evaluate import WET_COMPOUNDS
from src.simulator.race import PLAN_COLUMNS, Race

TYRE_CHANGE, PASS_THROUGH = "tyre_change", "pass_through"
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
USED_AGE = {"SOFT": 4.0, "MEDIUM": 3.0, "HARD": 2.0}
ALLOCATION = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}
SPRINT_ALLOCATION = {"SOFT": 6, "MEDIUM": 4, "HARD": 2}


class PlanError(ValueError):
    """An edit that cannot be made (a lap outside the plan, no stop where one is expected)."""


# --- a driver's plan as passages ------------------------------------------------------------------

def _passages(rows: pd.DataFrame) -> Dict[int, dict]:
    """{in-lap: passage} of one driver's rows (in lap order); a tyre change carries the set it fits."""
    by_lap = rows.set_index("lap_number")
    out = {}
    for lap, row in by_lap[by_lap["is_in_lap"]].iterrows():
        passage = {"kind": row["pit_kind"]}
        if row["pit_kind"] == TYRE_CHANGE and lap + 1 in by_lap.index:
            fitted = by_lap.loc[lap + 1]
            passage.update(compound=fitted["compound"], age=float(fitted["tyre_age"]), fresh=fitted["fresh_tyre"])
        out[int(lap)] = passage
    return out


def _rebuild(driver: str, first: pd.Series, laps: List[int], passages: Dict[int, dict],
             previous: Optional[dict] = None) -> pd.DataFrame:
    """The driver's rows on `laps`, from the tyres on the first of them (its real row) and the passages.
    previous: the passage at the end of the lap before the first (it makes the first lap an out-lap)."""
    compound, age, stint_lap, fresh = first["compound"], float(first["tyre_age"]), int(first["stint_lap"]), first["fresh_tyre"]
    rows = []
    for i, lap in enumerate(laps):
        if i > 0:
            previous = passages.get(lap - 1)
            if previous is not None and previous["kind"] == TYRE_CHANGE:     # a new set from this out-lap
                compound, age, stint_lap, fresh = previous["compound"], previous["age"], 1, previous["fresh"]
            else:
                age, stint_lap = age + 1.0, stint_lap + 1
        passage = passages.get(lap)
        rows.append({"driver_code": driver, "lap_number": lap, "compound": compound, "tyre_age": age,
                     "stint_lap": stint_lap, "fresh_tyre": fresh,
                     "is_wet_tyre": isinstance(compound, str) and compound in WET_COMPOUNDS,
                     "is_in_lap": passage is not None,
                     "is_out_lap": (lap - 1) in passages or (i == 0 and previous is not None),
                     "is_tyre_change": passage is not None and passage["kind"] == TYRE_CHANGE,
                     "pit_kind": passage["kind"] if passage is not None else None})
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def _edit(plan: pd.DataFrame, driver: str, change) -> pd.DataFrame:
    """
    Apply change(passages, laps) to one driver's passages and rebuild that driver's rows from the first lap
    the change touches; the laps before it keep their real rows (FastF1's new-set flag sometimes flips
    inside a stint, so rebuilding untouched laps would change them: 54 of 888 drivers on 2023-2024).
    """
    mine = plan[plan["driver_code"] == driver].sort_values("lap_number").reset_index(drop=True)
    if mine.empty:
        raise PlanError(f"{driver} has no laps left in this plan.")
    laps = mine["lap_number"].astype(int).tolist()
    before = _passages(mine)
    passages = {lap: dict(p) for lap, p in before.items()}
    change(passages, laps)
    touched = [lap for lap in set(before) | set(passages) if before.get(lap) != passages.get(lap)]
    if not touched:
        return plan.copy()
    first = min(touched)
    kept = mine[mine["lap_number"] < first]
    rebuilt = _rebuild(driver, mine[mine["lap_number"] == first].iloc[0], [lap for lap in laps if lap >= first],
                       passages, previous=passages.get(first - 1))
    out = pd.concat([plan[plan["driver_code"] != driver], kept[PLAN_COLUMNS], rebuilt], ignore_index=True)
    return out.sort_values(["driver_code", "lap_number"]).reset_index(drop=True)


def box(plan: pd.DataFrame, driver: str, lap: int, compound: str, fresh: bool = True) -> pd.DataFrame:
    """A tyre change at the end of `lap` for a set of `compound` (new, or used at USED_AGE)."""
    def change(passages, laps):
        if lap not in laps or lap == laps[-1]:
            raise PlanError(f"{driver} cannot stop at the end of lap {lap}: its planned laps are {laps[0]}-{laps[-1]}.")
        passages[lap] = {"kind": TYRE_CHANGE, "compound": compound, "age": 1.0 if fresh else USED_AGE[compound],
                         "fresh": 1.0 if fresh else 0.0}
    return _edit(plan, driver, change)


def stay_out(plan: pd.DataFrame, driver: str, in_lap: int) -> pd.DataFrame:
    """No pit passage at the end of `in_lap`: the car stays out on its tyres."""
    def change(passages, laps):
        if in_lap not in passages:
            raise PlanError(f"{driver} has no stop at the end of lap {in_lap}.")
        del passages[in_lap]
    return _edit(plan, driver, change)


def move_stop(plan: pd.DataFrame, driver: str, in_lap: int, to_lap: int) -> pd.DataFrame:
    """The stop made at the end of `in_lap`, made at the end of `to_lap` instead (the same set fitted)."""
    def change(passages, laps):
        if in_lap not in passages or passages[in_lap]["kind"] != TYRE_CHANGE:
            raise PlanError(f"{driver} has no tyre change at the end of lap {in_lap}.")
        if to_lap not in laps or to_lap == laps[-1]:
            raise PlanError(f"{driver} cannot stop at the end of lap {to_lap}: its planned laps are {laps[0]}-{laps[-1]}.")
        passages[to_lap] = passages.pop(in_lap)
    return _edit(plan, driver, change)


def stops_of(plan: pd.DataFrame, driver: str) -> List[int]:
    """The in-laps of a driver's planned tyre changes."""
    mine = plan[(plan["driver_code"] == driver) & plan["is_tyre_change"]]
    return sorted(int(lap) for lap in mine["lap_number"])


# --- the rules a plan must keep -------------------------------------------------------------------

def stints(race: Race, plan: pd.DataFrame, driver: str) -> List[str]:
    """The compound of each stint over the whole race, history (the race's rows before the plan) included."""
    first_planned = int(plan.loc[plan["driver_code"] == driver, "lap_number"].min())
    history = race.laps[(race.laps["driver_code"] == driver) & (race.laps["lap_number"] < first_planned)]
    rows = pd.concat([history[["lap_number", "compound", "is_in_lap", "is_tyre_change"]],
                      plan.loc[plan["driver_code"] == driver, ["lap_number", "compound", "is_in_lap", "is_tyre_change"]]])
    rows = rows.sort_values("lap_number")
    compounds, previous_change = [], True
    for compound, change in zip(rows["compound"], rows["is_tyre_change"].astype(bool)):
        if previous_change and isinstance(compound, str):
            compounds.append(compound)
        previous_change = change
    return compounds


def problems(race: Race, plan: pd.DataFrame, driver: str, sprint_weekend: bool = False) -> List[str]:
    """What breaks the rules in a driver's race (history + plan); empty when the plan is legal. The two
    compounds rule only binds a car that reaches the flag (a retirement may have run one)."""
    used = stints(race, plan, driver)
    out = []
    finishes = int(plan.loc[plan["driver_code"] == driver, "lap_number"].max()) >= race.total_laps
    if finishes and not any(c in WET_COMPOUNDS for c in used) and len({c for c in used if c in DRY_COMPOUNDS}) < 2:
        out.append("a dry race needs two different dry compounds")
    allocation = SPRINT_ALLOCATION if sprint_weekend else ALLOCATION
    for compound, sets in allocation.items():
        if used.count(compound) > sets:
            out.append(f"{used.count(compound)} {compound.lower()} stints, {sets} sets in the allocation")
    return out


def new_problems(race: Race, real_plan: pd.DataFrame, plan: pd.DataFrame, driver: str,
                 sprint_weekend: bool = False) -> List[str]:
    """The problems an edited plan has that the real one did not: what a "what if" must not add. (The
    real plans of 2023-2024 count 21 problems by the strict stint count: red-flag refits, retirements.)"""
    real = set(problems(race, real_plan, driver, sprint_weekend))
    return [p for p in problems(race, plan, driver, sprint_weekend) if p not in real]
