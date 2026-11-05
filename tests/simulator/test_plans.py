"""
Phase 4 step 2: changing a plan (src/simulator/plans.py), on Qatar 2024.

An edit that changes nothing gives back the real plan exactly. Verstappen's one stop (lap 35, mediums to
hards) moved to lap 30: hards from lap 31 at age 1, and the field's trips through the pit lane behind the
safety car on laps 36-37 keep his tyres on. An extra stop, a used set, staying out on mediums all race (two
dry compounds needed), three hard stints (two sets), a stop outside the plan.
"""

import pandas as pd
import pytest

from src.simulator import plans
from src.simulator import race as R

QATAR = (2024, 23)


@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    race = R.load(*QATAR, root=golden_gold_lake)
    return race, R.plan(race, 6)


def rows_of(plan, driver):
    return plan[plan["driver_code"] == driver].set_index("lap_number")


def test_no_change_is_the_real_plan(qatar):
    race, plan = qatar
    for driver in plan["driver_code"].unique():
        same = plans._edit(plan, driver, lambda passages, laps: None)
        assert same.sort_values(["driver_code", "lap_number"]).reset_index(drop=True).equals(
            plan.sort_values(["driver_code", "lap_number"]).reset_index(drop=True))
    assert plans.stops_of(plan, "VER") == [35]
    assert all(plans.new_problems(race, plan, plan, d) == [] for d in plan["driver_code"].unique())


def test_moving_verstappens_stop_to_lap_30(qatar):
    race, plan = qatar
    moved = rows_of(plans.move_stop(plan, "VER", 35, 30), "VER")
    assert moved.at[30, "is_in_lap"] and moved.at[30, "pit_kind"] == "tyre_change"
    assert moved.loc[31, ["compound", "tyre_age", "stint_lap"]].tolist() == ["HARD", 1.0, 1]
    assert moved.at[31, "is_out_lap"] and not moved.at[35, "is_in_lap"] and not moved.at[36, "is_out_lap"]
    # Laps 36-37: led through the pit lane, no tyre change: the hards stay on, one lap older each lap.
    assert moved.loc[36:38, "tyre_age"].tolist() == [6.0, 7.0, 8.0]
    assert moved.loc[36:37, "pit_kind"].tolist() == ["pass_through", "pass_through"]
    # Before lap 30 nothing changed.
    assert moved.loc[7:29].equals(rows_of(plan, "VER").loc[7:29])
    assert plans.new_problems(race, plan, plans.move_stop(plan, "VER", 35, 30), "VER") == []


def test_an_extra_stop_and_a_used_set(qatar):
    race, plan = qatar
    extra = rows_of(plans.box(plan, "VER", 20, "HARD"), "VER")
    assert extra.loc[21, ["compound", "tyre_age", "stint_lap", "fresh_tyre"]].tolist() == ["HARD", 1.0, 1, 1.0]
    assert extra.at[35, "pit_kind"] == "tyre_change" and extra.loc[36, "tyre_age"] == 1.0   # the real stop stays
    used = rows_of(plans.box(plan, "VER", 20, "SOFT", fresh=False), "VER")
    assert used.loc[21, ["compound", "tyre_age", "fresh_tyre"]].tolist() == ["SOFT", plans.USED_AGE["SOFT"], 0.0]


def test_the_rules(qatar):
    race, plan = qatar
    stayed = plans.stay_out(plan, "VER", 35)
    assert plans.stops_of(stayed, "VER") == [] and set(rows_of(stayed, "VER")["compound"]) == {"MEDIUM"}
    assert plans.new_problems(race, plan, stayed, "VER") == ["a dry race needs two different dry compounds"]
    three_hards = plans.box(plans.box(plan, "VER", 15, "HARD"), "VER", 25, "HARD")
    assert plans.stints(race, three_hards, "VER") == ["MEDIUM", "HARD", "HARD", "HARD"]
    assert plans.new_problems(race, plan, three_hards, "VER") == ["3 hard stints, 2 sets in the allocation"]
    four_mediums = plans.box(plans.box(plans.box(plan, "VER", 12, "MEDIUM"), "VER", 20, "MEDIUM"), "VER", 28, "MEDIUM")
    assert "4 medium stints, 3 sets in the allocation" in plans.problems(race, four_mediums, "VER")
    assert all("medium" not in p for p in plans.problems(race, four_mediums, "VER", sprint_weekend=True))


def test_edits_that_cannot_be_made(qatar):
    _, plan = qatar
    with pytest.raises(plans.PlanError):
        plans.box(plan, "VER", 3, "HARD")                    # before the plan starts (lap 6 was the start)
    with pytest.raises(plans.PlanError):
        plans.box(plan, "VER", 57, "HARD")                   # the last lap: nothing after it
    with pytest.raises(plans.PlanError):
        plans.stay_out(plan, "VER", 20)                      # no stop there
    with pytest.raises(plans.PlanError):
        plans.move_stop(plan, "VER", 36, 30)                 # lap 36 was a pass-through, not a tyre change
