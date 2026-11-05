"""
Phase 4 step 1: what a replay takes from a real race (src/simulator/race.py), on the reference races.

Golden facts from sources other than the simulator: Race Control's safety cars in Qatar 2024 (laps 1-4,
35-39, a VSC on lap 40 that became a safety car, 40-42), Verstappen's one stop on lap 35 then the field
led through the pit lane on laps 36-37, the order at the line (Verstappen, Leclerc, Piastri, Russell),
Russell's pole with Verstappen second (his grid penalty). And the engine's bookkeeping: fed every car's
REAL lap times, with every rule switched off, it reproduces the real race exactly, every crossing to the
millisecond and the order at the flag (Qatar's safety cars and lapped cars, British's rain). The real overtakes are
counted by hand on a made-up three-car race (one on lap 2, one on lap 3, a car in the pit lane not counted, a lap behind
the safety car not counted).
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analytics.neutralisations import SAFETY_CAR
from src.models.evaluate import SeasonLocked
from src.simulator import race as R
from src.simulator.engine import simulate
from src.simulator.race import DEPLOY, END, FULL, GREEN
from src.simulator.rules import Rules

QATAR, BRITISH = (2024, 23), (2024, 12)
PHASES = ("deploy", "full", "end", "single")
NO_RULES = Rules(pit_loss_sec=0.0, sc_pit_loss_sec=0.0, vsc_pit_loss_sec=0.0, queue_gap_sec=0.0, follow_gap_sec=0.0,
                 sc_leader_ratio=dict.fromkeys(PHASES, 1.0), sc_catch_ratio=dict.fromkeys(PHASES, 1.0),
                 vsc_ratio=dict.fromkeys(PHASES, 1.0), pass_max=1.0, pass_slope=1.0, pass_midpoint=-1e9,
                 pass_train_factor=1.0)


@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    return R.load(*QATAR, root=golden_gold_lake)


@pytest.fixture(scope="module")
def british(golden_gold_lake):
    return R.load(*BRITISH, root=golden_gold_lake)


def real_lap_times(race, planned):
    """Every planned lap's real duration (lap end - the lap before's end); laps a car never drove (a lapped
    finisher run on to the distance) get 1000 s: they come after the flag and are never used."""
    ends = race.laps.set_index(["driver_code", "lap_number"])["lap_end_sec"]
    before = ends.reindex(pd.MultiIndex.from_arrays([planned["driver_code"], planned["lap_number"] - 1])).to_numpy()
    after = ends.reindex(pd.MultiIndex.from_arrays([planned["driver_code"], planned["lap_number"]])).to_numpy()
    return pd.Series(after - before, index=planned.index).fillna(1000.0)


def test_qatar_events_in_the_leaders_laps(qatar):
    assert qatar.periods[["kind", "first_lap", "last_lap"]].values.tolist() == [
        ["SC", 1, 4], ["SC", 35, 39], ["VSC", 40, 40], ["SC", 40, 42]]
    conditions = R.lap_conditions(qatar)
    assert conditions.loc[5, "kind"] == GREEN and conditions.loc[43, "kind"] == GREEN
    assert tuple(conditions.loc[35]) == (SAFETY_CAR, DEPLOY) and tuple(conditions.loc[36]) == (SAFETY_CAR, FULL)
    assert tuple(conditions.loc[39]) == (SAFETY_CAR, END)
    assert tuple(conditions.loc[40]) == (SAFETY_CAR, DEPLOY)        # the VSC became a safety car: the stronger kind


def test_the_status_of_any_set_of_periods():
    # A VSC on laps 10-12 that the safety car of laps 12-14 overlaps on lap 12 (the stronger kind), a red flag on 20.
    status = R.status_of([("VSC", 10, 12), ("SC", 12, 14), ("red flag", 20, 20)], 30)
    assert status == {10: ("VSC", DEPLOY), 11: ("VSC", FULL), 12: ("SC", DEPLOY), 13: ("SC", FULL), 14: ("SC", END),
                      20: ("red flag", "single")}
    assert R.status_of([("SC", 28, 35)], 30) == {28: ("SC", DEPLOY), 29: ("SC", FULL), 30: ("SC", FULL)}   # cut at the flag
    assert R.status_of([], 30) == {}


def test_qatar_starts_when_every_car_has_a_clean_lap(qatar):
    # Safety car on laps 1-4: lap 5 is the restart, so nobody has a clean lap when it ends; lap 6 is the first.
    assert R.first_start_lap(qatar) == 6
    state = R.start_state(qatar, 6)
    assert len(state) == 18 and set(state["laps_done"]) == {6}
    assert list(state["clock_sec"]) == sorted(state["clock_sec"])      # in order on the road
    assert R.first_start_lap(qatar, 30) == 30


def test_qatar_plan_holds_the_real_decisions(qatar):
    plan = R.plan(qatar, 6)
    ver = plan[plan["driver_code"] == "VER"].set_index("lap_number")
    assert ver.index.min() == 7 and ver.index.max() == 57
    assert ver.at[35, "is_in_lap"] and ver.at[35, "pit_kind"] == "tyre_change" and ver.at[35, "compound"] == "MEDIUM"
    assert ver.at[36, "pit_kind"] == "pass_through" and ver.at[37, "pit_kind"] == "pass_through"
    assert ver.at[38, "is_out_lap"] and ver.at[38, "compound"] == "HARD" and not ver.at[38, "is_in_lap"]
    retired = set(plan["driver_code"]) - set(qatar.finishers)
    assert retired == {"PER", "HUL", "STR"}                               # running at lap 6, retired later
    assert plan[plan["driver_code"] == "PER"]["lap_number"].max() == 39


def test_qatar_scoring_ranks(qatar):
    real = R.finish_ranks(qatar)
    assert [d for d, _ in sorted(real.items(), key=lambda x: x[1])][:4] == ["VER", "LEC", "PIA", "RUS"]
    hold = R.hold_order_ranks(qatar, 6, qatar.finishers)
    assert [d for d, _ in sorted(hold.items(), key=lambda x: x[1])][:5] == ["VER", "NOR", "RUS", "PIA", "LEC"]
    assert qatar.grid["RUS"] == 1 and qatar.grid["VER"] == 2
    assert not qatar.mixed


@pytest.mark.parametrize("which, start_lap", [("qatar", 6), ("qatar", 34), ("british", 5)])
def test_real_lap_times_reproduce_the_race_exactly(which, start_lap, request):
    race = request.getfixturevalue(which)
    state, planned = R.start_state(race, start_lap), R.plan(race, start_lap)
    sim = simulate(state, planned, real_lap_times(race, planned), R.lap_conditions(race), NO_RULES, race.total_laps)
    both = sim.laps.merge(race.laps[["driver_code", "lap_number", "lap_end_sec"]], on=["driver_code", "lap_number"])
    both = both.dropna(subset=["lap_end_sec"])
    assert len(both) > 300
    assert np.abs(both["clock_sec"] - both["lap_end_sec"]).max() < 1e-9
    real = R.finish_ranks(race, R.running_at(race, start_lap))
    assert [d for d in sim.finish["driver_code"] if d in real] == sorted(real, key=real.get)


def test_british_is_mixed(british):
    assert british.mixed and british.periods.empty and R.first_start_lap(british) == 5


def test_the_test_season_stays_locked(golden_gold_lake):
    with pytest.raises(SeasonLocked):
        R.development_races(golden_gold_lake, seasons=(2024, 2025))
    assert R.development_races(golden_gold_lake) == (BRITISH, QATAR)


def test_a_car_that_retired_can_be_run_on_to_the_flag(qatar):
    real, run_on = R.plan(qatar, 6), R.plan(qatar, 6, run_on=True)
    last = lambda plan, d: int(plan[plan["driver_code"] == d]["lap_number"].max())
    tail = lambda d: run_on[run_on["driver_code"] == d].sort_values("lap_number").iloc[-1]
    # Perez (lap 39), Hulkenberg (40) and Stroll (8) retired: their last tyres are carried on to lap 57.
    assert [last(real, d) for d in ("PER", "HUL", "STR")] == [39, 40, 8]
    assert [last(run_on, d) for d in ("PER", "HUL", "STR")] == [57, 57, 57]
    assert [(tail(d)["compound"], tail(d)["tyre_age"]) for d in ("PER", "HUL", "STR")] == [("HARD", 22.0), ("MEDIUM", 26.0), ("HARD", 50.0)]
    # Stroll's lap 8 was an in-lap that ended in the pits: he is not in the pit lane any more.
    lap8 = lambda plan: plan[(plan["driver_code"] == "STR") & (plan["lap_number"] == 8)].iloc[0]
    assert lap8(real)["is_in_lap"] and lap8(real)["pit_kind"] == "ended"
    assert not lap8(run_on)["is_in_lap"] and pd.isna(lap8(run_on)["pit_kind"])
    # The cars that took the flag are the same in both.
    keep = lambda plan: plan[plan["driver_code"].isin(qatar.finishers)].sort_values(["driver_code", "lap_number"]).reset_index(drop=True)
    assert keep(run_on).equals(keep(real))


def test_the_real_overtakes_are_counted():
    ends = {"AAA": [100, 200, 300, 400], "BBB": [101, 201.5, 302, 401], "CCC": [102, 200.5, 299, 420]}
    rows = [{"driver_code": d, "lap_number": lap, "lap_end_sec": float(t), "is_pit_lap": d == "CCC" and lap == 4}
            for d, times in ends.items() for lap, t in enumerate(times, start=1)]
    race = R.Race(season=2024, round=99, event_name="Made-up", total_laps=4, laps=pd.DataFrame(rows),
                  finishers=("AAA", "BBB", "CCC"), periods=pd.DataFrame(columns=["kind", "first_lap", "last_lap"]), grid={})
    # Lap 2: CCC gets past BBB; lap 3: CCC past AAA; lap 4: CCC is in the pit lane (not an overtake).
    assert R.overtakes(race, 1) == 2 and R.overtakes(race, 2) == 1
    behind_a_safety_car = replace(race, periods=pd.DataFrame({"kind": ["SC"], "first_lap": [3], "last_lap": [3]}))
    assert R.overtakes(behind_a_safety_car, 1) == 1                               # lap 3 does not count
