"""
Phase 4 step 1: the engine (src/simulator/engine.py), on hand-built races worked out by hand.

Clear-air laps add up; a green stop costs exactly the pit loss, split between in-lap and out-lap; a
drive-through costs the loss minus a stop; a red-flag stop is free. A faster car is held FOLLOW gap
behind when it cannot pass, and passes after exactly the number of laps the chances say (p = 0.2 a lap:
the fourth). A safety car closes the field to the QUEUE gap behind the safety car, which stays where
the leader is when the leader pits; a stop under it costs its loss and drops the car behind the queue.
No passing under a VSC; its gaps are kept. Lapped cars and cars in the pit lane hold nobody, a retiring
car is not ranked, and the flag falls on the first car to complete the distance. The teams are called on the first
lap of each neutralisation, and a stop they add is a planned stop. Behind a car in a DRS train (within 1 s of the car
in front of it) the chance is halved: the pass comes on the 7th lap stuck instead of the 4th.
"""

import numpy as np
import pytest

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator.engine import laps_neutralised, simulate, stop_loss
from src.simulator.race import DEPLOY, END, FULL
from src.simulator.rules import TYPICAL_STATIONARY_SEC
from tests.simulator.hand_race import RULES, clocks, order_after, race_inputs


def run(cars, total_laps, rules=RULES, **kwargs):
    start, planned, pace, conditions = race_inputs(cars, total_laps, **kwargs)
    return simulate(start, planned, pace, conditions, rules, total_laps)


def test_clear_air_laps_add_up():
    sim = run({"AAA": (0.0, 90.0), "BBB": (10.0, 90.5)}, total_laps=5)
    assert clocks(sim, "AAA") == pytest.approx([90, 180, 270, 360, 450])
    assert clocks(sim, "BBB") == pytest.approx([10 + 90.5 * k for k in range(1, 6)])
    assert sim.finish["driver_code"].tolist() == ["AAA", "BBB"]
    assert not sim.laps["held"].any() and sim.passes == 0


def test_a_green_stop_costs_the_pit_loss_split_over_in_and_out_lap():
    sim = run({"AAA": (0.0, 90.0)}, total_laps=5, pits={"AAA": {3: "tyre_change"}})
    laps = np.diff([0.0] + clocks(sim, "AAA"))
    assert laps == pytest.approx([90, 90, 90 + 0.2 * 20, 90 + 0.8 * 20, 90])       # 4 s in, 16 s out
    assert sum(laps) == pytest.approx(5 * 90 + RULES.pit_loss_sec)


def test_what_a_passage_costs():
    assert stop_loss(RULES, "green", "pass_through") == pytest.approx(20.0 - TYPICAL_STATIONARY_SEC)
    assert stop_loss(RULES, SAFETY_CAR, "tyre_change") == 10.0
    assert stop_loss(RULES, VIRTUAL_SAFETY_CAR, "tyre_change") == 12.0
    assert stop_loss(RULES, RED_FLAG, "tyre_change") == 0.0
    assert stop_loss(RULES, SAFETY_CAR, "pass_through") == 0.0          # the field led through the pit lane


def test_a_faster_car_that_cannot_pass_is_held_the_follow_gap_behind():
    rules = RULES.with_changes(pass_midpoint=50.0)                    # no chance of passing
    sim = run({"AAA": (0.0, 90.0), "BBB": (2.0, 89.8)}, total_laps=20, rules=rules)
    gap = np.array(clocks(sim, "BBB")) - np.array(clocks(sim, "AAA"))
    # 2 s behind, 0.2 s a lap faster: 0.6 s behind after lap 7 on its own, so held from lap 7 on.
    assert gap[:6] == pytest.approx([1.8, 1.6, 1.4, 1.2, 1.0, 0.8])
    assert gap[6:] == pytest.approx([0.7] * 14)
    held = sim.laps[sim.laps["driver_code"] == "BBB"].sort_values("lap_number")["held"].tolist()
    assert held == [False] * 6 + [True] * 14
    assert sim.finish["driver_code"].tolist() == ["AAA", "BBB"] and sim.passes == 0


def test_passes_once_the_chances_add_up_to_one_half():
    # 0.5 s a lap faster; midpoint set so that the chance is exactly 0.2 a lap: -ln(0.8) = 0.223 a lap,
    # ln 2 = 0.693 is reached on the 4th lap stuck. Then it is by: 0.2 s behind at the line (not held),
    # ahead at the end of lap 5.
    rules = RULES.with_changes(pass_midpoint=0.5 + np.log(3.0) / 4.0)
    assert rules.pass_chance(0.5) == pytest.approx(0.2)
    sim = run({"AAA": (0.0, 90.0), "BBB": (0.3, 89.5)}, total_laps=8, rules=rules)
    gap = np.array(clocks(sim, "BBB")) - np.array(clocks(sim, "AAA"))
    assert gap[:3] == pytest.approx([0.7, 0.7, 0.7])
    assert gap[3] == pytest.approx(0.2)
    assert gap[4] == pytest.approx(-0.3)
    assert order_after(sim, 5) == ["BBB", "AAA"] and sim.passes == 1
    assert sim.finish["driver_code"].tolist() == ["BBB", "AAA"]


def test_a_safety_car_closes_the_field_to_the_queue_gap():
    sc = {1: (SAFETY_CAR, DEPLOY), 2: (SAFETY_CAR, FULL), 3: (SAFETY_CAR, FULL), 4: (SAFETY_CAR, END)}
    sim = run({"AAA": (0.0, 90.0), "BBB": (10.0, 90.0), "CCC": (20.0, 90.0)}, total_laps=6, conditions=sc)
    a, b, c = (np.array(clocks(sim, d)) for d in ("AAA", "BBB", "CCC"))
    assert a[0] == pytest.approx(117.0) and b[0] == pytest.approx(127.0)    # the lap it comes out: x 1.3 for all
    assert a[1] == pytest.approx(117 + 1.6 * 90)                            # the safety car's pace
    assert (b - a)[1:4] == pytest.approx([0.5] * 3)                          # caught up in one lap (x 1.2)
    assert (c - b)[1:4] == pytest.approx([0.5] * 3)
    assert sim.finish["driver_code"].tolist() == ["AAA", "BBB", "CCC"]


def test_a_stop_under_the_safety_car_costs_its_loss_and_the_queue_closes_up():
    sc = {1: (SAFETY_CAR, DEPLOY), 2: (SAFETY_CAR, FULL), 3: (SAFETY_CAR, FULL), 4: (SAFETY_CAR, END)}
    sim = run({"AAA": (0.0, 90.0), "BBB": (10.0, 90.0), "CCC": (20.0, 90.0)}, total_laps=6, conditions=sc,
              pits={"AAA": {2: "tyre_change"}})
    a, b, c = (np.array(clocks(sim, d)) for d in ("AAA", "BBB", "CCC"))
    # Lap 2: the leader pits. The safety car stays where it was (117 + 144 = 261); BBB closes up to it
    # (127 + 1.2 x 90 = 235 on its own), CCC behind BBB. AAA: the safety car's pace + 2 s (its in-lap share).
    assert b[1] == pytest.approx(261.0) and c[1] == pytest.approx(261.5)
    assert a[1] == pytest.approx(261.0 + 0.2 * 10)
    # Lap 3: AAA's out-lap (+ 8 s) puts it behind the queue: the stop cost it two places.
    assert b[2] == pytest.approx(261 + 144) and c[2] == pytest.approx(261 + 144 + 0.5)
    assert a[2] == pytest.approx(263.0 + 144 + 8)
    assert order_after(sim, 3) == ["BBB", "CCC", "AAA"]


def test_no_passing_under_a_vsc_and_its_gaps_are_kept():
    vsc = {3: (VIRTUAL_SAFETY_CAR, DEPLOY), 4: (VIRTUAL_SAFETY_CAR, FULL), 5: (VIRTUAL_SAFETY_CAR, END)}
    rules = RULES.with_changes(pass_midpoint=-5.0)                    # would pass at once under green
    sim = run({"AAA": (0.0, 90.0), "BBB": (0.75, 85.0)}, total_laps=6, rules=rules, conditions=vsc,
              laps_done=3)
    # Lap 4 (a VSC's full lap): BBB is 5 s a lap faster, but no passing: held 0.7 s behind.
    lap4 = sim.laps[sim.laps["lap_number"] == 4].set_index("driver_code")["clock_sec"]
    assert lap4["BBB"] - lap4["AAA"] == pytest.approx(0.7)
    assert lap4["AAA"] == pytest.approx(90 * 1.4)


def test_cars_in_the_pit_lane_and_lapped_cars_hold_nobody():
    rules = RULES.with_changes(pass_midpoint=50.0)
    # AAA on its in-lap is 4 s slower: BBB, 1 s behind on the same pace, gets by without a fight.
    sim = run({"AAA": (0.0, 90.0), "BBB": (1.0, 90.0)}, total_laps=4, rules=rules, pits={"AAA": {2: "tyre_change"}})
    lap2 = sim.laps[sim.laps["lap_number"] == 2].set_index("driver_code")
    assert not lap2.at["BBB", "held"] and lap2.at["BBB", "clock_sec"] == pytest.approx(181.0)
    assert lap2.at["AAA", "clock_sec"] == pytest.approx(184.0)
    # A lapped car (one lap fewer done) just ahead on the road does not hold the leader (blue flags).
    lapped = run({"LED": (0.0, 85.0), "LAP": (-0.3, 90.0)}, total_laps=4, rules=rules, laps_done={"LED": 1, "LAP": 0})
    assert not lapped.laps[lapped.laps["driver_code"] == "LED"]["held"].any()
    assert clocks(lapped, "LED") == pytest.approx([85.0, 170.0, 255.0])


def test_a_retiring_car_is_not_ranked_and_the_flag_falls_on_the_first_car_to_finish():
    sim = run({"AAA": (0.0, 90.0), "BBB": (5.0, 90.0), "CCC": (10.0, 95.0)}, total_laps=5, last_lap={"BBB": 3})
    assert sim.finish["driver_code"].tolist() == ["AAA", "CCC"]
    assert sim.finish["laps"].tolist() == [5, 5]
    assert clocks(sim, "BBB") == pytest.approx([95.0, 185.0, 275.0])          # stops after its last lap


def test_a_car_lapped_during_the_race_finishes_a_lap_down():
    # 10 s a lap slower over 10 laps: lapped. The flag falls at 500 s (AAA, lap 10); BBB crosses at 540 s
    # with 9 laps and finishes there (not after another lap).
    sim = run({"AAA": (0.0, 50.0), "BBB": (0.5, 60.0)}, total_laps=10)
    row = sim.finish.set_index("driver_code")
    assert row.at["AAA", "laps"] == 10 and row.at["AAA", "clock_sec"] == pytest.approx(500.0)
    assert row.at["BBB", "laps"] == 9 and row.at["BBB", "clock_sec"] == pytest.approx(540.5)


def test_monte_carlo_draws_passes():
    rules = RULES.with_changes(pass_midpoint=0.5 + np.log(3.0) / 4.0)    # 0.2 a lap
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (0.3, 89.5)}, 8)
    laps_to_pass = []
    for seed in range(200):
        sim = simulate(start, planned, pace, conditions, rules, 8, rng=np.random.default_rng(seed))
        gaps = np.array(clocks(sim, "BBB")) - np.array(clocks(sim, "AAA"))
        laps_to_pass.append(int(np.argmax(gaps < 0.69)) + 1 if (gaps < 0.69).any() else 99)
    # The lap it gets by follows the geometric distribution of p = 0.2: P(first lap) = 0.2.
    assert np.mean(np.array(laps_to_pass) == 1) == pytest.approx(0.2, abs=0.07)


def test_a_slow_stop_costs_its_extra_seconds_split_like_the_loss():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0)}, 5, pits={"AAA": {3: "tyre_change"}})
    sim = simulate(start, planned, pace, conditions, RULES, 5, stop_extra={("AAA", 3): 3.0})
    laps = np.diff([0.0] + clocks(sim, "AAA"))
    assert laps == pytest.approx([90, 90, 90 + 0.2 * 23, 90 + 0.8 * 23, 90])          # a 23 s stop, not 20


def test_the_teams_are_called_when_a_neutralisation_comes_out():
    seen = []

    def hook(event):          # the engine's dictionaries go on changing: keep copies
        seen.append((event.lap, event.kind, event.length, list(event.order), event.leader, dict(event.done)))

    conditions = {3: (SAFETY_CAR, DEPLOY), 4: (SAFETY_CAR, FULL), 5: (SAFETY_CAR, END),
                  8: (VIRTUAL_SAFETY_CAR, DEPLOY), 9: (VIRTUAL_SAFETY_CAR, END)}
    start, planned, pace, table = race_inputs({"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}, 12, conditions=conditions)
    simulate(start, planned, pace, table, RULES, 12, on_neutralisation=hook)
    # Called once per neutralisation, on the lap it comes out, before the leader starts it; green laps and a
    # neutralisation's later laps do not call it.
    assert seen == [(3, SAFETY_CAR, 3, ["AAA", "BBB"], "AAA", {"AAA": 2, "BBB": 2}),
                    (8, VIRTUAL_SAFETY_CAR, 2, ["AAA", "BBB"], "AAA", {"AAA": 7, "BBB": 7})]


def test_two_safety_cars_in_a_row_are_two_neutralisations():
    status = {35: (SAFETY_CAR, DEPLOY), 36: (SAFETY_CAR, FULL), 37: (SAFETY_CAR, END), 38: (SAFETY_CAR, DEPLOY),
              39: (SAFETY_CAR, END)}
    assert laps_neutralised(status, 35) == 3 and laps_neutralised(status, 38) == 2 and laps_neutralised({}, 10) == 1


def test_a_stop_added_by_the_hook_is_a_planned_stop():
    cars = {"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}
    conditions = {3: (SAFETY_CAR, DEPLOY), 4: (SAFETY_CAR, END)}
    planned_stop = run(cars, 8, pits={"AAA": {3: "tyre_change"}}, conditions=conditions)
    start, planned, pace, table = race_inputs(cars, 8, conditions=conditions)

    def hook(event):          # the same stop, made by the teams when the safety car comes out
        event.plans["AAA"][3] = (90.0, True, False, "tyre_change")
        event.plans["AAA"][4] = (90.0, False, True, None)
        event.out_lap_passage[("AAA", 4)] = "tyre_change"

    sim = simulate(start, planned, pace, table, RULES, 8, on_neutralisation=hook)
    for driver in cars:
        assert clocks(sim, driver) == pytest.approx(clocks(planned_stop, driver))
    assert clocks(sim, "AAA") != pytest.approx(clocks(run(cars, 8, conditions=conditions), "AAA"))      # and it cost something


def test_a_car_in_a_drs_train_is_harder_to_pass():
    # As above (0.5 s a lap faster, 0.2 a lap), but the car ahead (BBB) runs 0.9 s behind AAA all race: inside 1 s, a
    # DRS train (and outside the 0.7 s at which it would be held). The chance is 0.2 x 0.5 = 0.1 a lap: -ln(0.9) = 0.105
    # a lap reaches ln 2 on the 7th lap stuck, ahead at the line one lap later. No train effect (factor 1): the 4th
    # lap, ahead at the end of the 5th.
    rules = RULES.with_changes(pass_midpoint=0.5 + np.log(3.0) / 4.0, pass_train_factor=0.5)
    cars = {"AAA": (0.0, 90.0), "BBB": (0.9, 90.0), "CCC": (1.2, 89.5)}

    def first_lap_ahead(sim):
        return next(lap for lap in range(1, 11) if order_after(sim, lap).index("CCC") < order_after(sim, lap).index("BBB"))

    assert first_lap_ahead(run(cars, total_laps=10, rules=rules.with_changes(pass_train_factor=1.0))) == 5
    assert first_lap_ahead(run(cars, total_laps=10, rules=rules)) == 8
