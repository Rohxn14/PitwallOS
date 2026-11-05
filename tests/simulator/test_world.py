"""
Phase 4 step 3b: the Monte Carlo with the events drawn (src/simulator/world.py), on hand-built races.

With nothing drawn and nobody answering, a run is the plain Monte Carlo of step 3a (same seed, same ranks). A
safety car that comes out on lap 4 of a race where the leader is 20 s clear: the field closes up behind the
safety car; if the teams keep their plans the leader stays ahead, but if the leader boxes (a stop under it costs
the 10 s of the made-up rules) he comes out behind the car that stayed out. A car boxing moves its next planned
stop to the neutralisation, the same set of tyres, or takes one more when it has none left (softs with 15 laps or
fewer to go, else hards); a car that stops under it anyway, a lap too late to gain anything, or on wet tyres is
left alone. A changed plan is re-priced with the tyre terms: a set fitted 8 laps earlier is 8 laps older, 0.1 s a
lap slower, once the old stop's lap is passed. The pace misses come from their own generator: what the events do
never changes them. Step 3c: the real race's own neutralisations run through the world give step 3a's ranks run for
run; a car's speed miss is drawn from its place group's pool (a fourth car drawn 2 s a lap faster than the first three
wins every run); every run counts its overtakes.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator import engine, montecarlo, plans, race as R, teams as T, world as W
from src.simulator.events import KINDS, NO_EVENTS, EventModel
from src.simulator.teams import BoxModel, GreenAges, Stop, Teams, TyreTerms
from tests.simulator.hand_race import RULES, race_inputs

USUAL = GreenAges({})
NO_TERMS = TyreTerms(offset=np.zeros(4), wear=np.zeros(4))


def box_rule(intercept, position=0.0, mood_sd=0.0):
    """A box rule that does not look at the tyres: logistic(intercept + position x position)."""
    return BoxModel(mean=np.zeros(6), scale=np.ones(6), coef=np.array([intercept, 0, 0, 0, 0, 0, position], float),
                    sigma=mood_sd, first_lap_share={k: 1.0 for k in KINDS})


class Fixed(EventModel):
    """Neutralisations that are not drawn: these ones, every run."""
    def __init__(self, periods):
        super().__init__()
        self.periods = periods

    def draw(self, rng, after_lap, total_laps):
        return list(self.periods)


def world(events=NO_EVENTS, box=None):
    return W.World(events=events, box=box or box_rule(-30.0), ages=USUAL, terms=NO_TERMS)


def uncertainty(offsets=(0.0,), draws=("offset",)):
    zero = np.array([0.0])
    return montecarlo.Uncertainty(offsets={p: np.asarray(offsets, float) for p in montecarlo.PARTS}, stint_shifts=zero,
                                  lap_noise=zero, stop_extra=zero, draws=tuple(draws))


def run_one(cars, total_laps, periods, box, pits=None, planned_changes=None):
    """One run by hand: the Teams, the Draw, the engine. Returns (draw, simulation, teams, planned, pace)."""
    start, planned, pace, _ = race_inputs(cars, total_laps, pits=pits)
    if planned_changes:
        planned = planned_changes(planned)
    teams = Teams(planned, box, USUAL, NO_TERMS, total_laps)
    pace = pace.to_numpy(float)
    draw = teams.begin(pace, np.random.default_rng(0))
    sim = engine.simulate_plans(start, draw.plans, draw.passage, R.status_of(periods, total_laps), RULES, total_laps,
                                record_laps=False, on_neutralisation=draw.react)
    return draw, sim, teams, planned, pace


def test_without_events_or_answers_it_is_the_plain_monte_carlo():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (0.5, 90.0), "CCC": (4.0, 90.2)}, 12,
                                                   pits={"AAA": {6: "tyre_change"}})
    drawn = uncertainty([-0.5, 0.5], draws=("offset", "pass"))
    plain = montecarlo.run(start, planned, pace, conditions, RULES, 12, drawn, runs=60, seed=4, start_lap=0)
    ours = W.run(start, planned, pace, RULES, 12, drawn, world(), runs=60, seed=4, start_lap=0, events=False, react=False)
    assert ours.ranks.equals(plain.ranks)
    # Events that never come (a hazard of zero) leave it the same: the events have their own random stream.
    quiet = W.run(start, planned, pace, RULES, 12, drawn, world(Fixed([])), runs=60, seed=4, start_lap=0)
    assert quiet.ranks.equals(plain.ranks)


def test_a_boxing_leader_hands_the_lead_to_the_car_behind():
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (20.0, 90.0)}, 12)
    safety_car = Fixed([(SAFETY_CAR, 4, 6)])
    only_the_leader = box_rule(50.0, -40.0)             # first: 50 - 40 = +10 (boxes); second: 50 - 80 = -30 (stays out)

    def ranks(box, react):
        return W.run(start, planned, pace, RULES, 12, uncertainty(), world(safety_car, box), runs=5, seed=1,
                     start_lap=0, react=react).ranks

    kept, answered, nobody = ranks(only_the_leader, False), ranks(only_the_leader, True), ranks(box_rule(-30.0), True)
    assert (kept["AAA"] == 1).all() and (kept["BBB"] == 2).all()                  # plans kept: the leader stays ahead
    assert (answered["BBB"] == 1).all() and (answered["AAA"] == 2).all()          # his stop under it cost 10 s
    assert (nobody["AAA"] == 1).all()                                             # no one boxes: nothing changes


def test_a_car_with_no_stop_left_takes_one_more_and_a_planned_stop_moves():
    cars = {"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}
    draw, sim, teams, _, _ = run_one(cars, 24, [(SAFETY_CAR, 4, 6)], box_rule(30.0), pits={"AAA": {12: "tyre_change"}})
    planned_set = teams.cars["AAA"].stops[0]
    assert draw.boxed == [("AAA", 4), ("BBB", 4)]
    # AAA's stop moved from lap 12 to lap 4, fitting the set it had planned; BBB, with none planned, takes a new
    # set of hards (20 laps left: more than 15).
    assert draw.stops["AAA"] == (Stop(4, "tyre_change", planned_set.code, planned_set.compound, planned_set.age),)
    assert draw.stops["BBB"] == (Stop(4, "tyre_change", T.HARD, "HARD", 1.0),)
    plan = draw.plans["AAA"]
    assert plan[4][1] and plan[5][2] and not plan[12][1] and not plan[13][2]      # in-lap 4, out-lap 5; lap 12 is ordinary now
    assert draw.passage[("AAA", 5)] == "tyre_change" and ("AAA", 13) not in draw.passage
    assert plan[5][0] == 90.0                                                      # no tyre terms: the pace is unchanged
    assert sim.finish["laps"].tolist() == [24, 24]
    late, *_ = run_one(cars, 24, [(SAFETY_CAR, 14, 16)], box_rule(30.0))           # 10 laps left: softs
    assert late.stops["BBB"] == (Stop(14, "tyre_change", T.SOFT, "SOFT", 1.0),)


def test_the_late_the_wet_and_those_already_stopping_are_left_alone():
    cars = {"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}
    # Already stopping under it (lap 5 is inside the safety car's laps 4-6): AAA's plan is not touched.
    draw, *_ = run_one(cars, 24, [(SAFETY_CAR, 4, 6)], box_rule(30.0), pits={"AAA": {5: "tyre_change"}})
    assert draw.boxed == [("BBB", 4)] and [s.in_lap for s in draw.stops["AAA"]] == [5]
    # Two laps from the flag: nothing to gain, nobody is asked.
    draw, *_ = run_one(cars, 24, [(SAFETY_CAR, 23, 24)], box_rule(30.0))
    assert draw.boxed == []
    # The rule says no: the plans are the base plans, to the last tuple.
    draw, _, _, planned, pace = run_one(cars, 24, [(SAFETY_CAR, 4, 6)], box_rule(-30.0), pits={"AAA": {12: "tyre_change"}})
    assert draw.boxed == [] and draw.plans == engine._plans(planned.reset_index(drop=True), pace)

    def on_intermediates(planned):                       # BBB is on intermediates: no tyre terms, no answer
        planned = planned.copy()
        mine = planned["driver_code"] == "BBB"
        planned.loc[mine, "compound"] = "INTERMEDIATE"
        planned.loc[mine, "is_wet_tyre"] = True
        return planned

    draw, *_ = run_one(cars, 24, [(SAFETY_CAR, 4, 6)], box_rule(30.0), planned_changes=on_intermediates)
    assert draw.boxed == [("AAA", 4)]


def test_a_changed_plan_is_re_priced_with_the_tyre_terms():
    cars = {"AAA": (0.0, 90.0), "BBB": (30.0, 90.0)}
    start, planned, pace, _ = race_inputs(cars, 24)
    planned = plans.box(planned, "AAA", 12, "MEDIUM")            # a new set of mediums at lap 13 (age 1 ... 12)
    terms = TyreTerms(offset=np.zeros(4), wear=np.array([0.0, 0.1, 0.0, 0.0]))                 # 0.1 s a lap of medium age
    teams = Teams(planned, box_rule(-30.0), USUAL, terms, 24)
    draw = teams.begin(np.full(len(planned), 90.0), np.random.default_rng(0))
    event = engine.Neutralisation(lap=4, kind=SAFETY_CAR, length=3, order=["AAA", "BBB"], leader="AAA",
                                  done={"AAA": 3, "BBB": 3}, plans=draw.plans, out_lap_passage=draw.passage)
    before = {lap: entry[0] for lap, entry in draw.plans["AAA"].items()}
    assert teams.cars["AAA"].stops == (Stop(12, "tyre_change", T.MEDIUM, "MEDIUM", 1.0),)
    draw.stops["AAA"] = (replace(teams.cars["AAA"].stops[0], in_lap=4),)               # the stop moved from lap 12 to lap 4
    draw._apply("AAA", event)
    after = {lap: entry[0] for lap, entry in draw.plans["AAA"].items()}
    # Laps 5-12: the new set is 4 laps younger than the old one (age lap - 4 against lap): 0.4 s faster. From lap 13
    # on the planned set would have been new (age lap - 12), the moved one has run 8 laps more (lap - 4): 0.8 s slower.
    for lap in range(1, 25):
        expected = 0.0 if lap <= 4 else -0.4 if lap <= 12 else 0.8
        assert after[lap] - before[lap] == pytest.approx(expected, abs=1e-9), lap
    assert draw.plans["AAA"][4][1] and draw.plans["AAA"][5][2] and not draw.plans["AAA"][12][1]


def test_the_same_seed_gives_the_same_events_and_answers():
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (3.0, 90.0), "CCC": (6.0, 90.0)}, 30)
    often = EventModel(hazard={SAFETY_CAR: 0.05, VIRTUAL_SAFETY_CAR: 0.0, RED_FLAG: 0.0},
                       lengths={SAFETY_CAR: np.array([2, 3]), VIRTUAL_SAFETY_CAR: np.zeros(0, int), RED_FLAG: np.zeros(0, int)})
    noisy = uncertainty([-0.4, 0.4], draws=("offset", "pass"))

    def ranks(seed):
        return W.run(start, planned, pace, RULES, 30, noisy, world(often, box_rule(0.0, mood_sd=1.0)), runs=40,
                     seed=seed, start_lap=0).ranks

    assert ranks(8).equals(ranks(8)) and not ranks(8).equals(ranks(9))


# --- retirements -------------------------------------------------------------------------------------------------

from src.simulator.events import RetirementModel


def test_a_retiring_car_ends_where_its_plan_is_cut():
    cars = {"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}
    start, planned, pace, _ = race_inputs(cars, 20)
    draw = Teams(planned, box_rule(-30.0), USUAL, NO_TERMS, 20).begin(pace.to_numpy(float), np.random.default_rng(0), retire={"AAA": 9})
    assert max(draw.plans["AAA"]) == 9 and max(draw.plans["BBB"]) == 20
    sim = engine.simulate_plans(start, draw.plans, draw.passage, {}, RULES, 20, record_laps=False)
    assert sim.finish["driver_code"].tolist() == ["BBB"]                           # he is not ranked


def test_a_retiring_car_is_not_asked_beyond_its_last_lap():
    cars = {"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}
    start, planned, pace, _ = race_inputs(cars, 24)
    teams = Teams(planned, box_rule(30.0), USUAL, NO_TERMS, 24)
    # BBB drives to lap 6 and stops there: the safety car of laps 4-6 asks him on lap 4 (he can stop on lap 4), and the
    # one of laps 6-7 on lap 6, his last lap: no stop on a last lap, and no laps added after it.
    draw = teams.begin(pace.to_numpy(float), np.random.default_rng(0), retire={"BBB": 6})
    engine.simulate_plans(start, draw.plans, draw.passage, R.status_of([(SAFETY_CAR, 4, 5), (SAFETY_CAR, 6, 7)], 24), RULES, 24,
                          record_laps=False, on_neutralisation=draw.react)
    assert draw.boxed == [("AAA", 4), ("BBB", 4), ("AAA", 6)] and max(draw.plans["BBB"]) == 6


def test_the_drawn_retirements_follow_the_hazard_and_are_missing_from_the_ranks():
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (3.0, 90.0), "CCC": (6.0, 90.0)}, 40)
    w = W.World(events=NO_EVENTS, box=box_rule(-30.0), ages=USUAL, terms=NO_TERMS, retirements=RetirementModel(hazard=0.02))
    out = W.run(start, planned, pace, RULES, 40, uncertainty(), w, runs=400, seed=2, start_lap=0, events=False, react=False, retire=True)
    assert out.ranks["AAA"].isna().mean() == pytest.approx(1 - 0.98 ** 39, abs=0.07)       # 55 %: retires on a lap before the last
    kept = W.run(start, planned, pace, RULES, 40, uncertainty(), w, runs=20, seed=2, start_lap=0, events=False, react=False)
    assert not kept.ranks.isna().any().any()                                                 # not drawn: all three finish
    certain = W.World(events=NO_EVENTS, box=box_rule(-30.0), ages=USUAL, terms=NO_TERMS, retirements=RetirementModel(hazard=1.0))
    nobody = W.run(start, planned, pace, RULES, 40, uncertainty(), certain, runs=3, seed=2, start_lap=0, events=False, react=False, retire=True)
    assert nobody.ranks.isna().all().all() and (nobody.chance(3) == 0.0).all()


# --- step 3c ---------------------------------------------------------------------------------------------------

from src.simulator.events import FixedEvents
from src.simulator.race import DEPLOY, END, FULL


def test_the_real_events_run_through_the_world_are_step_3a():
    conditions = {4: (SAFETY_CAR, DEPLOY), 5: (SAFETY_CAR, FULL), 6: (SAFETY_CAR, END)}
    start, planned, pace, table = race_inputs({"AAA": (0.0, 90.0), "BBB": (1.0, 89.8), "CCC": (5.0, 90.1)}, 14,
                                              pits={"BBB": {5: "tyre_change"}}, conditions=conditions)
    drawn = uncertainty([-0.4, 0.0, 0.4], draws=("offset", "pass"))
    plain = montecarlo.run(start, planned, pace, table, RULES, 14, drawn, runs=50, seed=6, start_lap=0)
    real = W.World(events=FixedEvents(periods=((SAFETY_CAR, 4, 6),)), box=box_rule(30.0), ages=USUAL, terms=NO_TERMS)
    ours = W.run(start, planned, pace, RULES, 14, drawn, real, runs=50, seed=6, start_lap=0, react=False)
    assert ours.ranks.equals(plain.ranks) and len(ours.passes) == 50


def test_speed_misses_drawn_by_place_at_the_start():
    assert [W.place_group(p) for p in (1, 3, 4, 6, 7, 10, 11, 15, 16, 20)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (1.0, 90.0), "CCC": (2.0, 90.0), "DDD": (3.0, 90.0)}, 20)
    assert W.start_places(start) == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
    # The first three are drawn 1 s a lap slower than forecast, the fourth 1 s faster: it wins every run. With the one
    # pool for everybody (a miss of 0) nothing moves and AAA wins.
    by_place = W.PlaceOffsets({(part, g): np.array([1.0 if g == 0 else -1.0]) for part in montecarlo.PARTS for g in range(5)})
    zero = uncertainty([0.0], draws=("offset",))
    ours = W.run(start, planned, pace, RULES, 20, zero, world(), runs=5, seed=1, start_lap=0, events=False, react=False,
                 place_offsets=by_place)
    assert (ours.ranks["DDD"] == 1).all()
    one_pool = W.run(start, planned, pace, RULES, 20, zero, world(), runs=5, seed=1, start_lap=0, events=False, react=False)
    assert (one_pool.ranks["AAA"] == 1).all()


def test_every_run_counts_its_overtakes():
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (0.5, 88.0)}, 10)       # 2 s a lap faster: past at once
    ours = W.run(start, planned, pace, RULES, 10, uncertainty(), world(), runs=4, seed=1, start_lap=0, events=False, react=False)
    assert ours.passes.tolist() == [1, 1, 1, 1]
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 88.0), "BBB": (0.5, 90.0)}, 10)       # slower behind: never
    assert W.run(start, planned, pace, RULES, 10, uncertainty(), world(), runs=4, seed=1, start_lap=0, events=False,
                 react=False).passes.tolist() == [0, 0, 0, 0]
