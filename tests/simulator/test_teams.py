"""
Phase 4 step 3b: the teams in a drawn race (src/simulator/teams.py), on Qatar 2024 and on hand-built plans.

Qatar's safety cars are Race Control's (laps 1-4, 35-39, 40-42). Of its 26 tyre changes, 15 after the first start
lap (6) were made under a safety car: 12 on lap 35, one on lap 36, two on lap 40, and the field was led through the
pit lane on laps 36-37. With the usual stopping ages fixed at 18 laps (medium) and 23 (hard), the base plan moves
the stops of cars that were not due and drops the rest: Bottas, stopped on lap 24 and again on lap 35 on 11-lap-old
hards, would have stopped again on lap 35 + (23 - 11) = 47; Russell (12-lap-old hards) on 46; Alonso's second stop
(one lap after the first, on new tyres) and the lap-40 stops of Lawson and Tsunoda (5-lap-old hards: 40 + 18 = 58,
past the last 4 laps) disappear; Verstappen's, on 35-lap-old mediums, was due anyway and stays. The reaction rule
is fitted on made-up events with a known truth; the tyre terms equal rung 2's own; a moved stop rebuilds the tyres
exactly as the plan editor (plans.py) does.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.models.linear import LinearDecomposition
from src.simulator import engine, pace, plans, race as R, teams as T
from src.simulator.teams import (BoxModel, GreenAges, Stop, TyreTerms, build_cars, decontaminate, tyres, box_rows)
from tests.simulator.hand_race import race_inputs

QATAR = (2024, 23)
USUAL = GreenAges({"SOFT": np.full(9, 15.0), "MEDIUM": np.full(9, 18.0), "HARD": np.full(9, 23.0)})
NO_TERMS = TyreTerms(offset=np.zeros(4), wear=np.zeros(4))


@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    return R.load(*QATAR, root=golden_gold_lake)


# --- how old a tyre usually is when it is changed ---------------------------------------------------------------

def test_the_green_stops_of_qatar(qatar):
    ages = GreenAges.fit([qatar])
    # The nine tyre changes under green after lap 1 (lap-1 stops and the safety car's are not green stops).
    assert ages.ages["MEDIUM"].tolist() == [24, 24, 27, 34, 34, 35] and ages.ages["HARD"].tolist() == [6, 30]
    assert ages.ages["SOFT"].tolist() == [20]
    assert ages.median("MEDIUM") == pytest.approx(30.5)                  # six stops: its own median
    assert ages.median("HARD") == 23.0 and ages.median("SOFT") == 15.0   # fewer than five: the 2023 + 2024 medians
    assert ages.median("INTERMEDIATE") == 20.0 and ages.median(None) == 20.0


# --- the base plan ----------------------------------------------------------------------------------------------

def test_qatars_base_plan_takes_the_safety_cars_out(qatar):
    conditions = R.lap_conditions(qatar)
    planned = R.plan(qatar, 6)
    base = decontaminate(qatar, planned, conditions, USUAL)
    stops = lambda plan, d: plans.stops_of(plan, d)
    assert stops(planned, "BOT") == [24, 35] and stops(base, "BOT") == [24, 47]       # 35 + (23 - 11)
    assert stops(planned, "RUS") == [23, 35] and stops(base, "RUS") == [23, 46]       # 35 + (23 - 12)
    assert stops(planned, "ALO") == [35, 36] and stops(base, "ALO") == [35]
    assert stops(planned, "LAW") == [35, 40] and stops(base, "LAW") == [35]
    assert stops(planned, "TSU") == [35, 40] and stops(base, "TSU") == [35]
    assert stops(base, "VER") == stops(planned, "VER") == [35]                         # due anyway: 35-lap-old mediums
    assert stops(base, "HAM") == stops(planned, "HAM") == [34]                         # a green stop stays
    # The 32 passages through the pit lane under the safety car (laps 35-37) are gone; the two drive-throughs made
    # under green (Norris on lap 45, Hamilton on 47: penalties) stay. Nothing else about the rows changes.
    assert (planned["pit_kind"] == "pass_through").sum() == 34 and (base["pit_kind"] == "pass_through").sum() == 2
    assert set(base.loc[base["pit_kind"] == "pass_through", "driver_code"]) == {"NOR", "HAM"}
    rows = lambda plan: plan[["driver_code", "lap_number"]].sort_values(["driver_code", "lap_number"]).reset_index(drop=True)
    assert rows(base).equals(rows(planned))                                      # the same laps of the same cars
    assert {d: plans.new_problems(qatar, planned, base, d) for d in set(planned["driver_code"])} == \
        {d: [] for d in set(planned["driver_code"])}
    # BOT's moved stop fits the set he really fitted (the used mediums of lap 36): on the out-lap of lap 48 they
    # have the age and the compound they had on lap 36, and lap 47 is the in-lap.
    real_bot = planned[planned["driver_code"] == "BOT"].set_index("lap_number")
    bot = base[base["driver_code"] == "BOT"].set_index("lap_number")
    assert bot.loc[47, "is_in_lap"] and bot.loc[48, "is_out_lap"] and not bot.loc[35, "is_in_lap"]
    assert (bot.loc[48, "compound"], bot.loc[48, "tyre_age"]) == (real_bot.loc[36, "compound"], real_bot.loc[36, "tyre_age"])


def test_a_race_without_events_has_nothing_to_take_out(qatar):
    planned = R.plan(qatar, 44)                    # after the last safety car (lap 42)
    assert decontaminate(qatar, planned, R.lap_conditions(qatar), USUAL).equals(planned.reset_index(drop=True))


# --- who boxes when a neutralisation comes out ------------------------------------------------------------------

def test_the_cars_that_boxed_under_qatars_safety_cars(qatar):
    rows = box_rows(qatar)
    first, second = rows[rows["lap"] == 35], rows[rows["lap"] == 40]
    assert set(rows["lap"]) == {35, 40}                                   # lap 1-4 is before the start of any replay
    assert len(first) == 15 and first["boxed"].sum() == 11 and (first.loc[first["boxed"], "offset"] == 0).all()
    assert (first["length"] == 5).all() and (second["length"] == 3).all()   # two safety cars, not one of eight laps
    assert len(second) == 16 and second["boxed"].sum() == 2 and set(second.loc[second["boxed"], "driver_code"]) == {"LAW", "TSU"}
    ver = first[first["driver_code"] == "VER"].iloc[0]
    assert (ver["position"], ver["compound"], ver["age"], ver["boxed"]) == (1, "MEDIUM", 35.0, True)
    pia = first[first["driver_code"] == "PIA"].iloc[0]                    # fourth on the road, on tyres one lap old
    assert (pia["position"], pia["age"], pia["boxed"]) == (4, 1.0, False)


def made_up_events(n_events=60, seed=3, sigma=1.3):
    """Events with a known truth: boxing is likelier on old tyres and on a safety car, much less on young tyres,
    and every event has its own mood."""
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(n_events):
        kind = SAFETY_CAR if rng.random() < 0.7 else VIRTUAL_SAFETY_CAR
        mood = sigma * rng.standard_normal()
        for position in range(1, 21):
            age = float(rng.integers(1, 36))
            eta = -1.0 + (0.8 if kind == SAFETY_CAR else 0.0) - (2.0 if age <= 5 else 0.0) + 1.6 * min(age / 18.0, 1.5) + mood
            rows.append({"season": 2024, "round": 1 + event // 10, "lap": 10 + event, "kind": kind, "length": 3,
                         "driver_code": f"D{position}", "position": position, "compound": "MEDIUM", "age": age,
                         "boxed": bool(rng.random() < 1 / (1 + np.exp(-eta))), "offset": 0})
    return pd.DataFrame(rows)


def test_the_box_rule_finds_the_age_effect_and_the_mood_of_made_up_events():
    model = BoxModel.fit(made_up_events(), USUAL)
    assert model.periods == 60 and model.cars == 1200
    assert 0.8 < model.sigma < 2.0                                          # truth 1.3
    sc = lambda age: model.chance(SAFETY_CAR, age, "MEDIUM", 10, 0.0, USUAL)
    assert sc(3) < 0.25 < 0.5 < sc(25)                                      # young tyres rarely box, old ones mostly
    assert sc(12) < sc(18) < sc(30)
    assert model.chance(SAFETY_CAR, 20, "MEDIUM", 10, 0.0, USUAL) > model.chance(VIRTUAL_SAFETY_CAR, 20, "MEDIUM", 10, 0.0, USUAL)
    assert model.chance(SAFETY_CAR, 20, "MEDIUM", 10, -2.0, USUAL) < sc(20) < model.chance(SAFETY_CAR, 20, "MEDIUM", 10, 2.0, USUAL)


def test_few_events_fall_back_to_the_measured_rates():
    model = BoxModel.fit(pd.DataFrame(columns=T.BOX_COLUMNS), USUAL)
    assert model.periods == 0
    for kind, rate in {SAFETY_CAR: 0.44, VIRTUAL_SAFETY_CAR: 0.16, RED_FLAG: 0.64}.items():
        assert model.chance(kind, 20.0, "MEDIUM", 5, 0.0, USUAL) == pytest.approx(rate)


def test_the_share_that_box_on_the_first_lap_is_measured():
    # Two safety cars, four cars each, all boxed, three of the four on the first lap: 6 of 8. With the measured 0.94
    # counted as two more boxers: (6 + 2 x 0.94) / (8 + 2) = 0.788. No VSC seen: the measured 0.81 stays. Red flag: 1.
    rows = pd.DataFrame([{"season": 2024, "round": 1, "lap": 10 + 20 * e, "kind": SAFETY_CAR, "length": 3,
                          "driver_code": f"D{i}", "position": i, "compound": "MEDIUM", "age": 20.0, "boxed": True,
                          "offset": 1 if i == 4 else 0} for e in range(2) for i in range(1, 5)])
    model = BoxModel.fit(rows, USUAL)                                       # two events: too few to fit, the shares still count
    assert model.periods == 0
    assert model.first_lap_share[SAFETY_CAR] == pytest.approx((6 + 2 * 0.94) / 10)
    assert model.first_lap_share[VIRTUAL_SAFETY_CAR] == pytest.approx(0.81) and model.first_lap_share[RED_FLAG] == 1.0


# --- the tyres' share of a lap time ----------------------------------------------------------------------------

def test_the_tyre_terms_are_rung_2s_own():
    rung = LinearDecomposition()
    rung.terms = pd.DataFrame({"coef": {"offset SOFT": -0.4, "offset HARD": 0.15, "wear SOFT": 0.07, "wear MEDIUM": 0.04,
                                        "wear HARD": 0.03, "fuel": 0.03}, "se": 0.01})
    terms = TyreTerms.from_rung2(rung)
    laps = pd.DataFrame({"lap_number": [10, 11, 12, 13], "compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"],
                         "tyre_age": [5.0, 8.0, 3.0, 4.0]})
    other = pd.DataFrame({"lap_number": [10, 11, 12, 13], "compound": ["HARD", "HARD", "SOFT", "MEDIUM"],
                          "tyre_age": [1.0, 2.0, 20.0, 9.0]})
    code = lambda frame: np.array([T.CODE.get(c, T.OTHER) for c in frame["compound"]])
    ours = terms.value(code(laps), laps["tyre_age"].to_numpy()) - terms.value(code(other), other["tyre_age"].to_numpy())
    theirs = pace.planned_terms(rung, laps, 50) - pace.planned_terms(rung, other, 50)       # the fuel cancels (same laps)
    assert ours == pytest.approx(theirs.to_numpy(), abs=1e-12)
    assert terms.value(np.array([T.SOFT]), np.array([5.0]))[0] == pytest.approx(-0.4 + 0.07 * 5)
    assert terms.value(np.array([T.OTHER]), np.array([5.0]))[0] == 0.0                       # intermediates: nothing


# --- a car's plan as arrays -------------------------------------------------------------------------------------

def hand_plan():
    start, planned, pace_, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}, 24)
    return plans.box(planned, "AAA", 10, "HARD")                 # AAA: mediums to lap 10, new hards from lap 11


def test_the_cars_of_a_plan_and_their_stops():
    planned = hand_plan().reset_index(drop=True)
    cars = build_cars(planned, NO_TERMS)
    aaa = cars["AAA"]
    assert aaa.stops == (Stop(10, "tyre_change", T.HARD, "HARD", 1.0),) and cars["BBB"].stops == ()
    assert aaa.laps.tolist() == list(range(1, 25)) and aaa.code[:10].tolist() == [T.MEDIUM] * 10 and aaa.code[10:].tolist() == [T.HARD] * 14
    assert aaa.age[10] == 1.0 and aaa.age[23] == 14.0 and not aaa.first_is_out
    assert (planned.iloc[aaa.rows]["driver_code"] == "AAA").all()


def test_a_moved_stop_rebuilds_the_tyres_like_the_plan_editor():
    planned = hand_plan().reset_index(drop=True)
    aaa = build_cars(planned, NO_TERMS)["AAA"]
    for edited, stops in [(plans.move_stop(planned, "AAA", 10, 6), (Stop(6, "tyre_change", T.HARD, "HARD", 1.0),)),
                          (plans.move_stop(planned, "AAA", 10, 15), (Stop(15, "tyre_change", T.HARD, "HARD", 1.0),)),
                          (plans.stay_out(planned, "AAA", 10), ()),
                          (plans.box(planned, "AAA", 18, "SOFT"), aaa.stops + (Stop(18, "tyre_change", T.SOFT, "SOFT", 1.0),))]:
        code, age = tyres(aaa, stops)
        mine = edited[edited["driver_code"] == "AAA"].sort_values("lap_number")
        assert age.tolist() == mine["tyre_age"].tolist()
        assert code.tolist() == [T.CODE[c] for c in mine["compound"]]
    code, age = tyres(aaa, aaa.stops)
    assert code.tolist() == aaa.code.tolist() and age.tolist() == aaa.age.tolist()          # no change, no change


def test_the_cars_that_retired_get_the_stops_a_team_would_make(qatar):
    run_on = T.decontaminate(qatar, R.plan(qatar, 6, run_on=True), R.lap_conditions(qatar), USUAL)
    plan = T.stops_for_run_on(qatar, run_on, USUAL)
    stops = lambda d: plans.stops_of(plan, d)
    # Perez' carried-on hards are 22 laps old at the flag (usual 23): no more stop. Hulkenberg's mediums, 9 laps old on
    # lap 40, are 18 on lap 49 (usual 18): a stop, with 8 laps to go on softs. Stroll's hards, new on lap 8, reach 23 on
    # lap 30: a stop, onto hards (27 laps to go); the next set is not old before the last four laps.
    assert stops("PER") == [35] and stops("HUL") == [31, 49] and stops("STR") == [7, 30]
    hul = plan[plan["driver_code"] == "HUL"].set_index("lap_number")
    assert (hul.loc[50, "compound"], hul.loc[50, "tyre_age"]) == ("SOFT", 1.0) and hul.loc[49, "is_in_lap"]
    stro = plan[plan["driver_code"] == "STR"].set_index("lap_number")
    assert (stro.loc[31, "compound"], stro.loc[31, "tyre_age"]) == ("HARD", 1.0)
    # Every car that took the flag has the stops of its (decontaminated) real plan.
    base = T.decontaminate(qatar, R.plan(qatar, 6), R.lap_conditions(qatar), USUAL)
    finishers = [d for d in qatar.finishers if d in set(plan["driver_code"])]
    assert len(finishers) == 15 and all(plans.stops_of(plan, d) == plans.stops_of(base, d) for d in finishers)


# --- step 4: the tyre life ------------------------------------------------------------------------------------

def test_a_compounds_life_is_the_age_sets_almost_never_ran_past():
    # 100 soft sets came off at ages 1 .. 100: 99 % of them by 99.01 laps, so the life is 99 (rounded down).
    life = T.TyreLife({"SOFT": np.arange(1.0, 101.0)})
    assert life.limit("SOFT") == 99.0
    # Too few sets (fewer than MIN_SETS) or none: the life measured on 2023 + 2024.
    few = T.TyreLife({"MEDIUM": np.array([10.0, 50.0])})
    assert few.limit("MEDIUM") == T.DEFAULT_LIFE["MEDIUM"] and few.limit("HARD") == T.DEFAULT_LIFE["HARD"]
    assert few.limits() == T.DEFAULT_LIFE


def test_the_life_of_qatars_sets(qatar):
    life = T.TyreLife.fit([qatar])
    # Every set's age when it came off (Gold's stints): Verstappen's mediums came off at the end of lap 35, his hards
    # took the flag.
    ver = qatar.laps[qatar.laps["driver_code"] == "VER"].set_index("lap_number")
    assert ver.at[35, "tyre_age"] in life.ages["MEDIUM"] and ver.at[57, "tyre_age"] in life.ages["HARD"]
    assert all(np.all(np.diff(a) >= 0) for a in life.ages.values())          # kept sorted
    assert sum(len(a) for a in life.ages.values()) == qatar.laps.groupby(["driver_code", "stint_number"]).ngroups
