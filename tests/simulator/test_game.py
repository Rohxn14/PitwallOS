"""
Phase 4 step 4: the strategist game (src/simulator/game.py).

The fastest finish by hand: with rung 2's wear and offsets a 57-lap race from new mediums is one stop on lap 27 for
hards (the cost 0.041 s^2 - 2.236 s + c is lowest at s = 27.3); wear on the current tyres only makes the earliest stop
the best; the two-compounds rule forces a stop that costs time, and with four laps left only "box now" keeps it; no
set left, no stop on that compound.

The same futures for every option: a pass is drawn from a number fixed in advance for (car, car ahead, lap), so the
pass comes on the lap whose number is below the chance; two plans of one car that differ in a stop under a drawn
safety car keep every run's safety cars with world.run(common=True), not without it (the answers drew different
numbers and every later run moved).

Your race at a moment: the exact replay of Qatar 2024 (real lap times, rules off) seen through snapshot_of is the real
race's snapshot at every lap. A game on Qatar in a three-race lake (the fold that never saw it): Verstappen starts on
his real mediums with none of the real team's stops; staying out at every call brings the game to the end of lap 34
for the safety car of lap 35 and to lap 39 for the one of lap 40; a call after a moment never moves the race before it;
two identical options get identical chances; the bot plays a whole game and the result names the real team's place in
the same race and the real one (Verstappen won). The command line plays a game on saved models.
"""

import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analytics.neutralisations import SAFETY_CAR
from src.pipeline import lake
from src.simulator import engine, game as G, montecarlo, plans, race as R, world as W
from src.simulator.events import EventModel
from src.simulator.teams import BoxModel, GreenAges, TyreTerms
from tests.simulator.hand_race import RULES, order_after, race_inputs
from tests.simulator.test_race import NO_RULES, real_lap_times

QATAR, HUNGARY, BRITISH = (2024, 23), (2024, 13), (2024, 12)
# offsets and wear by compound code (soft, medium, hard, other): rung 2's on 2023 + 2024 (CLAUDE.md, Phase 3 step 2)
RUNG2 = TyreTerms(offset=np.array([-0.394, 0.0, 0.131, 0.0]), wear=np.array([0.0749, 0.0451, 0.0371, 0.0]))
PLENTY = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}


# --- the fastest finish -------------------------------------------------------------------------------------

def test_one_stop_on_lap_27_for_hards_from_new_mediums():
    terms = TyreTerms(offset=np.array([5.0, 0.0, 0.131, 0.0]), wear=np.array([0.0749, 0.045, 0.037, 0.0]))
    stops = G.fastest_finish("MEDIUM", 0.0, 0, 57, ["MEDIUM"], PLENTY, terms, 22.0)
    # mediums for s laps, then hards: 0.045 s(s+1)/2 + 0.131 (57-s) + 0.037 (57-s)(58-s)/2 = 0.041 s^2 - 2.236 s + c
    assert stops == [(27, "HARD")]
    by_hand = {s: 0.045 * s * (s + 1) / 2 + 0.131 * (57 - s) + 0.037 * (57 - s) * (58 - s) / 2 for s in range(2, 55)}
    assert min(by_hand, key=by_hand.get) == 27


def test_wear_on_the_current_tyres_only_makes_the_earliest_stop_the_best():
    terms = TyreTerms(offset=np.array([1.0, 0.0, 0.0, 0.0]), wear=np.array([0.0, 0.05, 0.0, 0.0]))
    # Both compounds used already: staying out is allowed, and costs 0.05 x (11 + ... + 40) = 38.25 s; a stop at the
    # end of lap 12 costs 20 + 0.05 x (11 + 12) = 21.15 s (hards: no wear, no offset; softs 1 s a lap slower).
    assert G.fastest_finish("MEDIUM", 10.0, 10, 40, ["SOFT", "MEDIUM"], PLENTY, terms, 20.0) == [(12, "HARD")]
    # Staying out for at least 3 laps (the game's STAY_OUT_LAPS): the earliest stop is then at the end of lap 14.
    assert G.fastest_finish("MEDIUM", 10.0, 10, 40, ["SOFT", "MEDIUM"], PLENTY, terms, 20.0, stay_laps=3) == [(14, "HARD")]
    # Boxing now for softs: 29 laps on them cost 29 s; the shortest stint allowed (8 laps, 8 s) and a second stop for
    # hards (20 s) cost 28 s.
    assert G.fastest_finish("MEDIUM", 10.0, 10, 40, ["SOFT", "MEDIUM"], PLENTY, terms, 20.0,
                            box_now="SOFT") == [(11, "SOFT"), (19, "HARD")]


def test_the_two_compounds_rule_and_the_sets():
    flat = TyreTerms(offset=np.array([1.0, 0.0, 0.0, 0.0]), wear=np.zeros(4))
    # Nothing wears: staying out would cost nothing, but one medium stint breaks the rule: one stop, for hards.
    assert G.fastest_finish("MEDIUM", 5.0, 10, 40, ["MEDIUM"], PLENTY, flat, 20.0) == [(12, "HARD")]
    assert G.fastest_finish("MEDIUM", 5.0, 10, 40, ["MEDIUM", "HARD"], PLENTY, flat, 20.0) == []
    # Four laps left: a stop must be at the end of the next lap at the latest, so only "box now" keeps the rule.
    assert G.fastest_finish("MEDIUM", 5.0, 36, 40, ["MEDIUM"], PLENTY, flat, 20.0) is None
    assert G.fastest_finish("MEDIUM", 5.0, 36, 40, ["MEDIUM"], PLENTY, flat, 20.0, box_now="HARD") == [(37, "HARD")]
    # No hard set left: softs (1 s a lap slower) are the only way to keep the rule.
    assert G.fastest_finish("MEDIUM", 5.0, 10, 40, ["MEDIUM"], {**PLENTY, "HARD": 0}, flat, 20.0)[0][1] == "SOFT"
    assert G.fastest_finish("MEDIUM", 5.0, 10, 40, ["MEDIUM"], {**PLENTY, "HARD": 0}, flat, 20.0, box_now="HARD") is None


LIFE = {"SOFT": 29.0, "MEDIUM": 41.0, "HARD": 57.0}          # teams.DEFAULT_LIFE: 2023 + 2024


def test_no_set_runs_past_its_life():
    # Rohan's case (Bahrain 2024, a 24 s stop): boxing for softs at the end of lap 22 of 57. By the straight line, 34
    # laps on them cost 31.2 s, less than a second stop (33.5 s at best: two soft stints of 17 laps and the 24 s), so
    # "softs to the flag" won; but no soft set ran more than 31 laps in 2023-2024. With the life, a second stop.
    to_the_flag = G.fastest_finish("MEDIUM", 22.0, 22, 57, ["MEDIUM"], PLENTY, RUNG2, 24.0, box_now="SOFT")
    assert to_the_flag == [(23, "SOFT")]
    stops = G.fastest_finish("MEDIUM", 22.0, 22, 57, ["MEDIUM"], PLENTY, RUNG2, 24.0, box_now="SOFT", life=LIFE)
    assert stops[0] == (23, "SOFT") and len(stops) == 2 and stops[1][0] - 23 <= LIFE["SOFT"]
    # The one stop of 27 laps on mediums and 30 on hards is within both lives: unchanged.
    terms = TyreTerms(offset=np.array([5.0, 0.0, 0.131, 0.0]), wear=np.array([0.0749, 0.045, 0.037, 0.0]))
    assert G.fastest_finish("MEDIUM", 0.0, 0, 57, ["MEDIUM"], PLENTY, terms, 22.0, life=LIFE) == [(27, "HARD")]
    # A set at its life: staying out is not on offer, boxing is.
    assert G.fastest_finish("SOFT", 28.0, 30, 57, ["SOFT", "HARD"], PLENTY, RUNG2, 22.0, stay_laps=3, life=LIFE) is None
    assert G.fastest_finish("SOFT", 28.0, 30, 57, ["SOFT", "HARD"], PLENTY, RUNG2, 22.0, box_now="HARD", life=LIFE)[0] == (31, "HARD")
    # Staying out is still fine while the set has the laps: softs 20 laps old, 9 laps of life left, 12 laps to go.
    assert G.fastest_finish("SOFT", 20.0, 45, 57, ["SOFT", "HARD"], PLENTY, RUNG2, 22.0, stay_laps=3, life=LIFE) is not None


# --- the same futures for every option -----------------------------------------------------------------------

def test_a_pass_comes_on_the_lap_whose_fixed_number_is_below_the_chance():
    # BBB starts 0.6 s behind AAA and is 1 s a lap faster: a pass chance of 0.8 / (1 + e^0) = 0.4 every lap stuck.
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 91.0), "BBB": (0.6, 90.0), "CCC": (30.0, 92.0)}, 8)
    draws = engine.PassDraws(start.index, 8, np.random.default_rng(0))
    a, b = draws.index["AAA"], draws.index["BBB"]
    draws.table[b, a, :] = 0.9                      # above 0.4: no pass ...
    draws.table[b, a, 4] = 0.1                      # ... until lap 4
    sim = engine.simulate(start, planned, pace, conditions, RULES, 8, draws=draws)
    assert order_after(sim, 3)[:2] == ["AAA", "BBB"] and order_after(sim, 4)[:2] == ["BBB", "AAA"]
    # Another car's plan does not move that luck: CCC, far behind, now stops on lap 5.
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 91.0), "BBB": (0.6, 90.0), "CCC": (30.0, 92.0)}, 8,
                                                   pits={"CCC": {5: "tyre_change"}})
    again = engine.simulate(start, planned, pace, conditions, RULES, 8, draws=draws)
    assert order_after(again, 4)[:2] == ["BBB", "AAA"] and order_after(again, 3)[:2] == ["AAA", "BBB"]


class Recording(EventModel):
    """Draws like the model and remembers every run's neutralisations."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def draw(self, rng, after_lap, total_laps):
        out = super().draw(rng, after_lap, total_laps)
        self.seen.append(tuple(out))
        return out


def test_two_plans_of_one_car_keep_every_runs_safety_cars_only_with_common_futures():
    start, planned, pace, _ = race_inputs({"AAA": (0.0, 90.0), "BBB": (3.0, 90.0), "CCC": (6.0, 90.0)}, 20)
    stopping = plans.box(planned, "CCC", 6, "HARD")                       # CCC stops on lap 6 in one plan
    coin = BoxModel(mean=np.zeros(6), scale=np.ones(6), coef=np.zeros(7), sigma=0.0,
                    first_lap_share={k: 1.0 for k in (SAFETY_CAR, "VSC", "red flag")})   # every car: one chance in two
    zero = np.array([0.0])
    drawn = montecarlo.Uncertainty(offsets={p: np.array([-0.3, 0.3]) for p in montecarlo.PARTS},
                                   stint_shifts=zero, lap_noise=zero, stop_extra=zero, draws=("offset", "pass"))
    seen = {}
    for common in (True, False):
        for name, plan in (("stays out", planned), ("stops", stopping)):
            events = Recording(hazard={SAFETY_CAR: 0.15, "VSC": 0.0, "red flag": 0.0},
                               lengths={SAFETY_CAR: np.array([2]), "VSC": np.zeros(0, int), "red flag": np.zeros(0, int)})
            world = W.World(events=events, box=coin, ages=GreenAges({}), terms=TyreTerms(np.zeros(4), np.zeros(4)))
            W.run(start, plan, pd.Series(90.0, index=plan.index), RULES, 20, drawn, world, runs=40, seed=3,
                  start_lap=0, common=common)
            seen[(common, name)] = events.seen
    assert seen[(True, "stays out")] == seen[(True, "stops")]              # the same safety cars in every run
    assert seen[(False, "stays out")] != seen[(False, "stops")]            # the answers moved every later run


def test_your_neighbours_are_the_cars_just_ahead_and_just_behind_on_your_lap():
    state = pd.DataFrame({"laps_done": [10, 10, 10, 10, 9, 10], "clock_sec": [95.0, 100.0, 101.5, 102.0, 102.5, 103.0]},
                         index=pd.Index(["EEE", "AAA", "BBB", "YOU", "DDD", "CCC"], name="driver_code"))
    # DDD is between YOU and CCC on the road but a lap down: not a neighbour; AAA and EEE are further ahead.
    assert G.neighbours(state, "YOU") == [("BBB", -0.5), ("CCC", 1.0)]
    assert G.neighbours(state, "EEE") == [("AAA", 5.0)]                  # the leader on the road: only a car behind


# --- your race at a moment ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    return R.load(*QATAR, root=golden_gold_lake)


def test_the_race_at_a_moment_is_the_real_ones_when_the_simulation_is(qatar):
    state, planned = R.start_state(qatar, 6), R.plan(qatar, 6)
    sim = engine.simulate(state, planned, real_lap_times(qatar, planned), R.lap_conditions(qatar), NO_RULES,
                          qatar.total_laps)
    crossings = G.crossings_of(state, sim)
    for lap in (7, 20, 34, 39, 45, 56):
        # the cars running at the start (Colapinto and Ocon went out on lap 1; Stroll, Perez, Hulkenberg later)
        ours, real = R.snapshot_of(crossings, lap), R.snapshot(qatar, lap).loc[lambda s: s.index.isin(state.index)]
        assert list(ours.index) == list(real.index)
        assert (ours["laps_done"] == real["laps_done"]).all()
        assert np.abs(ours["clock_sec"] - real["clock_sec"]).max() < 1e-9


# --- a game on Qatar ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models(game_lake):
    """The game's models on a lake of the three reference races, saved in it (conftest: game_lake)."""
    return game_lake


def qatar_game(models, futures=30):
    fitted, root = models
    return G.new_game(R.load(*QATAR, root=root), "VER", fitted, seed=5, futures=futures)


def test_a_game_starts_on_the_real_tyres_with_none_of_the_real_stops(models):
    game = qatar_game(models)
    real = game.real_plan[game.real_plan["driver_code"] == "VER"].sort_values("lap_number")
    mine = game.plan.sort_values("lap_number")
    assert mine["lap_number"].tolist() == real["lap_number"].tolist()
    assert mine["compound"].iloc[0] == real["compound"].iloc[0] == "MEDIUM"
    # The plan is the fastest finish from the start: replace it by staying out on everything, ask again, same stops.
    bare = game.plan
    for in_lap in plans.stops_of(bare, "VER"):
        bare = plans.stay_out(bare, "VER", in_lap)
    again = G.fastest_after(replace(game, plan=bare), game.start_lap)
    assert G.stops_in(game.plan, "VER") == again and len(again) >= 1      # mediums only so far: a stop is needed
    assert plans.problems(game.race, game.plan, "VER") == []              # the plan keeps the rules
    assert len(game.labels) == 20 and game.labels["VER"] == "YOU" and game.labels["RUS"] == "car A"    # RUS on pole


def stay_out(game, moment, options, table):
    return next(i for i, o in enumerate(options) if o.box is None) if any(o.box is None for o in options) else 0


def test_staying_out_brings_the_game_to_both_safety_cars(models):
    game = qatar_game(models, futures=10)
    G.play(game, stay_out, show=lambda text: None)
    reasons = {d.lap: d.reasons for d in game.decisions}
    assert any("safety car is coming out" in r for r in reasons.get(34, []))       # lap 35's safety car
    assert any("safety car is coming out" in r for r in reasons.get(39, []))       # lap 40's (a VSC, then a safety car)
    assert all(d.lap < game.total_laps - 3 for d in game.decisions)
    # Staying out means at least 3 more laps: a planned stop is never asked about again on the next lap.
    planned = [d.lap for d in game.decisions if any("planned stop" in r for r in d.reasons)]
    assert planned and all(b - a >= G.STAY_OUT_LAPS for a, b in zip(planned, planned[1:]))
    # "Has just pitted" is only ever the car just ahead or just behind on the road (the race up to a call never changes,
    # so the final race shows what the game saw).
    crossings = G.crossings_of(game.start, G.played(game))
    named = {label: driver for driver, label in game.labels.items()}
    pitted = [(d.lap, r) for d in game.decisions for r in d.reasons if "has just pitted" in r]
    for lap, reason in pitted:
        rival = named[" ".join(reason.split()[:2])]
        assert rival in [car for car, _ in G.neighbours(G.state_at(game, crossings, lap - 1), "VER")]


def test_a_call_never_moves_the_race_before_it(models):
    game = qatar_game(models, futures=10)
    before = G.played(game)
    moment = G.next_moment(game, before, game.start_lap)
    options = G.options_at(game, moment)
    boxing = next(o for o in options if o.box is not None)
    after = G.played(replace(game, plan=boxing.plan))
    seen = R.snapshot_of(G.crossings_of(game.start, before), moment.lap)["clock_sec"].max()
    a = G.crossings_of(game.start, before)
    b = G.crossings_of(game.start, after)
    a, b = (t[t["lap_end_sec"] <= seen].sort_values(["driver_code", "lap_number"]).reset_index(drop=True) for t in (a, b))
    assert len(a) > 100 and a.equals(b)


def test_identical_options_get_identical_chances(models):
    game = qatar_game(models, futures=40)
    moment = G.next_moment(game, G.played(game), game.start_lap)
    options = G.options_at(game, moment)
    table = G.odds(game, moment, [options[0], options[0], options[-1]])
    assert table.iloc[0].equals(table.iloc[1])
    assert ((table["win"] <= table["podium"]) & (table["podium"] <= table["points"])).all()
    assert table[["win", "podium", "points", "dnf"]].stack().between(0, 1).all()


def test_the_bot_plays_a_whole_game(models):
    game = qatar_game(models, futures=10)
    shown = []
    result = G.play(game, G.most_points, show=shown.append)
    assert game.decisions and result.place is not None and result.real_team_place is not None
    assert result.real_place == 1                                     # Verstappen won Qatar 2024
    assert result.real_stops == [(35, "HARD")]
    text = "\n".join(shown)
    assert "Chequered flag" in text and "2024 Qatar Grand Prix" in text and "VER" not in shown[0]   # blind until the end


def test_a_typed_answer_names_a_car_by_place_or_by_name():
    order, labels = ["RUS", "VER", "NOR"], {"RUS": "car A", "VER": "YOU", "NOR": "car C"}
    for answer in ("3", "P3", "p3", " p 3", "C", "c", "car C", "CAR c", "NOR", "nor"):
        assert G.car_from_answer(answer, order, labels) == "NOR", answer
    assert G.car_from_answer("1", order, labels) == "RUS"
    for nothing in ("4", "0", "Z", "P", ""):
        assert G.car_from_answer(nothing, order, labels) is None, nothing


def test_one_turn_and_the_running_order(models):
    game = qatar_game(models, futures=5)
    turn = G.next_turn(game, game.start_lap)
    order = G.running_order(game, turn.moment.state)
    assert order["place"].tolist() == list(range(1, len(order) + 1))
    assert order["yours"].sum() == 1 and order.loc[order["yours"], "car"].item() == "YOU"
    assert order["to the leader"].iloc[0] == "leader" and order["gap ahead"].iloc[0] == ""
    assert len(turn.options) == len(turn.table) >= 2
    after = G.take(game, turn, len(turn.options) - 1)
    assert after == turn.moment.lap and game.decisions[-1].chosen == turn.options[-1].label
    assert game.plan is turn.options[-1].plan


def test_the_bot_on_every_race_writes_its_report(models, capsys):
    fitted, root = models
    assert G.main(["--lake", root, "--evaluate", "--futures", "3"]) == 0
    games = lake.read_report(G.BOT_REPORT, root)
    assert set(zip(games["season"], games["round"])) == {QATAR, HUNGARY}          # the dry races, three cars each
    assert len(games) == 6 and (games["decisions"] >= 1).all() and games["real_place"].notna().all()
    out = capsys.readouterr().out
    assert "Big moments per game" in out and "The bot against the real team's own plan" in out


def test_the_command_line_plays_on_saved_models(models, capsys):
    fitted, root = models
    assert G.main(["--lake", root, "--season", "2024", "--round", "23", "--driver", "VER", "--bot", "--futures", "5",
                   "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Fitting" not in out and "Chequered flag" in out and "--seed 1 --season 2024 --round 23 --driver VER" in out
