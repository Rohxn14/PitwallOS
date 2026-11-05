"""
Phase 4 step 3: the Monte Carlo (src/simulator/montecarlo.py), on hand-built races with known answers.

With nothing drawn, every run is the deterministic race. Two equal cars whose whole-race offsets are drawn
from {-0.5, +0.5} s a lap: only one draw in four (the second car -0.5, the first +0.5) makes the second
car faster, so it wins about a quarter of the runs. The same seed gives the same runs. Stints are counted
from the tyre changes; every planned stop gets one drawn extra; lap noise is drawn lap by lap from its pool.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulator import engine, montecarlo
from src.simulator.montecarlo import Uncertainty
from tests.simulator.hand_race import RULES, race_inputs

NOTHING = np.array([0.0])


def uncertainty(offsets=NOTHING, stint_shifts=NOTHING, lap_noise=NOTHING, stop_extra=NOTHING, draws=montecarlo.ALL_DRAWS):
    return Uncertainty(offsets={p: np.asarray(offsets, float) for p in montecarlo.PARTS}, stint_shifts=np.asarray(stint_shifts, float),
                       lap_noise=np.asarray(lap_noise, float), stop_extra=np.asarray(stop_extra, float), draws=tuple(draws))


def test_the_part_of_the_race():
    assert [montecarlo.part_of_race(lap, 60) for lap in (5, 19, 20, 39, 40, 59)] == \
        ["early", "early", "middle", "middle", "late", "late"]


def test_with_nothing_drawn_every_run_is_the_deterministic_race():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (3.0, 89.8), "CCC": (6.0, 90.3)}, 10)
    outcomes = montecarlo.run(start, planned, pace, conditions, RULES, 10, uncertainty(draws=()), runs=5, seed=1, start_lap=0)
    deterministic = engine.simulate(start, planned, pace, conditions, RULES, 10).finish.set_index("driver_code")["rank"]
    assert (outcomes.ranks.nunique() == 1).all()
    assert outcomes.ranks.iloc[0].to_dict() == deterministic.astype(float).to_dict()


def test_one_draw_in_four_makes_the_second_car_win():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (0.5, 90.0)}, 12)
    outcomes = montecarlo.run(start, planned, pace, conditions, RULES, 12, uncertainty([-0.5, 0.5], draws=("offset",)),
                              runs=600, seed=7, start_lap=0)
    assert outcomes.chance(1)["BBB"] == pytest.approx(0.25, abs=0.06)
    assert outcomes.chance(1)["AAA"] + outcomes.chance(1)["BBB"] == pytest.approx(1.0)
    table = outcomes.table()
    assert list(table.columns) == ["win", "podium", "points", "mean_rank"] and table.index[0] == "AAA"


def test_the_same_seed_gives_the_same_runs():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (0.5, 90.0)}, 12)
    noisy = uncertainty([-0.5, 0.5], lap_noise=[-0.3, 0.0, 0.3], draws=("offset", "lap", "pass"))
    first = montecarlo.run(start, planned, pace, conditions, RULES, 12, noisy, runs=50, seed=3, start_lap=0)
    again = montecarlo.run(start, planned, pace, conditions, RULES, 12, noisy, runs=50, seed=3, start_lap=0)
    assert first.ranks.equals(again.ranks)


def test_stints_stops_and_lap_noise_are_drawn_where_they_belong():
    start, planned, pace, conditions = race_inputs({"AAA": (0.0, 90.0), "BBB": (5.0, 90.0)}, 10,
                                                   pits={"AAA": {4: "tyre_change", 7: "tyre_change"}})
    prepared = montecarlo.prepare(planned)
    aaa = planned["driver_code"].to_numpy() == "AAA"
    # AAA: stint 0 on laps 1-4 (the in-lap included), stint 1 on laps 5-7, stint 2 on laps 8-10.
    stints = prepared.stint_of_row[aaa]
    assert len(set(stints[:4])) == 1 and len(set(stints[4:7])) == 1 and len(set(stints[7:])) == 1
    assert len(set(stints)) == 3 and prepared.stints == 4
    assert prepared.stops == [("AAA", 4), ("AAA", 7)]
    rng = np.random.default_rng(0)
    extra = montecarlo.draw_stops(prepared, uncertainty(stop_extra=[1.5, 6.0], draws=("stop",)), rng)
    assert set(extra) == {("AAA", 4), ("AAA", 7)} and set(extra.values()) <= {1.5, 6.0}
    assert montecarlo.draw_stops(prepared, uncertainty(stop_extra=[1.5], draws=()), rng) == {}
    drawn = montecarlo.draw_pace(pace.to_numpy(float), prepared, uncertainty(lap_noise=[-0.2, 0.4], draws=("lap",)), "early", rng)
    assert set(np.round(drawn - pace.to_numpy(float), 9)) <= {-0.2, 0.4}
    shifted = montecarlo.draw_pace(pace.to_numpy(float), prepared, uncertainty(stint_shifts=[1.0, 2.0], draws=("stint",)), "early", rng)
    per_stint = pd.Series(shifted - pace.to_numpy(float)).groupby(prepared.stint_of_row).nunique()
    assert (per_stint == 1).all()                                      # one shift for every lap of a stint
