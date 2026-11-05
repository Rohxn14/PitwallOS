"""
Phase 4 step 3b: the events drawn (src/simulator/events.py), worked out by hand.

The hazard of a kind is its events over the laps at risk: the laps after the start lap, minus the later laps of
each event (the lap an event starts on was at risk). Qatar 2024 from its first start lap (6): a safety car on lap 35
(laps 35-39) and one on lap 40 (laps 40-42; a VSC on lap 40 starts at the same lap, the stronger kind counts), so
2 events over 57 - 6 - (4 + 2) = 45 laps. A hazard of 1 starts an event on every lap at risk; nothing is drawn
without one; the draws follow the hazard.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator import race as R
from src.simulator.events import KINDS, NO_EVENTS, EventModel, Observed, events_after

QATAR, BRITISH = (2024, 23), (2024, 12)


def periods(*rows):
    return pd.DataFrame(rows, columns=["kind", "first_lap", "last_lap"])


def model(hazard=None, lengths=None):
    return EventModel(hazard={**{k: 0.0 for k in KINDS}, **(hazard or {})},
                      lengths={**{k: np.zeros(0, int) for k in KINDS}, **{k: np.array(v) for k, v in (lengths or {}).items()}})


def test_one_event_per_starting_lap_and_none_before_the_start():
    table = periods((SAFETY_CAR, 1, 4), (SAFETY_CAR, 35, 39), (VIRTUAL_SAFETY_CAR, 40, 40), (SAFETY_CAR, 40, 42))
    assert events_after(table, 6) == ((SAFETY_CAR, 35, 5), (SAFETY_CAR, 40, 3))         # the VSC on lap 40 is the SC's
    assert events_after(table, 35) == ((SAFETY_CAR, 40, 3),)                            # lap 35 is the start: not after it
    assert events_after(periods(), 5) == ()


def test_the_hazard_is_events_over_laps_at_risk():
    fitted = EventModel.from_observed([Observed(60, 5, ((SAFETY_CAR, 20, 4), (VIRTUAL_SAFETY_CAR, 40, 1))),
                                       Observed(50, 5, ())])
    # race 1: 55 laps after the start, 3 of them later laps of the safety car -> 52; race 2: 45 -> 97 laps at risk
    assert fitted.laps_at_risk == 97 and fitted.races == 2
    assert fitted.hazard[SAFETY_CAR] == pytest.approx(1 / 97) and fitted.hazard[VIRTUAL_SAFETY_CAR] == pytest.approx(1 / 97)
    assert fitted.hazard[RED_FLAG] == 0.0
    assert fitted.lengths[SAFETY_CAR].tolist() == [4] and fitted.lengths[VIRTUAL_SAFETY_CAR].tolist() == [1]
    assert fitted.total_hazard == pytest.approx(2 / 97)


def test_qatar_gives_two_safety_cars_in_forty_five_laps(golden_gold_lake):
    qatar = R.load(*QATAR, root=golden_gold_lake)
    british = R.load(*BRITISH, root=golden_gold_lake)               # rain: not a dry race, left out
    fitted = EventModel.fit([qatar, british])
    assert fitted.races == 1 and fitted.laps_at_risk == 45
    assert fitted.hazard[SAFETY_CAR] == pytest.approx(2 / 45) and fitted.hazard[VIRTUAL_SAFETY_CAR] == 0.0
    assert sorted(fitted.lengths[SAFETY_CAR].tolist()) == [3, 5]
    assert "safety car 0.0444 a lap" in fitted.describe()


def test_a_hazard_of_one_starts_an_event_on_every_lap_at_risk():
    certain = model({SAFETY_CAR: 1.0}, {SAFETY_CAR: [2]})
    # laps 6-7, 8-9, 10-11, and the last one cut at the flag
    assert certain.draw(np.random.default_rng(0), 5, 12) == [
        (SAFETY_CAR, 6, 7), (SAFETY_CAR, 8, 9), (SAFETY_CAR, 10, 11), (SAFETY_CAR, 12, 12)]
    assert certain.draw(np.random.default_rng(0), 12, 12) == []                  # no lap left after the start


def test_nothing_is_drawn_without_a_hazard():
    assert NO_EVENTS.draw(np.random.default_rng(0), 5, 60) == []
    assert model({SAFETY_CAR: 0.0}).draw(np.random.default_rng(0), 5, 60) == []


def test_the_draws_follow_the_hazard():
    # SC 0.01 and VSC 0.02 a lap at risk, lengths 3 and 1: one third of the events are safety cars, and the
    # chance of a 100-lap race with none is (1 - 0.03)^100 = 4.8 %
    drawn = model({SAFETY_CAR: 0.01, VIRTUAL_SAFETY_CAR: 0.02}, {SAFETY_CAR: [3], VIRTUAL_SAFETY_CAR: [1]})
    rng = np.random.default_rng(11)
    races = [drawn.draw(rng, 0, 100) for _ in range(4000)]
    events = [p for race in races for p in race]
    assert np.mean([p[0] == SAFETY_CAR for p in events]) == pytest.approx(1 / 3, abs=0.02)
    assert np.mean([len(race) == 0 for race in races]) == pytest.approx(0.97 ** 100, abs=0.012)
    assert {p[2] - p[1] + 1 for p in events if p[0] == SAFETY_CAR} <= {1, 2, 3}      # 3 laps, less when the flag cuts it
    assert all(a[2] < b[1] for race in races for a, b in zip(race, race[1:]))        # never inside an earlier event


def test_the_same_seed_gives_the_same_events():
    drawn = model({SAFETY_CAR: 0.02, RED_FLAG: 0.005}, {SAFETY_CAR: [2, 3, 5], RED_FLAG: [1]})
    assert drawn.draw(np.random.default_rng(5), 5, 60) == drawn.draw(np.random.default_rng(5), 5, 60)


# --- retirements -------------------------------------------------------------------------------------------------

from src.simulator.events import RetirementModel


def test_qatar_has_three_retirements_in_834_car_laps(golden_gold_lake):
    # From lap 6 eighteen cars are running; Perez (lap 39), Hulkenberg (40) and Stroll (8) do not take the flag:
    # 33 + 34 + 2 laps at risk, and 51 laps for each of the fifteen that finish.
    qatar = R.load(*QATAR, root=golden_gold_lake)
    british = R.load(*BRITISH, root=golden_gold_lake)               # rain: left out
    fitted = RetirementModel.fit([qatar, british])
    assert (fitted.retirements, fitted.car_laps) == (3, 33 + 34 + 2 + 15 * 51)
    assert fitted.hazard == pytest.approx(3 / 834) and "3 in 834 car-laps" in fitted.describe()


def test_a_certain_hazard_retires_every_car_with_laps_left():
    drawn = RetirementModel(hazard=1.0).draw(np.random.default_rng(0), {"AAA": 30, "BBB": 10, "CCC": 6}, 5, 30)
    assert drawn == {"AAA": 6, "BBB": 6}                              # CCC's plan already ends on lap 6
    assert RetirementModel(hazard=0.0).draw(np.random.default_rng(0), {"AAA": 30}, 5, 30) == {}


def test_the_retirements_follow_the_hazard():
    # One car, 100 laps, 1 % a lap: it retires before the flag unless 99 laps in a row are clear: 1 - 0.99^99 = 63 %.
    rng, model = np.random.default_rng(3), RetirementModel(hazard=0.01)
    share = np.mean([bool(model.draw(rng, {"AAA": 100}, 0, 100)) for _ in range(5000)])
    assert share == pytest.approx(1 - 0.99 ** 99, abs=0.02)
    assert model.draw(np.random.default_rng(4), {"AAA": 100}, 0, 100) == model.draw(np.random.default_rng(4), {"AAA": 100}, 0, 100)
