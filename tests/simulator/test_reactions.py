"""
Phase 4 step 2: rivals' answers (src/simulator/reactions.py), on Hungary 2024.

The threats of the race (a car stops under green less than 3 s behind the car ahead): Norris on Piastri at
the end of lap 45, 2.051 s behind, answered two laps later (Piastri's stop on lap 47); Norris's first stop
(lap 17, 3.5 s behind) is not a threat. The cover model recovers a made-up logistic and, with too few
threats, gives the measured rate to every threat. A "what if" where Norris stops on lap 43 instead: Piastri,
1.8 s ahead, may answer by moving his stop to lap 44; no answer to a stop that is not new, to a car more
than 3 s ahead, or from a car with no stop left.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.simulator import plans
from src.simulator import race as R
from src.simulator import reactions
from src.simulator.reactions import INPUTS, CoverModel

HUNGARY = (2024, 13)


@pytest.fixture(scope="module")
def hungary(tmp_path_factory):
    from src.pipeline.cleaner import build_silver
    from src.pipeline.feature_store import build_gold
    from tests.conftest import HUNGARY_FIXTURE_LAKE
    root = tmp_path_factory.mktemp("hungary_gold") / "lake"
    shutil.copytree(HUNGARY_FIXTURE_LAKE, root)
    build_silver(*HUNGARY, "R", lake_root=str(root), telemetry=False)
    build_gold(*HUNGARY, "R", lake_root=str(root))
    return R.load(*HUNGARY, root=str(root)), str(root)


def real_crossings(race):
    """The race's own crossings, as a simulation's laps would give them."""
    return race.laps.rename(columns={"lap_end_sec": "clock_sec"})[["driver_code", "lap_number", "clock_sec"]]


def test_the_threats_of_hungary(hungary):
    race, root = hungary
    threats = reactions.threats(race, root)
    norris = threats[(threats["driver_code"] == "NOR") & (threats["in_lap"] == 45)].iloc[0]
    assert norris["car_ahead"] == "PIA" and norris["gap"] == pytest.approx(2.051) and bool(norris["covered"])
    assert not ((threats["driver_code"] == "NOR") & (threats["in_lap"] == 17)).any()       # 3.5 s behind
    assert threats["covered"].sum() == 1 and len(threats) == 15


def test_the_cover_model_recovers_a_made_up_logistic():
    rng = np.random.default_rng(3)
    n = 4000
    rows = pd.DataFrame({"ahead_age": rng.uniform(2, 35, n), "age_difference": rng.normal(0, 6, n),
                         "gap": rng.uniform(0, 3, n), "laps_left": rng.uniform(5, 30, n)})
    z = -1.0 + 0.08 * (rows["ahead_age"] - 15) + 0.1 * rows["age_difference"]
    rows["covered"] = rng.random(n) < 1 / (1 + np.exp(-z))
    model = CoverModel.fit(rows)
    p = model.chances(rows)
    for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)):
        inside = (p >= low) & (p < high)
        if inside.sum() > 100:
            assert rows.loc[inside, "covered"].mean() == pytest.approx(p[inside].mean(), abs=0.05)
    few = CoverModel.fit(rows.head(5))
    assert few.chance(20, 0, 1, 20) == pytest.approx(reactions.DEFAULT_COVER_RATE)


def test_piastri_may_answer_norris_stopping_on_lap_43(hungary):
    race, _ = hungary
    model = CoverModel.fit(pd.DataFrame(columns=INPUTS + ["covered"]))       # the measured rate for all
    real = R.plan(race, 42)
    earlier = plans.move_stop(real, "NOR", 45, 43)
    answer = reactions.answer(race, real, earlier, "NOR", real_crossings(race), R.lap_conditions(race), model)
    assert answer.driver == "PIA" and answer.stop_lap == 43 and 0 < answer.gap < 3
    assert answer.chance == pytest.approx(reactions.DEFAULT_COVER_RATE)
    assert plans.stops_of(real, "PIA") == [47] and plans.stops_of(answer.plan, "PIA") == [44]
    # Nothing new: no answer.
    assert reactions.answer(race, real, real, "NOR", real_crossings(race), R.lap_conditions(race), model) is None


def test_no_answer_from_far_away_or_from_a_car_with_no_stop_left(hungary):
    race, _ = hungary
    model = CoverModel.fit(pd.DataFrame(columns=INPUTS + ["covered"]))
    real = R.plan(race, 42)
    crossings = real_crossings(race)
    # A car more than 3 s behind the car ahead when lap 42 ended, stopping at the end of lap 43: no threat.
    at_42 = crossings[crossings["lap_number"] == 42].set_index("driver_code")["clock_sec"].sort_values()
    gaps = at_42.diff()
    far = [d for d in gaps.index[1:] if gaps[d] >= reactions.THREAT_GAP_SEC and d in set(real["driver_code"])
           and 43 not in plans.stops_of(real, d) and 44 not in plans.stops_of(real, d)][0]
    alone = plans.box(real, far, 43, "HARD")
    assert reactions.answer(race, real, alone, far, crossings, R.lap_conditions(race), model) is None
    # Piastri with no stop left (staying out) cannot answer Norris.
    no_stop = plans.stay_out(plans.move_stop(real, "NOR", 45, 43), "PIA", 47)
    assert reactions.answer(race, real, no_stop, "NOR", crossings, R.lap_conditions(race), model) is None
