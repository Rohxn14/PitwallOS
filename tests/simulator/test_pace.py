"""
Phase 4 step 1: lap times for planned laps (src/simulator/pace.py), on races that follow the formula.

With no stop between the start lap S and lap S+k, a forecast is exactly the rung's own predict(laps, k)
minus its traffic term (rungs 2, 3 and 4). After a stop it follows the plan: on exact physics, the new
tyres' laps are predicted exactly. Hindsight pace is each lap's own time without its traffic term. The
pace prior recovers made-up coefficients, and only moves each driver's level; the team's last race is
read from races BEFORE the one simulated, never from it or later ones.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.boosted import Boosted
from src.models.linear import LinearDecomposition, next_lap_terms
from src.models.persistence import Persistence
from src.models.state_space import StateSpace
from src.simulator import pace
from src.simulator.race import PLAN_COLUMNS
from tests.models.simulated_race import PHYSICS, PLANS, formula_race

START = 5


def season(n_races, **kwargs):
    return pd.concat([formula_race(2023, r, seed=r, **kwargs) for r in range(1, n_races + 1)], ignore_index=True)


def plan_of(race_laps, start=START):
    """The plan of a formula race after the start lap: its own tyres and pit laps."""
    planned = race_laps[race_laps["lap_number"] > start].assign(
        is_tyre_change=lambda d: d["is_in_lap"], pit_kind=lambda d: np.where(d["is_in_lap"], "tyre_change", None))
    return planned[PLAN_COLUMNS].reset_index(drop=True)


@pytest.fixture(scope="module")
def races():
    """Six races with wander and noise (training), and one to forecast: the formula's own gaps cycling."""
    laps = season(7, step_sd=0.05, noise_sd=0.1)
    return laps[laps["round"] < 7], laps[laps["round"] == 7].reset_index(drop=True)


def traffic_term(rung2, rows, k):
    coef = rung2.coefficients()
    traffic = [c for c in coef.index if c.startswith("traffic")]
    return next_lap_terms(rows, k)[traffic] @ coef[traffic]


@pytest.mark.parametrize("make", [LinearDecomposition, StateSpace, Boosted])
def test_before_a_stop_the_forecast_is_the_rungs_own_prediction_without_traffic(races, make):
    train, race = races
    rung = make().fit(train)
    hist = evaluate.model_inputs(race[race["lap_number"] <= START])
    planned = plan_of(race)
    forecast = pace.forecast(rung, hist, planned, traffic=pace.HELD)
    rung2 = rung if isinstance(rung, LinearDecomposition) else rung.rung2 if isinstance(rung, StateSpace) else rung.rung3.rung2
    checked = 0
    for driver, (_, _, stop, _) in PLANS.items():
        at_start = (hist["driver_code"] == driver) & (hist["lap_number"] == START)
        for lap in range(START + 1, stop):                     # the laps before its stop: lap S's tyres, older
            k = lap - START
            expected = (rung.predict(hist, k) - traffic_term(rung2, hist, k))[at_start].iloc[0]
            got = forecast[(planned["driver_code"] == driver) & (planned["lap_number"] == lap)].iloc[0]
            assert got == pytest.approx(expected, abs=1e-9)
            checked += 1
    assert checked == sum(stop - START - 1 for _, _, stop, _ in PLANS.values())


def test_on_exact_physics_every_planned_lap_is_right_even_after_a_stop():
    laps = season(4, gap_cycle=[4.0])                                 # exact physics, clear air all race
    rung = LinearDecomposition().fit(laps[laps["round"] < 4])
    race = laps[laps["round"] == 4].reset_index(drop=True)
    planned = plan_of(race)
    forecast = pace.forecast(rung, evaluate.model_inputs(race[race["lap_number"] <= START]), planned)
    real = race.set_index(["driver_code", "lap_number"])["lap_time_sec"]
    on_plan = real.reindex(pd.MultiIndex.from_frame(planned[["driver_code", "lap_number"]])).to_numpy()
    flying = ~(planned["is_in_lap"] | planned["is_out_lap"]).to_numpy()   # pit laps carry the made-up 20 s
    assert np.abs(forecast.to_numpy()[flying] - on_plan[flying]).max() < 1e-6
    after = flying & (planned["lap_number"] > planned["driver_code"].map({d: p[2] for d, p in PLANS.items()})).to_numpy()
    assert after.sum() > 40                                           # new tyres, another compound


def test_persistence_ignores_the_plan(races):
    _, race = races
    hist = evaluate.model_inputs(race[race["lap_number"] <= START])
    forecast = pace.forecast(Persistence(), hist, plan_of(race))
    assert forecast.groupby(plan_of(race)["driver_code"]).nunique().eq(1).all()


def test_hindsight_is_each_laps_own_time_without_its_traffic():
    laps = season(4)                                                  # exact physics, gaps cycling
    rung = LinearDecomposition().fit(laps[laps["round"] < 4])
    race = laps[laps["round"] == 4].reset_index(drop=True)
    planned = plan_of(race)
    got = pace.hindsight(rung, evaluate.model_inputs(race), planned)
    rows = race.set_index(["driver_code", "lap_number"]).reindex(pd.MultiIndex.from_frame(planned[["driver_code", "lap_number"]]))
    coef = pd.Series(PHYSICS)
    traffic = [c for c in coef.index if c.startswith("traffic")]
    from src.models.linear import lap_terms
    own_traffic = lap_terms(rows["compound"], rows["tyre_age"], rows["fuel_mass_est"], rows["gap_on_track_start_sec"])[traffic] @ coef[traffic]
    clean = rows["is_clean_lap"].to_numpy(bool)
    assert np.abs(got.to_numpy()[clean] - (rows["lap_time_sec"] - own_traffic).to_numpy()[clean]).max() < 1e-6


def test_realised_relative_pace_is_against_the_median_driver():
    laps = season(4, gap_cycle=[4.0])
    rung = LinearDecomposition().fit(laps[laps["round"] < 4])
    race = laps[laps["round"] == 4]
    relative = pace.realised_relative_pace(rung, race, START)
    assert relative.to_dict() == pytest.approx({"AAA": -0.5, "BBB": 0.0, "CCC": 0.5})   # bases 90.0, 90.5, 91.0


def test_the_prior_recovers_made_up_coefficients():
    rng = np.random.default_rng(0)
    rows = []
    for lap in (5, 15, 25, 35, 45):
        n = 400
        rel, grid, team = rng.normal(0, 0.5, n), rng.integers(-9, 10, n).astype(float), rng.normal(0, 0.4, n)
        w = lap / (lap + 30.0)
        rows.append(pd.DataFrame({"lap": lap, "rel": rel, "grid": grid, "team": team,
                                  "realised": w * rel + (1 - w) * (0.07 * grid + 0.5 * team)}))
    prior = pace.PacePrior().fit(pd.concat(rows))
    assert prior.per_place == pytest.approx(0.07, abs=0.003) and prior.per_team == pytest.approx(0.5, abs=0.03)
    assert prior.weight(5) == pytest.approx(5 / 35, abs=0.01)


def test_the_prior_moves_only_each_drivers_level(races):
    train, race = races
    rung = LinearDecomposition().fit(train)
    planned = plan_of(race)
    forecast = pace.forecast(rung, evaluate.model_inputs(race[race["lap_number"] <= START]), planned)
    terms = pace.planned_terms(rung, planned, int(race["total_race_laps"].iloc[0]))
    prior = pace.PacePrior(s0=20.0, per_place=0.1, per_team=0.5)
    moved = prior.adjust(forecast, planned, terms, grid={"AAA": 3, "BBB": 1, "CCC": 2}, team_last={"AAA": 0.2}, lap=START)
    shift = (moved - forecast).groupby(planned["driver_code"])
    assert (shift.max() - shift.min()).max() < 1e-9                 # one shift per driver: the shape is kept
    rel = pace.implied_base(forecast, planned, terms)
    w = START / (START + 20.0)
    expected_aaa = w * rel["AAA"] + (1 - w) * (0.1 * 1 + 0.5 * 0.2)  # grid 3 against the median 2
    assert (rel["AAA"] + shift.first()["AAA"]) == pytest.approx(expected_aaa)
    unmoved = pace.PacePrior(s0=1e-9).adjust(forecast, planned, terms, {}, {}, START)   # w = 1: the race alone
    assert np.allclose(unmoved, forecast, atol=1e-6)


def test_the_teams_last_race_comes_from_earlier_races_only():
    form = pd.DataFrame({"season": [2023, 2023, 2024, 2024, 2024], "round": [21, 22, 1, 2, 3],
                         "team": ["Red", "Red", "Red", "Blue", "Red"], "pace": [0.1, 0.2, 0.3, -0.4, 0.9]})
    laps = pd.DataFrame({"driver_code": ["AAA", "BBB", "CCC"], "team_code": ["Red", "Blue", "Green"]})
    race = SimpleNamespace(season=2024, round=2, laps=laps)
    # Red's latest before 2024 round 2 is round 1 (0.3), not round 3; Blue's only one is this race: none.
    assert pace.team_last_race(race, form) == {"AAA": 0.3}
