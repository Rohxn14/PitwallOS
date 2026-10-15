"""
Phase 3 rung 3: the state-space model (src/models/state_space.py).

The Kalman filter worked by hand, its two limits (no averaging = rung 2's latest reading, no wander = the
running mean), and a made-up season whose bases wander with a known step and noise: the ratio chosen comes
out near the truth. On Qatar 2024, Verstappen's restart lap after the safety car (laps 35-42 without a
reading) gets the larger weight the filter's arithmetic says it should. No leak; the CLI.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.linear import LinearDecomposition
from src.models.state_space import StateSpace, level_filter, main, steady_gain
from src.pipeline import lake
from tests.models.simulated_race import formula_race


def one_driver(values, rho, restart=None):
    reading = pd.Series(values, dtype=float)
    key = pd.Series("AAA", index=reading.index)
    restart = pd.Series(False, index=reading.index) if restart is None else pd.Series(restart, index=reading.index)
    level, var = level_filter(reading, key, rho, restart)
    return level.tolist(), var.tolist()


# --- the filter, by hand -------------------------------------------------------------------------

def test_the_filter_worked_by_hand():
    level, var = one_driver([10.0, 12.0, np.nan, 11.0], rho=1.0)
    # lap 0: the first reading starts the level: 10, variance 1 (one reading's noise)
    # lap 1: variance 1 + 1 = 2, gain 2/3: level 10 + 2/3 x (12 - 10) = 11.333, variance 2/3
    # lap 2: no reading: variance 2/3 + 1 = 5/3, the level stays
    # lap 3: variance 5/3 + 1 = 8/3, gain 8/11: level 11.333 + 8/11 x (11 - 11.333) = 11.0909, variance 8/11
    assert level == pytest.approx([10.0, 34 / 3, 34 / 3, 122 / 11])
    assert var == pytest.approx([1.0, 2 / 3, 5 / 3, 8 / 11])


def test_after_laps_without_a_reading_the_next_one_is_trusted_more():
    settled, _ = one_driver([0.0] * 20 + [1.0], rho=1.0)
    after_gap, _ = one_driver([0.0] * 20 + [np.nan] * 3 + [1.0], rho=1.0)
    assert settled[-1] == pytest.approx(steady_gain(1.0), abs=1e-6)     # the settled weight: 0.618
    assert after_gap[-1] > settled[-1] + 0.1                            # a stop or a safety car: trust it more


def test_the_two_limits_latest_reading_and_running_mean():
    level, _ = one_driver([10.0, 12.0, 11.0, 13.0], rho=1e9)
    assert level == pytest.approx([10.0, 12.0, 11.0, 13.0], abs=1e-6)  # no averaging: rung 2's base
    level, _ = one_driver([10.0, 12.0, 11.0, 13.0], rho=1e-12)
    assert level == pytest.approx([10.0, 11.0, 11.0, 11.5], abs=1e-6)  # no wander: the running mean


def test_a_wet_reading_restarts_the_level_and_drivers_are_filtered_apart():
    level, var = one_driver([10.0, 12.0, 11.0], rho=1.0, restart=[False, False, True])
    assert level[2] == 11.0 and var[2] == 1.0
    reading = pd.Series([10.0, 20.0, 12.0, 22.0])
    key = pd.Series(["AAA", "BBB", "AAA", "BBB"])
    level, _ = level_filter(reading, key, 1.0, pd.Series(False, index=reading.index))
    assert level.tolist() == pytest.approx([10.0, 20.0, 10 + 4 / 3, 20 + 4 / 3])


# --- fitting --------------------------------------------------------------------------------------

def test_on_a_made_up_season_the_ratio_and_the_noise_come_out_near_the_truth():
    # Every driver's base takes a random step of 0.2 s a lap and every lap has 0.2 s of noise: q / h = 1.
    season = pd.concat([formula_race(2023, r, step_sd=0.2, noise_sd=0.2, seed=r) for r in range(1, 41)],
                       ignore_index=True)
    rung = StateSpace().fit(season)
    assert 0.5 < rung.rho < 2.0
    assert np.sqrt(rung.h) == pytest.approx(0.2, rel=0.25)


# --- the reference races --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference(golden_gold_lake):
    return evaluate.load_laps([2024], root=golden_gold_lake)


@pytest.fixture(scope="module")
def fitted(reference):
    return StateSpace().fit(reference)


def test_with_no_averaging_rung_3_is_rung_2(reference, fitted):
    rung3 = StateSpace().fit(reference)
    rung3.rho = 1e9
    scored = reference["scored"]
    pd.testing.assert_series_equal(rung3.predict(reference)[scored],
                                   LinearDecomposition().fit(reference).predict(reference)[scored],
                                   check_names=False, atol=1e-5)


def test_qatar_verstappen_restart_lap_gets_the_weight_the_filter_says(reference, fitted):
    # Laps 35-42: safety car, his stop, safety car: no reading for 8 laps after lap 34 (1:24.468).
    q = reference[reference["round"] == 23]
    reading = fitted.rung2.base_readings(q)
    level, var = fitted.filter(q)
    ver = q["driver_code"] == "VER"
    row = {n: q.index[ver & (q["lap_number"] == n)][0] for n in (34, 43)}
    assert reading.loc[q.index[ver & q["lap_number"].between(35, 42)]].isna().all()
    weight = (level[row[43]] - level[row[34]]) / (reading[row[43]] - level[row[34]])
    before = var[row[34]] + 9 * fitted.rho                  # 9 laps of wander: laps 35 to 43
    assert weight == pytest.approx(before / (before + 1.0))
    assert weight > steady_gain(fitted.rho)


def test_rung_3_never_uses_a_later_lap(reference, fitted):
    assert evaluate.future_leaks(fitted, reference, every=5) == 0
    assert evaluate.future_leaks(fitted, reference, every=5, laps_ahead=3) == 0


def test_rung_3_predicts_every_scored_lap_one_and_five_laps_ahead(reference):
    for k in (1, 5):
        predictions = evaluate.cross_validate(StateSpace, reference, laps_ahead=k)
        assert predictions[evaluate.ahead_answers(reference, k)["scored"]].notna().all()


def test_the_bands_widen_further_ahead_and_are_wider_in_the_wet(reference, fitted):
    scored = reference["scored"]
    one, five = fitted.predict_sd(reference, 1), fitted.predict_sd(reference, 5)
    assert (five[scored] > one[scored]).all()
    assert fitted.h_wet > fitted.h                          # British's laps on inters are noisier


def test_command_line_compares_the_rungs_at_every_horizon(golden_gold_lake, tmp_path, capsys):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    assert main(["--season", "2024", "--lake", str(root)]) == 0
    out = capsys.readouterr().out
    assert "state_space: 2 races of 2024" in out and "Against linear" in out and "Laps ahead" in out
    horizons = lake.read_report("pace_model_horizons", str(root))
    assert horizons["rung"].tolist()[:3] == ["persistence", "linear", "state_space"]
    assert sorted(set(horizons["laps_ahead"])) == [1, 3, 5, 10]
    assert lake.read_report("pace_model_state_space", str(root))["round"].tolist() == [12, 23]
