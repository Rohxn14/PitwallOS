"""
Phase 3 rung 4: LightGBM corrections of rung 3 (src/models/boosted.py).

Rung 3's inputs worked by hand on a race with exact physics. A made-up season with an effect the formula
does not have (the first flying lap on new tyres 0.6 s faster): the trees find it on races they never saw,
and so does the hand-written baseline; with nothing to learn, the correction is 0. Wet tyres get none, and
too few training races give none. On Qatar 2024: lap N+k's inputs are what Gold recorded for lap N+k, and
the row before Verstappen's restart lap says what Race Control and the timing say (safety car, 8 laps since
his last clean lap). No leak; every scored lap predicted; the CLI.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.boosted import (AFTER_NEUTRALISED, AFTER_STOP, AHEAD_FEATURES, LAP_KINDS, NEXT_LAP_FEATURES,
                                ORDINARY, Boosted, SpecialLaps, features, lap_kind, main, rung3_inputs)
from src.models.state_space import StateSpace
from src.pipeline import lake
from tests.models.simulated_race import PLANS, formula_race

QATAR = 23
EFFECT = -0.6          # the made-up first flying lap on new tyres


def season(n_races, **effects):
    return pd.concat([formula_race(2023, r, seed=r, **effects) for r in range(1, n_races + 1)], ignore_index=True)


@pytest.fixture(scope="module")
def exact():
    """Races that follow the formula exactly, in clear air all race: rung 3 predicts every scored lap without
    error at any number of laps ahead (with changing gaps it would not: it holds the gap at lap N's)."""
    return season(12, gap_cycle=[4.0])


@pytest.fixture(scope="module")
def with_effect():
    """120 races with the hidden effect and a little noise, in clear air: 90 to learn from, 30 to test on."""
    laps = season(120, noise_sd=0.05, first_flying_lap_sec=EFFECT, gap_cycle=[4.0])
    held_out = laps["round"] > 90
    return laps[~held_out], laps[held_out]


# --- the inputs -----------------------------------------------------------------------------------

def test_rung_3_inputs_by_hand_on_exact_physics(exact):
    rung3 = StateSpace().fit(exact)
    race = exact[exact["round"] == 1]
    aaa = race[race["driver_code"] == "AAA"].set_index("lap_number")
    inputs = rung3_inputs(rung3, race).loc[race["driver_code"] == "AAA"].set_axis(aaa.index)
    stop = PLANS["AAA"][2]                                    # in-lap 12, out-lap 13: no reading on either
    assert np.isnan(inputs.loc[1, "laps_since_reading"])      # lap 1: no clean lap yet
    assert inputs.loc[stop - 1:stop + 2, "laps_since_reading"].tolist() == [0, 1, 2, 0]
    # Exact physics: rung 3's prediction of every clean lap was right, and its level never moved.
    clean_after_lap_2 = aaa["is_clean_lap"] & (aaa.index > 2)
    assert inputs.loc[clean_after_lap_2, "last_error"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert inputs["level_drift"].dropna().abs().max() == pytest.approx(0.0, abs=1e-6)
    assert inputs["level_drift"].isna().sum() == 6           # laps 1-6: no level 5 laps earlier (first reading lap 2)
    assert (inputs["odd_reading_skipped"] == 0).all()


def test_with_nothing_to_learn_the_correction_is_zero(exact):
    rung = Boosted().fit(exact)
    for k in (1, 4):
        assert rung.correction(exact, k)[exact["scored"]].abs().max() == pytest.approx(0.0, abs=1e-6)


# --- learning what rung 3 misses ------------------------------------------------------------------

def test_the_trees_find_an_effect_rung_3_cannot_see_on_races_they_never_saw(with_effect):
    train, test = with_effect
    rung = Boosted().fit(train)
    assert rung.trees["next lap"] > 0
    correction = rung.correction(test)
    kind = lap_kind(test, rung3_inputs(rung.rung3, test))
    scored = test["scored"]
    # The row of the out-lap predicts the first flying lap: about the 0.6 s rung 3 does not know about.
    assert correction[scored & (kind == AFTER_STOP)].median() == pytest.approx(EFFECT, abs=0.15)
    # Far from a stop there is little to correct.
    settled = scored & (kind == ORDINARY) & (test["tyre_age"] >= 8)
    assert correction[settled].abs().median() < 0.05
    rung3_mae = (rung.rung3.predict(test) - test[evaluate.TARGET])[scored].abs().mean()
    rung4_mae = (rung.predict(test) - test[evaluate.TARGET])[scored].abs().mean()
    assert rung4_mae < rung3_mae - 0.01


def test_the_hand_written_baseline_finds_the_same_effect(with_effect):
    train, _ = with_effect
    rung = SpecialLaps().fit(train)
    assert list(rung.medians.index) == list(LAP_KINDS)
    assert rung.medians.loc[AFTER_STOP, 1] == pytest.approx(EFFECT, abs=0.15)
    assert abs(rung.medians.loc[ORDINARY, 1]) < 0.05


def test_too_few_races_to_cross_validate_means_no_correction(exact):
    one_race = exact[exact["round"] == 1]
    rung = Boosted().fit(one_race)
    assert rung.next_lap is None and rung.ahead is None
    pd.testing.assert_series_equal(rung.predict(one_race), rung.rung3.predict(one_race), check_names=False)


# --- the reference races --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference(golden_gold_lake):
    return evaluate.load_laps([2024], root=golden_gold_lake)


@pytest.fixture(scope="module")
def fitted(reference):
    return Boosted().fit(reference)


def test_lap_n_plus_k_inputs_are_what_gold_recorded_for_lap_n_plus_k(reference, fitted):
    q = reference[reference["round"] == QATAR].reset_index(drop=True)
    by_driver = q.groupby("driver_code")
    later = by_driver.shift(-3)
    same_stint = (later["lap_number"] == q["lap_number"] + 3) & (later["stint_lap"] == q["stint_lap"] + 3)
    f = features(q, rung3_inputs(fitted.rung3, q), 3)[same_stint]
    assert same_stint.sum() > 700
    assert (f["predicted_lap"] == later.loc[same_stint, "lap_number"]).all()
    assert (f["predicted_tyre_age"] == later.loc[same_stint, "tyre_age"]).all()
    assert (f["predicted_stint_lap"] == later.loc[same_stint, "stint_lap"]).all()
    assert (f["predicted_progress"] - later.loc[same_stint, "race_progress"]).abs().max() < 1e-4   # Gold rounds it
    assert list(features(q, rung3_inputs(fitted.rung3, q), 1)[AHEAD_FEATURES].columns) == AHEAD_FEATURES
    assert "laps_ahead" not in NEXT_LAP_FEATURES


def test_qatar_verstappen_restart_lap_is_predicted_from_a_safety_car_row(reference, fitted):
    # Race Control: safety car on the leader's laps 40-42. Verstappen: last clean lap 34 (1:24.468), in-lap 35,
    # through the pit lane behind the safety car on 36-37, on hards from lap 36; lap 43 is the restart lap.
    q = reference[reference["round"] == QATAR]
    ver = q["driver_code"] == "VER"
    row = {n: q.index[ver & (q["lap_number"] == n)][0] for n in (33, 34, 41, 42)}
    inputs = rung3_inputs(fitted.rung3, q)
    one_lap = features(q, inputs, 1)
    assert q.loc[row[42], "scored"]
    assert inputs.loc[row[42], "laps_since_reading"] == 8               # laps 35 to 42: no reading
    assert lap_kind(q, inputs)[row[42]] == AFTER_NEUTRALISED
    assert one_lap.loc[row[42], ["is_sc", "compound", "predicted_tyre_age", "predicted_lap"]].tolist() == \
        [1.0, 2.0, 8.0, 43.0]                                          # under the SC, hards, 8 laps old on lap 43
    # Rung 3's error on lap N: lap N's time against what rung 3 predicted for it when lap N-1 ended.
    rung3 = fitted.rung3.predict(q)
    assert inputs.loc[row[34], "last_error"] == pytest.approx(84.468 - rung3[row[33]])
    assert inputs.loc[row[42], "last_error"] > 50                       # lap 42 behind the safety car: 149.083 s


def test_wet_tyre_laps_get_rung_3_as_it_is(reference, fitted):
    wet = reference["is_wet_tyre"].astype(bool)
    assert wet.sum() > 50                                               # British: inters from lap 28 (Hamilton) on
    pd.testing.assert_series_equal(fitted.predict(reference)[wet], fitted.rung3.predict(reference)[wet],
                                   check_names=False)


def test_rung_4_never_uses_a_later_lap(reference, fitted):
    assert fitted.next_lap is not None and fitted.ahead is not None
    assert evaluate.future_leaks(fitted, reference, every=5) == 0
    assert evaluate.future_leaks(fitted, reference, every=5, laps_ahead=3) == 0


def test_rung_4_predicts_every_scored_lap_one_and_five_laps_ahead(reference):
    predictions = evaluate.horizon_predictions(Boosted, reference, (1, 5))
    for k in (1, 5):
        assert predictions[k][evaluate.ahead_answers(reference, k)["scored"]].notna().all()


def test_command_line_compares_every_rung_and_writes_the_reports(golden_gold_lake, tmp_path, capsys):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    assert main(["--season", "2024", "--lake", str(root)]) == 0
    out = capsys.readouterr().out
    assert "boosted: 2 races of 2024" in out and "Against state_space" in out and "Against special_laps" in out
    assert "Laps ahead" in out and "What the trees use" in out and "by what lap N was" in out
    horizons = lake.read_report("pace_model_horizons", str(root))
    assert horizons["rung"].tolist()[:5] == ["persistence", "linear", "state_space", "special_laps", "boosted"]
    assert lake.read_report("pace_model_boosted", str(root))["round"].tolist() == [12, 23]
    assert set(lake.read_report("pace_model_boosted_inputs", str(root))["input"]) == set(AHEAD_FEATURES)
