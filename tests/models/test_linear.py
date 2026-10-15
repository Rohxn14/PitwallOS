"""
Phase 3 rung 2: the linear decomposition (src/models/linear.py).

A race made from the formula's own physics (known offsets, wear, fuel, traffic): the fit recovers
every term and the prediction is exact. The odd-reading guard worked by hand. On Qatar 2024, the terms
predicted for lap N+1 are the ones Gold recorded for lap N+1, and Verstappen's restart lap is his lap 34
corrected by the terms. British 2024 (inters) is not a dry race, so it never trains the terms.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.linear import (LinearDecomposition, PHASE_2, THIS_LAP, base_source, drop_odd_readings, dry_races,
                               lap_terms, main, next_lap_terms, own_terms)
from src.models.persistence import Persistence
from src.pipeline import lake
from tests.models.simulated_race import PHYSICS, formula_race

BRITISH, QATAR = (2024, 12), (2024, 23)


@pytest.fixture(scope="module")
def made_up():
    return pd.concat([formula_race(2023, 1), formula_race(2023, 2, base_shift=3.0)], ignore_index=True)


# --- the formula --------------------------------------------------------------------------------

def test_the_terms_of_one_lap_by_hand():
    terms = lap_terms(pd.Series(["SOFT", "INTERMEDIATE", "HARD"]), pd.Series([3.0, 5.0, 12.0]),
                      pd.Series([100.0, 80.0, 50.0]), pd.Series([0.7, 1.2, np.nan]))
    soft, inter, hard = (terms.iloc[i] for i in range(3))
    assert soft[["offset SOFT", "wear SOFT", "fuel", "traffic 0.5-1 s"]].tolist() == [1.0, 3.0, 100.0, 1.0]
    assert soft.drop(["offset SOFT", "wear SOFT", "fuel", "traffic 0.5-1 s"]).eq(0).all()
    assert inter[[c for c in terms if c.startswith(("offset", "wear"))]].eq(0).all()   # inters: no tyre terms
    assert inter[["fuel", "traffic 1-1.5 s"]].tolist() == [80.0, 1.0]
    assert hard[["offset HARD", "wear HARD"]].tolist() == [1.0, 12.0]
    assert hard[[c for c in terms if c.startswith("traffic")]].eq(0).all()             # no car ahead: clear air


def test_the_fit_recovers_the_physics_of_a_made_up_race(made_up):
    rung = LinearDecomposition().fit(made_up)
    assert rung.training_races == 2
    for term, value in PHYSICS.items():
        assert rung.terms.loc[term, "coef"] == pytest.approx(value, abs=1e-6), term


def test_on_its_own_physics_every_lap_is_predicted_exactly(made_up):
    rung = LinearDecomposition().fit(made_up)
    scored = made_up["scored"]
    linear = evaluate.predictions_of(rung, made_up)
    assert (linear - made_up[evaluate.TARGET])[scored].abs().max() == pytest.approx(0.0, abs=1e-6)
    # Persistence cannot know the fuel, the wear, the stops or the traffic.
    persistence = evaluate.predictions_of(Persistence(), made_up)
    assert (persistence - made_up[evaluate.TARGET])[scored].abs().mean() > 0.2


def test_after_a_stop_the_base_carries_over_to_the_new_tyres(made_up):
    # AAA stops on lap 12 (in-lap) and 13 (out-lap): the lap-14 prediction is made from lap 11's base.
    rung = LinearDecomposition().fit(made_up)
    race = made_up[made_up["round"] == 1]
    aaa = race["driver_code"] == "AAA"
    row_13 = race.index[aaa & (race["lap_number"] == 13)][0]
    assert base_source(rung, race)[row_13] != THIS_LAP
    assert rung.predict(race)[row_13] == pytest.approx(race.loc[row_13, evaluate.TARGET], abs=1e-6)


# --- the odd-reading guard, by hand --------------------------------------------------------------

def guarded(values):
    reading = pd.Series(values, dtype=float)
    return drop_odd_readings(reading, pd.Series("AAA", index=reading.index)).tolist()


def test_a_single_odd_lap_is_skipped():
    # 95.0 is more than 3 s slower than the median of 90.0 and 90.2: a restart, a lock-up.
    assert guarded([90.0, 90.2, 95.0, 90.1]) == pytest.approx([90.0, 90.2, np.nan, 90.1], nan_ok=True)


def test_a_new_level_is_followed_from_the_second_odd_lap():
    # Rain: every lap 4 s slower than the last. Only the first slow lap is skipped.
    assert guarded([90.0, 94.0, 98.0, 102.0]) == pytest.approx([90.0, np.nan, 98.0, 102.0], nan_ok=True)


def test_a_faster_lap_is_never_odd_and_drivers_are_judged_apart():
    assert guarded([95.0, 90.0, 89.0]) == [95.0, 90.0, 89.0]
    reading = pd.Series([90.0, 100.0, 95.0, 100.2])
    drivers = pd.Series(["AAA", "BBB", "AAA", "BBB"])
    assert drop_odd_readings(reading, drivers).tolist() == pytest.approx([90.0, 100.0, np.nan, 100.2], nan_ok=True)


# --- the reference races --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference(golden_gold_lake):
    return {race: evaluate.load_race(*race, root=golden_gold_lake) for race in (BRITISH, QATAR)}


def test_british_is_not_a_dry_race_and_qatar_is(reference):
    assert not dry_races(reference[BRITISH]).any()       # inters from lap 28 (Hamilton) on
    assert dry_races(reference[QATAR]).all()


def test_the_terms_for_lap_n_plus_1_are_the_ones_gold_recorded_for_it(reference):
    q = reference[QATAR].reset_index(drop=True)
    nxt = q.groupby("driver_code").shift(-1)
    same_tyres = (nxt["lap_number"] == q["lap_number"] + 1) & ~q["is_in_lap"] & nxt["is_out_lap"].eq(False)
    predicted, recorded = next_lap_terms(q)[same_tyres], own_terms(nxt[same_tyres].assign(
        total_race_laps=q["total_race_laps"]))
    assert len(predicted) > 800                                          # every driver, most of the race
    pd.testing.assert_frame_equal(predicted.drop(columns="fuel"), recorded.drop(columns="fuel"))
    assert (predicted["fuel"] - recorded["fuel"]).abs().max() < 0.006    # Gold rounds fuel to 0.01 kg


def test_three_laps_ahead_the_tyre_and_fuel_terms_are_the_ones_gold_recorded(reference):
    # The gap three laps ahead is not known in advance (it is held at lap N's), so only tyres and fuel.
    q = reference[QATAR].reset_index(drop=True)
    by_driver = q.groupby("driver_code")
    later = by_driver.shift(-3)
    no_stop = later["lap_number"] == q["lap_number"] + 3
    for j in range(3):                                           # laps N .. N+2 are not in-laps
        no_stop &= by_driver["is_in_lap"].shift(-j).eq(False)
    predicted = next_lap_terms(q, laps_ahead=3)[no_stop]
    recorded = own_terms(later[no_stop].assign(total_race_laps=q["total_race_laps"]))
    tyres = [c for c in predicted if c.startswith(("offset", "wear"))]
    assert len(predicted) > 700
    pd.testing.assert_frame_equal(predicted[tyres], recorded[tyres])
    assert (predicted["fuel"] - recorded["fuel"]).abs().max() < 0.006    # Gold rounds fuel to 0.01 kg


def test_qatar_verstappen_restart_lap_is_lap_34_corrected_by_the_terms(reference):
    # Laps 35-42: safety car, his stop, safety car. Lap 34 (1:24.468) is the last clean lap before it.
    q = reference[QATAR]
    rung = LinearDecomposition().fit(q)
    coef = rung.coefficients()
    ver = q[q["driver_code"] == "VER"].set_index("lap_number")
    row_34, row_42 = q.index[(q["driver_code"] == "VER") & q["lap_number"].isin([34, 42])]
    by_hand = ver.loc[34, "lap_time_sec"] + (next_lap_terms(q).loc[row_42] - own_terms(q).loc[row_34]) @ coef
    assert rung.predict(q)[row_42] == pytest.approx(by_hand, abs=1e-9)
    assert by_hand < ver.loc[34, "lap_time_sec"]          # 9 laps of fuel burnt and fresh tyres: faster


def test_with_no_dry_race_to_learn_from_it_is_the_latest_good_clean_lap(reference):
    rung = LinearDecomposition().fit(reference[BRITISH])
    assert rung.training_races == 0
    assert (rung.coefficients() == 0).all()
    british = reference[BRITISH]
    own_lap = base_source(rung, british) == THIS_LAP
    pd.testing.assert_series_equal(rung.predict(british)[own_lap], Persistence().predict(british)[own_lap],
                                   check_names=False)


def test_rung_2_never_uses_a_later_lap(reference):
    rung = LinearDecomposition().fit(reference[QATAR])
    for laps in reference.values():
        assert evaluate.future_leaks(rung, laps, every=5) == 0


def test_rung_2_predicts_every_scored_lap_of_both_reference_races(golden_gold_lake):
    laps = evaluate.load_laps([2024], root=golden_gold_lake)
    predictions = evaluate.cross_validate(LinearDecomposition, laps)
    assert predictions[laps["scored"]].notna().all()


def test_command_line_compares_with_rung_1_and_writes_both_reports(golden_gold_lake, tmp_path, capsys):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    assert main(["--season", "2024", "--lake", str(root)]) == 0
    out = capsys.readouterr().out
    assert "linear: 2 races of 2024" in out and "Against persistence" in out
    assert "Terms fitted on the 1 dry races" in out
    terms = lake.read_report("pace_model_linear_terms", str(root))
    assert set(PHASE_2) <= set(terms["term"])
    assert lake.read_report("pace_model_linear", str(root))["round"].tolist() == [12, 23]
