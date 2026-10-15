"""
Phase 3 rung 1: persistence (src/models/persistence.py, typed by Rohan).

Hand-worked predictions on the simulated race; golden ones on Qatar 2024 from F1's lap times; the
leakage check; the command line on both reference races.
"""

import shutil

import pandas as pd
import pytest

from src.models import evaluate
from src.models.persistence import (EARLIER_CLEAN_LAP, NO_PREDICTION, THIS_LAP, Persistence, main,
                                    prediction_source)
from src.pipeline import lake
from tests.models.simulated_race import simulated_race

QATAR = (2024, 23)


def predictions_by_lap(laps, driver):
    """Persistence's prediction made at the end of each of this driver's laps, by lap number."""
    predicted = Persistence().predict(evaluate.model_inputs(laps))
    mine = laps["driver_code"] == driver
    return pd.Series(predicted[mine].to_numpy(), index=laps.loc[mine, "lap_number"])


def test_the_prediction_is_the_last_clean_lap():
    aaa = predictions_by_lap(simulated_race(), "AAA")
    assert aaa[2] == 90.0 and aaa[4] == 91.2 and aaa[7] == 91.8     # lap N itself
    assert pd.isna(aaa[1])           # lap 1 is never clean and nothing came before it (never scored)


def test_after_a_safety_car_lap_it_uses_the_last_clean_lap_before_it():
    bbb = predictions_by_lap(simulated_race(), "BBB")
    assert bbb[5] == 91.0            # lap 5 (120 s behind the safety car) is not clean: lap 4's time
    assert bbb[6] == 91.5            # lap 6 is clean again


def test_every_later_lap_gets_the_same_guess():
    laps = simulated_race()
    pd.testing.assert_series_equal(Persistence().predict(laps, laps_ahead=5), Persistence().predict(laps))


def test_it_learns_nothing():
    rung = Persistence()
    assert rung.fit(simulated_race(2023, 5)) is rung


def test_prediction_source_names_the_rule_used():
    laps = simulated_race()
    bbb = prediction_source(laps)[laps["driver_code"] == "BBB"].tolist()
    assert bbb == [NO_PREDICTION, THIS_LAP, THIS_LAP, THIS_LAP, EARLIER_CLEAN_LAP, THIS_LAP, THIS_LAP, THIS_LAP]


@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    return evaluate.load_race(*QATAR, root=golden_gold_lake)


def test_qatar_hamilton_lap_46_is_predicted_from_lap_45(qatar):
    # F1's lap times: Hamilton lap 45 1:25.280, lap 46 1:24.726.
    ham = predictions_by_lap(qatar, "HAM")
    assert ham[45] == pytest.approx(85.280)
    actual = qatar.query("driver_code == 'HAM' and lap_number == 45")[evaluate.TARGET].item()
    assert ham[45] - actual == pytest.approx(0.554, abs=1e-3)


def test_qatar_verstappen_after_the_safety_car_is_predicted_from_lap_34(qatar):
    # Laps 35-42: safety car, stop, safety car. Verstappen's last clean lap was lap 34, 1:24.468;
    # the restart lap 43 took 1:24.509.
    ver = predictions_by_lap(qatar, "VER")
    assert ver[42] == pytest.approx(84.468)


def test_persistence_never_uses_a_later_lap(qatar):
    assert evaluate.future_leaks(Persistence(), qatar, every=5) == 0


def test_persistence_predicts_every_scored_lap_of_both_reference_races(golden_gold_lake):
    laps = evaluate.load_laps([2024], root=golden_gold_lake)
    predictions = evaluate.cross_validate(Persistence, laps)
    assert predictions[laps["scored"]].notna().all()


def test_command_line_scores_the_reference_races_and_writes_a_report(golden_gold_lake, tmp_path, capsys):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    assert main(["--season", "2024", "--lake", str(root)]) == 0
    out = capsys.readouterr().out
    assert "persistence: 2 races of 2024" in out
    assert "Headline: MAE on dry laps" in out and "Predictions by rule" in out
    report = lake.read_report("pace_model_persistence", str(root))
    assert report[["season", "round"]].values.tolist() == [[2024, 12], [2024, 23]]


def test_command_line_refuses_the_test_season(tmp_path):
    with pytest.raises(evaluate.SeasonLocked):
        main(["--season", "2025", "--lake", str(tmp_path)])
