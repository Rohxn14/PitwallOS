"""
Phase 4, the end: the simulator's test season (src/simulator/holdout.py).

A copy of the game's three-race lake in which Qatar 2024 is relabelled as a 2025 race (and the game's saved models, which
had seen Qatar, removed): the command line fits the simulator once on British and Hungary 2024, judges it on "2025"
Qatar by the three steps' measures, and writes its four reports; none of the misses it draws comes from the test race.
"""

import shutil

import pytest

from src.pipeline import lake
from src.simulator import holdout
from tests.models.test_holdout import relabel_as

QATAR = 23


@pytest.fixture(scope="module")
def lake_with_a_test_race(game_lake, tmp_path_factory):
    _, source = game_lake
    root = tmp_path_factory.mktemp("sim_holdout") / "lake"
    shutil.copytree(source, root)
    shutil.rmtree(root / lake.MODELS_DIR)                  # fitted with Qatar among them: not for this test
    relabel_as(str(root), 2024, 2025, QATAR)
    # The simulator also reads the race's official result (Bronze): move it with the race.
    shutil.move(str(lake.session_dir(lake.BRONZE, 2024, QATAR, "R", str(root))),
                str(lake.session_dir(lake.BRONZE, 2025, QATAR, "R", str(root))))
    return str(root)


def test_the_simulator_is_fitted_on_2024_and_judged_on_2025(lake_with_a_test_race, capsys):
    root = lake_with_a_test_race
    assert holdout.main(["--train-season", "2024", "--test-season", "2025", "--lake", root, "--runs", "10"]) == 0
    out = capsys.readouterr().out
    assert "Test season 2025: 1 races (1 dry); the simulator fitted once on the 2 races of 2024." in out
    assert "THE TEST SEASON" in out and "places off per car at the flag from lap 5" in out
    assert "Neutralisations:" in out and "From lap 5" in out
    replay_rows = lake.read_report(holdout.REPORTS["replay"], root)
    assert set(zip(replay_rows["season"], replay_rows["round"])) == {(2025, QATAR)}
    assert set(replay_rows["source"]) == set(holdout.SOURCES)
    chances = lake.read_report(holdout.REPORTS["chances"], root)
    assert set(chances["config"]) == {"+ retirements", "history"} and set(chances["season"]) == {2025}
    pairs = lake.read_report(holdout.REPORTS["pairs"], root)
    assert set(pairs["first_lap"]) <= {35, 40}                          # Qatar's safety cars
    assert "The development races" not in out                            # no development reports in this lake


def test_the_misses_drawn_never_come_from_the_test_season(lake_with_a_test_race):
    from src.simulator import game
    models = game.load_models(lake_with_a_test_race, seasons=(2024,), log=lambda text: None)
    assert set(models.misses.offsets["season"]) == {2024} and set(models.misses.places_all["season"]) == {2024}
