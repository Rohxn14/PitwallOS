"""
Phase 3 step 5: the test season (src/models/holdout.py).

A lake in which Qatar 2024 is relabelled as a 2025 race: the command line fits every rung once on British
2024 and scores "2025", lines it up with the development report when there is one, and writes its reports.
The test season can never be trained on. against_development worked by hand.
"""

import shutil

import pandas as pd
import pytest

from src.models.evaluate import SeasonLocked
from src.models.holdout import against_development, main
from src.pipeline import lake

QATAR = 23


def relabel_as(root, season_from, season_to, round_number):
    """Move one race's Silver laps and Gold features to another season (the season column too)."""
    for layer, table in ((lake.SILVER, "laps"), (lake.GOLD, lake.GOLD_TABLE)):
        old = lake.table_path(layer, table, season_from, round_number, "R", root)
        df = lake.read_table(old).assign(season=season_to)
        lake.write_table(df, lake.table_path(layer, table, season_to, round_number, "R", root), lake.read_metadata(old))
        shutil.rmtree(lake.session_dir(layer, season_from, round_number, "R", root))


@pytest.fixture
def lake_with_a_test_race(golden_gold_lake, tmp_path):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    relabel_as(str(root), 2024, 2025, QATAR)
    return str(root)


def test_command_line_fits_on_2024_and_scores_2025(lake_with_a_test_race, capsys):
    root = lake_with_a_test_race
    assert main(["--train-season", "2024", "--test-season", "2025", "--lake", root]) == 0
    out = capsys.readouterr().out
    assert "Test season 2025: 1 races" in out and "fitted once on the 1 races of 2024" in out
    assert "boosted: 1 races of 2025" in out and "Laps ahead on the test season" in out
    assert "bands on the test season" in out and "by what lap N was" in out
    assert "Development (cross-validated" not in out             # no development report in this lake
    horizons = lake.read_report("pace_model_test_horizons", root)
    assert horizons["rung"].tolist()[:5] == ["persistence", "linear", "state_space", "special_laps", "boosted"]
    assert sorted(set(horizons["laps_ahead"])) == [1, 3, 5, 10]
    races = lake.read_report("pace_model_test_races", root)
    assert races[["season", "round"]].values.tolist() == [[2025, QATAR]]
    assert {"persistence", "boosted"} <= set(races.columns)


def test_with_a_development_report_the_two_are_lined_up(lake_with_a_test_race, capsys):
    root = lake_with_a_test_race
    development = pd.DataFrame({"laps_ahead": [1, 1], "rung": ["persistence", "boosted"], "dry_mae": [0.5, 0.4]})
    lake.write_report(development, "pace_model_horizons", root)
    assert main(["--train-season", "2024", "--test-season", "2025", "--lake", root]) == 0
    assert "Development (cross-validated" in capsys.readouterr().out


def test_the_test_season_can_never_be_trained_on(lake_with_a_test_race):
    with pytest.raises(SeasonLocked):
        main(["--train-season", "2024", "2025", "--test-season", "2025", "--lake", lake_with_a_test_race])


def test_against_development_by_hand():
    test = pd.DataFrame({"laps_ahead": [1, 1, 3], "rung": ["a", "b", "a"], "dry_mae": [0.50, 0.40, 0.70]})
    development = pd.DataFrame({"laps_ahead": [1, 1], "rung": ["a", "b"], "dry_mae": [0.45, 0.42]})
    table = against_development(test, development)
    assert table["change"].tolist()[:2] == pytest.approx([0.05, -0.02])
    assert pd.isna(table.loc[2, "development_mae"])               # a horizon the development report lacks
