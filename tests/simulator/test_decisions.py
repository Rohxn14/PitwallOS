"""
Phase 4 step 2: decisions (src/simulator/decisions.py), in a lake of the three reference races.

A "what if" that changes nothing is the real race. Verstappen's options in Qatar at the end of lap 20 (box
next lap on each dry compound instead of his lap-35 stop): mediums again break the two-compounds rule. The
real undercut of Hungary 2024 (Norris on Piastri, lap 45, 2.051 s behind, it worked by 5.428 s, Phase 2's
golden case) replayed out of fold from lap 44: the row holds the real facts and the simulator's call.
Neighbours at Qatar's safety cars that chose differently. The command line on the three races.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.pipeline import lake
from src.simulator import decisions, plans, race as R, replay

QATAR, HUNGARY = (2024, 23), (2024, 13)


@pytest.fixture(scope="module")
def three_races(golden_gold_lake, tmp_path_factory):
    from src.pipeline.cleaner import build_silver
    from src.pipeline.feature_store import build_gold
    from tests.conftest import HUNGARY_FIXTURE_LAKE
    root = tmp_path_factory.mktemp("three") / "lake"
    shutil.copytree(golden_gold_lake, root)
    shutil.copytree(HUNGARY_FIXTURE_LAKE / "bronze" / "season=2024" / "round=13",
                    root / "bronze" / "season=2024" / "round=13")
    build_silver(*HUNGARY, "R", lake_root=str(root), telemetry=False)
    build_gold(*HUNGARY, "R", lake_root=str(root))
    return str(root)


@pytest.fixture(scope="module")
def folds(three_races):
    laps = evaluate.load_laps((2024,), three_races)
    races = {key: R.load(*key, root=three_races) for key in evaluate.races_of(laps)}
    fitted = {key: replay.fit_fold(laps, [key], races, three_races) for key in races}
    return races, fitted


def test_a_change_that_changes_nothing_is_the_real_race(folds):
    races, fitted = folds
    result = decisions.what_if(races[QATAR], fitted[QATAR], 20, "VER", lambda plan: plan)
    assert result.changed.laps["clock_sec"].equals(result.real.laps["clock_sec"])
    assert result.answer is None and result.problems == []
    assert result.expected_rank() == decisions.rank_of(result.real, "VER")


def test_verstappens_options_at_lap_20(folds):
    races, fitted = folds
    table = decisions.options(races[QATAR], fitted[QATAR], 20, "VER")
    assert table["option"].tolist() == ["as planned (stop lap 35)", "box lap 21, new soft (instead)",
                                        "box lap 21, new medium (instead)", "box lap 21, new hard (instead)"]
    problems = dict(zip(table["option"], table["problems"]))
    assert problems["box lap 21, new medium (instead)"] == "a dry race needs two different dry compounds"
    assert problems["box lap 21, new hard (instead)"] == "" and problems["as planned (stop lap 35)"] == ""
    assert table["expected_rank"].between(1, 20).all() and table["chance"].between(0, 1).all()


def test_the_hungary_undercut_replayed_out_of_fold(folds, three_races):
    races, fitted = folds
    rows = pd.DataFrame(decisions.undercut_calls(races[HUNGARY], fitted[HUNGARY], sources=("hindsight", decisions.BEST),
                                                 root=three_races))
    norris = rows[(rows["driver"] == "NOR") & (rows["in_lap"] == 45)]
    assert set(norris["source"]) == {"hindsight", decisions.BEST}
    first = norris.iloc[0]
    assert first["car_ahead"] == "PIA" and first["car_ahead_in_lap"] == 47
    assert first["gap_before_sec"] == pytest.approx(2.051) and first["real_gain_sec"] == pytest.approx(5.428)
    assert bool(first["real_worked"]) and norris["sim_gap_after_sec"].notna().all()
    assert set(norris["sim_worked"]) <= {True, False}
    values = pd.DataFrame(decisions.undercut_value(races[HUNGARY], fitted[HUNGARY], root=three_races))
    nor = values[(values["driver"] == "NOR") & (values["in_lap"] == 45)].iloc[0]
    assert nor["value_sec"] == pytest.approx(nor["gap_waiting_sec"] - nor["gap_undercut_sec"])


def test_neighbours_that_chose_differently_at_qatars_safety_cars(folds):
    races, fitted = folds
    rows = pd.DataFrame(decisions.neutralisation_pairs(races[QATAR], fitted[QATAR], sources=(decisions.BEST,)))
    assert len(rows) and set(rows["first_lap"]) <= {35, 40}
    assert (rows["boxed"].eq(rows["ahead"]) | rows["boxed"].eq(rows["behind"])).all()
    assert rows[decisions.BEST].isin([True, False]).all() and rows["real_ahead_first"].isin([True, False]).all()


def test_command_line_options_for_one_moment(three_races, capsys):
    assert decisions.main(["--season", "2024", "--lake", three_races, "--options", "2024", "13", "NOR", "43"]) == 0
    out = capsys.readouterr().out
    assert "Hungarian Grand Prix 2024, NOR at the end of lap 43" in out
    assert "as planned (stop lap 45)" in out and "box lap 44, new hard (instead)" in out


def test_command_line(three_races, tmp_path, capsys):
    root = tmp_path / "lake"
    shutil.copytree(three_races, root)
    assert decisions.main(["--season", "2024", "--lake", str(root)]) == 0
    out = capsys.readouterr().out
    assert "Undercut calls:" in out and "Neutralisations:" in out and "Undercut value" in out
    calls = lake.read_report(decisions.CALLS_REPORT, str(root))
    assert set(calls["season"]) == {2024} and (calls["round"] == 13).any()


def test_an_answer_to_a_stop_on_the_first_planned_lap(folds):
    races, fitted = folds
    # Norris boxes at the end of lap 44 instead of 45, from the end of lap 43: the lap before the stop is the
    # start itself, and Piastri, 2 s ahead then, may answer.
    result = decisions.what_if(races[HUNGARY], fitted[HUNGARY], 43, "NOR",
                               lambda plan: plans.move_stop(plan, "NOR", 45, 44))
    assert result.answer is not None and result.answer.driver == "PIA" and 0 < result.chance < 1
    assert plans.stops_of(result.answer.plan, "PIA") == [45]
