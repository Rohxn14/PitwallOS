"""
Phase 4 step 1: replaying real races (src/simulator/replay.py), on the two reference races.

The command line replays both out of fold (each with rungs and rules from the other race only) and writes
its report: every start (lap 6, half distance, the lap before each of Qatar's safety cars) with every pace
source. The test season cannot be replayed. A replay reads nothing of the race after its start lap
except the plan and the events: scrambling every later lap time changes no forecast and no simulated lap.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import SeasonLocked
from src.models.linear import LinearDecomposition
from src.models.state_space import StateSpace
from src.models import evaluate
from src.pipeline import lake
from src.simulator import engine, pace, race as R, replay
from src.simulator.rules import Rules

QATAR = (2024, 23)


@pytest.fixture
def lake_copy(golden_gold_lake, tmp_path):
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    return str(root)


def test_command_line_replays_both_races_out_of_fold(lake_copy, capsys):
    assert replay.main(["--season", "2024", "--lake", lake_copy]) == 0
    out = capsys.readouterr().out
    assert "2 races replayed" in out and "From lap 5" in out and "Mixed races" in out
    rows = lake.read_report(replay.REPORT, lake_copy)
    assert set(rows["source"]) == set(replay.SOURCES)
    qatar = rows[(rows["season"] == 2024) & (rows["round"] == 23)]
    assert set(qatar["start"]) == {"lap 5", "halfway", "SC lap 35", "VSC lap 40", "SC lap 40"}
    assert qatar.loc[qatar["start"] == "lap 5", "start_lap"].iloc[0] == 6        # nobody had a clean lap at 5
    assert (qatar["cars"] == qatar.groupby("start")["cars"].transform("first")).all()
    assert qatar["places_off"].between(0, 15).all() and qatar["no_pace"].eq(0).all()
    british = rows[rows["round"] == 12]
    assert british["mixed"].all() and set(british["start"]) == {"lap 5", "halfway"}


def test_the_test_season_cannot_be_replayed(lake_copy):
    with pytest.raises(SeasonLocked):
        replay.main(["--season", "2025", "--lake", lake_copy])


def test_nothing_after_the_start_is_read_but_the_plan_and_the_events(golden_gold_lake):
    race = R.load(*QATAR, root=golden_gold_lake)
    start = 20
    rungs = [LinearDecomposition().fit(race.laps), StateSpace().fit(race.laps)]
    planned = R.plan(race, start)

    scrambled = R.load(*QATAR, root=golden_gold_lake)
    later = scrambled.laps["lap_number"] > start
    rng = np.random.default_rng(0)
    for column in ("lap_time_sec", "gap_on_track_start_sec", "gap_on_track_end_sec", "gap_ahead_sec",
                   "last_clean_lap_time_sec", "roll3_clean_std_sec", "position"):
        scrambled.laps.loc[later, column] = rng.uniform(50, 150, int(later.sum()))
    assert R.plan(scrambled, start).equals(planned)
    rules, conditions = Rules(), R.lap_conditions(race)
    for rung in rungs:
        before = pace.forecast(rung, R.history(race, start), planned)
        after = pace.forecast(rung, R.history(scrambled, start), planned)
        assert np.allclose(before, after, equal_nan=True)
        a = engine.simulate(R.start_state(race, start), planned, before, conditions, rules, race.total_laps)
        b = engine.simulate(R.start_state(scrambled, start), planned, after, conditions, rules, race.total_laps)
        assert a.laps["clock_sec"].equals(b.laps["clock_sec"])
