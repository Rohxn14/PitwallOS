"""
The race report and its sanity checks (src/analytics/report.py), Phase 2 step 6.

The report must carry each step's golden fact for the reference races (Race Control's safety cars at
Qatar, its 26 tyre changes, Norris on Piastri at Hungary), and the sanity checks must sort every
number correctly: in range, outside but explained (Qatar 2024's hard tyres got faster: the track
gained rubber), low confidence (safety-car stop losses), not checked (a wet race's wear), not
measured, or outside and unexplained, which makes the command fail.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from src.analytics import report as rep
from tests.conftest import HUNGARY, HUNGARY_FIXTURE_LAKE

BRITISH, QATAR = (2024, 12), (2024, 23)


@pytest.fixture(scope="module")
def reports(golden_silver_laps, hungary_silver_laps):
    return {QATAR: rep.build_report(golden_silver_laps[QATAR], "Qatar Grand Prix"),
            BRITISH: rep.build_report(golden_silver_laps[BRITISH], "British Grand Prix"),
            HUNGARY: rep.build_report(hungary_silver_laps, "Hungarian Grand Prix")}


def numbers(report):
    return rep.headline_numbers(report).set_index("quantity")


def checks(report):
    return rep.sanity_checks(report).set_index(["check", "item"])


def test_qatar_report_carries_each_steps_golden_fact(reports):
    r = reports[QATAR]
    assert (r.season, r.round_number, r.event_name) == (2024, 23, "Qatar Grand Prix")
    n = numbers(r)
    assert n.loc["neutralised periods", "value"] == "SC 1-4, SC 35-39, VSC 40, SC 40-42"     # Race Control
    assert n.loc["tyre changes", "value"] == 26                                               # step 2
    assert {"tyre wear soft, s per lap", "tyre wear medium, s per lap", "tyre wear hard, s per lap"} <= set(n.index)
    assert n.loc["cost of following, s per lap", "value"] == pytest.approx(0.404, abs=0.001)  # step 4
    assert np.isnan(n.loc["undercuts that worked, %", "value"]) and n.loc["undercuts that worked, %", "n"] == 0


def test_the_report_reuses_one_classification_for_every_step(reports, golden_silver_laps):
    r = reports[QATAR]
    assert len(r.pace.laps) == len(golden_silver_laps[QATAR])                                 # nothing dropped
    assert r.dirty_air.laps.index.isin(r.pace.laps.index).all()


def test_qatar_hard_tyres_are_outside_their_range_and_explained(reports):
    c = checks(reports[QATAR])
    hard = c.loc[("tyre wear, s per lap", "HARD")]
    assert hard["value"] < hard["low"] and hard["status"] == rep.EXPLAINED_OUTSIDE
    assert "rubber" in hard["explanation"]
    assert (c["status"] != rep.UNEXPLAINED).all()


def test_a_wet_races_wear_and_dirty_air_are_not_checked(reports):
    c = checks(reports[BRITISH])
    assert c.loc[("cost of following, s per lap", ""), "status"] == rep.NOT_CHECKED
    assert c.loc[("fuel check, s per kg", ""), "status"] == rep.NOT_CHECKED
    assert c.loc[("pace laps, % of laps", ""), "status"] == rep.OK                            # still checked


def test_hungary_undercut_is_in_the_report_and_its_gain_in_range(reports):
    r = reports[HUNGARY]
    n = numbers(r)
    assert n.loc["undercuts that worked, %", "value"] == 100.0 and n.loc["undercuts that worked, %", "n"] == 1
    c = checks(r)
    assert c.loc[("undercut gain, s", "NOR on PIA, laps 45/47"), "status"] == rep.OK
    assert c.loc[("undercut gain, s", "NOR on PIA, laps 45/47"), "value"] == pytest.approx(5.428)


@pytest.mark.parametrize("season, round_number, check, item, value, mixed, status", [
    (2030, 1, "green pit loss, s", "", 22.0, False, rep.OK),
    (2030, 1, "green pit loss, s", "", 35.0, False, rep.OK),                     # the range includes its ends
    (2030, 1, "green pit loss, s", "", 40.0, False, rep.UNEXPLAINED),
    (2024, 23, "tyre wear, s per lap", "HARD", -0.057, False, rep.EXPLAINED_OUTSIDE),
    (2030, 1, "SC/VSC stop loss, s", "SC", -5.0, False, rep.LOW),
    (2030, 1, "tyre wear, s per lap", "SOFT", 9.9, True, rep.NOT_CHECKED),
    (2030, 1, "fuel check, s per kg", "", np.nan, False, rep.NOT_MEASURED),
    (2030, 1, "pace laps, % of laps", "", 10.0, True, rep.UNEXPLAINED),          # mixed exempts only steps 3 and 4
])
def test_judge(season, round_number, check, item, value, mixed, status):
    assert rep.judge(season, round_number, check, item, value, mixed)[0] == status


def test_every_explanation_belongs_to_a_checked_number():
    for (season, round_number, check, item), why in rep.EXPLAINED.items():
        assert check in rep.RANGES and why
    assert set(rep.LOW_CONFIDENCE) <= set(rep.RANGES) and set(rep.NOT_IN_MIXED) <= set(rep.RANGES)


@pytest.fixture
def three_race_lake(golden_lake):
    from src.pipeline.cleaner import build_silver

    shutil.copytree(Path(HUNGARY_FIXTURE_LAKE) / "bronze", Path(golden_lake) / "bronze", dirs_exist_ok=True)
    for race in (BRITISH, QATAR, HUNGARY):
        build_silver(*race, "R", lake_root=golden_lake, telemetry=False)
    return golden_lake


def test_command_line_writes_the_report_and_passes(three_race_lake, capsys):
    from src.pipeline import lake

    assert rep.main(["--season", "2024", "--lake", three_race_lake]) == 0
    sanity = lake.read_report("sanity", three_race_lake)
    assert sorted(sanity["round"].unique()) == [12, 13, 23]
    assert (sanity["status"] == rep.EXPLAINED_OUTSIDE).sum() == 1                  # Qatar's hard tyres
    numbers_ = lake.read_report("race_report", three_race_lake)
    assert set(numbers_["step"]) == {1, 2, 3, 4, 5}
    out = capsys.readouterr().out
    assert "numbers checked in 3 races" in out and "rubber" in out


def test_command_line_prints_one_race(three_race_lake, capsys):
    assert rep.main(["--season", "2024", "--round", "13", "--lake", three_race_lake]) == 0
    out = capsys.readouterr().out
    assert "Hungarian Grand Prix 2024 (round 13)" in out and "NOR on PIA, laps 45/47" in out


def test_command_line_fails_on_an_unexplained_number(three_race_lake, monkeypatch, capsys):
    monkeypatch.setitem(rep.RANGES, "green pit loss, s", (0.0, 1.0))               # every race now fails it
    assert rep.main(["--season", "2024", "--lake", three_race_lake]) == 1
    assert rep.UNEXPLAINED in capsys.readouterr().out
