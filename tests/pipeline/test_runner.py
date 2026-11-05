"""
runner.py: one race end to end, offline from the golden lake's laps-only Bronze; and what it downloads.

Nothing here touches the network: the two download calls are replaced by fakes that record what they were asked.
A race takes laps from FastF1 and telemetry from OpenF1 (F1's archive needs a login); what Bronze has is never
downloaded again.
"""

import shutil

import pytest

from src.pipeline import lake, runner
from src.pipeline.runner import bronze_step, run_pipeline
from tests.conftest import FIXTURE_LAKE

BRITISH = (2024, 12)


def test_laps_only_run_from_bronze(golden_lake):
    result = run_pipeline(2024, 12, lake_root=golden_lake, skip_extract=True, telemetry=False)
    assert result["gold"]["feature_rows"] == 960
    assert result["silver"]["telemetry_path"] is None


def test_telemetry_cannot_be_built_from_a_laps_only_bronze(golden_lake):
    with pytest.raises(ValueError, match="ingested laps only"):
        run_pipeline(2024, 12, lake_root=golden_lake, skip_extract=True, telemetry=True)


@pytest.fixture
def downloads(monkeypatch):
    """Records the downloads; the FastF1 one really puts a Bronze in place, as a download would."""
    calls = []

    def fake_extract(year, round_number, session, root, telemetry):
        calls.append(("fastf1", telemetry))
        relative = lake.session_dir(lake.BRONZE, year, round_number, "R", FIXTURE_LAKE).relative_to(FIXTURE_LAKE)
        shutil.copytree(FIXTURE_LAKE / relative, lake.session_dir(lake.BRONZE, year, round_number, session, root))
        return {"status": "ok", "fastf1_warnings": ["a warning"]}

    monkeypatch.setattr(runner, "extract_session", fake_extract)
    monkeypatch.setattr(runner.openf1, "add_telemetry", lambda *a, **k: calls.append(("openf1", a[:3])))
    return calls


def test_a_new_race_takes_laps_from_fastf1_and_telemetry_from_openf1(tmp_path, downloads):
    summary = bronze_step(*BRITISH, "R", str(tmp_path), skip_extract=False, telemetry=True)
    assert downloads == [("fastf1", False), ("openf1", (2024, 12, "R"))]
    assert summary["status"] == "ok" and summary["fastf1_warnings"] == ["a warning"]


def test_a_new_race_laps_only_never_asks_openf1(tmp_path, downloads):
    bronze_step(*BRITISH, "R", str(tmp_path), skip_extract=False, telemetry=False)
    assert downloads == [("fastf1", False)]


def test_a_race_in_bronze_is_not_downloaded_again(golden_lake, downloads):
    summary = bronze_step(*BRITISH, "R", golden_lake, skip_extract=False, telemetry=False)
    assert downloads == [] and summary["status"] == "skipped" and summary["lap_rows"] > 0


def test_a_laps_only_bronze_gains_only_the_telemetry(golden_lake, downloads):
    bronze_step(*BRITISH, "R", golden_lake, skip_extract=False, telemetry=True)
    assert downloads == [("openf1", (2024, 12, "R"))]


def test_telemetry_already_there_downloads_nothing(golden_lake, downloads, monkeypatch):
    monkeypatch.setattr(runner.lake, "bronze_has_telemetry", lambda *a, **k: True)
    bronze_step(*BRITISH, "R", golden_lake, skip_extract=False, telemetry=True)
    assert downloads == []


def test_skip_extract_never_uses_the_network(golden_lake, downloads):
    bronze_step(*BRITISH, "R", golden_lake, skip_extract=True, telemetry=False)
    assert downloads == []


def test_a_qualifying_session_asks_fastf1_for_its_telemetry(tmp_path, downloads, monkeypatch):
    monkeypatch.setattr(runner.lake, "bronze_has_telemetry", lambda *a, **k: True)
    bronze_step(2024, 12, "Q", str(tmp_path), skip_extract=False, telemetry=True)
    assert downloads == [("fastf1", True)]


def test_a_session_other_than_a_race_gets_no_openf1_telemetry(golden_lake, downloads, monkeypatch):
    monkeypatch.setattr(runner.lake, "has_bronze", lambda *a, **k: True)
    monkeypatch.setattr(runner.lake, "bronze_has_telemetry", lambda *a, **k: False)
    with pytest.raises(ValueError, match="only added for races"):
        bronze_step(*BRITISH, "Q", golden_lake, skip_extract=False, telemetry=True)
    assert downloads == []
