"""
runner.py: one race end to end, and what it downloads.

Nothing here touches the network: the download (openf1_bronze.ingest_session) is replaced by a fake that records what
it was asked and puts a real Bronze from OpenF1 in place (Qatar 2024's small tables). A Bronze that is already there is
never downloaded again; the FastF1-shaped fixture Bronze (the golden lake) is used as it is.
"""

import shutil

import pytest

from src.pipeline import lake, runner
from src.pipeline.runner import bronze_step, run_pipeline
from tests.pipeline.test_openf1_translate import QATAR_FIXTURE as QATAR_OPENF1

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
    """Records the downloads; each one really puts a Bronze from OpenF1 in place, as a download would."""
    calls = []

    def fake_ingest(year, round_number, session, root, telemetry=True):
        calls.append(("openf1 bronze", telemetry))
        bronze = lake.session_dir(lake.BRONZE, year, round_number, session, root)
        if not bronze.exists():
            shutil.copytree(QATAR_OPENF1, bronze)              # a real Bronze from OpenF1 (small tables only)
        return {"status": "ingested"}

    monkeypatch.setattr(runner.openf1_bronze, "ingest_session", fake_ingest)
    return calls


def test_a_new_race_comes_from_openf1_alone(tmp_path, downloads):
    summary = bronze_step(2024, 23, "R", str(tmp_path), skip_extract=False, telemetry=False)
    assert downloads == [("openf1 bronze", False)]
    assert summary["status"] == "ingested" and summary["lap_rows"] == 944 and summary["event_name"] == "Qatar Grand Prix"


def test_an_openf1_bronze_is_asked_again_and_fetches_only_what_it_lacks(tmp_path, downloads):
    bronze_step(2024, 23, "R", str(tmp_path), skip_extract=False, telemetry=False)
    bronze_step(2024, 23, "R", str(tmp_path), skip_extract=False, telemetry=True)
    assert downloads == [("openf1 bronze", False), ("openf1 bronze", True)]   # ingest_session decides what is missing


def test_a_new_race_end_to_end_from_openf1(tmp_path, downloads):
    result = run_pipeline(2024, 23, lake_root=str(tmp_path), telemetry=False)
    assert result["gold"]["feature_rows"] > 900 and result["silver"]["lap_report"]["clock_method"] == "session started"


def test_a_fixture_bronze_that_is_not_from_openf1_is_used_as_it_is(golden_lake, downloads):
    summary = bronze_step(*BRITISH, "R", golden_lake, skip_extract=False, telemetry=False)
    assert downloads == [] and summary["status"] == "skipped" and summary["lap_rows"] > 0


def test_skip_extract_never_uses_the_network(golden_lake, downloads):
    bronze_step(*BRITISH, "R", golden_lake, skip_extract=True, telemetry=False)
    assert downloads == []


def test_qualifying_does_not_go_through_the_pipeline(tmp_path, downloads):
    with pytest.raises(ValueError, match="only races and sprints"):
        bronze_step(2024, 12, "Q", str(tmp_path), skip_extract=False, telemetry=True)
    assert downloads == []
