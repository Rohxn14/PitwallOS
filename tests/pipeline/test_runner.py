"""
runner.py: one race end to end, offline from the golden lake's laps-only Bronze; and what it downloads.

Nothing here touches the network: the two download calls are replaced by fakes that record what they were asked.
A race comes from OpenF1 alone (F1's archive needs a login); a race already in a FastF1 Bronze keeps it and gains
only OpenF1's telemetry; what Bronze has is never downloaded again.
"""

import shutil

import pytest

from src.pipeline import lake, runner
from src.pipeline.runner import bronze_step, run_pipeline
from tests.conftest import FIXTURE_LAKE
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
    """Records the downloads; the FastF1 one really puts a Bronze in place, as a download would."""
    calls = []

    def fake_extract(year, round_number, session, root, telemetry):
        calls.append(("fastf1", telemetry))
        relative = lake.session_dir(lake.BRONZE, year, round_number, "R", FIXTURE_LAKE).relative_to(FIXTURE_LAKE)
        shutil.copytree(FIXTURE_LAKE / relative, lake.session_dir(lake.BRONZE, year, round_number, session, root))
        return {"status": "ok", "fastf1_warnings": ["a warning"]}

    def fake_ingest(year, round_number, session, root, telemetry=True):
        calls.append(("openf1 bronze", telemetry))
        bronze = lake.session_dir(lake.BRONZE, year, round_number, session, root)
        if not bronze.exists():
            shutil.copytree(QATAR_OPENF1, bronze)              # a real Bronze from OpenF1 (small tables only)
        return {"status": "ingested"}

    monkeypatch.setattr(runner, "extract_session", fake_extract)
    monkeypatch.setattr(runner.openf1, "add_telemetry", lambda *a, **k: calls.append(("openf1", a[:3])))
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
