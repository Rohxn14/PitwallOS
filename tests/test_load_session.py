"""
f1_data.load_session: where a session to replay comes from, and what it says when it cannot be had.

A session from 2023 on comes from the lake (tests/lib/test_lake_session.py, test_lake_qualifying.py). What the lake
lacks is downloaded from OpenF1 first. When that fails, or the session is of a kind or a season the lake cannot hold,
the replay stops at once with a message that says why (it used to crash in its worker processes with KeyError '1').

Nothing here touches the network or the real lake: the lake is an empty temporary folder and the download is faked.
"""

import pytest

from src import f1_data
from src.pipeline import openf1_bronze, runner


@pytest.fixture
def downloads(monkeypatch):
    """Every attempt to fetch a session from OpenF1; each one fails, as without a network."""
    asked = []

    def no_network(year, round_number, session_type, lake_root, telemetry):
        asked.append((year, round_number, session_type, "pipeline"))
        raise ConnectionError("no network in tests")

    def no_download(year, round_number, session_type, lake_root):
        asked.append((year, round_number, session_type, "bronze"))
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(runner, "run_pipeline", no_network)
    monkeypatch.setattr(openf1_bronze, "ingest_session", no_download)
    return asked


def test_a_race_the_lake_lacks_is_built_from_openf1_and_a_failure_says_why(tmp_path, downloads):
    with pytest.raises(f1_data.TelemetryNotAvailable, match="2026 round 15 'R' is not in the lake and could not be "
                                                            "downloaded from OpenF1 .ConnectionError: no network"):
        f1_data.load_session(2026, 15, "R", lake_root=str(tmp_path))
    assert downloads == [(2026, 15, "R", "pipeline")]               # Bronze, Silver and Gold: the replay reads Silver


def test_a_sprint_goes_the_same_way_as_a_race(tmp_path, downloads):
    with pytest.raises(f1_data.TelemetryNotAvailable):
        f1_data.load_session(2024, 23, "S", lake_root=str(tmp_path))
    assert downloads == [(2024, 23, "S", "pipeline")]


@pytest.mark.parametrize("session_type", ["Q", "SQ"])
def test_a_qualifying_session_the_lake_lacks_is_downloaded_into_bronze_only(tmp_path, downloads, session_type):
    with pytest.raises(f1_data.TelemetryNotAvailable, match="could not be downloaded"):
        f1_data.load_session(2026, 15, session_type, lake_root=str(tmp_path))
    assert downloads == [(2026, 15, session_type, "bronze")]        # qualifying gets no Silver


def test_a_season_before_openf1_starts_says_so_without_asking(tmp_path, downloads):
    with pytest.raises(f1_data.TelemetryNotAvailable, match="OpenF1, which starts in 2023"):
        f1_data.load_session(2022, 1, "R", lake_root=str(tmp_path))
    assert downloads == []


def test_a_kind_of_session_that_cannot_be_replayed_says_so(tmp_path, downloads):
    with pytest.raises(f1_data.TelemetryNotAvailable, match="only races .R., sprints .S., qualifying"):
        f1_data.load_session(2026, 15, "FP1", lake_root=str(tmp_path))
    assert downloads == []
