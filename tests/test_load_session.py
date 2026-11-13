"""
f1_data.load_session: where a session to replay comes from, and what it says when it cannot be had.

A race from 2023 on comes from the lake (tests/lib/test_lake_session.py). When the lake does not have it and it cannot
be downloaded from OpenF1, and for every other kind of session, FastF1 is asked. FastF1 only logs a failed download
(401 from F1's archive since October 2026) and returns anyway, as seen on 2025 round 12: laps 825, car data 0
drivers. The replay then crashed in its worker processes with KeyError '1': a session without car data stops at once
with a message that says why.

Nothing here touches the network or the real lake: the lake is an empty temporary folder and the download is faked.
"""

import pytest

from src import f1_data
from src.pipeline import runner


class FakeSession:
    def __init__(self, car_data):
        self.car_data = car_data
        self.loaded_with = None

    def load(self, **options):
        self.loaded_with = options


@pytest.fixture
def downloads(monkeypatch):
    """Every attempt to build a race from OpenF1; each one fails, as without a network."""
    asked = []

    def no_network(year, round_number, session_type, lake_root, telemetry):
        asked.append((year, round_number, session_type, telemetry))
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(runner, "run_pipeline", no_network)
    return asked


def _with_session(monkeypatch, session):
    monkeypatch.setattr(f1_data.fastf1, "get_session", lambda *args: session)


def test_a_race_the_lake_lacks_is_asked_of_openf1_then_of_fastf1(monkeypatch, tmp_path, downloads):
    session = FakeSession({"1": object()})
    _with_session(monkeypatch, session)
    assert f1_data.load_session(2026, 15, "R", lake_root=str(tmp_path)) is session
    assert downloads == [(2026, 15, "R", True)]                     # OpenF1 first, with telemetry
    assert session.loaded_with == {"telemetry": True, "weather": True}


def test_a_qualifying_session_goes_straight_to_fastf1(monkeypatch, tmp_path, downloads):
    session = FakeSession({"1": object()})
    _with_session(monkeypatch, session)
    assert f1_data.load_session(2026, 15, "Q", lake_root=str(tmp_path)) is session
    assert downloads == []                                          # the lake has races only so far


def test_a_race_before_2023_goes_straight_to_fastf1(monkeypatch, tmp_path, downloads):
    session = FakeSession({"1": object()})
    _with_session(monkeypatch, session)
    assert f1_data.load_session(2022, 1, "R", lake_root=str(tmp_path)) is session and downloads == []   # OpenF1 starts in 2023


def test_a_session_without_car_data_says_why(monkeypatch, tmp_path, downloads):
    _with_session(monkeypatch, FakeSession({}))
    with pytest.raises(f1_data.TelemetryNotAvailable, match="2025 round 12 'R'.*needed a login"):
        f1_data.load_session(2025, 12, "R", lake_root=str(tmp_path))
