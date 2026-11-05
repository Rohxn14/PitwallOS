"""
f1_data.load_session: a session without car data stops at once with a message that says why.

FastF1 only logs a failed download (401 from F1's archive since October 2026) and returns anyway, as seen on 2025
round 12: laps 825, car data 0 drivers. The replay then crashed in its worker processes with KeyError '1'.
"""

import pytest

from src import f1_data


class FakeSession:
    def __init__(self, car_data):
        self.car_data = car_data
        self.loaded_with = None

    def load(self, **options):
        self.loaded_with = options


def _with_session(monkeypatch, session):
    monkeypatch.setattr(f1_data.fastf1, "get_session", lambda *args: session)


def test_a_session_with_car_data_loads(monkeypatch):
    session = FakeSession({"1": object()})
    _with_session(monkeypatch, session)
    assert f1_data.load_session(2026, 15, "R") is session
    assert session.loaded_with == {"telemetry": True, "weather": True}


def test_a_session_without_car_data_says_why(monkeypatch):
    _with_session(monkeypatch, FakeSession({}))
    with pytest.raises(f1_data.TelemetryNotAvailable, match="2025 round 12 'R'.*needed a login"):
        f1_data.load_session(2025, 12, "R")
