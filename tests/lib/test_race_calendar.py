"""
race_calendar.py: the season's weekends from OpenF1 and the circuits from MultiViewer, kept in the lake.

The calendar is OpenF1's real list of 2023's race sessions and meetings (tests/fixtures/openf1/calendar_2023_*), plus
the other sessions of two weekends, written here with their real names and times: Monaco (an ordinary weekend) and
Qatar (a sprint weekend). No network: the fetch is a fake that counts its calls, the lake a temporary folder.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.lib import race_calendar
from src.pipeline import lake

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"
RACES = json.loads((FIXTURES / "calendar_2023_sessions.json").read_text(encoding="utf-8"))
MEETINGS = json.loads((FIXTURES / "calendar_2023_meetings.json").read_text(encoding="utf-8"))
MONACO, QATAR = 1210, 1221        # OpenF1's meeting keys


def other(meeting, name, start):
    return {"meeting_key": meeting, "session_name": name, "date_start": start, "is_cancelled": False}


SESSIONS = RACES + [
    other(MONACO, "Practice 1", "2023-05-26T11:30:00+00:00"), other(MONACO, "Qualifying", "2023-05-27T14:00:00+00:00"),
    other(QATAR, "Practice 1", "2023-10-06T13:30:00+00:00"), other(QATAR, "Qualifying", "2023-10-06T17:00:00+00:00"),
    other(QATAR, "Sprint Shootout", "2023-10-07T13:00:00+00:00"), other(QATAR, "Sprint", "2023-10-07T17:30:00+00:00"),
]


class Fetch:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, endpoint, query):
        self.calls.append(endpoint)
        if self.fail:
            raise ConnectionError("no network")
        return SESSIONS if endpoint == "sessions" else MEETINGS


def test_the_weekends_of_2023_in_the_pickers_shape():
    weekends = race_calendar.build_weekends(SESSIONS, MEETINGS, 2023)
    assert [w["round_number"] for w in weekends] == list(range(1, 23))            # Imola was cancelled: 22, not 23
    monaco, qatar, vegas = weekends[5], weekends[16], weekends[20]
    assert (monaco["event_name"], monaco["date"], monaco["country"], monaco["type"]) == (
        "Monaco Grand Prix", "2023-05-28", "Monaco", race_calendar.CONVENTIONAL)
    assert list(monaco["session_dates"]) == ["Practice 1", "Qualifying", "Race"]  # in the order they are run
    assert qatar["type"] == race_calendar.SPRINT and "sprint" in qatar["type"]    # what the picker looks for
    assert list(qatar["session_dates"]) == ["Practice 1", "Sprint Shootout", "Sprint", "Qualifying", "Race"]
    assert vegas["event_name"] == "Las Vegas Grand Prix" and vegas["date"] == "2023-11-18"   # Saturday night there
    assert monaco["meeting_key"] == MONACO and monaco["circuit_key"] is not None


def test_a_session_counts_once_its_scheduled_start_has_passed():
    qatar = race_calendar.build_weekends(SESSIONS, MEETINGS, 2023)[16]
    saturday = datetime(2023, 10, 7, 15, 0, tzinfo=timezone.utc)
    assert race_calendar.has_started(qatar, "Sprint Shootout", saturday)
    assert not race_calendar.has_started(qatar, "Sprint", saturday) and not race_calendar.has_started(qatar, "Race", saturday)
    assert not race_calendar.has_started(qatar, "Practice 3", saturday)           # a sprint weekend has none


def test_a_finished_season_is_fetched_once_and_then_read_from_the_lake(tmp_path):
    fetch = Fetch()
    later = pd.Timestamp("2026-11-14T00:00:00+00:00")
    first = race_calendar.weekends(2023, tmp_path, fetch, now=later)
    again = race_calendar.weekends(2023, tmp_path, fetch, now=later + pd.Timedelta(days=400))
    assert first == again and len(first) == 22 and fetch.calls == ["sessions", "meetings"]
    kept = lake.read_json(lake.reference_path("calendar_2023", tmp_path))
    assert kept["year"] == 2023 and len(kept["weekends"]) == 22


def test_the_running_season_is_fetched_again_after_half_a_day_and_survives_without_a_network(tmp_path):
    during = pd.Timestamp("2023-06-01T08:00:00+00:00")                            # 2023 is "this year"
    fetch = Fetch()
    race_calendar.weekends(2023, tmp_path, fetch, now=during)
    race_calendar.weekends(2023, tmp_path, fetch, now=during + pd.Timedelta(hours=11))
    assert len(fetch.calls) == 2                                                  # fresh enough: the lake's copy
    race_calendar.weekends(2023, tmp_path, fetch, now=during + pd.Timedelta(hours=13))
    assert len(fetch.calls) == 4                                                  # a race may have been cancelled since
    offline = race_calendar.weekends(2023, tmp_path, Fetch(fail=True), now=during + pd.Timedelta(days=3))
    assert len(offline) == 22                                                     # an old calendar beats none


def test_without_a_network_or_a_copy_it_says_so_and_before_2023_there_is_nothing(tmp_path):
    with pytest.raises(race_calendar.CalendarError, match="2023 calendar could not be fetched.*no copy"):
        race_calendar.weekends(2023, tmp_path, Fetch(fail=True))
    fetch = Fetch()
    assert race_calendar.weekends(2022, tmp_path, fetch) == [] and fetch.calls == []     # OpenF1 starts in 2023
    assert race_calendar.seasons(datetime(2026, 11, 14).date()) == [2023, 2024, 2025, 2026]


# --- circuits ----------------------------------------------------------------------------------------------------

class Answer:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def test_a_circuit_is_asked_of_multiviewer_by_name_once_and_kept(tmp_path):
    asked = []

    def get(url, headers, timeout):
        asked.append((url, headers["User-Agent"]))
        return Answer(200, {"rotation": 61, "corners": [{"number": 1}], "circuitName": "Lusail"})

    assert race_calendar.rotation(2024, 150, tmp_path, get) == 61.0
    assert race_calendar.circuit(2024, 150, tmp_path, get)["circuitName"] == "Lusail"
    assert asked == [("https://api.multiviewer.app/api/v1/circuits/150/2024", race_calendar.USER_AGENT)]   # once
    assert "PitwallOS" in race_calendar.USER_AGENT                # it refuses a client that does not say who it is


def test_a_circuit_multiviewer_refuses_or_cannot_be_reached_is_an_error_and_nothing_is_kept(tmp_path):
    with pytest.raises(race_calendar.CalendarError, match="HTTP 403"):
        race_calendar.circuit(2024, 150, tmp_path, lambda *a, **k: Answer(403))

    def down(*args, **kwargs):
        raise requests.ConnectionError("down")

    with pytest.raises(race_calendar.CalendarError, match="could not be reached"):
        race_calendar.circuit(2024, 150, tmp_path, down)
    assert not lake.reference_path("circuit_150_2024", tmp_path).exists()
