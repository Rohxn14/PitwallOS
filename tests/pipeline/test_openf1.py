"""
openf1.py: OpenF1's telemetry added to a race already in Bronze, without the network (every answer is faked).

The fake OpenF1 serves the Qatar 2024 fixture (tests/fixtures/openf1/: VER and NOR's car data and location for laps
1-6, the whole lap table, cut from the real download): the other 18 drivers have nothing, as a driver OpenF1 lacks.
"""

from pathlib import Path

import pandas as pd
import pytest
import requests

from src.pipeline import lake, openf1

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"
QATAR = (2024, 23)
SESSIONS = [   # real OpenF1 rows (trimmed): a cancelled race, a Saturday-night race, two ordinary ones
    {"session_key": 9086, "date_start": "2023-05-21T13:00:00+00:00", "location": "Imola", "is_cancelled": True},
    {"session_key": 9157, "date_start": "2023-09-03T13:00:00+00:00", "location": "Monza", "is_cancelled": False},
    {"session_key": 9189, "date_start": "2023-11-19T06:00:00+00:00", "location": "Las Vegas", "is_cancelled": False},
    {"session_key": 9655, "date_start": "2024-12-01T16:00:00+00:00", "location": "Lusail", "is_cancelled": False},
]


class FakeOpenF1:
    """Answers like OpenF1 from the Qatar fixture; records every request; can be told to fail one driver."""

    def __init__(self, broken_driver=None, empty=False):
        self.tables = {name: pd.read_parquet(FIXTURES / f"qatar_{name}.parquet") for name in lake.BRONZE_OPENF1_TABLES}
        self.calls, self.broken_driver, self.empty = [], broken_driver, empty

    def __call__(self, endpoint, query):
        self.calls.append((endpoint, dict(query)))
        if endpoint == "sessions":
            return SESSIONS
        if self.broken_driver is not None and query.get("driver_number") == self.broken_driver:
            raise openf1.OpenF1Error("HTTP 502 for this driver, after every retry")
        if self.empty:
            return []
        table = self.tables[{"car_data": openf1.CAR_TABLE, "location": openf1.LOCATION_TABLE,
                             "laps": openf1.LAPS_TABLE}[endpoint]]
        if "driver_number" in query:
            table = table[table["driver_number"] == query["driver_number"]]
        return table.to_dict("records")


# --- which session ---------------------------------------------------------------------------------

def test_the_session_is_found_by_its_date():
    fetch = lambda endpoint, query: SESSIONS
    assert openf1.find_session(fetch, 2023, "2023-09-03 00:00:00", "Monza")["session_key"] == 9157
    # Las Vegas races on Saturday night: the local date is the day before OpenF1's UTC start
    assert openf1.find_session(fetch, 2023, "2023-11-18 00:00:00", "Las Vegas")["session_key"] == 9189
    with pytest.raises(openf1.OpenF1Error, match="0 OpenF1 sessions"):        # Imola 2023 was cancelled
        openf1.find_session(fetch, 2023, "2023-05-21 00:00:00", "Imola")


def test_two_races_on_the_same_day_are_told_apart_by_the_place():
    twins = SESSIONS + [{"session_key": 1, "date_start": "2024-12-01T12:00:00+00:00", "location": "Elsewhere"}]
    fetch = lambda endpoint, query: twins
    assert openf1.find_session(fetch, 2024, "2024-12-01 00:00:00", "Lusail")["session_key"] == 9655
    with pytest.raises(openf1.OpenF1Error, match="2 OpenF1 sessions"):
        openf1.find_session(fetch, 2024, "2024-12-01 00:00:00", None)


# --- the download, written to Bronze ------------------------------------------------------------------

def test_telemetry_is_added_to_bronze_as_delivered(golden_lake):
    fake = FakeOpenF1()
    out = openf1.add_telemetry(*QATAR, root=golden_lake, fetch=fake, workers=2)
    assert out["session_key"] == 9655
    drivers = openf1.drivers_of(*QATAR, root=golden_lake)
    assert len(drivers) == 20 and len(fake.calls) == 1 + 2 * 20 + 1              # sessions, 2 per driver, laps
    assert all(q["session_key"] == 9655 for e, q in fake.calls if e != "sessions")
    for name in lake.BRONZE_OPENF1_TABLES:                                      # as delivered: same rows, same columns
        written = lake.read_table(lake.table_path(lake.BRONZE, name, *QATAR, "R", golden_lake))
        assert len(written) == len(fake.tables[name]) and list(written.columns) == list(fake.tables[name].columns)
    manifest = lake.read_manifest(*QATAR, root=golden_lake)
    entry = manifest[lake.OPENF1]
    assert entry["session_key"] == 9655 and entry["drivers"] == drivers
    assert entry["missing"]["car_data"] == entry["missing"]["location"] == [d for d in drivers if d not in (1, 4)]
    assert manifest["tables"][openf1.CAR_TABLE]["rows"] == len(fake.tables[openf1.CAR_TABLE])
    assert lake.bronze_telemetry_source(*QATAR, root=golden_lake) == lake.OPENF1
    assert lake.bronze_has_telemetry(*QATAR, root=golden_lake)


def test_a_failed_request_leaves_bronze_as_it_was(golden_lake):
    before = lake.read_manifest(*QATAR, root=golden_lake)
    with pytest.raises(openf1.OpenF1Error):
        openf1.add_telemetry(*QATAR, root=golden_lake, fetch=FakeOpenF1(broken_driver=44), workers=2)
    assert lake.read_manifest(*QATAR, root=golden_lake) == before
    assert not any(lake.table_path(lake.BRONZE, n, *QATAR, "R", golden_lake).exists() for n in lake.BRONZE_OPENF1_TABLES)
    assert lake.bronze_telemetry_source(*QATAR, root=golden_lake) is None


def test_a_session_with_no_telemetry_at_all_is_an_error_not_an_empty_table(golden_lake):
    with pytest.raises(openf1.OpenF1Error, match="no openf1_car_data"):
        openf1.add_telemetry(*QATAR, root=golden_lake, fetch=FakeOpenF1(empty=True), workers=2)
    assert lake.OPENF1 not in lake.read_manifest(*QATAR, root=golden_lake)


def test_fastf1s_own_tables_win_and_leftovers_without_the_manifest_entry_do_not_count(golden_lake):
    for name in lake.BRONZE_OPENF1_TABLES:              # tables written, then a crash before the manifest
        lake.write_table(pd.DataFrame({"x": [1]}), lake.table_path(lake.BRONZE, name, *QATAR, "R", golden_lake))
    assert lake.bronze_telemetry_source(*QATAR, root=golden_lake) is None
    for name in lake.BRONZE_TELEMETRY_TABLES:
        lake.write_table(pd.DataFrame({"x": [1]}), lake.table_path(lake.BRONZE, name, *QATAR, "R", golden_lake))
    assert lake.bronze_telemetry_source(*QATAR, root=golden_lake) == lake.FASTF1


# --- talking to OpenF1: answers, retries, manners ---------------------------------------------------------

class Answer:
    def __init__(self, status, body):
        self.status_code, self.body, self.text = status, body, str(body)

    def json(self):
        return self.body


def scripted(*answers):
    """A fake requests.get that gives these answers in turn (an exception class is raised)."""
    queue = list(answers)

    def get(url, params, timeout):
        answer = queue.pop(0)
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer("connection reset")
        return answer
    return get


def fetch_with(*answers):
    waits = []
    fetch = openf1.http_fetch(pace=openf1.Pace(clock=lambda: 0.0, sleep=lambda s: None), get=scripted(*answers),
                              sleep=waits.append)
    return fetch, waits


def test_rows_come_back_and_no_results_is_no_rows():
    fetch, waits = fetch_with(Answer(200, [{"speed": 300}]), Answer(404, {"detail": "No results found."}))
    assert fetch("car_data", {}) == [{"speed": 300}]
    assert fetch("car_data", {}) == [] and waits == []


def test_too_many_requests_waits_a_minute_and_server_errors_are_retried():
    fetch, waits = fetch_with(Answer(429, {"detail": "Too many requests"}), Answer(503, "busy"), Answer(502, "busy"),
                              Answer(200, []))
    assert fetch("laps", {}) == [] and waits == [openf1.RATE_LIMIT_WAIT_SEC, 10, 30]


def test_a_connection_that_keeps_failing_gives_up_after_the_last_wait():
    fetch, waits = fetch_with(*[requests.ConnectionError] * 5)
    with pytest.raises(openf1.OpenF1Error, match="ConnectionError"):
        fetch("car_data", {})
    assert waits == list(openf1.RETRY_WAITS_SEC)


def test_an_answer_that_is_not_rows_or_a_refusal_is_an_error_at_once():
    fetch, waits = fetch_with(Answer(200, {"detail": "something else"}))
    with pytest.raises(openf1.OpenF1Error, match="expected a list"):
        fetch("car_data", {})
    fetch, waits = fetch_with(Answer(400, {"detail": "bad query"}))
    with pytest.raises(openf1.OpenF1Error, match="HTTP 400"):
        fetch("car_data", {})
    assert waits == []


def test_requests_are_spaced_to_the_free_tier():
    slept = []
    pace = openf1.Pace(per_minute=30, clock=lambda: 0.0, sleep=slept.append)   # everyone asks at the same moment
    for _ in range(3):
        pace.wait()
    assert slept == [2.0, 4.0]                       # 30 a minute = one start every 2 s
