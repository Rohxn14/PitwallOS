"""
openf1.py: the client. How answers, refusals and failures are handled, and how requests are spaced. No network: the
`get` it is given is a fake. (What is asked of OpenF1, and where the answers go: test_openf1_bronze.py.)
"""

import pytest
import requests

from src.pipeline import openf1


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
