"""
backfill.py: many races into the lake, without the network.

The golden lake holds real Bronze for the 2024 British GP (round 12) and Qatar GP (round 23),
so a backfill over those rounds rebuilds them offline. Any other round has no Bronze: there the
download is faked, to fail, so retries and failure isolation can be tested.
"""

import pandas as pd
import pytest

from src.pipeline import backfill, contracts, extractor, lake
from src.pipeline.backfill import FAILED, OK, RetryPolicy

BRITISH, QATAR = 12, 23
QUICK = RetryPolicy(waits_sec=(1.0, 2.0), rate_limit_wait_sec=50.0, rate_limit_max_waits=2)


@pytest.fixture
def downloads(monkeypatch):
    """Every download attempt (by round). Each one fails: tests never touch the network."""
    calls = []

    def fake_extract(year, round_number, session, root, telemetry=True):
        calls.append(round_number)
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(extractor, "extract_session", fake_extract)
    return calls


def run(lake_root, rounds, **kwargs):
    """Backfill 2024 over `rounds`. Returns (report indexed by round, the waits it asked for)."""
    waits = []
    report = backfill.backfill(2024, rounds, root=lake_root, policy=QUICK, sleep=waits.append, **kwargs)
    return report.set_index("round"), waits


def saved_report(lake_root):
    return lake.read_report("backfill_2024", lake_root)


# --- golden: the report's numbers are the known facts of both races ---------------

def test_season_report_has_the_known_facts_of_both_races(golden_lake, downloads):
    report, _ = run(golden_lake, [BRITISH, QATAR])

    assert downloads == []                                   # Bronze was there: nothing downloaded
    assert (report["status"] == OK).all() and (report["source"] == backfill.FROM_BRONZE).all()

    qatar = report.loc[QATAR]
    assert qatar["event_name"] == "Qatar Grand Prix"
    assert qatar["sc_lap_numbers"] == "1-4, 34-42" and qatar["vsc_lap_numbers"] == "40"
    assert qatar["tyre_changes"] == 26 and qatar["pass_throughs"] == 34

    british = report.loc[BRITISH]
    assert british["sc_lap_numbers"] == "" and british["vsc_lap_numbers"] == ""
    assert british["tyre_changes"] == 45 and british["clean_laps"] == 850
    assert british["laps"] == british["gold_rows"] == british["bronze_lap_rows"] == 960
    assert british["dropped_rows"] == 0 and british["total_laps"] == british["max_lap"] == 52
    assert not british["bronze_telemetry"]                   # the fixture is laps only

    assert saved_report(golden_lake)["round"].tolist() == [BRITISH, QATAR]


# --- resumable ---------------------------------------------------------------------

def test_a_second_run_does_no_work_but_still_reports(golden_lake, downloads):
    run(golden_lake, [BRITISH])
    gold = lake.gold_features_path(2024, BRITISH, root=golden_lake)
    written = gold.stat().st_mtime_ns

    report, _ = run(golden_lake, [BRITISH])
    assert report.loc[BRITISH, "source"] == backfill.ALREADY_BUILT and report.loc[BRITISH, "status"] == OK
    assert gold.stat().st_mtime_ns == written                # Gold untouched
    assert report.loc[BRITISH, "tyre_changes"] == 45         # counts read back from the lake


def test_rebuild_redoes_silver_and_gold_from_bronze_offline(golden_lake, downloads):
    run(golden_lake, [BRITISH])
    report, _ = run(golden_lake, [BRITISH], rebuild=True)
    assert report.loc[BRITISH, "source"] == backfill.FROM_BRONZE
    assert downloads == []


def test_wanting_telemetry_from_a_laps_only_bronze_means_downloading_again(golden_lake, downloads):
    report, _ = run(golden_lake, [BRITISH], telemetry=True)
    assert downloads == [BRITISH] * 3                        # tried, not silently built without telemetry
    assert report.loc[BRITISH, "status"] == FAILED and report.loc[BRITISH, "stage"] == "bronze"


# --- isolated and retried -----------------------------------------------------------

def test_a_failing_race_does_not_stop_the_others(golden_lake, downloads):
    report, waits = run(golden_lake, [1, BRITISH])           # round 1 has no Bronze: its download fails

    assert report.loc[1, "status"] == FAILED and report.loc[1, "stage"] == "bronze"
    assert report.loc[1, "error"] == "ConnectionError: no network in tests"
    assert report.loc[BRITISH, "status"] == OK
    assert downloads == [1, 1, 1] and waits == [1.0, 2.0]    # first try + one retry per wait


def test_fastf1_rate_limit_gets_the_long_wait(golden_lake, monkeypatch):
    calls = []

    def fake_extract(year, round_number, session, root, telemetry=True):
        calls.append(round_number)
        if len(calls) <= 2:
            raise extractor.ExtractionError("FastF1 could not load part of 2024 round 1 'R': Failed to load "
                                            "weather data! (RateLimitExceededError: any API: 500 calls/h)")
        raise ConnectionError("down")

    monkeypatch.setattr(extractor, "extract_session", fake_extract)
    report, waits = run(golden_lake, [1])
    assert waits == [50.0, 50.0, 1.0, 2.0]                   # two long waits, then the ordinary ones
    assert report.loc[1, "error"] == "ConnectionError: down"


def test_a_contract_failure_is_reported_at_its_stage_and_never_retried(golden_lake, downloads, monkeypatch):
    def broken_silver(*args, **kwargs):
        raise contracts.ContractError("Silver laps broke its data contract:\n column  check  failures")

    monkeypatch.setattr(backfill, "build_silver", broken_silver)
    report, waits = run(golden_lake, [BRITISH])

    row = report.loc[BRITISH]
    assert row["status"] == FAILED and row["stage"] == "silver" and waits == []
    assert row["event_name"] == "British Grand Prix"         # named from the Bronze manifest
    assert row["error"] == "ContractError: Silver laps broke its data contract: | column  check  failures"


# --- reported -----------------------------------------------------------------------

def test_the_season_report_keeps_rows_from_earlier_runs(golden_lake, downloads):
    run(golden_lake, [QATAR])
    run(golden_lake, [BRITISH])
    run(golden_lake, [QATAR])                                # rerunning a round replaces its row
    saved = saved_report(golden_lake)
    assert saved["round"].tolist() == [BRITISH, QATAR]
    assert saved["status"].tolist() == [OK, OK]


def test_command_line_exit_code_says_whether_every_race_succeeded(golden_lake, downloads, monkeypatch):
    args = ["--season", "2024", "--rounds", str(BRITISH), str(QATAR), "--lake", golden_lake]
    assert backfill.main(args) == 0

    # A failure that is never retried (so the real retry waits never start): one race fails at Gold.
    real_gold = backfill.build_gold
    def gold_fails_for_qatar(year, round_number, *rest):
        if round_number == QATAR:
            raise contracts.ContractError("Gold lap features broke its data contract")
        return real_gold(year, round_number, *rest)

    monkeypatch.setattr(backfill, "build_gold", gold_fails_for_qatar)
    assert backfill.main(args + ["--rebuild"]) == 1
    assert saved_report(golden_lake).set_index("round")["status"].to_dict() == {BRITISH: OK, QATAR: FAILED}


# --- which races ----------------------------------------------------------------------

def test_only_races_that_have_been_run_are_built(monkeypatch):
    bahrain = pd.Timestamp("2024-03-02 15:00", tz="UTC")
    monkeypatch.setattr(extractor, "race_rounds", lambda season: [
        {"round": 1, "event_name": "Bahrain Grand Prix", "race_start_utc": bahrain},
        {"round": 2, "event_name": "Saudi Arabian Grand Prix", "race_start_utc": bahrain + pd.Timedelta(days=7)},
        {"round": 3, "event_name": "No race session", "race_start_utc": None},
    ])

    run_now, not_yet = backfill.races_to_run(2024, now=bahrain + pd.Timedelta(days=1))
    assert [r["round"] for r in run_now] == [1] and [r["round"] for r in not_yet] == [2, 3]

    # The flag fell an hour ago: the timing data may not be published yet.
    run_now, _ = backfill.races_to_run(2024, now=bahrain + pd.Timedelta(hours=3))
    assert run_now == []


def test_explicit_rounds_skip_the_schedule(monkeypatch):
    def no_schedule(season):
        raise AssertionError("the schedule must not be read")

    monkeypatch.setattr(extractor, "race_rounds", no_schedule)
    run_now, not_yet = backfill.races_to_run(2024, rounds=[5, 6])
    assert [r["round"] for r in run_now] == [5, 6] and not_yet == []
