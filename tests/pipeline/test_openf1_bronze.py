"""
openf1_bronze.py: one race into Bronze from OpenF1 alone, as delivered, without the network (every answer is faked).

The calendar is REAL: OpenF1's own lists of 2023's race sessions and meetings (tests/fixtures/openf1/calendar_2023_*),
the season with the traps: Imola cancelled, Las Vegas on a Saturday night, session keys that are not in date order.
The race itself is a tiny made-up one (two cars, three laps) served under Monza's real keys: Bronze copies, it does
not compute, so what matters is that every value comes back unchanged.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import lake, openf1, openf1_bronze

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openf1"
SESSIONS = json.loads((FIXTURES / "calendar_2023_sessions.json").read_text(encoding="utf-8"))
MEETINGS = json.loads((FIXTURES / "calendar_2023_meetings.json").read_text(encoding="utf-8"))
MONZA = (2023, 14)
KEY, MEETING = 9157, 1218            # Monza 2023 in OpenF1
CARS = (1, 55)


def samples(driver, **values):
    return [{"date": f"2023-09-03T13:03:0{i}.000000+00:00", "session_key": KEY, "meeting_key": MEETING,
             "driver_number": driver, **{k: v + i for k, v in values.items()}} for i in range(3)]


def race_answers():
    """What OpenF1 would answer for the made-up race, per endpoint."""
    where = {"session_key": KEY, "meeting_key": MEETING}
    return {
        "sessions": [s for s in SESSIONS if s["session_key"] == KEY],
        "meetings": [m for m in MEETINGS if m["meeting_key"] == MEETING],
        "drivers": [{**where, "driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"},
                    {**where, "driver_number": 55, "name_acronym": "SAI", "team_name": "Ferrari"}],
        "laps": [{**where, "driver_number": d, "lap_number": n, "date_start": f"2023-09-03T13:0{2 + n}:00+00:00",
                  "lap_duration": None if n == 1 else 81.5 + d / 100, "segments_sector_1": [2049, 2051, None]}
                 for d in CARS for n in (1, 2, 3)],
        "stints": [{**where, "driver_number": d, "stint_number": 1, "lap_start": 1, "lap_end": 3,
                    "compound": "MEDIUM", "tyre_age_at_start": 0} for d in CARS],
        "race_control": [{**where, "date": "2023-09-03T13:03:00+00:00", "driver_number": None, "lap_number": 1,
                          "category": "Flag", "flag": "GREEN", "scope": "Track", "sector": None,
                          "qualifying_phase": None, "message": "GREEN LIGHT - PIT EXIT OPEN"}],
        "weather": [{**where, "date": "2023-09-03T13:00:10+00:00", "air_temperature": 28.4, "rainfall": 0}],
        "position": [{**where, "date": "2023-09-03T13:03:00+00:00", "driver_number": d, "position": p}
                     for p, d in enumerate((55, 1), start=1)],
        "session_result": [{**where, "driver_number": 1, "position": 1, "gap_to_leader": 0, "duration": 4413.9},
                           {**where, "driver_number": 55, "position": 2, "gap_to_leader": 6.064, "duration": 4420.0},
                           {**where, "driver_number": 2, "position": 3, "gap_to_leader": "+1 LAP", "duration": None}],
        "pit": [{**where, "date": "2023-09-03T13:05:10+00:00", "driver_number": 1, "lap_number": 2,
                 "lane_duration": 24.1, "stop_duration": 2.4}],
        "intervals": [{**where, "date": "2023-09-03T13:04:00+00:00", "driver_number": 55, "gap_to_leader": 1.2,
                       "interval": 1.2}],
        "overtakes": [],                                             # a race nobody was passed in: truly empty
        "car_data": [row for d in CARS for row in samples(d, speed=280, rpm=11000)],
        "location": [row for d in CARS for row in samples(d, x=100, y=-200)],
    }


class FakeOpenF1:
    """Answers like OpenF1: the real 2023 calendar and the made-up race. Records every request."""

    def __init__(self, **changed):
        self.answers = {**race_answers(), **changed}
        self.calls = []

    def __call__(self, endpoint, query):
        self.calls.append((endpoint, dict(query)))
        if "year" in query:                                   # the calendar's two questions
            assert query["year"] == 2023
            return SESSIONS if endpoint == "sessions" else MEETINGS
        rows = self.answers[endpoint]
        if "driver_number" in query:
            rows = [r for r in rows if r["driver_number"] == query["driver_number"]]
        return rows

    def asked(self, endpoint):
        return [q for e, q in self.calls if e == endpoint and "year" not in q]


def bronze(root, endpoint):
    return lake.read_table(lake.table_path(lake.BRONZE, lake.openf1_table(endpoint), *MONZA, "R", root))


# --- which race -------------------------------------------------------------------------------------------

def test_the_rounds_of_2023_come_out_as_the_championship_numbered_them():
    calendar = openf1_bronze.season_calendar(FakeOpenF1(), 2023)
    assert list(calendar["round"]) == list(range(1, 23))                 # 23 race sessions, Imola cancelled: 22
    by_round = calendar.set_index("round")
    assert "Imola" not in set(calendar["location"])
    assert by_round.loc[6, "location"] == "Monaco"                       # Imola would have been round 6
    assert by_round.loc[14, ["session_key", "event_name", "location"]].tolist() == [9157, "Italian Grand Prix", "Monza"]
    # OpenF1's keys are not in date order (Qatar 9221 was run before Austin 9213): the date decides, not the key
    assert by_round.loc[17, "session_key"] == 9221 and by_round.loc[18, "session_key"] == 9213
    assert by_round.loc[22, "country"] == "United Arab Emirates"


def test_a_race_is_dated_where_it_is_run_and_counts_as_run_six_hours_after_its_end():
    vegas = openf1_bronze.race_of(openf1_bronze.season_calendar(FakeOpenF1(), 2023), 2023, 21)
    assert vegas["start_utc"] == pd.Timestamp("2023-11-19T06:00:00+00:00")
    assert openf1_bronze.local_date(vegas["start_utc"], vegas["gmt_offset"]) == "2023-11-18"   # Saturday night there
    assert not openf1_bronze.has_run(vegas, now=vegas["end_utc"] + pd.Timedelta(hours=5))
    assert openf1_bronze.has_run(vegas, now=vegas["end_utc"] + pd.Timedelta(hours=6))


def test_a_round_the_season_does_not_have_is_an_error():
    with pytest.raises(openf1.OpenF1Error, match="no round 23"):
        openf1_bronze.ingest_session(2023, 23, root="unused", fetch=FakeOpenF1())


# --- as delivered ---------------------------------------------------------------------------------------------

def test_every_table_reaches_bronze_as_delivered_and_the_manifest_describes_it(tmp_path):
    fake = FakeOpenF1()
    out = openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=fake, workers=2)
    assert out["status"] == "ingested" and out["session_key"] == KEY and out["empty"] == ["overtakes"]

    for endpoint, rows in fake.answers.items():
        if endpoint == "overtakes":                                   # no rows: no table, never an empty one
            assert not lake.table_path(lake.BRONZE, lake.openf1_table(endpoint), *MONZA, "R", tmp_path).exists()
            continue
        written = bronze(tmp_path, endpoint)
        assert len(written) == len(rows) and list(written.columns) == list(rows[0])
    laps = bronze(tmp_path, "laps")
    assert laps["date_start"].tolist()[:2] == ["2023-09-03T13:03:00+00:00", "2023-09-03T13:04:00+00:00"]   # still text
    assert pd.isna(laps["lap_duration"].iloc[0]) and laps["lap_duration"].iloc[1] == 81.51          # unknown stays unknown
    assert [None if pd.isna(v) else int(v) for v in laps["segments_sector_1"].iloc[0]] == [2049, 2051, None]
    assert bronze(tmp_path, "car_data")["speed"].tolist() == [280, 281, 282] * 2

    # every question was asked about this session, the meeting by its own key
    assert fake.asked("meetings") == [{"meeting_key": MEETING}]
    assert all(q.get("session_key") == KEY for e, q in fake.calls if "year" not in q and e != "meetings")
    assert len(fake.calls) == 2 + 12 + 2 * len(CARS)                 # the calendar, the small tables, two per driver

    manifest = lake.read_manifest(*MONZA, root=tmp_path)
    assert manifest["source"] == lake.OPENF1 and manifest["telemetry"] is True and manifest["driver_count"] == 2
    assert (manifest["event_name"], manifest["location"], manifest["country"], manifest["circuit"],
            manifest["event_date"]) == ("Italian Grand Prix", "Monza", "Italy", "Monza", "2023-09-03")
    entry = manifest[lake.OPENF1]
    assert entry["session_key"] == KEY and entry["meeting_key"] == MEETING and entry["empty"] == ["overtakes"]
    assert entry["missing"] == {"car_data": [], "location": []}
    assert manifest["tables"]["openf1_laps"]["rows"] == 6
    assert manifest["tables"]["openf1_laps"]["origin"].startswith("downloaded 20")
    assert lake.read_metadata(lake.table_path(lake.BRONZE, "openf1_laps", *MONZA, "R", tmp_path))["openf1_session_key"] == "9157"
    assert lake.bronze_telemetry_source(*MONZA, root=tmp_path) == lake.OPENF1       # what Silver will ask


def test_a_column_mixing_numbers_and_text_is_kept_as_text_value_by_value():
    table = openf1_bronze.as_table(race_answers()["session_result"])
    assert table["gap_to_leader"].tolist() == ["0", "6.064", "+1 LAP"]            # what OpenF1 sent, as text
    assert table["duration"].tolist()[:2] == [4413.9, 4420.0] and pd.isna(table["duration"].iloc[2])   # numbers stay
    nothing = openf1_bronze.as_table([{"gap": None}, {"gap": "+2 LAPS"}, {"gap": 1.5}])
    assert pd.isna(nothing["gap"].iloc[0]) and nothing["gap"].tolist()[1:] == ["+2 LAPS", "1.5"]


# --- never an empty table standing in for missing data ---------------------------------------------------------

@pytest.mark.parametrize("endpoint", ["laps", "stints", "race_control", "session_result", "car_data"])
def test_a_required_table_with_no_rows_fails_the_race_and_writes_nothing(tmp_path, endpoint):
    with pytest.raises(openf1.OpenF1Error, match=f"no '?(openf1_)?{endpoint}'? at all"):
        openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=FakeOpenF1(**{endpoint: []}), workers=2)
    assert not lake.has_bronze(*MONZA, root=tmp_path)
    assert not list(Path(tmp_path).rglob("*.parquet"))


def test_a_failed_request_writes_nothing(tmp_path):
    class Broken(FakeOpenF1):
        def __call__(self, endpoint, query):
            if endpoint == "location" and query.get("driver_number") == 55:
                raise openf1.OpenF1Error("HTTP 502, after every retry")
            return super().__call__(endpoint, query)

    with pytest.raises(openf1.OpenF1Error, match="502"):
        openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=Broken(), workers=2)
    assert not lake.has_bronze(*MONZA, root=tmp_path) and not list(Path(tmp_path).rglob("*.parquet"))


def test_a_driver_without_telemetry_is_named_not_invented(tmp_path):
    answers = race_answers()
    fake = FakeOpenF1(car_data=[r for r in answers["car_data"] if r["driver_number"] != 55])
    out = openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=fake, workers=2)
    assert out["missing"] == {"car_data": [55], "location": []}
    assert lake.read_manifest(*MONZA, root=tmp_path)[lake.OPENF1]["missing"]["car_data"] == [55]
    assert set(bronze(tmp_path, "car_data")["driver_number"]) == {1}


# --- only what Bronze lacks --------------------------------------------------------------------------------------

def test_a_second_run_asks_for_nothing_and_a_laps_only_bronze_is_topped_up_with_telemetry(tmp_path):
    first = FakeOpenF1()
    out = openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=first, telemetry=False)
    assert out["status"] == "ingested" and not first.asked("car_data")
    assert lake.read_manifest(*MONZA, root=tmp_path)["telemetry"] is False
    assert lake.bronze_telemetry_source(*MONZA, root=tmp_path) is None

    again = FakeOpenF1()
    assert openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=again, telemetry=False)["status"] == "complete"
    assert [e for e, q in again.calls if "year" not in q] == []        # not even the empty 'overtakes' is asked again

    top_up = FakeOpenF1()
    out = openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=top_up, workers=2)
    assert sorted(out["fetched"]) == ["openf1_car_data", "openf1_location"]
    assert {e for e, q in top_up.calls if "year" not in q} == {"car_data", "location"}
    manifest = lake.read_manifest(*MONZA, root=tmp_path)
    assert manifest["telemetry"] is True and manifest[lake.OPENF1]["empty"] == ["overtakes"]
    assert set(manifest["tables"]) == {lake.openf1_table(e) for e in race_answers() if e != "overtakes"}


def test_telemetry_already_in_another_lake_is_copied_not_downloaded(tmp_path, golden_lake):
    """The old lake (FastF1's laps + OpenF1's telemetry) already holds 2 GB of exactly these tables."""
    old, new, qatar = tmp_path / "old", tmp_path / "new", (2024, 23)
    race = pd.DataFrame([{"season": 2024, "round": 23, "session_key": 9655, "meeting_key": 1251,
                          "event_name": "Qatar Grand Prix", "location": "Lusail", "country": "Qatar",
                          "circuit": "Lusail", "start_utc": pd.Timestamp("2024-12-01T16:00:00+00:00"),
                          "end_utc": pd.Timestamp("2024-12-01T18:00:00+00:00"), "gmt_offset": "03:00:00"}])
    car = pd.DataFrame(samples(1, speed=300)).assign(session_key=9655)
    manifest = lake.read_manifest(*qatar, root=golden_lake)
    for name in lake.BRONZE_OPENF1_TABLES:
        lake.write_table(car, lake.table_path(lake.BRONZE, name, *qatar, "R", old))
    lake.write_manifest({**manifest, lake.OPENF1: {"session_key": 9655, "downloaded_at_utc": "2026-10-25T06:21:43+00:00"}},
                        *qatar, root=old)

    fake = FakeOpenF1()
    out = openf1_bronze.ingest_session(*qatar, root=new, fetch=fake, calendar=race, reuse_from=old)
    assert out["reused"] == ["openf1_car_data", "openf1_location"] and not fake.asked("car_data")
    assert out["missing"] == {"car_data": [55], "location": [55]}          # the copy holds car 1 only
    tables = lake.read_manifest(*qatar, root=new)["tables"]
    assert tables["openf1_car_data"]["origin"] == f"copied from the lake '{old}', downloaded 2026-10-25T06:21:43+00:00"
    assert lake.read_table(lake.table_path(lake.BRONZE, "openf1_car_data", *qatar, "R", new))["speed"].tolist() == [300, 301, 302]

    other = pd.DataFrame(samples(1, speed=300))                              # Monza's samples under Qatar's round
    lake.write_table(other, lake.table_path(lake.BRONZE, "openf1_car_data", *qatar, "R", old))
    with pytest.raises(openf1.OpenF1Error, match="is session .9157., not 9655"):
        openf1_bronze.ingest_session(*qatar, root=tmp_path / "third", fetch=FakeOpenF1(), calendar=race, reuse_from=old)


# --- never on top of something else ---------------------------------------------------------------------------------

def test_it_refuses_a_bronze_that_came_from_fastf1(golden_lake):
    before = lake.read_manifest(2024, 23, root=golden_lake)
    race = pd.DataFrame([{"season": 2024, "round": 23, "session_key": 9655, "meeting_key": 1251,
                          "start_utc": pd.Timestamp("2024-12-01T16:00:00+00:00"),
                          "end_utc": pd.Timestamp("2024-12-01T18:00:00+00:00")}])
    with pytest.raises(openf1.OpenF1Error, match="did not come from OpenF1 alone"):
        openf1_bronze.ingest_session(2024, 23, root=golden_lake, fetch=FakeOpenF1(), calendar=race)
    assert lake.read_manifest(2024, 23, root=golden_lake) == before


def test_it_refuses_a_round_that_holds_another_session(tmp_path):
    openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=FakeOpenF1(), telemetry=False)
    moved = openf1_bronze.season_calendar(FakeOpenF1(), 2023)
    moved.loc[moved["round"] == 14, "session_key"] = 9165                    # the calendar now says round 14 is Singapore
    with pytest.raises(openf1.OpenF1Error, match="holds OpenF1 session 9157, the calendar says 9165"):
        openf1_bronze.ingest_session(*MONZA, root=tmp_path, fetch=FakeOpenF1(), calendar=moved)


def test_only_races_so_far(tmp_path):
    with pytest.raises(openf1.OpenF1Error, match="only races"):
        openf1_bronze.ingest_session(*MONZA, session="Q", root=tmp_path, fetch=FakeOpenF1())


# --- sprints ---------------------------------------------------------------------------------------------------------

class WithSprint(FakeOpenF1):
    """Monza's meeting also has a 'Sprint' session, key 9999 (made up: Monza 2023 had none)."""

    def __call__(self, endpoint, query):
        if endpoint == "sessions" and "session_name" in query and "year" not in query:
            self.calls.append((endpoint, dict(query)))
            assert query == {"meeting_key": MEETING, "session_name": "Sprint"}
            return [{"session_key": 9999, "meeting_key": MEETING, "date_start": "2023-09-02T14:30:00+00:00",
                     "date_end": "2023-09-02T15:30:00+00:00", "is_cancelled": False}]
        return super().__call__(endpoint, query)


def test_a_sprint_is_the_sprint_session_of_the_races_meeting_in_its_own_folder(tmp_path):
    fake = WithSprint()
    out = openf1_bronze.ingest_session(*MONZA, session="S", root=tmp_path, fetch=fake, telemetry=False)
    assert out["session_key"] == 9999
    assert all(q.get("session_key") == 9999 for e, q in fake.calls if "session_key" in q)
    manifest = lake.read_manifest(*MONZA, session="S", root=tmp_path)
    assert manifest["session"] == "S" and manifest[lake.OPENF1]["session_start_utc"] == "2023-09-02T14:30:00+00:00"
    assert manifest["event_date"] == "2023-09-02" and manifest["event_name"] == "Italian Grand Prix"
    assert lake.has_bronze(*MONZA, session="S", root=tmp_path) and not lake.has_bronze(*MONZA, root=tmp_path)
    # the race's listings do not see it: the analytics and the models are about races
    assert lake.list_races(lake.BRONZE, "openf1_laps", tmp_path) == []
    assert lake.list_races(lake.BRONZE, "openf1_laps", tmp_path, session="S") == [MONZA]
    assert len(lake.list_sessions(lake.BRONZE, "openf1_laps", tmp_path, session=None)) == 1


def test_a_weekend_without_a_sprint_says_so(tmp_path):
    class NoSprint(FakeOpenF1):
        def __call__(self, endpoint, query):
            if endpoint == "sessions" and "session_name" in query and "year" not in query:
                return []
            return super().__call__(endpoint, query)

    with pytest.raises(openf1.OpenF1Error, match="0 'Sprint' sessions"):
        openf1_bronze.ingest_session(*MONZA, session="S", root=tmp_path, fetch=NoSprint())


# --- many races --------------------------------------------------------------------------------------------------------

def test_a_season_run_skips_races_not_run_yet_and_one_failure_does_not_stop_the_rest(tmp_path):
    class OnlyMonza(FakeOpenF1):
        def __call__(self, endpoint, query):
            if query.get("session_key") not in (None, KEY) or query.get("meeting_key") not in (None, MEETING):
                raise openf1.OpenF1Error("HTTP 500, after every retry")
            return super().__call__(endpoint, query)

    said = []
    after_monza = pd.Timestamp("2023-09-04T00:00:00+00:00")                 # rounds 1-14 run, 15-22 still to come
    results = openf1_bronze.ingest_season(2023, root=tmp_path, fetch=OnlyMonza(), telemetry=False, now=after_monza,
                                          say=said.append)
    assert [r["round"] for r in results] == list(range(1, 15))
    assert [r["status"] for r in results] == ["failed"] * 13 + ["ingested"]
    assert "HTTP 500" in results[0]["error"] and len(said) == 14 and "Italian Grand Prix" in said[-1]
    assert lake.list_races(lake.BRONZE, "openf1_laps", tmp_path) == [MONZA]


def test_the_command_line_reports_and_fails_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(openf1_bronze, "http_fetch", lambda: FakeOpenF1())
    assert openf1_bronze.main(["--season", "2023", "--rounds", "14", "--lake", str(tmp_path), "--no-telemetry"]) == 0
    assert "1 of 1 races in Bronze" in capsys.readouterr().out
    monkeypatch.setattr(openf1_bronze, "http_fetch", lambda: FakeOpenF1(laps=[]))
    assert openf1_bronze.main(["--season", "2023", "--rounds", "13", "--lake", str(tmp_path), "--no-telemetry"]) == 1
    assert "FAILED" in capsys.readouterr().out
