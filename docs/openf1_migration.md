# Moving from FastF1 to OpenF1

A developer's guide to how PitwallOS changed its data source. It is written piece by piece as the work is done.

**Status:** piece 1 of 4 built (Bronze from OpenF1). The pipeline still runs on the FastF1 lake until the new one has been
compared with it.

## Why

Around 14 October 2026 Formula 1 put its timing archive (`livetiming.formula1.com/static`) behind a login. FastF1, the library
PitwallOS used for every lap table, can only read it with a paid F1 TV subscription. Without one:

- no new race can be added to the lake,
- the 70 races already there cannot be downloaded again,
- the replay opens only sessions left in the local FastF1 cache.

[OpenF1](https://openf1.org) is a community service that keeps a copy of the same live feed. It is free for past races from
2023 on, needs no login, and is licensed CC BY-NC-SA 4.0 (non-commercial, with attribution). PitwallOS already took its car
telemetry from it. This migration takes everything else from it too, and removes FastF1.

## How the change is made

**Build beside, compare, then swap.** The new lake is built in its own folder (`data_openf1/`) while the old one (`data/`) stays
untouched. The old lake is the answer key: every rebuilt race is compared with it before anything depends on the new one.
Changing the old lake in place was rejected because a mistake would have destroyed the only copy of data that can no longer be
downloaded.

**Change one layer, keep the rest.** The lake has three layers: Bronze (raw, as delivered), Silver (cleaned), Gold (ready for
models). Only Bronze changes source. A translation step turns OpenF1's tables into the shape the Silver cleaner already
reads, so the cleaner, Gold, the analytics and the models keep their code and their tests.

| Piece | What | State |
|---|---|---|
| 1 | Bronze: OpenF1's tables for a race, as delivered | built |
| 2 | Translation: OpenF1's tables into the lap table the cleaner reads | next |
| 3 | One clock for laps and telemetry, zero at lights out | measured, not built |
| 4 | Rebuild 70 races, compare with the old lake, swap; then the replay, then FastF1 removed | planned |

## Decisions so far

| Decision | Chosen | Why |
|---|---|---|
| Where the translation lives | In Silver, in memory | Bronze stays a plain copy of the source. A bug in the translation is fixed by rebuilding Silver, never by downloading again |
| Which race is "2024 round 23" | The season's `Race` sessions that were not cancelled, in date order | OpenF1 has no round numbers. Measured: all 70 races of 2023-2025 get the round number the lake already used |
| Table names | `openf1_<endpoint>` | Says where a table came from; the three OpenF1 tables already in the lake keep their names |
| A column mixing numbers and text | Stored as text, value by value | Parquet allows one type per column. Dropping or converting `"+1 LAP"` would lose what the source said |
| A required table with no rows | The race fails, nothing is written | An empty table would look like "no laps were driven" |
| Telemetry already downloaded (2 GB) | Copied from the old lake | Checked identical to a fresh download; saves about 3.5 hours of requests |
| Session time zero | Lights out | Formula 1's official race time counts from lights out. Bronze keeps OpenF1's UTC time stamps |
| More tables than needed today (`intervals`, `overtakes`) | Taken now | The project has just lost a source overnight. Raw data is cheap, access is not guaranteed |

## Piece 1: Bronze from OpenF1

### Purpose

Copy everything OpenF1 knows about one race into the lake, unchanged, so that every later step can be rebuilt offline.

### The data

OpenF1 answers one kind of question per address ("endpoint"). Each becomes one Parquet table in
`<lake>/bronze/season=YYYY/round=N/session=R/`:

| Table | Holds | Required |
|---|---|---|
| `openf1_sessions`, `openf1_meetings` | OpenF1's keys, the place, the scheduled start | yes |
| `openf1_drivers` | car number, three-letter code, team | yes |
| `openf1_laps` | per driver and lap: start (UTC), duration, sectors, speed traps | yes |
| `openf1_stints` | tyre sets: compound, first and last lap, age when fitted | yes |
| `openf1_race_control` | Race Control's messages: flags, safety car, deleted lap times | yes |
| `openf1_weather` | one reading a minute | yes |
| `openf1_position` | every change of position | yes |
| `openf1_session_result` | the classification | yes |
| `openf1_pit` | pit-lane passages: when, time in the lane, time stationary | no |
| `openf1_intervals` | gap to the leader and to the car ahead, about every 4 s | no |
| `openf1_overtakes` | who passed whom, when | no |
| `openf1_car_data`, `openf1_location` | telemetry, about 4 samples a second per driver | yes, unless `--no-telemetry` |

`_manifest.json` is written last and marks the Bronze as complete. It records `"source": "openf1"`, the race's identity,
OpenF1's session key, each table's row count and where it came from, the endpoints that were empty, and the drivers OpenF1
has no telemetry for.

### Files

**`src/pipeline/openf1_bronze.py`** (new)

| Function | What it does | Why it is built that way |
|---|---|---|
| `as_table(rows)` | OpenF1's rows as a table Parquet can hold | Only a column that mixes text with numbers is changed, to text. Measured on six races: `session_result.gap_to_leader` is the only such column |
| `season_calendar(fetch, year)` | One row per race of a season, with its round number | Sorted by date, not by session key: OpenF1's keys are not in date order (Qatar 2023 is 9221, Austin, a week later, 9213) |
| `race_of(calendar, year, round)` | The calendar row of one round | Fails loudly for a round the season does not have |
| `has_run(race, now)` | True six hours after the scheduled end | A season run must skip races still to come |
| `local_date(start_utc, gmt_offset)` | The race's date at the circuit | Las Vegas starts on Sunday morning UTC, Saturday night there |
| `held_endpoints(...)` | What this Bronze already answers | Lets a second run fetch only what is missing. Refuses a Bronze from FastF1, and a round holding another OpenF1 session |
| `telemetry_in(other_root, ...)` | OpenF1 telemetry already in another lake | Checked to be the same session before it is copied |
| `ingest_session(...)` | One race into Bronze | Everything is fetched into memory first, tables are written next, the manifest last: a failed request leaves nothing half-written |
| `ingest_season(...)` | Every race run so far, one after the other | One race failing is reported and never stops the others |
| `main()` | The command line | Exit code 1 if any race failed |

**`src/pipeline/openf1.py`** (changed): `download_per_driver` was split out of `download`, so the new module and the old
telemetry path share the per-driver download. The client itself (`http_fetch`, `Pace`) is unchanged: at most 25 requests a
minute, retries on server errors, a one-minute wait on "too many requests".

**`src/pipeline/lake.py`** (changed): `openf1_table(endpoint)` gives a table's name. No module builds a path or a name by hand.

**`.gitignore`**: `data_openf1/` and `data_fastf1/` are ignored like `data/`.

### Rules kept

1. As delivered: no renaming, no conversion, no filtering. Time stamps stay text.
2. Never an empty table standing in for missing data.
3. A driver without telemetry is named in the manifest, never invented.
4. Everything is downloaded before anything is written; the manifest is written last.
5. Only what Bronze lacks is downloaded.
6. Never on top of a Bronze from another source.

### Tests

`tests/pipeline/test_openf1_bronze.py`, 20 tests, no network. The calendar is OpenF1's real 2023 list
(`tests/fixtures/openf1/calendar_2023_*.json`), the season with the traps: Imola cancelled, Las Vegas on a Saturday night,
keys out of date order. The race is a made-up one of two cars and three laps: Bronze copies, so the test is that every value
comes back unchanged.

Each rule above has a test. The tests were themselves tested: the module was broken in ten ways (cancelled races kept, rounds
sorted by key, an empty required table accepted, the FastF1 guard removed, ...) and each break made a test fail.

### Known limits

- Races only. Qualifying and sprints come later.
- OpenF1 has no pit table for the first six races of 2023. Piece 2 has to find pit entries and exits for them another way, or
  leave them unknown.
- A round number is a race's place in the calendar. If a race were cancelled after later rounds were ingested, their numbers
  would move: the ingest then refuses, naming both sessions, instead of mixing two races in one folder.
- The telemetry of 2023-2025 was downloaded in October 2026 and copied; the other tables are from November 2026.

### How to run it

    python -m src.pipeline.openf1_bronze --season 2024 --rounds 23 --lake data_openf1
    python -m src.pipeline.openf1_bronze --season 2023 2024 2025 --lake data_openf1 --reuse-telemetry-from data
    python -m src.pipeline.openf1_bronze --season 2024 --lake data_openf1 --no-telemetry     # small tables only
    python -m pytest tests/pipeline/test_openf1_bronze.py

A race is 12 small requests plus 2 per driver: about 30 s with copied telemetry, about 3 minutes without.

### Measured

All 70 races of 2023-2025 were ingested into `data_openf1/` on 2026-11-11 and 12: **70 of 70, none failed.**

| Table | Rows |
|---|---|
| `openf1_laps` | 77,373 |
| `openf1_stints` | 3,977 |
| `openf1_pit` | 2,339 |
| `openf1_race_control` | 6,286 |
| `openf1_weather` | 11,069 |
| `openf1_position` | 34,456 |
| `openf1_session_result`, `openf1_drivers` | 1,397 each |
| `openf1_intervals` | 1,672,060 |
| `openf1_overtakes` | 17,589 |
| `openf1_car_data` | 48,789,989 |
| `openf1_location` | 50,033,694 |

Bronze is 0.62 GB. About 30 s a race with copied telemetry.

- **No pit table for 2023 rounds 1-6** (Bahrain to Monaco): OpenF1 answers "no results". Every later race has one.
- **Nine drivers in eight races have no telemetry** and are named in the manifests: 2023 Italy (22), Qatar (55), Sao Paulo (16);
  2024 Britain (10), Sao Paulo (18, 23); 2025 Austria (55), Britain (43), Italy (27). They are entered drivers who appear in
  OpenF1's driver list; whether each one failed to start is checked in piece 2, against the lap table.
- **Stable source:** Qatar 2024's lap table is identical in three downloads (October, early November, 11 November), and the
  copied telemetry is identical to the old lake's, row for row.
- **Expectation, written before the run:** 67 to 70 would succeed, any failure being a required table missing for an early
  2023 race. Result: 70. The early-2023 gap was real but fell on `pit`, which is optional.
