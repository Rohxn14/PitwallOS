# Moving from FastF1 to OpenF1

A developer's guide to how PitwallOS changed its data source. It is written piece by piece as the work is done.

**Status:** the lake in `data/` is built from OpenF1 alone (90 races, 2023 to 2026), new races are downloaded from OpenF1, and
the replay reads the lake. Sprints replay from the lake too, and the session picker's calendar and the map's rotation no longer
need FastF1. **Finished.** FastF1 is no longer a dependency: it is not imported, not installed, not in `requirements.txt`.
Everything the app shows comes from OpenF1 through the lake.

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
| 1 | Bronze: OpenF1's tables for a race, as delivered | done |
| 2 | Translation: OpenF1's tables into the lap table the cleaner reads, on one clock with zero at lights out | done |
| 3 | Rebuild 70 races, compare with the old lake | done |
| 4 | New races from OpenF1; pace models re-scored; the 2026 races; swap the lakes | done |
| 5 | The replay from the lake | done |
| 6 | The calendar and the circuits without FastF1; sprints from OpenF1 | done |
| 7 | Qualifying from OpenF1 | done |
| 8 | FastF1 removed | done |

## Decisions so far

| Decision | Chosen | Why |
|---|---|---|
| Where the translation lives | In Silver, in memory | Bronze stays a plain copy of the source. A bug in the translation is fixed by rebuilding Silver, never by downloading again |
| Which race is "2024 round 23" | The season's `Race` sessions that were not cancelled, in date order | OpenF1 has no round numbers. Measured: all 70 races of 2023-2025 get the round number the lake already used |
| Table names | `openf1_<endpoint>` | Says where a table came from; the three OpenF1 tables already in the lake keep their names |
| A column mixing numbers and text | Stored as text, value by value | Parquet allows one type per column. Dropping or converting `"+1 LAP"` would lose what the source said |
| A required table with no rows | The race fails, nothing is written | An empty table would look like "no laps were driven" |
| Telemetry already downloaded (2 GB) | Copied from the old lake | Checked identical to a fresh download; saves about 3.5 hours of requests |
| Session time zero | Lights out: Race Control's first "SESSION STARTED" message + 0.27 s | Formula 1's official race time counts from lights out. The message is logged 0.27 s early (measured, piece 2). Bronze keeps OpenF1's UTC time stamps |
| Pit laps without times | `PitInLap` / `PitOutLap` columns, honoured by the cleaner | OpenF1 has no pit table for six races. The pit laps are still known; their times are not, and stay null |
| Where the classification lives | Silver `results`, written when Silver is built | The Analyze window read FastF1's results from Bronze. `lake.results_path` picks whichever exists |
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


## Piece 2: the translation

### Purpose

Silver's cleaner reads one table per race in FastF1's shape: a row per driver and lap with the lap time, when the lap started
and ended, pit entry and exit, compound, tyre life, position and track status. OpenF1 spreads these over eight tables and
leaves some out. `src/pipeline/openf1_translate.py` rebuilds that table in memory each time Silver is built. It is pure:
tables in, tables out.

### The rules, and the measurement behind each

**The clock.** Session time 0 is lights out.

- OpenF1 merges F1's session-status feed into Race Control's messages. The first `SESSION STARTED` exists in all 70 races.
  In 67 it equals the start OpenF1 gives lap 1 to the millisecond. In the other three only the message is right: Japan and
  Monaco 2024 (lap 1 is dated from the restart after a red flag, 28 and 41 minutes later) and Miami 2025 (no lap-1 start).
- The message is logged before the lights go out. With zero at the message, the winner's last lap ends 0.27 s after the
  official race time in every race (53 races with complete winner laps: mean 0.270 s, sd 0.036 s). The telemetry agrees:
  cars leave the grid 0.66 s after the message, 0.39 s after message + 0.27 s, a normal reaction plus sampling delay.
- So zero = message + 0.27 s. At Qatar 2024 the winner's last lap then ends at 5465.355 s; the official time is 5465.323 s.
- At Zandvoort 2023 and Las Vegas 2024 FastF1's own start was 2.09 s and 0.45 s late. The new lake differs there on purpose.

**Lap starts and ends.** OpenF1 gives each lap a stamped start and an exact duration (F1's official lap time).

- The stamps of laps 2+ are 0.173 s late and jitter by about 40 ms. The durations are exact.
- So laps are chained: inside an unbroken run, each start is the previous start plus the previous duration. The run as a whole
  is placed at the median of (stamp - 0.173 s - chain). Lap 1 starts at zero and ends where lap 2 starts. A lap without a
  duration ends where the next lap starts.
- Against FastF1's crossing times: median difference 18 ms per race.
- Where they disagree most, Saudi Arabia 2025, the cars' positions decide: at the new lap-end times all drivers are at the same
  spot on the line (sd 1.2 m between drivers); at FastF1's times they are spread over 13.7 m (sd 5.3 m).

**Track status.** OpenF1 has no status feed. It is rebuilt from Race Control's messages:

| Message | Code |
|---|---|
| `SAFETY CAR DEPLOYED` | 4 |
| `VIRTUAL SAFETY CAR DEPLOYED`, `... ENDING` | 6, 7 |
| red flag | 5, until the next `SESSION STARTED` (not `TRACK CLEAR`: the track is clear long before the restart) |
| `TRACK CLEAR` | 1 |
| yellow or double yellow in a sector | 2 until that sector's `CLEAR`; hidden under a safety car or red flag |
| `SESSION STARTED` | clears yellows left from the laps to the grid (Italy 2025 had one that would have lasted 30 minutes) |

**Pits.** A pit row's `date` is the pit exit (FastF1's exit time + 0.03 to 0.12 s) and `lane_duration` is entry to exit. Entry =
date - lane_duration, on the in-lap. Without a pit table (2023 rounds 1-6) the in- and out-laps are flagged from the lap
table's `is_pit_out_lap`, with null times.

**Tyres.** Tyre life on lap n = age when fitted + laps since + 1, after two repairs:

- A stint that OpenF1 starts on the in-lap is moved to the out-lap (tyres only change in the pit lane). This also repairs the
  empty first stint OpenF1 writes after a red flag on lap 1.
- A stint with the same compound whose age simply carries on is the same set, driven through the pit lane: tyre life and
  stint number continue.

**The rest.** Position: the last position record up to a second after the lap ends. Deleted laps: Race Control's
"TIME ... DELETED ... LAP n" messages, minus any reinstated. Grid: each driver's first position record (OpenF1's
`starting_grid` is empty). Race distance: the winner's laps.

**What OpenF1 cannot say.** FastF1's `IsAccurate` was its own judgement. Here a lap is inaccurate only on evidence: no lap
time, or three sectors that do not add up to it. `FastF1Generated` is always false.

### Files

| File | Change |
|---|---|
| `src/pipeline/openf1_translate.py` | new: `lights_out`, `lap_times`, `status_timeline`, `lap_status`, `tyres_by_lap`, `pit_by_lap`, `deleted_laps`, `positions_at`, `lap_table`, `weather_table`, `results_table`, `pit_stop_record`, `translate`, `read_bronze` |
| `src/pipeline/cleaner.py` | `build_silver` translates when the manifest says `"source": "openf1"`, writes Silver `results`, and puts the telemetry on the same zero. `clean_laps` honours `PitInLap` / `PitOutLap` |
| `src/pipeline/lake.py` | `SILVER_RESULTS_TABLE`, `results_path` |
| `src/analytics/race_overview.py` | reads `lake.results_path` |
| `src/pipeline/quality.py` | reads the lap count from `openf1_laps` for an OpenF1 Bronze |

### Tests

`tests/pipeline/test_openf1_translate.py`, 19 tests. Small made-up tables for each rule, and a golden race: OpenF1's real
tables for Qatar 2024 (`tests/fixtures/openf1/qatar_2024/`, 103 KB). Golden facts the code never reads: the winner crosses
the line at the official race time, 1:31:05.323; the safety car is out on the leader's laps 1-4 and 35-42 with a virtual
safety car on lap 40; Verstappen leads lap 1 from second on the grid; 26 tyre changes. The module was broken in 13 ways and
every break made a test fail (one needed a test added first).

## Piece 3: the rebuild, compared with the FastF1 lake

    python -m src.pipeline.backfill --season 2024 --rounds 1 2 ... 24 --lake data_openf1 --telemetry --rebuild

All 70 races built Silver (laps, telemetry, results) and Gold: 0 failed.

### Lap by lap

77,259 laps exist in both lakes (old 77,713, new 77,373). Share of those laps on which a Silver column is identical:

| Column | Mean over races | Races at 99 % or more (of 70) |
|---|---|---|
| position | 99.7 % | 68 |
| compound | 99.1 % | 65 |
| tyre life | 98.1 % | 60 |
| in-lap / out-lap | 99.9 % / 99.9 % | 70 / 68 |
| safety car / virtual safety car / red flag | 99.7 % / 100.0 % / 99.9 % | 66 / 70 / 66 |
| green for the whole lap | 99.4 % | 62 |
| clean lap | 99.5 % | 63 |
| tyre change | 99.9 % | 68 |
| lap end within 0.1 s | 98.6 % | 60 |

**46 of 70 races match at 99 % or more on every one of these columns, 52 at 98 %, 60 at 95 %.** The expectations written
before measuring were "pretty much all" (Rohan) and "about 55" (Claude): both too high.

### What the analyses say

| Number | FastF1 lake | OpenF1 lake |
|---|---|---|
| Laps / clean laps / pace laps | 77,713 / 66,049 / 65,698 | 77,373 / 65,597 / 65,182 |
| Green pit stop loss, median | 22.5 s | 22.2 s |
| Tyre wear per lap, soft / medium / hard | +0.064 / +0.049 / +0.044 s | +0.063 / +0.050 / +0.044 s |
| Cost of following | +0.327 s/lap | +0.329 s/lap |
| Undercuts that worked | 45 % of 148 | 45 % of 146 |
| by gap, 0-1 / 1-2 / 2-3 s | 76 / 45 / 23 % | 76 / 45 / 25 % |
| Race report | 686 numbers, 0 unexplained | 686 numbers, 0 unexplained |

Every headline of Phase 2 is reproduced to within a few hundredths.

### Where the lakes differ, and which is right

| Difference | Verdict |
|---|---|
| Crossing times, Saudi Arabia 2025 (62 % within 0.1 s) | the new lake: the cars' positions at the line prove it |
| Race start at Zandvoort 2023 and Las Vegas 2024 | the new lake: FastF1's start was late |
| Lap 1 times, sector 1 of lap 1, lap times behind the safety car in the pit lane | different definitions; OpenF1 has values where FastF1 had none |
| Pit-lane time on every pit passage (the old lake: only where F1 published its pit-stop record) | the new lake has more |
| Azerbaijan 2023: 17 % of laps not green | the old lake: Race Control never cleared a yellow in sector 3 |
| Belgium 2025, Miami 2025, Australia 2025: tyre life | not settled: OpenF1's stints are irregular there (Miami was inconsistent in FastF1 too) |
| 340 laps missing, some deleted laps not found (Qatar 2024: 15 of 18), laps without an end (a car that stops on track) | the old lake |

### Known limits

- A sector yellow that Race Control never clears stays yellow until the next `TRACK CLEAR`. Real yellows last up to 13 minutes,
  so no time limit can tell a forgotten one from a long one. The error is on the safe side: laps lose the clean flag, none gain it.
- A lap with no duration and no next lap has no end, so no position, weather or status.
- Classified positions of retired cars are null (OpenF1 gives none).
- The two constants (0.173 s, 0.27 s) are measured medians, not physical facts. If OpenF1 changes how it stamps laps they must be
  measured again; the golden test on the official race time would fail first.
- The pace models were re-scored in piece 4.


## Piece 4: new races from OpenF1, the 2026 season, and the swap

### New races come from OpenF1

`runner.bronze_step` and `backfill.build_race` now ingest a race with no Bronze through `openf1_bronze.ingest_session`. The
backfill takes a season's races, and which of them have been run, from OpenF1's calendar (`openf1_bronze.season_calendar`,
`has_run`). The Analyze window builds a missing race the same way, in about half a minute. A race already in an old FastF1
Bronze keeps it. Qualifying and sprints still go through FastF1 until they are moved.

### Four more rules, found on races that broke

| Race | What broke | Rule added |
|---|---|---|
| Belgium 2025 | The start was behind the safety car (rain); OpenF1 has every car on mediums from lap 2, though they pitted for them on lap 12. Pace-model error 2.16 s, the worst race of 2025 | A start announced "RACE WILL START BEHIND THE SAFETY CAR" runs under 4 until the rolling or standing start. A stint that starts where the car never left the pit lane carries the previous set on |
| Italy 2026 | A 200.7 s lap after a red flag counted as a clean lap: it held an extra formation lap | After a red flag, if Race Control spoke of the safety car while the race was stopped, the restart runs under 4 until the start message (F1's own feed does the same: Monaco 2024, Mexico 2023, Zandvoort 2023); "EXTRA FORMATION LAP" puts it back. Without any start message (Sao Paulo 2023) it ends after 180 s |
| Netherlands 2026 | A lap ended 10 ms before it started (OpenF1's laps contradict each other after a 32-minute lap) | A lap whose end would come before its start has no end |
| Mexico 2026 | The clock was a day off: the race's messages carry a "SESSION STARTED" from the day before | Lights out is looked for no earlier than an hour before the scheduled start |

The tyre rule is limited twice: it is not applied where the lap table has holes (Miami 2025 holds 56 % of the laps), nor where
the car did leave the pit lane since the previous set was fitted, on another lap than the stint says (Miami 2025: out on lap
27, the stint from lap 33). There OpenF1 contradicts itself and the stint is kept as given.

After these rules, 48 of 70 races match the FastF1 lake at 99 % or better on every key column (46 before), 53 at 98 %, 61 at
95 %, and no race is worse on any column except Sao Paulo 2023 and Japan 2024, by about one lap per car: their formation lap
after the red flag is now marked as safety car, where F1's feed showed green.

### The pace models on the new lake

Fitted on 2023-2024, scored once on 2025, as in Phase 3. Mean error on dry laps, in seconds:

| Model | Next lap, FastF1 lake | Next lap, OpenF1 lake | 10 laps, FastF1 lake | 10 laps, OpenF1 lake |
|---|---|---|---|---|
| Persistence | 0.420 | 0.421 | 0.700 | 0.706 |
| Linear decomposition | 0.398 | 0.400 | 0.644 | 0.652 |
| State-space | 0.386 | 0.389 | 0.633 | 0.643 |
| LightGBM corrections | 0.353 | 0.355 | 0.569 | 0.582 |

Each rung still beats the one before it at every horizon (paired bootstrap over races). Before the Belgium fix the boosted
model scored 0.413: one race with wrong tyres and a wrong start was enough to move a season's headline by 0.06 s.

### 2026

All 20 races run so far were downloaded (about 2.3 minutes each, telemetry included) and built: 20 of 20. The winner's
rebuilt finish matches the official race time within 0.06 s in 18 of the 19 races where his laps are complete; Austria 2026 is
0.32 s off, for a reason not found. 2026's telemetry has more source faults than 2023-2025: 11 of 20 races pass every
quality check. Monaco 2026's car positions stop 45 minutes before the flag. Hungary and China 2026 carry samples from the day
before; Silver keeps only the samples inside the race's laps, so these do no harm.

The Analyze window opens every 2026 race, including the Italian Grand Prix of the screenshot that started this work.

### The swap

Two folder renames on the same disk, nothing copied or deleted:

    data/          the FastF1 lake  ->  data_fastf1/   (kept, read-only from now on)
    data_openf1/   the OpenF1 lake  ->  data/          (90 races: 2023-2026)

Both names are ignored by git and refused by the local push guard. The reports in `data/reports/` were all rebuilt on the
new lake (backfill, the six analytics steps, the 2025 model test, the telemetry quality report), and the README's charts and
numbers were redrawn from it.

What the swap did NOT change: the test fixtures. `tests/fixtures/lake/` and `lake_hungary/` are FastF1-shaped Bronze. That
shape is now the cleaner's input contract (the translation produces it), so they still test the cleaner exactly; the
translation has its own golden race (`tests/fixtures/openf1/qatar_2024/`). Golden numbers measured on FastF1's clock in those
tests stay as they are.


## Piece 5: the replay from the lake

### Purpose

The replay (Tom Shaw's `src/f1_data.py`, `main.py`, `src/interfaces/race_replay.py`) read FastF1's `Session` object, which
since the login exists only for sessions left in FastF1's cache: four races. The lake holds the same facts for all 90. This
piece lets the replay read the lake, so every race from 2023 on replays again.

### Design

`src/lib/lake_session.py` holds `LakeSession`: one race from the lake, offering what the replay reads from a FastF1 session
under the same names. It is an adapter again, at the other end of the system: the translation made OpenF1's tables look like
FastF1's for the cleaner; `LakeSession` makes the lake look like a FastF1 session for the replay. The replay's frame builder
(1,000 lines) and its window keep their code.

| The replay reads | From FastF1 it was | From the lake it is |
|---|---|---|
| `session.laps`, `.results`, `.weather_data` | FastF1's tables | the translation of Bronze's OpenF1 tables (same shape) |
| `session.track_status` | F1's status feed | the status timeline rebuilt from Race Control |
| `session.race_control_messages` | F1's messages | Bronze `openf1_race_control`, on the session clock |
| each lap's `get_telemetry()` | car and position samples merged per lap, in 20 worker processes | `driver_samples()`: Silver telemetry (10 Hz, one clock), no worker processes |
| the timing stream (official order, gaps) | FastF1's private `_extended_timing_data` | `timing_stream()`: OpenF1's `position` and `intervals` tables |
| team colours | `fastf1.plotting` | `driver_colors()`: OpenF1's `team_colour` |
| the track's outline and DRS zones | the fastest qualifying lap | `example_lap()`: the fastest clean race lap in Silver telemetry |
| the map's rotation | FastF1's circuit info | MultiViewer, through FastF1's client (the one thing still asked of FastF1's code for a race) |

Five places in the replay ask `hasattr(session, ...)` and take the lake's version: `load_session`, `get_driver_colors`,
`get_circuit_rotation`, `example_lap_telemetry` (new, used by `main.py` and the safety-car drawing), the driver loop of
`get_race_telemetry`, and the two readers of the timing stream.

`f1_data.load_session` decides the source: a race from 2023 on comes from the lake; if the lake lacks it, it is downloaded
from OpenF1 and built first (about 3 minutes, once); only if that fails is FastF1's cache asked. Qualifying, sprints and
seasons before 2023 still go to FastF1. The cached replay frames of a lake race have their own file name, so frames computed
from FastF1 are never mixed with them (their clocks differ).

### A change in Silver: the lap a car never finished

Found by the comparison below. A car that retires on track has a last lap with no end (it never crosses the line again).
Silver telemetry stopped at the car's last completed lap, so in the replay a retired car froze at the finish line instead of
where it stopped. Now `cleaner.stopped_at` keeps the telemetry of that lap until the car comes to rest (the first time it
stands still for 30 s), plus 5 s, and never past the race's end. What comes after is left out: a car on a recovery truck
still sends a position. The lap itself keeps a null end time.

### Measured: Baku 2026, both ways

Baku 2026 is in FastF1's cache and in the lake. Its replay was built from each and compared at 5,800 instants per driver.

| | Result |
|---|---|
| Distance between the same car in the two replays | median 0.54 m, 90 % under 2.1 m |
| Same race position at the same instant | 99.6 % |
| Final order, finished / retired state | identical for all 22 drivers |
| The 17 finishers: lap number | identical throughout |

The remaining differences are three retired cars, and the lake is right or more complete in each: Stroll's car, which the
FastF1 replay moves to map coordinate (0, 0) after he stops (FastF1's "no position" value) while the lake keeps it where it
stopped; Alonso and Colapinto, for whom OpenF1 has one more lap row (driving into the pits to retire).

Qatar 2024, which FastF1 could no longer replay at all, builds its frames from the lake in about 100 s (once; then they are
cached) with the official final order.

### Known limits

- The track outline comes from a race lap, so DRS zones are those used in that lap (the replay's own speed-based fallback
  applies). Qualifying is not in the lake yet.
- Gaps between cars update about every 4 s (OpenF1's `intervals`), not continuously.
- Qualifying and sprint replays, and seasons before 2023, still need FastF1's cache.
- The window itself needs a display and is checked by eye; its frames, lap times and order are tested without one.
- Monaco 2026's car positions stop 45 minutes before the flag (OpenF1): its replay stops moving there.

### Tests

`tests/lib/test_lake_session.py` (8): a temporary lake built from real OpenF1 data (Qatar 2024's tables, two cars'
telemetry). Which sessions are replayable; `load_session` never asks FastF1 for a race the lake has; the attributes the
replay reads; a driver's arrays; the official order from the timing stream (Russell on pole, Verstappen leading after two
minutes); the outline lap; the frames end to end; where a retired car's telemetry ends. `tests/test_load_session.py` (4): the
order of sources, with an empty temporary lake and a faked download.

## Piece 6: the calendar, the circuits and sprints without FastF1

### The calendar

The session picker asked FastF1 for the season's weekends. `src/lib/race_calendar.py` builds the same list from OpenF1's
`sessions` and `meetings` (two requests a season): round, event name, the race's date at the circuit, country, whether it is
a sprint weekend, and every session's scheduled start. The picker's functions in `f1_data.py`
(`get_race_weekends_by_year`, `get_race_weekends_by_place`, `get_all_unique_race_names`, `list_rounds`, `list_sprints`) now
call it. The picker offers 2023 onwards, where OpenF1's data starts; it used to offer 2018 onwards, seasons that could no
longer be opened.

### The circuits

The map's rotation came from MultiViewer through FastF1's client. MultiViewer refuses a request that does not name its
client (HTTP 403 to the default `python-requests`), and answers one that does. `race_calendar.circuit` asks it directly as
`PitwallOS (+repository URL)` and keeps the whole answer: rotation, corners with their numbers, marshal sectors. The corners
are the official reference the track model (Phase 5.1) will be checked against.

### Reference data in the lake

Both answers are small, change rarely and are needed offline, so they are kept as JSON under `<lake>/reference/`
(`lake.reference_path`, `write_json`, `read_json`):

| What | Fetched | Then |
|---|---|---|
| a finished season's calendar | once | read from the lake |
| the running season's calendar | again when the copy is older than 12 hours (a race may be cancelled) | the copy if there is no network |
| a circuit in a given year | once | read from the lake |

Opening the picker costs 3 s the first time and 0.01 s afterwards, and it works offline.

### Sprints

A sprint is a short race with the same OpenF1 tables. `openf1_bronze.ingest_session(..., session="S")` finds it as the
`Sprint` session of the race's meeting (`session_of`) and writes it to its own folder, `session=S`. The translation, Silver,
Gold and `LakeSession` take it unchanged; `f1_data.load_session` sends "S" to the lake like "R". A sprint is downloaded the
first time it is replayed (about 3 minutes). Checked on Qatar 2024's sprint: 380 laps, 20 classified, the replay's final order
Piastri, Norris, Russell, as the result sheet has it.

`lake.list_sessions` and `list_races` now list races unless asked otherwise: the six analyses listed every session, so a
sprint in the lake would have been counted as a second race at that round.

### Tests

`tests/lib/test_race_calendar.py` (7): 2023's real calendar plus the real session times of two weekends; the picker's
shape; a finished season fetched once; the running season fetched again after 12 hours and surviving without a network; the
error when there is neither network nor copy; a circuit asked once, by name, and kept; a refusal keeps nothing.
`tests/pipeline/test_openf1_bronze.py` (+2): a sprint lands in its own folder and the race listings do not see it; a weekend
without a sprint says so.

### Known limits

- Qualifying and sprint qualifying are not in the lake: their replays still need FastF1's cache.
- A sprint is not pre-downloaded; the first replay of each waits for the download.
- Analyze has no sprint view: the analyses are about races.

## Piece 7: qualifying from OpenF1

### Purpose

The qualifying replay shows each driver's fastest lap of Q1, Q2 and Q3 as a trace. It was the last thing in the app that
needed FastF1. `src/lib/lake_qualifying.py` (`LakeQualifying`) answers its three questions from Bronze.

### Design

Qualifying is ingested into Bronze like a race (`openf1_bronze`, sessions "Q" and "SQ"; sprint qualifying is looked for
under both names OpenF1 has used, "Sprint Qualifying" and 2023's "Sprint Shootout"). It gets **no Silver and no Gold**: the
cleaner is about races (a clock from lights out, laps chained to the flag, clean laps, stints), and the only consumer of
qualifying is this replay, which needs a few dozen laps cut from the raw tables. `openf1_bronze.RACE_LIKE` names the
sessions that go on to Silver ("R", "S").

| The replay needs | From OpenF1 |
|---|---|
| the results with each driver's Q1 / Q2 / Q3 time | `session_result.duration`, a list of three per driver |
| the laps of each part | Race Control has one `SESSION STARTED` and one `SESSION FINISHED` per part, numbered by `qualifying_phase`; a lap belongs to the part in which it started. A part stopped by a red flag and restarted is one part |
| a driver's fastest lap of a part | the lap of that part whose time equals his official best in the result table; failing that, the shortest timed lap that is not a pit-out lap. A faster lap deleted for track limits is therefore never chosen |
| that lap's telemetry | `car_data` and `location` between the lap's start (`date_start` - 0.173 s) and its end, on a clock from 0; position interpolated onto the speed samples; a missing reading (2026 has empty DRS values) becomes 0 |

`f1_data._quali_fastest_lap` takes the lake's lap and telemetry when the session offers them; the 280 lines that turn a lap
into frames are unchanged. `load_session` sends "Q" and "SQ" to the lake, downloading the session's Bronze first if it is
missing (about 2.5 minutes), and asks FastF1 only if that fails.

### Measured: Baku 2026 qualifying, both ways

| | Result |
|---|---|
| Position and Q1 / Q2 / Q3 times | identical for all 22 drivers |
| Fastest laps (22 + 15 + 10) | 47 in both, 47 with the same lap time |
| The same car on the same lap, distance between the two replays | median 2.0 m, 90 % under 4.9 m, worst lap 29 m |
| Sector times, compound, top speed (331 km/h) | identical |

Before the result table was used to pick the lap, 46 of 47 matched: the lake had taken a faster lap that did not count.
The four qualifying sessions FastF1's cache holds (2026 Australia, Barcelona, Baku, Bahrain) are all in the lake.

### Tests

`tests/lib/test_lake_qualifying.py` (10), on a real cut of Baku 2026 qualifying
(`tests/fixtures/openf1/baku_2026_qualifying/`): the three parts' times; a red-flagged part joined; a lap's part; the
official results (pole 1:42.526); the fastest lap by code and by number; a made-up faster lap not chosen; a lap's telemetry
(6.0 km, ends where it began); a lap without samples; the replay's frames; `load_session` never asking FastF1.

### Known limits

- Qualifying cannot be analysed, only replayed: it has no Silver.
- Sprint qualifying is wired and found under both names, but no real session has been downloaded or replayed.
- The 0.173 s lag of OpenF1's lap stamps was measured on races and is assumed for qualifying.

## Piece 8: FastF1 removed

With every session coming from the lake, FastF1 had no caller left that could succeed. It was removed in one piece, and the
test suite was then run in an environment where FastF1 is not installed at all.

### What was deleted

| Deleted | What it was |
|---|---|
| `src/pipeline/extractor.py` | Bronze from FastF1: loading a session, listening to FastF1's log for hidden failures, F1's pit-stop record |
| `openf1.find_session`, `drivers_of`, `download`, `add_telemetry`, its command line | adding OpenF1's telemetry to a Bronze whose laps came from FastF1 |
| `telemetry_clock.session_offset`, `lap_offsets`, `lap_one_vote`, `later_laps`, `ClockError` | matching OpenF1's clock to FastF1's, with a vote among the drivers. One source has one clock: nothing to match |
| `cleaner.telemetry_tables`' FastF1 branches, `lake.BRONZE_TELEMETRY_TABLES`, `lake.FASTF1` | reading FastF1's own telemetry tables |
| `telemetry_quality.line_speeds`, `openf1_clock` and the line-speed check | compared the car's speed with F1's speed at the line (`SpeedFL`), which OpenF1 does not carry |
| the runner's and the backfill's FastF1 paths, `--pit-stops`, `--verbose`, the rate-limit waits | downloading through FastF1, and waiting out FastF1's own hourly cap |
| `f1_data.enable_cache`, `_process_single_driver`, the FastF1 branches of six functions, the worker processes | the replay's FastF1 path: per-lap telemetry merged in 20 processes |
| `fastf1._api` and `fastf1.utils` in `race_replay.py` | the timing stream and a gap parser (now three lines) |
| the "FastF1 Cache Location" setting and its row in the settings dialog | where FastF1 kept its downloads |
| `tests/pipeline/test_bronze.py`, `test_shared_cache.py`, and the tests of each function above | the suite went from 664 tests to 617 |
| `fastf1` in `requirements.txt` | the dependency |

36 files changed: about 2,030 lines removed and 370 added.

### What was kept, and why

- **The FastF1 table shape.** `openf1_translate` produces FastF1's lap table because that is what the cleaner reads. The
  column names (`LapStartTime`, `TrackStatus`, `FastF1Generated`) are now simply the cleaner's input contract.
- **The fixtures in `tests/fixtures/lake/`.** They are real Bronze in that shape (British, Qatar and Hungary 2024), and the
  golden tests of Silver, Gold, the analytics and the models are built on them. A manifest without `"source"` marks such a
  Bronze; the cleaner reads its tables directly. They have no telemetry path any more.
- **FastF1's samples as a golden fact.** `tests/fixtures/openf1/baku_63_fastf1_*.parquet` are FastF1's own telemetry
  samples of one car, saved while FastF1 could still load them. A test checks that OpenF1's samples are the same, sample
  for sample. The proof that the two sources carry one feed stays, without the library.
- **`LATER_LAP_LAG_SEC = 0.173`.** Measured by the clock-matching code against FastF1; the translation still uses it.
- **Seasons before 2023 are gone from the app.** OpenF1 starts in 2023. `load_session` says so instead of trying.

### What the replay says when it cannot open a session

`f1_data.load_session` raises `TelemetryNotAvailable` with the reason: the season is before 2023, the kind of session
cannot be replayed, or the session is not in the lake and the download from OpenF1 failed (with the error).

### Tests

The suite passes with FastF1 uninstalled. `tests/conftest.py` gained `openf1_qatar_lake`, a lake built once per test run from
OpenF1's real tables for Qatar 2024 (Bronze, Silver with two cars' telemetry, Gold); the tests of Silver telemetry, of the
telemetry quality report and of the replay adapter share it. `tests/test_load_session.py` was rewritten around the new
behaviour (6 tests).

## What the migration cost and what it bought

| | Before (October 2026) | After |
|---|---|---|
| Source | FastF1, behind a paid login; OpenF1 for telemetry | OpenF1 only, free |
| Races in the lake | 70 (2023-2025) | 90 (2023-2026) |
| Can a new race be added? | no | yes, about 3 minutes |
| Replay | 4 races | every race, sprint and qualifying session from 2023 |
| Clocks | two, matched by a vote | one, from lights out |
| Session time zero | FastF1's, about an hour before the start | lights out: a lap's end is the official race time |
| Seasons before 2023 | in the picker (2018 on), not openable since the login | not offered |
| Analytics headlines, model scores | - | reproduced (0.355 s against 0.353 s for the best model) |

What it showed about the design: the lake's rule that only Bronze knows the source held. The cleaner, Gold, six analyses
and four models changed by about ten lines between them.
