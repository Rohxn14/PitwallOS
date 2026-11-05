# Phase 5 step 5.0: car telemetry in the lake

A developer's guide to everything step 5.0 built: what it is for, where the data comes from, every file and function,
why each piece is built the way it is (and what was tried or rejected, with the measured reasons), the rules it keeps,
how it is tested, what it measured, and what it does not do yet. Written 2026-10-25.

## 1. What this step is for

Phase 5 uses **telemetry**: what each car reports about itself several times a second (speed, engine revs, gear,
throttle, brake, DRS) and where it is on the track (x, y, z). Until now the lake held only **timing** (one row per lap:
lap time, sectors, tyres, pit stops, track status). Timing says *how long* a lap took; telemetry says *how* it was
driven: where a car brakes, how fast it takes each corner, how much a worn tyre costs in each corner.

Step 5.0 puts telemetry for every race of 2023, 2024 and 2025 (70 races) into the lake and measures how complete and
trustworthy it is, **before** anything is built on it. Its consumers:

| Step | Uses the samples for |
|---|---|
| 5.1 track model | a reference line per circuit, corners found from the speed trace, braking zones, DRS zones |
| 5.2 lap features | per car, lap and corner: minimum speed, braking point, throttle pick-up, top speed |
| 5.4 tyres | corner-speed fade against tyre age: does it warn of a tyre going off before the lap time does? |
| 5.8 the game | positions on the track for "watch the race" |

## 2. Words used below

| Word | Meaning |
|---|---|
| sample | one reading from the car, with its time stamp. The source sends ~4 a second (every 0.24 s median, 0.44 s at the 95th percentile) |
| car data / location | the two streams: car data = speed, rpm, gear, throttle, brake, DRS; location = x, y, z (tenths of a metre). Each has its own time stamps |
| session time | seconds since the timing session's clock started; every time in our lake (lap starts, pit times) is on this clock |
| UTC | the time of day (world time). OpenF1 stamps samples with it |
| t0 | the UTC moment when session time was 0: one number per race. `session time = UTC - t0` |
| tick | one point of Silver's 10 Hz grid (every 0.1 s, the same instants for every car) |
| hole | a stretch where the source samples are more than 2 s apart: Silver leaves the values there empty (`is_data_gap`) |
| no reading | the value 104 on the brake or the throttle: the car sent nothing |

## 3. Where the data comes from, and why OpenF1

FastF1 (the Python library the pipeline's Bronze layer uses) downloads from F1's live-timing archive
(`livetiming.formula1.com/static`). Since 14-15 October 2026 that archive answers **401 Unauthorized** to anyone without
an F1 TV login (FastF1 issue #978; FastF1 3.9.0, released 2026-10-24, adds the login; F1 TV Access or higher is
needed). The lake's timing for the 70 races was downloaded in Phase 1, before the lock, so it is safe. Telemetry was
never downloaded (Phase 1 was laps only): none of the 70 races had it.

**OpenF1** (`api.openf1.org`) is a community service that recorded the same F1 feed and serves every session since 2023,
free and without a login, for "educational purposes, personal learning projects, research and non-commercial fan
engagement" (its terms), at most 3 requests a second and 30 a minute.

**Is it the same data?** Checked before choosing it, on Baku 2026, where FastF1 had already downloaded the telemetry
(before the lock): for driver 63, every one of FastF1's 36,103 car samples and 37,092 position samples has an OpenF1
sample within 1 ms, and speed, rpm, gear, throttle, x, y and z are **identical**. (The time stamps differ by 0 or 1 ms
because the two round the millisecond differently.) So OpenF1 is a copy of the same feed, not an estimate.

Differences that matter: OpenF1 keeps the raw brake value (0 / 100 / 104) where FastF1 made it a yes/no; OpenF1 gives
no DRS value for 2026 cars (they have no DRS) where FastF1 wrote 0; OpenF1 starts in 2023; it is slower (rate limit).

## 4. The flow

    OpenF1 API ──openf1.py──> Bronze (as delivered)              ──cleaner.py──> Silver               ──> report
                              openf1_car_data   (per sample)     telemetry_clock  telemetry.parquet     telemetry_quality.csv
                              openf1_location   (per sample)     (t0, FastF1's    (10 Hz grid, every
                              openf1_laps       (per lap)         column names)    car, laps marked)
                              _manifest.json "openf1" entry

Bronze keeps what OpenF1 delivered, untouched. Silver turns it into the lake's shape: the session clock, one grid for
every car, holes marked. The quality report reads Silver (and a little of Bronze) and says how far to trust it.

Commands:

    python -m src.pipeline.backfill --season 2023 2024 2025 --telemetry   # download what is missing, build Silver
    python -m src.pipeline.openf1 --season 2024 --rounds 23               # only the download, for named rounds
    python -m src.pipeline.telemetry_quality --season 2023 2024 2025      # data/reports/telemetry_quality.csv

## 5. The clock problem: two estimates, one checking the other

OpenF1 stamps each sample with UTC ("2024-12-01T16:03:12.345000+00:00"). Our lake counts in session time (Bronze laps'
`LapStartTime`, `Time`, `PitInTime` ...). To put OpenF1's samples next to our laps we need **t0** for each race.

FastF1 works t0 out when it loads telemetry, but our Bronze was ingested laps only, so it never did (`LapStartDate` is
empty in all 70 races). What both sources record, for every driver, is when each lap started: OpenF1's lap table in UTC
(`date_start`), Bronze's in session time (`LapStartTime`). Each pair gives one estimate of t0 (`date_start -
LapStartTime`). Two ways of using them:

**Estimate 1, lap 1 (exact when sound).** The median over drivers of their lap-1 pair.

Example, Qatar 2024, Verstappen: OpenF1 says lap 1 started at 16:03:33.083 UTC; Bronze says 54 min 49.896 s of session
time. t0 = 16:03:33.083 - 0:54:49.896 = **15:08:43.187**. All 20 drivers give exactly the same t0. Measured on the four
2026 races where FastF1 knows the true t0 (Australia, Barcelona, Baku, Bahrain, all in FastF1's cache with telemetry):
1 ms off, for every driver of every race.

*The vote.* A few cars can carry a wrong lap 1. São Paulo 2023 (red flag on lap 1): the four cars caught in the crash
(Albon, Magnussen, Ricciardo, Piastri) have the restart as their lap-1 start, 27 minutes late; the 15 others agree to
the millisecond. Mexico 2023: Pérez, out at the first corner, 70 minutes away; 18 agree. Australia 2023: Leclerc, out on
lap 1. So drivers more than 5 ms from the median are **outliers** (named in the report), and lap 1 counts only if more
than half of the drivers, and at least 3, agree. A majority cannot be outvoted by a few wrong laps; a split vote is not
guessed.

**Estimate 2, the later laps (about ±0.05 s, never fooled by one lap).** OpenF1 stamps laps 2 onwards *later* than they
started (the stamp comes from a later message): measured over the 56 races of 2023-2025 where both estimates agree, by
**0.173 s** (median of the races' medians; race-to-race sd 0.025 s; by season 0.149 / 0.190 / 0.182). So t0 = median over
every lap from 2 on of (`date_start - LapStartTime`) - 0.173 s. A wrong lap shifts one pair out of ~1,000: the median
does not move.

**The rule.** Lap 1 when the two agree within 0.25 s (10 sd of the later laps' own scatter); the later laps when they do
not, or when lap 1 cannot vote. Why the check is needed: in some races **every** car's lap 1 is wrong in the same way, so
the vote cannot see it:

| Race | Lap 1 vs the later laps | Why |
|---|---|---|
| Japan 2024 | 28 min | red flag on lap 1: lap 1 dated from the restart in one source, from the start in the other |
| Monaco 2024 | 41 min | the same |
| Zandvoort 2023 | 2.1 s | (a wet start); unknown |
| Las Vegas 2024 | 0.45 s | unknown |
| Miami 2025 | - | OpenF1 has no lap-1 start (and only ~33 of 57 laps per car) |

**How we know the later laps are right there:** a check that uses neither estimate. The finish line is a fixed place on
the circuit map, the same every year. Where are the cars (x, y) at the moments the lap table says they crossed it?

| Race | On the lap-1 clock | On the later laps' clock | Compared with |
|---|---|---|---|
| Las Vegas 2024 | 35.6 m from the line | **4.3 m** | Las Vegas 2023 |
| Zandvoort 2023 | 174 m from the line | **7.2 m** | Zandvoort 2024 |

(4-7 m is the spread of the crossings themselves.) This check is now part of the quality report.

**Alternatives rejected:** the later laps alone (±0.05 s where lap 1 gives 1 ms, in 65 of 70 races); Race Control
messages (1-second resolution); the car's position at the line without another year to compare (circular: needs the
line, which needs the clock).

The clock refuses (`ClockError`, Silver not built) only when neither estimate is possible: no lap 1 that can vote and no
later lap. A wrong t0 would silently shift every sample of the race; a refusal is visible.

## 6. The files

### `src/pipeline/openf1.py` (Bronze: the download)

| Piece | What it does | Why this way |
|---|---|---|
| `find_session` | OpenF1's race session for our race: a session named "Race" in that year, not cancelled, starting (UTC) within a day of the manifest's local `event_date`; the circuit's `location` breaks a tie | By **date**, never by country: 2023 has two Italian races in OpenF1 (Imola, cancelled for the floods, and Monza). Within a day: Las Vegas races on Saturday night local time, which is Sunday in UTC |
| `drivers_of` | the car numbers in Bronze's laps | We want telemetry for exactly the cars that raced |
| `download` | per driver: car data and location; then the session's lap table. 3 at a time (threads) | ~5 MB and ~10 s per driver per stream: one request per driver keeps each answer small. Each answer becomes a table inside its worker thread: a race is ~1.4 million samples, as Python dicts that would be ~1 GB of memory |
| `add_telemetry` | runs the above and writes the three Bronze tables **as delivered** + the manifest's `"openf1"` entry (session key, drivers, who OpenF1 had nothing for, when) | **The manifest is written last**: a crash or a failed request leaves Bronze exactly as it was (`lake.bronze_telemetry_source` ignores tables without the manifest entry). A driver OpenF1 has no samples for is written down (`missing`), never filled in. A session with no car data at all is an error, not an empty table |
| `http_fetch` | one request: rows; "404 No results found" = no rows; 429 (too many requests) waits 60 s (up to 10 times); a dropped connection or a server error (5xx) retried after 10, 30, 60, 120 s; anything else raises `OpenF1Error` | Long downloads meet network blips; a refusal for a bad query should fail at once |
| `Pace` | spaces the *starts* of requests, across the threads, to 25 a minute | OpenF1's free tier: 30 a minute, 3 a second. 25 leaves room |
| CLI | `--season --rounds [--lake]` | Download only, for named races |

Measured: ~100 s a race (2 requests per driver + 1), ~690,000 car and ~700,000 position samples, ~8 MB of Bronze.

### `src/pipeline/lake.py` (where things live)

- `BRONZE_OPENF1_TABLES = ("openf1_car_data", "openf1_location", "openf1_laps")`, `FASTF1`, `OPENF1`.
- `bronze_telemetry_source(...)` -> `"fastf1"`, `"openf1"` or `None`. FastF1's own tables win when both exist (the
  four cached 2026 races). OpenF1's tables count only if the manifest has its `"openf1"` entry.
- `bronze_has_telemetry(...)` = either source (it used to mean "FastF1's tables").

### `src/pipeline/telemetry_clock.py` (Silver: the clock and the shape)

| Piece | What it does |
|---|---|
| `utc(stamps)` | OpenF1's ISO text -> UTC times without a time zone (like FastF1's `Date`), millisecond precision kept |
| `lap_offsets(openf1_laps, bronze_laps)` | one row per (driver, lap) in both tables: `offset = date_start - LapStartTime`. Car numbers paired as numbers (Bronze stores text, OpenF1 numbers) |
| `lap_one_vote(pairs)` | estimate 1: the median of the lap-1 offsets, if more than half (and at least 3) are within 5 ms of it; returns the drivers that agreed, their spread and the outliers |
| `later_laps(pairs)` | estimate 2: the median of the offsets of laps 2+ minus `LATER_LAP_LAG_SEC`; `None` without such laps |
| `session_offset(openf1_laps, bronze_laps)` | the rule above. Returns `(t0, report)`: `clock_method` ("lap 1" / "later laps"), `clock_check_sec` (lap 1 minus the later laps, ~0 when both are right), `clock_note` (why the later laps), `clock_drivers`, `clock_spread_ms`, `clock_outliers`, `clock_laps`, `t0_utc` |
| `_on_session_clock(table, t0)` | the columns every sample shares: `DriverNumber` (text), `Date` (UTC), `SessionTime = Date - t0` |
| `car_samples(car_data, t0)` | OpenF1's car data with FastF1's column names (`Speed`, `RPM`, `nGear`, `Throttle`, `Brake`, `DRS`) |
| `position_samples(location, t0)` | the same for `X`, `Y`, `Z` |
| constants | `NO_READING = 104`, `MAX_SPREAD_MS = 5`, `MIN_DRIVERS = 3`, `LATER_LAP_LAG_SEC = 0.173`, `MAX_DISAGREEMENT_SEC = 0.25`; `ClockError` |

**Why FastF1's column names?** Silver's resampler (`cleaner.resample_driver`) already reads them, and it is tested.
Renaming once at the border is safer than teaching the resampler a second vocabulary.

**The brake.** The raw values are 0 (off), 100 (on) and 104. 104 appears on exactly the same samples as throttle 104
(467 samples of Verstappen's Qatar race): it means the car sent no reading. FastF1 turns any non-zero brake into
"pressed", so FastF1-based telemetry calls those moments braking. Here: 0 = off, anything else = pressed, **104 =
unknown** (a nullable boolean). The throttle keeps 104; the resampler already treats throttle above 100 as missing.

**Gear and DRS** stay the raw codes, in a 16-bit integer: OpenF1's gear carries codes that are not gears (9-25, and 128
at walking pace in the pit lane: 2,249 of the samples of the first 55 races), which do not fit the 8-bit type a gear
needs; what is a gear is decided by Silver's resampler (below). DRS codes 10, 12 and 14 mean open (Silver's `drs_open`).
For 2026 cars OpenF1 gives no DRS value at all: they have no DRS, and an empty value says so (FastF1 wrote 0).

### `src/pipeline/cleaner.py` (Silver: resampling, unchanged except two points)

`build_silver(..., telemetry=True)` now asks `telemetry_tables(...)` for the two streams: FastF1's tables if Bronze has
them, otherwise OpenF1's put through `telemetry_clock`. `telemetry_clock` is imported only inside that branch, so
nothing else depends on it. Silver's report gains `telemetry_source` and the clock report.

What the resampler does (unchanged, worth knowing): for each driver it builds the grid of 0.1 s ticks from the start
of their lap 1 to the end of their last lap (the same instants for every car, so cars can be compared at the same
moment), then for each tick:

- continuous signals (speed, rpm, throttle, x, y, z) are **interpolated** between the samples either side;
- discrete signals (gear, brake, DRS) are **held** from the last sample (a gear is never "3.5");
- where the samples either side are more than 2 s apart, the values are left empty and `is_data_gap` is set
  (never invent a value across a hole);
- `lap_number` and `lap_elapsed_sec` come from Silver's laps (which lap the tick falls in);
- `lap_distance_m` = speed integrated over time, restarting at each lap.

Changed:

- the brake is held as a **nullable** boolean (it used to be forced to yes/no: no reading became "pressed");
- a gear code outside 0-8 (`GEAR_MAX`) is **unknown** before it is held (so the gear after a code 128 is not "128"
  for the next ticks, and Silver's contract, gear in 0-8, holds); the source samples that carried one are counted in
  Silver's report (`samples_not_a_gear`).

### `src/pipeline/backfill.py`

`--telemetry` on a race whose Bronze has its laps but no telemetry now downloads OpenF1's (`source` =
`"bronze+openf1"` in the season report) instead of asking FastF1 to download the race again (which now needs the login).

### `src/pipeline/telemetry_quality.py` (the report)

One row per race (`data/reports/telemetry_quality.csv`). Each check, and why it is a real check:

| Check | Measure | Limit | What it catches |
|---|---|---|---|
| coverage | drivers who completed a lap vs drivers with telemetry; the missing car numbers | none missing | a car OpenF1 lacks |
| clock | OpenF1 races: the clock's report (method, lap 1 vs the later laps, outliers), recomputed from Bronze | Silver refuses when no estimate is possible | a wrong session or lap 1 |
| holes | share of ticks inside holes on **racing laps**: not red-flag laps (while a race is stopped the cars sit in the pit lane and send nothing) and not the lap a car stopped on (its last lap without a lap time: Russell, Australia 2023, engine fire on lap 18, sends nothing for the 38 minutes his unfinished lap stays open). The share over all ticks is reported apart (`hole_share`); number of holes, longest | 0.1 % of racing ticks | feed dropouts |
| full laps | clean laps (Silver's `is_clean_lap`) with telemetry and no hole | 95 % of clean laps | the laps Phase 5 can measure |
| lap length | median distance of the full laps; share more than 2 % from it | 1 % of full laps | speed sensor faults, lap boundaries in the wrong place |
| speed vs position | distance from integrating speed / length of the path drawn by x, y; median over full laps | 0.99-1.01 | the two streams come from different sensors; if both are right they agree |
| line speed | the car's own speed at the tick nearest each lap end minus F1's timing-loop speed at the line (Bronze `SpeedFL`): the **offset** (median) and its **steadiness** (interquartile range, IQR) | IQR 5 km/h | a wrong clock makes it jump lap to lap |
| finish line | the median (x, y) of the cars at their lap ends (pit laps left out), measured from a landmark the clock cannot move: the **slowest point of the lap** (median over clean laps of where each lap is slowest, a hairpin or chicane; its laps lie a median 2.8 m from it). Across all races at the same circuit (`location`), each race's distance from the circuit's median (`line_offset_m`) | 15 m (~0.2 s at racing speed) | a clock that is right within the race but shifted: the check that caught Las Vegas 2024 and Zandvoort 2023. From the landmark, not on the bare map, because Suzuka's map itself moved about 30 m in 2025: its slowest corner by (-8.8, +30.1) m, its line by (-6.8, +27.3) m |
| no reading | share of ticks with the brake unknown | (reported) | |
| DRS open | share of ticks with DRS open | (reported) | 2026: zero |

A race outside any limit gets the reasons in `problems` ("" = all passed).

**The limits and the expectation.** Before measuring, the expectation (Claude's) was: holes under 0.1 % of ticks, 95 %
of clean laps complete, lap lengths within 2 % of the race's median, speed and position within 1 % of each other, the
line speed within 5 km/h. The last was **wrong**: on the 44 sound races of 2023-2024 the difference is a steady offset
per circuit, -1 to +12 km/h (the loop is not exactly on the line, and cars accelerate or brake there), varying only
0.9-2.8 km/h from lap to lap; where the clock was minutes out it varied by more than 100 km/h. So the limit is on the
steadiness, 5 km/h, **set after looking** (twice the worst sound race). The finish line was added after the clock errors
were found, and measured from the slowest corner once Suzuka's map was found to have moved. The holes limit was kept,
but on racing laps: red-flag stoppages and stopped cars are not dropouts.

## 7. The pipeline's rules, kept

1. **Each layer reads only the one before.** Bronze = OpenF1 as delivered; Silver reads Bronze; the report reads Silver
   and Bronze.
2. **Never invent a value.** Missing drivers are written down; holes stay empty; brake "no reading" stays unknown; 2026
   DRS stays empty.
3. **Never delete an inconvenient sample.** The resampler marks holes; nothing is dropped.
4. **Everything dropped is counted.** Drivers without telemetry are listed (Silver report `drivers_skipped`, the quality
   report `drivers_missing`).
5. **No path by hand.** Every path comes from `lake.py`.
6. Silver and Gold never import FastF1 (a test enforces it); `telemetry_clock` does not either.

## 8. Tests (no test touches the network)

| File | Tests | What they prove |
|---|---|---|
| `tests/pipeline/test_telemetry_clock.py` | 20 | t0 = FastF1's own t0 within 2 ms on the four 2026 races; a driver missing from one source is left out; a few wrong lap 1s are outvoted and named; a split vote or too few drivers stop the clock; the real red-flag races Mexico and São Paulo 2023 (Pérez; the four restart cars outvoted); lap 1 checked against the later laps on six real races (Qatar 2024 keeps lap 1; Japan, Monaco, Las Vegas 2024, Zandvoort 2023 and Miami 2025 use the later laps, t0 within 0.05 s of the whole race's); no lap 1 and no later lap refuses; Baku driver 63: every one of FastF1's 661 car and 670 position samples has its OpenF1 twin, on the same session time within 2 ms, with identical values; brake 104 is unknown, gear codes fit; Silver from OpenF1 on the Qatar fixture race: VER and NOR laps 2-6 within 2 % of Lusail's 5,419 m, no holes |
| `tests/pipeline/test_openf1.py` | 11 | sessions by date (cancelled Imola, Las Vegas a day later, two on a day told apart by place); written as delivered (42 requests, 18 drivers missing, recorded); a failed request leaves Bronze untouched; no car data at all is an error; FastF1's tables win, crash leftovers do not count; 404 / 429 / 5xx / dropped connections; the request pace |
| `tests/pipeline/test_telemetry_quality.py` | 10 | each measure by hand (runs of holes, red-flag laps and a stopped car's last lap are not racing, a lap measured by speed and by position, the tick nearest the line, the finish line, the slowest point, a race far from its circuit's line and a whole map that moved, the problems); Qatar with two cars on six laps: the report says exactly that |
| `tests/pipeline/test_silver_telemetry.py` | +1 | a gear code that is not a gear (128) is unknown, not held over |
| `tests/pipeline/test_backfill.py` | +1, 1 changed | `--telemetry` downloads from OpenF1 (not FastF1) and then builds Silver with telemetry |

Fixtures: `tests/fixtures/openf1/` (~450 KB), cut from the real downloads: lap 1 of the four 2026 races with FastF1's
t0; lap 1 of Mexico and São Paulo 2023; laps 1-15 of six races for the clock cases (Miami 2025 whole); Baku 2026 driver
63, three minutes, both sources; Qatar 2024 VER and NOR, laps 1-6, and OpenF1's whole lap table.

    python -m pytest tests/pipeline/test_telemetry_clock.py tests/pipeline/test_openf1.py tests/pipeline/test_telemetry_quality.py tests/pipeline/test_silver_telemetry.py -v

## 9. What was measured

**The download** (2026-10-25): 70 of 70 races, 118 minutes (96-172 s a race), no driver missing in any race. Per race
488,000-983,000 car samples and 496,000-1,008,000 position samples. OpenF1's Bronze tables: 0.59 GB for the 70 races.

**Silver:** 70 of 70 races built, 1,388 car-races, 74.0 million ticks, 2.08 GB; 5-7 s a race.

**What the first build found** (the clock as first written: lap 1 only, every driver within 5 ms). The safety checks
stopped every one of these; nothing wrong was written:

1. gear codes that are not gears in 10 races (Silver's contract failed in 8; code 128 did not fit an 8-bit integer in
   2): -> unknown gear in the resampler, raw codes as 16-bit integers in the clock;
2. a few wrong lap 1s in 3 races (Sao Paulo, Mexico, Australia 2023): -> the vote;
3. every car's lap 1 wrong in 5 races (found by comparing lap 1 with the later laps once the first build passed):
   -> the later laps as the check and the fallback.

**The clock, 70 races:** lap 1 in 65 (every agreeing driver at 0.0 ms from the others; lap 1 within 0.069 s of the later
laps in all of them), the later laps in 5 (Zandvoort 2023, Japan 2024, Monaco 2024, Las Vegas 2024, Miami 2025). Drivers
outvoted: Leclerc in Australia 2023, Perez in Mexico 2023, Ricciardo, Magnussen, Albon and Piastri in Sao Paulo 2023.

**The quality report, 70 races** (`data/reports/telemetry_quality.csv`): **63 pass every check.**

| Measure | Result |
|---|---|
| holes on racing laps | median 0.000 % of ticks, max 0.46 % (Zandvoort 2023); over all ticks max 1.59 % (Australia 2023) |
| clean laps complete | median 100 %, min 89.4 % (Zandvoort 2023) |
| lap length vs the official length | median -1.04 % (66 races with a known official length; -0.58 to -1.79 %; shortest on twisty circuits: Monaco -1.69, Mexico City -1.60, Sao Paulo -1.56: the official length follows the middle of the track, drivers cut the corners) |
| speed vs position | median 1.0019 (0.9974-1.0075) |
| line speed | offset -0.9 to +12.3 km/h per circuit; steadiness (IQR) median 1.5, max 4.3 km/h |
| finish line, seen from the slowest corner, against the circuit's other races | median 1.5 m, max 7.3 m |
| brake "no reading" | median 0.073 % of ticks |

The 7 races that do not pass, each a real fault of the source, not of the pipeline:

| Race | Problem | What it is |
|---|---|---|
| Zandvoort 2023 | holes 0.46 %, 89.4 % of clean laps complete | the feed went silent for all 20 cars at once (e.g. 6.4 s at 6,277 s), several times |
| Sao Paulo 2023 | holes 0.11 % | Tsunoda, 100 s without a sample on lap 3 (green) |
| Abu Dhabi 2023, Belgium 2024, Sao Paulo 2024 | holes 0.11-0.15 % (longest 5-6 s); Sao Paulo 2024 93.6 % of clean laps complete | short outages |
| Italy 2025 | 1.1 % of full laps off the median length | Albon's speed reads 0 on laps 2-10 (85 s laps measured as 0 m) |
| Abu Dhabi 2025 | 1.5 % of full laps off the median length | Antonelli's speed reads 2-5 % high on laps 2-12 and 22 |

**Against the expectation (written before measuring):**

| Expected | Measured | |
|---|---|---|
| 70 of 70 races, at most ~5 car-races missing | 70 of 70, none missing | right |
| holes under 0.1 % of ticks in most races | 65 of 70 (median 0.000 %) | right |
| a clean lap 0.5-2 % shorter than the official length | 0.58-1.79 % shorter | right |
| the lap-1 clock within 2 ms on every race | 65 of 70; 5 races needed the later laps, 3 had outvoted drivers | **wrong** |
| the line speed within 5 km/h | a steady offset per circuit of up to 12 km/h | **wrong** (the check now looks at its steadiness) |

## 10. Known limits and open points

- **OpenF1 is a community service** (non-commercial terms, 2023 onwards, 30 requests a minute). Bronze keeps what it
  delivered, so nothing in the lake depends on it after the download; new races need it (or F1 TV + FastF1 3.9.0).
- **The later-laps clock is good to about +-0.05 s** (5 races), the lap-1 clock to about 1 ms (65 races). 0.05 s is
  4 m at 300 km/h: fine for per-corner measures, not for anything that needs the line to the metre in those 5 races.
- **Gear codes 9-25 and 128** are unknown (their meaning is not documented; 128 appears at walking pace in the pit lane).
- **DRS:** OpenF1 also carries codes 8, 9, 11, 13 and 15; only 10, 12 and 14 count as open (FastF1's convention).
  Not yet checked against anything (step 5.5 will, against where the DRS zones are).
- **The throttle** is interpolated across a "no reading", like any missing sample; the brake is not (it is held).
- **A faulty sensor on one car** shows only as a race-level problem here (Italy and Abu Dhabi 2025). Steps 5.1-5.2 must
  judge each lap (its length against the field's, speed against position) and leave such laps out.
- **`lap_distance_m` is integrated speed**: about 1 % shorter than the official length, and it drifts within a lap.
  Step 5.1's track model gives a proper distance along a reference line.
- **The map of a circuit can move between years** (Suzuka 2025: about 30 m). Positions from different years are not in
  one frame; step 5.1 must align each race's map (for example on its own landmarks) before comparing years.
- **The timing loop at the line is not exactly on it**: the line speed has a steady offset per circuit (-0.9 to +12.3
  km/h); the loop's place is unknown.
- **OpenF1's lap table** is not complete everywhere (Miami 2025: about 33 of 57 laps per car); only the clock uses it.
- **The replay** still reads FastF1: it works only for races FastF1 has cached (four 2026 races). Moving it onto the
  lake's Silver telemetry is planned with step 5.8.
