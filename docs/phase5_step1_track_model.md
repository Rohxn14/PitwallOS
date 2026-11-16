# Phase 5 step 5.1: the track model

A developer's guide to how PitwallOS builds a map of each circuit from telemetry, and finds its corners.

## Purpose

Every telemetry question that follows is a question about a place on the track: how fast a driver takes turn 4, where he
brakes for turn 1, where a worn tyre first costs time. Silver telemetry cannot answer "where" well: its `lap_distance_m`
is each car's own speed added up over time, about 1 % short, drifting, and different for every car and lap.

This step builds, for each race:

- **the reference line**: one path round the circuit, the same for every car, with a distance along it;
- **the corners**: where they are on that line, found from the data alone;
- **the shared lap distance**: any point of any lap placed on that line.

It is checked against the official corner list of every circuit.

## The data

| Input | From | Used for |
|---|---|---|
| Silver telemetry | 10 Hz, every car: x, y, speed, lap number, `lap_distance_m`, `is_data_gap` | the line, its speed profile |
| Silver laps | `is_clean_lap` | which laps to trust |
| the official corners | MultiViewer, `src/lib/race_calendar.circuit` (kept in `data/reference/`) | the answer key only |

The official corners are in the telemetry's own coordinates: their median distance from our line is 2 m at Lusail.

Output, in Gold, per race:

| Table | One row per | Columns |
|---|---|---|
| `track_line` | point of the line, every ~5 m | `distance_m`, `x`, `y`, `speed_kmh` (median of all cars), `curvature` (1/m, + = left), `laps` |
| `track_corners` | corner, per method | `distance_m`, `x`, `y`, `speed_kmh`, `curvature`, `method` (`speed`, `map`, `both`, `official`), `number` (official only) |

Both are checked against a data contract before they are written (`contracts.GOLD_TRACK_LINE`, `GOLD_TRACK_CORNERS`).
The report `data/reports/track_model.csv` has one row per race with every score.

## How it works

### 1. Good laps (`good_laps`)

A lap builds the line if it is a clean lap (green, no pit, not lap 1), its telemetry has no hole and no missing position,
and its length is within 2 % of the race's median. Qatar 2024: 660 laps. At most 150 are used, spread over the race.

### 2. The reference line (`reference_line`)

Each lap is put on the same grid by its **share of the lap done** (0 at the line, 1 back at it), not by metres. A car
whose own distance is 1 % short still has its turn 4 at the same share. At each of about 1,000 points, the line is the
**median** of all laps' x and y. A median, not a mean: one car running 20 m wide for a stretch does not move it.

Then, round the closed loop (the last point joins the first):

- the distance along the line, from the start/finish line;
- the median speed, smoothed over 30 m;
- the curvature: how fast the direction changes per metre, from the path smoothed over 30 m.

### 3. Finding corners, three ways

| Method | A corner is | Sees | Cannot see |
|---|---|---|---|
| **speed** (`corners_from_speed`), the baseline | a minimum of the median speed at least 10 km/h below the highest speed on each side before a lower minimum | every place the cars slow down for | a corner taken flat out; one of two corners taken in one braking |
| **map** (`corners_from_map`) | a bend: a stretch curving one way tighter than 600 m radius, turning 12° or more in all | every bend of the path, fast or slow | a corner whose bend is too gentle |
| **both** (`corners_from_both`) | every bend of the map, plus speed minima no bend is within 40 m of | both | |

Two refinements of the map method, each found on real data:

- **A double apex is two corners.** Two corners turning the same way in a row form one unbroken bend. Where a bend's
  curvature has two peaks with a dip below half of the smaller one between them, and each part turns at least 12°, it is
  two corners. (`_apexes`, `SPLIT_DIP = 0.5`.)
- **A corner is at the middle of its tightest stretch**, the stretch within 10 % of its tightest curvature (likewise the
  middle of a flat-bottomed speed minimum), not at its edge. (`_middle_of_flat`.)

### 4. Scoring against the official list (`official_corners`, `score`)

Each official corner is placed on the line (the nearest point). Found and official corners are paired closest first,
within 75 m along the line (the short way round, across the start/finish line), each used once. A race scores: found,
matched, missed (with the official numbers), extra, `exact_count` (as many found as official), `all_right` (exact and
every one matched), and the median distance of the matched pairs.

### 5. The shared lap distance (`project`)

Any point (x, y) gets the distance of the nearest point of the line (a k-d tree, so a whole race's telemetry takes a
second). Every car, every lap, gets the same distance at the same place. Step 5.2 uses it for per-corner features.

## Decisions and the evidence behind them

| Decision | Alternatives | Why |
|---|---|---|
| One line per race, not per circuit | one map per circuit, all years | Suzuka's map moved by about 30 m in 2025; the official list is per year. A race's own line needs no alignment. Comparing years is a later step |
| Grid by share of lap, not metres | resample on each car's own distance | each car's distance is short by its own amount; shares line up turn 4 with turn 4 |
| Median of laps | mean | a mean follows a car off the track; measured: the made-up test with one car 20 m wide moves the mean, not the median |
| Thresholds (600 m, 12°, 10 km/h, 30 m) set before scoring, not fitted | tune them to the answer key | a sweep of 36 settings on 2023-2024 found no setting better than ours by more than 0.5 %. Tuning to the key would only fit it |
| The double-apex split | none | chosen on 2023-2024 (+37 corners matched, +5 extra), confirmed on 2025-2026, which it was not tuned on (+31, +8) |
| A race with broken positions fails | write the line anyway | Hungary 2026: OpenF1's positions stand still and jump; the contract refuses the line. The telemetry quality report had flagged that race already |

## Measured

87 races scored: all 90 except Hungary 2026 (refused, above) and two 2026 circuits MultiViewer has no 2026 entry for
(Madrid, Bahrain's second date).

**The line.** Its length is a median 0.12 % shorter than MultiViewer's outline of the circuit (5 % to 95 % of races: -0.64 %
to +0.01 %). Silver's own lap distance is about 1 % short. On 200 laps of Qatar 2024 projected onto the line, the distance
goes backwards on 0.04 % of samples; every lap has the same length (5,383 m), where Silver's own lap lengths spread by
9.7 m (sd).

**The corners**, 1,451 official corners in 87 races:

| Method | Matched | Extra | Exact count | Every corner right | Median placement error |
|---|---|---|---|---|---|
| speed | 707 (49 %) | 1 | 0 races | 0 | 19.6 m |
| map | 1,283 (88 %) | 91 | 11 races | 8 | 9.7 m |
| both | 1,283 (88 %) | 120 | 13 races | 7 | 9.7 m |

By how fast the corner is taken (map method):

| Speed at the official corner | Found |
|---|---|
| under 150 km/h | 647 of 650 (99.5 %) |
| 150 to 200 | 87 % |
| 200 to 250 | 79 % |
| 250 to 300 | 83 % |
| over 300 | 81 % |

By circuit (2023-2025, 24 circuits): every official corner found in every year at Monza, Spa, Melbourne and Zandvoort
(Melbourne and Zandvoort with extras, so not the exact count); exact count in at least one year at 3 circuits (Monza, Spa,
Montréal). Hardest: São Paulo (67 % of corners found), Shanghai (69 %), Las Vegas (75 %), Austin (83 %). Most extras: Monaco
(18 over three years), Melbourne (15), Suzuka (12): bends the official list does not number.

**The expectations, written before measuring.**

| Who | Expected | Measured |
|---|---|---|
| Rohan | the exact official count on "pretty much all" of 24 circuits (more than 18) | 3 circuits in at least one year, 2 in every year |
| Claude | fewer than 10 exact from speed alone | 0 |
| Claude | more with the map's curvature | 3 circuits, 11 races: more, but few |
| Claude | nearly every corner where cars really brake found | 99.5 % of corners taken under 150 km/h |

Why an exact count is rare: the official list numbers things that are not one place a car turns. A chicane is two
corners 50 to 100 m apart (35 % of official corners are within 150 m of the next one); a long spiral is one or three
numbers depending on the circuit; a flat-out kink is numbered at some circuits and not others. A method that gets 88 % of
corners within 10 m of the marker gets the count exactly right at few circuits, because one merged chicane or one extra
kink anywhere breaks it.

## Tests

`tests/telemetry/test_track_model.py`, 10 tests:

- made-up circuits drawn from straights and arcs, with every corner's place known: a rectangle with four round corners
  (found both ways within 15 m; the line as long as the path within 0.3 %), the same with two double apexes (six corners on
  the map, four for the speed), the same with a slight S on each straight (no corner);
- one car running 20 m wide does not move the line;
- a single tight bend is never split by its own noise;
- the shared distance of points on and five metres off the line;
- scoring: closest first, each corner once, across the start/finish line;
- which laps are good;
- the golden race: Qatar 2024 from OpenF1 (two cars, one good lap each) against Lusail's official corners
  (`tests/fixtures/multiviewer/circuit_150_2024.json`): the line within 1 % of the official 5,419 m, at least 14 of 16
  corners from the map, no extra from the speed.

The module was broken in nine ways (no double-apex split, the edge of a flat apex, laps not normalised, a mean instead of
the median, holes allowed, no wrap across the line, an official corner used twice, the speed drop and the minimum turn
ignored); each break made a test fail (two needed a test added first).

## Known limits

- The line is per race; nothing yet aligns two years of a circuit.
- Corners on long same-direction curves (Shanghai turns 1 to 3) and fast chicanes are the misses.
- A race whose positions are faulty gets no line (Hungary 2026).
- DRS zones, braking zones, straights and the pit lane are not in this step (they were in the plan; 5.2 will need the
  braking points and can find them from the same line).
- The shared distance is not written into Silver: 5.2 projects what it needs.

## How to run it

    python -m src.telemetry.track_model --season 2023 2024 2025 2026        # about 2 minutes; data/reports/track_model.csv
    python -m pytest tests/telemetry -q

From Python:

    from src.telemetry import track_model as tm
    from src.pipeline import lake
    line = lake.read_table(lake.table_path(lake.GOLD, tm.LINE_TABLE, 2024, 23))
    tm.project(line, x, y)                     # shared lap distances of any points
