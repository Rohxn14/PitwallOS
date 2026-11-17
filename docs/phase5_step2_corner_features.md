# Phase 5 step 5.2: what each car did in each corner

A developer's guide to how PitwallOS turns a race's telemetry into per-lap, per-corner numbers: speeds, braking points,
throttle pick-up, and the time spent in each part of the lap.

## Purpose

Step 5.1 gave every race a reference line and the place of each corner on it. This step reads every lap of every car
against that line and writes the numbers a race engineer looks at, corner by corner:

- how long the car spent in each corner's part of the lap;
- how slow it went, where, and how fast it came in and went out;
- where it started braking, and how soon after the slowest point it was back on the throttle;

and, per lap, its top speed and the shares of the lap at full throttle, on the brakes, and coasting.

Its consumers: 5.3 (two drivers compared corner by corner), 5.4 (does a tyre wearing out show in corner speeds before it
shows in the lap time?) and 5.7 (do any of these numbers improve the pace model?).

## The data

| Input | From | Used for |
|---|---|---|
| Silver telemetry | 10 Hz, every car: time, x, y, speed, throttle, brake, `lap_distance_m`, `is_data_gap` | everything |
| Silver laps | `lap_time_sec`, `is_clean_lap` | the check only (lap times are never used to make a feature) |
| Gold `track_line`, `track_corners` | step 5.1 | where each sample is; where each corner is |

Output, in Gold, per race:

| Table | One row per | Columns |
|---|---|---|
| `corner_features` | car, lap, corner | `corner_order` (1, 2, ... along the lap), `corner` (its name: "4", or "1A"), `corner_source` (`official` or `map`), `corner_distance_m`, `is_complete`, `segment_time_sec`, `corner_speed_kmh`, `min_speed_kmh`, `min_speed_distance_m`, `entry_speed_kmh`, `brake_point_m`, `exit_speed_kmh`, `throttle_on_m`, `full_throttle_share` |
| `lap_telemetry` | car, lap | `is_complete`, `top_speed_kmh`, `full_throttle_share`, `brake_share`, `coast_share`, `segments_time_sum_sec`, `telemetry_ticks`, `lap_time_sec`, `is_clean_lap` |

Both are checked against contracts before they are written (`contracts.GOLD_CORNER_FEATURES`, `GOLD_LAP_TELEMETRY`):
the types and ranges, one row per car, lap and corner, every feature null on a row that is not complete, speeds and a
time on every row that is, and the slowest speed never above the entry or exit speed.

The report `data/reports/corner_features.csv` has one row per race: corners, rows, clean laps, the share of clean laps'
corners that are complete, and how far the segment times of a lap are from its official lap time.

## How it works

### 1. Which corners (`corner_list`)

The official corners where the race has them (MultiViewer's list, placed on the line in 5.1), so "turn 4" means what
everyone means by it. MultiViewer sometimes adds a letter: the Hungaroring has turns 1 and 1A, 12 and 12A. The letter is
kept (`corner` = "1A"), and `corner_order` numbers the corners in the order they are driven. A race whose circuit has no
official list (two 2026 circuits) uses the bends 5.1 found on the map, numbered in order.

### 2. Where each sample is (`lap_on_line`, `track_model.project_lap`)

Each sample is placed on the line (its distance along it, in metres). 5.1's `project` takes the nearest point of the
line. That is wrong where a circuit passes close to itself: the two sides of a hairpin, a crossover, a straight next to
another. So `project_lap` does it in two steps:

1. of the line's points near the sample (the 24 nearest, no more than 30 m further than the nearest), the one whose
   distance is closest to where the car's OWN distance says it is (its share of the lap times the line's length)
   tells which stretch of the track the car is on;
2. the answer is the nearest point of the line on that stretch (within 60 m along the line of the one in step 1),
   placed on the stretch BETWEEN that point and its neighbour, not snapped to it (the line has a point every 5 m).

The car's own distance only chooses the stretch. It is 1 % short and drifts, so it must not choose the point.

Then the two measures of distance are combined (a **complementary filter**). The map is right on average but jumpy:
OpenF1 sends about 3.7 positions a second, so at 10 Hz a position repeats and can be a quarter of a second old. The car's
own distance (its speed added up) is smooth from one sample to the next but drifts over a lap. The distance used is the
car's own distance plus the map's correction to it, averaged over 3 s (`FUSE_TICKS = 31`).

Then, across the start/finish line: a sample just before the line at the start of a lap gets a small negative distance
(not 5,380 m), and a sample just past it at the end gets a little more than the length. Last, the distance is made to
never go backwards (a running maximum): one noisy position cannot move a car back down the track.

"Share of the lap" is the car's own distance divided by the LINE's length, not by the car's own last distance: a car that
stops halfway round has done half a lap, not a whole one.

### 3. The segments (`segment_edges`)

The lap is cut halfway between consecutive corners. Corner k's segment runs from halfway after corner k-1 to halfway
before corner k+1; the first starts at the start/finish line and the last ends there. Every metre belongs to exactly one
segment.

    line ---- T1 ---- | ---- T2 -- | -- T3 ------- | ------- T4 ---- line
         segment 1      segment 2    segment 3       segment 4

The time of a segment is the time between its two cuts, each cut's time interpolated between the two samples either
side of it. The first and the last cut are the official crossings of the timing line (Silver's `lap_start_sec`,
`lap_end_sec`, to the millisecond, on the telemetry's clock), so **a lap's segment times add up exactly to its official
lap time** (`lap_ends`). Where the official time is unknown, or more than 0.5 s from the telemetry's own lap (Miami 2025's
lap 1: 2,300 s of telemetry for a 92 s lap), the lap's first sample and its last plus a tick are used.

**The check** is therefore on the cuts INSIDE the lap, against the **official sector times**, which nothing else here reads
(`sector_check`). On every clean lap, where was the car on our map when the official sector 1 (and 2) ended? The sector
line is where it lands on most laps (the median); a lap's error is how far from it the car was, turned into time by its
speed there.

### 4. The features of one segment (`segment_features`)

On the samples inside the segment:

| Feature | Rule |
|---|---|
| `min_speed_kmh`, `min_speed_distance_m` | the slowest sample, and its place |
| `corner_speed_kmh` | the speed at the corner's own marker |
| `brake_point_m` | the last unbroken run on the brakes that ends at or before the slowest sample: how far before the slowest sample it began. Null: no brakes in the segment before the slowest point |
| `entry_speed_kmh` | the speed where that braking began; without braking, 100 m before the slowest point (or at the segment's start) |
| `exit_speed_kmh` | 100 m after the slowest point (or at the segment's end) |
| `throttle_on_m` | how far after the slowest point the throttle first reaches 50 % |
| `full_throttle_share` | the share of the segment's samples at 99 % throttle or more (unknown readings left out) |

Why both `min_speed_kmh` and `corner_speed_kmh`: in a corner the cars brake for, the slowest point is the apex, a few
metres from the marker (median +10 m in corners taken under 100 km/h). In a corner taken flat out, the car is still
accelerating out of the corner before, so the slowest point of the segment is its START (a median 60 to 105 m before the
marker in corners over 200 km/h). The minimum speed there is not a corner speed; the speed at the marker is.

### 5. Holes (`lap_features`)

A segment with a telemetry hole in it (Silver's `is_data_gap`), a sample without a position, or fewer than 3 samples (a car
that stopped before reaching it) keeps its row, with `is_complete` False and every feature null, its time too. A lap with
any hole has `is_complete` False and null lap features. Nothing is filled in (pipeline rule 2) and no row is dropped (rule 3).

### 6. The lap (`lap_features`)

Top speed (highest sample), full-throttle share (99 % or more), brake share, coast share (throttle under 5 % and no
brake: "lift and coast"), and the sum of the segment times. Each share is of the samples whose reading is known.

## Decisions and the evidence behind them

| Decision | Alternatives | Why |
|---|---|---|
| Official corners, letters kept | the map's bends; numbers only | "turn 4" must mean turn 4 to a reader. Numbers only made two rows "turn 1" at the Hungaroring: the contract refused all four Hungarian races until the letter was kept |
| Cut halfway between corners | a fixed window round each corner (say 150 m each side); braking zone to exit | halfway cuts cover the lap exactly once, so segment times add up to the lap (with the official crossings as its ends, exactly). Fixed windows overlap at chicanes (35 % of official corners are within 150 m of the next) and leave straights out |
| Lap ends at the official line crossings | the lap's first sample and its last plus a tick (the first version); the moment worked out from distance and speed | the samples round each end to a tick: up to 0.05 s in the first and the last corner. Working the crossing out from position was worse (0.043-0.053 s spread against the official crossings, 0.029 s for the rounding): positions are stale |
| Distance fused with the car's own (3 s) | the map alone; the car's own distance alone | the map's positions repeat (3.7 Hz); the car's own distance drifts 1 %. Fused: cuts 14-20 ms from official sector times instead of 22-34 ms on 8 races; 1 s and 6 s also tried, 3 s kept as proposed before the run |
| Checked against official sector times | segment sums against official lap times | with official ends the sums equal the lap time by construction; the sectors test the cuts inside the lap, which a corner comparison depends on |
| Guided projection (`project_lap`) | the nearest point (5.1's `project`); the car's own distance | the nearest point jumps to the other side of a hairpin (a test builds one: 1,500 m wrong); the car's own distance is 1 % short and drifts. Measured: the end of official sector 1 lands at the same metre of the line on every clean lap with sd 3-5 m in 8 races, where the car's own distance spreads 3-6 m and up to 199 m (Zandvoort 2025, one bad lap). A sample at 300 km/h covers 8 m |
| Distances never go backwards | raw projections | segment cuts need distance to grow with time; a car's position jitters by a metre or two |
| Brake point from the last braking run before the slowest point | the first brake sample in the segment | a short dab on the brakes earlier in a long segment would move the "brake point" hundreds of metres |
| Both `min_speed_kmh` and `corner_speed_kmh` | one of them | measured above: in fast corners the minimum is the segment's start, 60-105 m before the corner |
| Entry and exit kept inside the segment | 100 m either side wherever it falls | the first run reached into the previous corner's segment, where the car was slower than at this "apex": the contract (minimum above entry speed) refused the race |
| Incomplete rows kept, features null | drop them; interpolate across holes | rules 2 and 3. A consumer can count them; the report does |
| Share of lap from the line's length | from the car's own last distance | a car that stops halfway (a retirement, positions that end early) looked like a whole lap: its last samples were put past the line |

## Measured

89 races: every race with a track model (all 90 but Hungary 2026, whose line 5.1 refuses). 98,148 laps, 1,621,924
corner rows, 53 MB of Gold. 87 races on official corners, 2 on the map's bends (Madrid 2026 and Bahrain's second 2026
date, which MultiViewer has no list for). About 5 s a race.

**The check: where the cuts land in time, against the official sector times** (clean complete laps, sectors 1 and 2):

| | Median of races | Middle half of races | Worst race |
|---|---|---|---|
| a typical lap's error | 20 ms | 17 to 24 ms | 70 ms (Austria 2026) |
| the worst 5 % of laps | 70 ms | 56 to 93 ms | 257 ms (Austria 2026) |

Austria 2026 is the race whose winner's finish is also 0.32 s from the official time (found during the move to OpenF1,
unexplained): a fault in that race's timing, not in this step.

**What the repair bought** (found by step 5.3, on 8 races against official sector times; typical / worst 5 %):

| Version | Error |
|---|---|
| first version: snapped to the line's points, lap pinned to its samples | 26-47 / 92-133 ms |
| placed between the line's points | 22-40 / 90-126 ms |
| fused with the car's own distance over 3 s | 14-36 / 40-95 ms |

The first version's own check (segment sums against official lap times, a median 0.029 s) measured the rounding of the
lap's ends to a 0.1 s tick, and this guide had called it "the 10 Hz grid, not the method". It was the method: the first
and the last corner carried up to 0.05 s of that rounding each, as much as the gap between teammates in a corner.

**Complete features on clean laps: 99.9 %** (81,760 of 82,051 laps). 88 of 89 races at 95 % or more, 86 at 99 %. The
lowest: Monaco 2026, 90 %: OpenF1's positions stop 45 minutes early there (the 5.0 quality report flagged it).

**Where samples land.** The end of official sector 1, put on the line, on every clean lap of 8 races (600 to 1,170 laps
each): spread (sd) 3.0 to 5.0 m. Sector 2: 3.2 to 3.9 m, except Zandvoort 2025 (sd 82 m from a few laps, robust spread
4.5 m). A sample at 300 km/h covers 8 m.

**What the corners look like**, 1,490 corners (each race's median over clean laps), by the speed at the corner:

| Speed at the corner | Corners | Braked for by most cars | Median brake point | Median throttle pick-up | Slowest point vs the marker |
|---|---|---|---|---|---|
| under 100 km/h | 272 | 92 % | 94 m | 20 m after | 10 m after |
| 100-150 | 401 | 75 % | 99 m | 13 m after | 14 m after |
| 150-200 | 290 | 54 % | 78 m | at once | 5 m after |
| 200-250 | 290 | 24 % | 55 m | at once | 61 m BEFORE |
| over 250 | 237 | 15 % | 41 m | at once | 105 m BEFORE |

The last column is why `corner_speed_kmh` exists. Slow corners not braked for by most cars are mostly the second half of a
chicane or a pair: the braking happened in the corner before.

**An example a reader can check, Qatar 2024, Verstappen against Norris** (medians of clean laps):

| | VER | NOR |
|---|---|---|
| lap time | 84.748 s | 84.938 s |
| top speed | 302 km/h | 304 km/h |
| full throttle | 61 % of the lap | 56 % |
| coasting | 2.8 % | 4.2 % |

Verstappen gains 0.03 to 0.08 s in each of turns 1 to 7 and 15 (the slower corners and the first sector); Norris gains
0.03 to 0.04 s in turns 8, 11, 12, 13 and 16 (the fast ones). Median differences per corner add up to 0.30 s, not the
lap's 0.19 s: medians do not add. (Step 5.3 compares PAIRED laps, lap N with lap N: there Verstappen is 0.03 s a lap
faster (median), 0.084 s (mean). The 0.19 s above mixes laps from different points of the race.)

**The expectations, written before measuring.**

| Who | Expected | Measured |
|---|---|---|
| Claude | segment sums within a median 0.05 s of the official lap time | 0.029 s (worst race 0.033 s); later found to be the lap's rounded ends, see above. The ends are now the official crossings |
| Claude | complete features on over 95 % of clean laps | 99.9 % (88 of 89 races over 95 %) |

## Tests

`tests/telemetry/test_corner_features.py`, 17 tests (four added by the repair: stale positions fused, official ends only where they agree, the sector check, a sample between two points of the line):

- a lap driven by hand round 5.1's made-up rectangle, simulated at 1 ms and sampled at 10 Hz: 288 km/h on the straights,
  braking over 150 m to 108 km/h at three corners, only a lift at the third, throttle at 20 % for 30 m after each apex.
  Every feature has a known answer: minimum and corner speeds, the slowest point's place, brake points (150 m), entry
  (288 km/h; without brakes 228 km/h), exit speeds, throttle pick-up (30 m), full-throttle shares worked out by hand;
- segment times against the 1 ms simulation, adding up to the lap within a tick; lap top speed, brake and coast shares;
- a hole in one corner empties that corner and no other; a sample without a position is a hole too;
- a car stopping after the first corner has only that corner;
- the distance across the line at both ends, never backwards;
- where two stretches run side by side (a made-up hairpin), a sample lands on the one the car is on, and a whole lap 6 m
  off-centre is right on its own distance 1 % short;
- the cuts, the corner list (official, letters, the map fallback);
- the golden race, Qatar 2024 from OpenF1 (the shared test lake: VER and NOR, laps 1-7): segment times of the clean laps
  equal the official lap times within a tick, turn 6 is the slowest corner (80-110 km/h), both brake for turns 1, 2, 4, 6,
  7, 10 and 16 and not for 3, 5, 8, 11, 12, 13; lap 7, whose positions stop early in the fixture, is left incomplete;
- a race whose map 5.1 refuses keeps no old map and no corner tables (the Hungary 2026 case, below).

The module was broken in 19 ways (cuts at the corners, the fastest point as the apex, the brake flag inverted, the brake
point at the last brake sample, holes ignored, a missing position not a hole, exit before the apex, any throttle as
"picked up", the lap's ends not pinned, the map instead of the official list, full throttle inverted, coasting as "no
throttle OR no brake", distances allowed backwards, no unwrap across the line, entry reaching anywhere, top speed as a mean,
the share of lap from the lap's own last distance, and the projection as the plain nearest point or as the car's own
distance). Each break made a test fail; three needed a test first. After the repair: 23 breaks (official ends ignored or used when they disagree, no fusion, positions snapped to the line's points, the sector error left in metres), all caught; snapping needed a test first.

## Known limits

- **Corner numbers follow MultiViewer.** A chicane is two corners and two segments; a long spiral can be one.
- **Positions at 3.7 Hz, samples at 10 Hz.** A cut inside the lap lands about 20 ms from the truth (worst 5 %: 70 ms);
  a braking point is known to about one sample (8 m at 300 km/h). Single laps are noisy; compare many.
- **The sector check learns where the sector line is from the same laps** (the median): a bias shared by every lap would
  not show.
- **Brake is on/off** (OpenF1 gives 0 or 100), so there is no braking pressure, only where braking starts and stops.
- **`min_speed_kmh` is not a corner speed in fast corners** (see above): use `corner_speed_kmh` there.
- **Not yet used by anything.** 5.3, 5.4 and 5.7 are its consumers; per the roadmap a feature that does not improve the
  pace model is dropped.
- **Each race on its own line.** Comparing a driver across years at the same corner needs aligned lines (not built).
- **Hungary 2026 has no corner features**: 5.1 refuses its line. Before this step, an old line written before 5.1's
  contract existed was still in Gold, and this step built on it (turns 12A and 13 came out 5 m apart). Now a refused
  track model removes the race's old map (`lake.remove_table`), and this step refuses a race without a map and removes
  its own old tables.

## How to run it

    python -m src.telemetry.track_model --season 2023 2024 2025 2026        # step 5.1 first (~2 min)
    python -m src.telemetry.corner_features --season 2023 2024 2025 2026    # ~5 s a race; data/reports/corner_features.csv
    python -m pytest tests/telemetry -q

From Python:

    from src.pipeline import lake
    from src.telemetry import corner_features as cf
    f = lake.read_table(lake.table_path(lake.GOLD, cf.CORNER_TABLE, 2024, 23))      # every car, lap, corner
    t = lake.read_table(lake.table_path(lake.GOLD, cf.LAP_TABLE, 2024, 23))         # every car, lap
