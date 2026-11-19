# Phase 5 step 5.3: two drivers compared, corner by corner

A developer's guide to how PitwallOS compares two drivers of one race corner by corner, and to the Analyze window's
Compare tab that shows it.

## Purpose

Pick two drivers. Where on the lap does one gain time on the other, and how: braking later, carrying more speed through
the corner, leaving it faster, getting back on the throttle sooner? Teammates are the natural pair (the same car, so the
difference is mostly the driver), but any two drivers can be compared.

This step is also the first consumer of step 5.2's corner tables, and it found two flaws in them, fixed in 5.2 (its
guide, "What the repair bought").

## The data

| Input | From | Used for |
|---|---|---|
| Gold `corner_features` | step 5.2: per car, lap, corner | segment times, braking point, corner and exit speed, throttle |
| Gold `lap_telemetry` | step 5.2: per car, lap | which laps are clean and whole; official lap times |
| Gold `lap_features` | `gap_on_track_start_sec` | laps spent following a car |
| Silver `laps` | `tyre_life` | the tyre-age difference pairing does not remove |

Output: a `Comparison` (in memory, for the tab) and the season report `data/reports/teammates.csv` (one row per
teammate pair per race, from the faster driver's side).

## How it works (`src/telemetry/compare.py`)

### 1. Paired laps (`paired_laps`)

Only laps BOTH drivers ran clean (green, not lap 1, no pit stop) and whole (no telemetry hole), compared lap N with lap
N. Both carry the same fuel, on the same track, at the same point of the race. This was Rohan's choice, over each
driver's fastest lap (one noisy lap each, possibly at very different fuel loads) and all clean laps unpaired (laps from
different points of the race mixed).

What pairing does NOT remove is said, never hidden: the tyre-age difference (`tyre_age_gap`, median over the paired
laps) and the laps each driver spent following a car (`following`).

### 2. Gaps (`compare`)

On every paired lap and every corner: A's segment time minus B's. **Negative = A faster.** Then per corner:

| Column | Meaning |
|---|---|
| `gap_median_sec` | a typical lap: what the chart's bars and the table show |
| `gap_mean_sec` | adds up: the corners' means make exactly the mean lap gap (a lap's segments add up to the lap, step 5.2) |
| `a_faster_share` | on what share of paired laps A was faster there |
| `<metric>_diff` | median A minus B of braking point, corner speed, exit speed, throttle pick-up |
| `<metric>_a_better`, `_b_better` | on what share of paired laps each driver had the better value |
| `winner` | A, B, or "" (level) |
| `how` | the differences the winner has on at least 2 of 3 paired laps, in words ("brakes 8 m later, 2 km/h faster at the corner"); otherwise "no single difference" |

Two summaries on purpose: a lock-up costing 0.6 s on one lap of three moves the mean by 0.2 s and leaves the median
where it was. A test does exactly that.

The **build-up** is, on each paired lap, the running total of the corner gaps; at each corner, the median of those
running totals and their middle half. It ends at the median lap gap. (It is not the sum of the bars: medians do not
add up.)

Lap gaps come from the official lap times (to the millisecond).

### 3. One race for the tab (`load_race`, `compare_race`)

The tab reads a race's tables once (`RaceCorners`); each new pair is computed from them in a fraction of a second. When
the tables are missing, `load_race` raises `NotComparable` saying why and what builds them: a race in the lake laps only
(the Analyze window builds a missing race that way), a race whose track model 5.1 refuses (Hungary 2026), or corner
features not built.

### 4. Every teammate pair (`teammates`, `race_teammates`, `main`)

Teams with exactly two drivers in the race; each pair said from the faster driver's side (on the mean paired lap).

## The Compare tab (`src/gui/compare_tab.py`, `compare_charts.py`)

- Two driver lists, **no default pair** (Rohan's choice: the tab does not suggest a story), and a Swap button (every
  sign flips). Nothing is shown until two different drivers are picked.
- A summary line: paired laps, who is faster by how much (median and mean), corners won, the seconds made in corners
  under 150 km/h, the tyre-age difference, the laps each spent following a car, and any caution (few paired laps; tyres
  5+ laps apart; one driver following on 25 % more of the laps than the other).
- One chart, two rows sharing the x axis (the corners in the order driven), so each corner's bar sits under the point
  where it moved the gap:
  - top: the build-up, the median as a line, the middle half of the laps as a band, the area between the line and zero
    tinted in the colour of whoever is ahead;
  - bottom: each corner's median gap, a bar in the colour of the corner's winner; the biggest gain of each driver
    labelled.
- A table of every corner: speed, median and mean gap, A faster on what share of laps, winner, how.
- Colours: A blue `#3987E5`, B orange `#D95926` (the reference palette's first two slots, dark steps), validated on the
  window's panel `#1A1A1A`: normal-vision ΔE 31.8, colour-blind 26.8, both 3:1 or more against the panel. Identity is
  never colour alone: a legend names both drivers; tooltips and the table repeat every number.

## Corner tables for every new race (`src/telemetry/build.py`)

`build_race_telemetry(year, round, session, root)`: the track model, then the corner features, for one race. The runner
(`run_pipeline`) calls it after Gold whenever it builds telemetry; backfill calls it when it (re)built a race's Silver
or when the corner tables are missing. It never raises: a race whose positions are faulty keeps its laps and features,
and the summary says why it has no corner tables. If the track model fails for any reason, the race's old track and
corner tables are removed: Gold's corner tables are always built from the current Silver telemetry, or absent.

The official corner list comes from `track_model.circuit_for` (MultiViewer, kept in `data/reference/`; offline, the
lake's copy). Without it, the map's bends are numbered instead, and the summary says why.

## Decisions and the evidence behind them

| Decision | Alternatives | Why |
|---|---|---|
| Paired laps | each driver's fastest lap; all clean laps unpaired | Rohan's choice; pairing cancels fuel and track evolution. Qatar 2024 shows what it changes: VER 0.03 s a lap faster than NOR on paired laps (median), not the 0.19 s of each driver's median over all his clean laps |
| Laps spent following a car kept, and counted | only laps where neither follows; only laps where both or neither follow | measured on 2023-2024 (430 pair-races): a real gap carries over to the next race, so the better choice agrees better with the pair's next race. All paired laps: rank correlation 0.33, same faster driver 62 %; neither following: 0.29, 63 % (and 80 fewer usable pair-races); both or neither: 0.32, 66 %. Differences within luck (~1 standard error); filtering halves the laps for nothing measurable |
| Median shown, mean adds up | one of them | a lock-up moves a mean, not a median; only means share out a lap gap |
| "How" only at 2 of 3 laps | the median difference alone | a median difference of 3 m on a 10 Hz grid is noise; consistency over laps is evidence. Judgement, not measured |
| One figure, two rows, shared x | two figures | the bars line up exactly under the build-up; two figures drift by the width of their axis labels |
| x = corners in driving order, evenly spaced | distance along the lap | corner labels never collide (chicanes 50 m apart would); the bars and the build-up share positions |
| No default pair | the winner and his teammate | Rohan's choice |
| Corner tables built by the pipeline | the two commands by hand | a race added later would have no Compare tab otherwise; every phase must have a consumer |

## Measured

**Every teammate pair**, 2023-2026: 847 pairs in 89 races; 2023-2025 with 10+ paired laps: 626.

| Gap between teammates (paired laps) | Pairs | Corners of 16 won by the faster | Share of the gap from corners under 150 km/h |
|---|---|---|---|
| under 0.05 s | 59 | 8.0 | 75 % |
| 0.05-0.10 s | 60 | 8.8 | 99 % |
| 0.10-0.20 s | 111 | 9.3 | 67 % |
| 0.20-0.30 s | 78 | 10.3 | 58 % |
| 0.30-0.50 s | 141 | 10.9 | 67 % |
| over 0.50 s | 174 | 12.6 | 63 % |

Corners under 150 km/h are 43 % of corners and hold about two thirds of the gap (median 65 % in pairs 0.1 s or more
apart). 130 of 626 pairs ran tyres 5 or more laps apart in age.

**The expectations, written before measuring.**

| Who | Expected | Measured |
|---|---|---|
| Rohan | a teammate 0.2 s faster wins about 11 of 16 corners | 9.6 (pairs 0.1-0.3 s apart; middle half 8.5-10.9) |
| Claude | 10 to 11 | 9.6: both a little high. A 0.2 s gap is spread thin |
| Claude | corners under 150 km/h hold more than half of the gap | 65 % typical; less than half in 39 % of pairs |
| Claude | (Hungary 2024) traffic explains why Piastri comes out faster than Norris | wrong: on the 50 laps neither followed a car, Piastri is still 0.065 s faster. And "Norris was faster" was the race's story, not a measurement |

**One race says little.** The same driver is faster again in the pair's next race only 62 % of the time (2023-2024),
whichever laps are used.

## Tests

- `tests/telemetry/test_compare.py` (11): a made-up race where every answer is known (A gains 0.1 s in corner 1 by
  braking 10 m later on every lap; B gains 0.05 s in corner 3 carrying 4 km/h more; lap 4 not clean for B): paired laps,
  gaps, winners, "how", means adding up, the build-up, a lock-up moving the mean and not the median, swapping sides, a
  difference on only some laps not called "how", refusals, teammates, tyre age, following counts; the golden race,
  Qatar 2024 (VER against NOR on their one paired lap: the official lap times, 86.502 - 86.683 s).
- `tests/telemetry/test_build.py` (5): one call builds the track model and the corner tables; without the official list
  the map's bends are numbered; a failing track model leaves no old corner tables; the runner builds them with
  telemetry and not without; backfill adds missing ones and leaves current ones alone.
- `tests/gui/test_compare_tab.py` (5): the chart (a point after each corner, a bar per corner, the last point is the
  lap, a legend naming both, bars are medians); the tab (nothing until two drivers, one driver is not a pair, swapping
  flips every sign, the same driver twice refused); the window on a race with telemetry and on one built laps only.

Broken on purpose: `compare.py` 12 ways, the tab and chart 9 ways; all caught, three after a test was added (the mean
shown as the median twice, a placeholder counted as a driver).

## Known limits

- **One race is noisy** (above): a corner's time is good to about 20 ms on one lap, and the faster teammate changes
  from race to race 38 % of the time.
- **Tyre age and traffic are counted, not removed.**
- **"How" names consistent differences, not causes**: braking 8 m later can be a different line, not braver braking.
  Brake is on/off in OpenF1: "brakes harder" cannot be measured.
- **A race built from the Analyze window has no telemetry** (laps only); the tab says so and gives the command.
- **The 2-of-3 rule and the 150 km/h line are judgement**, not measured.

## How to run it

    python main.py                                          # a race -> Analyze -> Compare
    python -m src.gui.race_analysis_window 2024 23          # straight to one race
    python -m src.telemetry.compare --season 2023 2024 2025  # every teammate pair: data/reports/teammates.csv
    python -m pytest tests/telemetry tests/gui/test_compare_tab.py -q

From Python:

    from src.telemetry import compare as cp
    c = cp.load_comparison(2024, 23, "VER", "NOR")
    c.corners[["corner", "gap_median_sec", "winner", "how"]]
