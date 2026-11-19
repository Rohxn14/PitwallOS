# Phase 5 step 5.4: tyres from telemetry

A developer's guide to what the corner tables say about tyre wear: where on the lap it costs time, and whether the
telemetry sees a worn tyre before the lap time does.

## Purpose

Phase 2 measured how much slower a tyre gets per lap of age, from lap times. Telemetry can say more: WHERE on the lap
the time goes, and perhaps WHEN it starts, before the lap time shows it. Rohan's guess, on record before this step:
*corner speeds warn of a tyre going off 3+ laps before the lap time does; fresh tyres show no early fade.*

## The data

| Input | From | Used for |
|---|---|---|
| Silver `laps` | through step 2.3 (`degradation_laps`): pace laps on slicks, dry track, consistent tyre record, stints of 6+ such laps | which laps, their stint, tyre age, fuel on board |
| Gold `corner_features` | step 5.2 | each corner's segment time; speed at the corner and exit speed |

Output: `data/reports/tyre_wear_corners.csv` (per race and compound: the lap's wear slope, the part made in slow
corners, the slow corners' share of the lap time) and `tyre_stints.csv` (per stint: when each signal saw the wear, the
early and whole-stint slopes, the "going off" lap of Rohan's first definition).

Races with any car on inters or wets are left out (the track changes under the tyre: step 2.3's rule).

## How it works (`src/telemetry/tyres.py`)

### 1. Where wear costs time (`corner_wear`, `slow_share`)

Each corner's segment time, fuel removed in proportion to its share of the lap (`corrected = segment - 0.03 s/kg x
fuel x segment / lap time`), gets a wear slope per compound: seconds per lap of tyre age, measured within each stint
(each stint keeps its own level; step 2.3's `within_stint_slope`). Only laps with every corner complete, so **the
corners' slopes add up exactly to the lap's slope** over the same laps: "slow corners carry 74 % of the wear" is a
sum, not a model. A slow corner is one whose median speed (over the race's clean laps) is under 150 km/h, as in 5.3.

### 2. Which signal sees the wear first (`fuel_corrected`, `running_t`, `detection_lap`, `stint_detection`, `who_first`)

Two signals per lap:
- **lap time**;
- **the slow corners' speed**: the mean over the race's slow corners of the speed at the corner and the exit speed
  (Rohan's choice of signal), only on laps where every slow corner is complete.

**Fuel, the same way for both.** A car gets faster as it burns fuel, which hides wear. Phase 2 removes 0.03 s per kg
from lap times; for corner speeds no such constant exists. Correcting one signal and not the other would favour it, so
both get their fuel effect estimated from the race by one method: step 2.3's `fuel_check`, now taking any column
(driver levels, compound offsets, a wear slope per compound and one fuel coefficient, fitted on all used laps; tyre age
resets at a stop and fuel does not, which tells them apart). Medians over races: 0.031 s per kg for the lap time
(Phase 2's own check: 0.031), -0.04 to -0.05 km/h per kg in the slow corners. A race where nobody's stints separate the
two (Australia and Belgium 2023, Miami 2025, Monaco and Zandvoort 2026) is left out and named.

**When is the wear clear?** On each stint, after k laps (k from 5), the least-squares slope of the signal on tyre age
over those k laps, divided by its standard error (`running_t`). The wear is "clear" at the first k where it is 2
standard errors the worse way (slower laps, lower speeds). Only the laps so far are used: a pit wall could run it
live. Also recorded: whether a trend the WRONG way (the tyre "getting better") was clear first.

**Who saw it first** (`who_first`): over stints where both signals exist on every lap, the lap each became clear;
lead = time's lap minus speed's lap (positive: the speed saw it first).

**Fresh tyres** (`early_slope`): the slope over each stint's first 5 laps, against the whole stint's.

### 3. Rohan's first definition, kept for the record (`going_off`, `pit_reaction`)

"Going off" = the lap time (fuel removed) 0.5 s above the stint's trend for 2 laps in a row, both laps judged against
the trend fitted on the laps before them. `pit_reaction` asks whether teams react: in stints that ended with a stop, how
often the stop came within 3 laps of the event, against chance (any lap where an event could have started).

## Decisions and the evidence behind them

| Decision | Alternatives | Why |
|---|---|---|
| "Which sees wear first" replaced "warning of going off" | the warning test on Rohan's 0.5 s jumps | measured first: the jump rule fires in 54-56 % of dry stints, a third of them on the same lap as 3+ other cars, and teams pit within 3 laps after 26 % (2023-24) and 18 % (2025-26) of them, against 40 % and 33 % by chance. Variants (against the field, first laps left out, traffic laps not counted) fire in 26-46 % and change nothing about the teams' reaction. The jumps are traffic, restarts and tyre management, not tyres; real cliffs are rare because teams stop before them. Rohan chose the new test |
| Fuel estimated per race for both signals | 0.03 s/kg for lap time, nothing for speed | an asymmetric correction favours one signal. The estimate for lap time (0.031 s/kg) matches Phase 2's check |
| 2 standard errors, 5 laps minimum, fixed before measuring | tuned thresholds | nothing to tune, so 2025-2026 is a clean check of 2023-2024 |
| Slow-corner speed = mean of corner and exit speed over slow corners, whole laps only | each corner separately; the slow corners' time | Rohan's choice of signal; a mean over some corners would move with which ones are missing |
| Corner wear from the segment times, fuel shared by time | speeds | times add up to the lap, so the shares are exact |

## Measured

**Where wear costs time.** Race-compounds whose lap wear is clear (slope more than 2 standard errors above zero):

| | 2023-2024 (41 dry races) | 2025-2026 (37 dry races) |
|---|---|---|
| race-compounds with clear wear | 73 of 106 | 70 of 98 |
| share of the wear made in slow corners (median; middle half) | **74 %** (56-83 %) | **74 %** (56-86 %) |
| slow corners' share of the lap time | 54 % | 54 % |
| lap wear per lap of age, median: soft / medium / hard | +0.077 / +0.056 / +0.046 s | +0.062 / +0.049 / +0.036 s |

Tyre wear is concentrated where grip and traction limit the car: three quarters of it in the slow corners, which take
about half the lap.

**Which signal sees the wear first** (stints measured by both signals on every lap):

| | 2023-2024 (39 races, 1,750 stints) | 2025-2026 (34 races, 1,512 stints) |
|---|---|---|
| wear clear in: lap time / slow-corner speed | 73 % / 74 % of stints | 73 % / 70 % |
| of stints both saw: speed first / same lap / time first | 41 % / 28 % / 31 % | 39 % / 30 % / 31 % |
| speed 3+ laps earlier | 22 % | 24 % |
| lead (time's lap minus speed's): median; mean; middle half | 0; +0.6; -1 to +2 laps | 0; +0.7; -1 to +2 laps |
| speed first more often than time first (sign test) | p = 0.0001 | p = 0.002 |
| a trend the wrong way seen first: time / speed | 27 % / 16 % | 27 % / 20 % |

**Fresh tyres** (stints of 10+ laps; slope over the first 5 laps against the whole stint):

| | 2023-2024 | 2025-2026 |
|---|---|---|
| lap time: first 5 laps / whole stint | +0.017 / +0.059 s per lap (worse on 54 % of stints) | +0.001 / +0.050 (50 %) |
| slow-corner speed: first 5 laps / whole stint | -0.23 / -0.16 km/h per lap (worse on 65 %, sign test p = 1e-32) | -0.15 / -0.13 (60 %, p = 2e-14) |

The lap time barely moves over a fresh tyre's first laps; the slow corners lose speed from the first lap, at least as
fast as over the whole stint.

**The expectations, written before measuring.**

| Who | Expected | Measured |
|---|---|---|
| Rohan (on record) | corner speeds warn 3+ laps before the lap time | in 22-24 % of stints; median 0 laps |
| Rohan (piece 2) | the speeds and the lap time see the wear "about the same time" | right: median 0; the speed first slightly more often (mean +0.6 laps) |
| Rohan | fresh tyres show no early fade | true of the lap time; false of the corner speeds (fading from lap 1 in 60-65 % of stints) |
| Rohan | a tyre goes off in about 15 % of dry stints | the rule fires in 54-56 %, but it is not tyres (teams do not react) |
| Claude | going off in about 5 % of stints | as above: the rule does not measure tyres |
| Claude | the speeds see the wear 0-1 laps from the lap time, possibly a lap later | 0 laps (median); not later: slightly earlier |
| Claude | no fade in corner speeds on fresh tyres | wrong: they fade from the first lap |

## Tests

`tests/telemetry/test_tyres.py`, 11 tests:
- golden: the 2024 Qatar GP's real laps (the FastF1-shaped fixture), each lap's time split into four corners in fixed
  shares: the corners' wear slopes add up to exactly step 2.3's compound slopes, each corner carries exactly its share,
  and a lap missing a corner is left out of every corner so the sum still holds;
- a clean trend is clear at 5 laps, the worse way for each signal; a trend the wrong way is flagged;
- noise takes longer to see through, and pure noise is called wear in under 30 % of made-up stints;
- who saw it first, counted only on stints measured on every lap;
- the jump rule needs two laps above the same trend (one lap is not a run; refitting after the first jump would hide
  the second: a test caught that bug);
- the teams' reaction against chance; the fuel effect recovered exactly for both signals by one method; the early slope
  is the first laps only; slow corners judged on clean laps, their speed averaged on whole laps only.

Broken on purpose 13 ways, all caught (five after tests were added; a fourteenth break changed nothing, because another
check already enforces the 5-lap minimum).

## Known limits

- **Detection is per stint and checks the slope after every lap**: repeated looks find a trend sooner than a single
  test would. Both signals get the same repeated looks, so the comparison is fair; the absolute "lap it became clear"
  is optimistic. Laps of a stint are not independent either (the standard errors are optimistic for both).
- **The lap time's fuel coefficient also absorbs track evolution** (the track gets faster as fuel burns): 27 % of
  stints first show the lap time getting "better", against 16-20 % for the speed. Part of the speed's earlier
  detection may be the lap time being fooled more often, not the speed seeing more.
- **The early fade in corner speeds is not explained here.** It could be the tyres losing peak grip from the first
  laps, drivers managing them, or the fuel effect on speed being estimated too small (a linear fuel error tilts every
  stint by the same amount). The lap time does not show it.
- **Real tyre cliffs are not measured**: they are rare, and the jump rule cannot tell them from traffic.
- **2025 was seen by the models in Phase 3**, not by this step; there were no thresholds to tune here.

## How to run it

    python -m src.telemetry.tyres --season 2023 2024 2025 2026      # ~3 min; data/reports/tyre_wear_corners.csv, tyre_stints.csv
    python -m pytest tests/telemetry/test_tyres.py -q

From Python:

    from src.telemetry import tyres as ty
    laps, segments, mixed = ty.race_stint_laps(2024, 1)
    ty.slow_share(ty.corner_wear(segments))            # Bahrain 2024: where the wear is
