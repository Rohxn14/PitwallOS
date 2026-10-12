# PitwallOS 🏎️

An end-to-end Formula 1 data engineering, analytics and machine-learning project: a data lake
built from official timing data, a strategy analytics engine, pace models and, in time, a race
simulator.

PitwallOS is built on top of **[F1 Race Replay](https://github.com/IAmTomShaw/f1-race-replay) by
[Tom Shaw](https://tomshaw.dev)**, a Python app that replays any race on a 2D track map with live
telemetry. The replay is still here and works as before (see [Race replay](#race-replay)).
PitwallOS adds everything that turns the data into analysis.

![Race Replay Preview](./resources/preview.png)

## The idea everything is built around

    lap time = base pace (driver, car, track)
             + fuel effect
             + tyre effect (compound, age)
             + traffic effect (dirty air)
             + track evolution
             + noise

Each phase of the project works on these terms. The analytics measure them from real races, the
models learn them, and the simulator uses them to replay strategies. Every phase is checked by
the one after it, and every number is tested against a known fact of a real race.

## What is built

### 1. The data lake (Bronze → Silver → Gold)

`src/pipeline/` downloads races with [FastF1](https://github.com/theOehrly/Fast-F1) and stores
them in three layers:

| Layer | Holds | Rules |
|---|---|---|
| **Bronze** | Raw FastF1 tables, as delivered (laps, weather, track status, race control, results, telemetry) | Never changed after download |
| **Silver** | Cleaned laps with flags: pit passage kind (tyre change, pass-through, retired), stints, safety car / VSC / red flag, clean lap, wet tyre, weather; telemetry on a shared 10 Hz clock | Never invent a value; never delete a lap, flag it |
| **Gold** | One row per lap, ready for models: fuel estimate, gaps, recent pace, next-lap target | Every feature is known when the lap ends (a test enforces no leakage) |

Every table is checked against a data contract (pandera) before it is written, and every dropped
row is counted.

### 2. Backfill: three full seasons

All 70 races of 2023, 2024 and 2025 are in the lake (77,713 laps). One command rebuilds a season,
resumes where it stopped and writes a data-quality report per race:

```bash
python -m src.pipeline.backfill --season 2023 2024 2025
```

### 3. Analytics, step by step

The analytics are being rebuilt one step at a time. Each step starts from a written design and a
predicted number, and ends with golden tests on real races.

- **Done: lap selection and neutralisations.** Every lap gets one status: a pace lap, or the
  first reason it is not (lap 1, safety car, pit lap, outlier...). A ledger counts every exclusion.
  Safety car, VSC and red-flag periods come from the official track status, counted in the
  leader's laps; they match Race Control's own log.
- **Next:** pit-stop loss, fuel-corrected tyre degradation, dirty air, measured undercuts.

### 4. The Analyze window

`python main.py` → pick a race → **Analyze**. Without running the replay, it shows the result,
the tyre strategies driven, the data quality and a **Pace** tab: lap-time charts per driver,
raw pace by compound, driver pace on each compound and the exclusion ledger. If a race is not in
the lake yet, the window builds it first.

## Roadmap

| Phase | Work | Status |
|---|---|---|
| 0. Pipeline | Bronze / Silver / Gold, data contracts, tests | Done |
| 1. Backfill | 2023-2025, a quality report per race | Done |
| 2. Analytics | Lap selection, pit loss, fuel-corrected degradation, dirty air, undercuts, race report | In progress (step 1 done) |
| 3. Pace model | A ladder of models, each beating the last on races it has never seen: persistence → linear decomposition → state-space tyre model → gradient boosting | Planned |
| 4. Race simulator | Replay the strategies actually driven, then Monte Carlo over safety-car timing and pace noise for "what if" questions | Planned |
| 5. Telemetry | Corner-by-corner driver comparison; minimum-speed fade as an early sign of tyre wear | Planned |
| 6. Presentation | A web front end over the simulator and a written case study of one race | Planned |

## Getting started

Requires Python 3.10+ (developed on 3.11; tests run on 3.10, 3.11 and 3.12).

```bash
git clone https://github.com/Rohxn14/PitwallOS
cd PitwallOS
python -m venv venv
# Windows: .\venv\Scripts\activate      macOS / Linux: source venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest            # run the test suite
python main.py              # pick a race, then Replay or Analyze
```

Build races into the lake:

```bash
python -m src.pipeline.runner --year 2024 --round 12 --no-telemetry   # one race, laps only
python -m src.pipeline.backfill --season 2024                         # a whole season
python -m src.analytics.laps --season 2024                            # pace laps for every race
```

The first download of a race takes a while; FastF1 caches it in `.fastf1-cache/` and the replay
and the pipeline share that cache. The lake lives in `data/` and can always be rebuilt.

If downloading fails, `python -m pip install --upgrade fastf1` usually fixes it.

## Race replay

The replay comes from F1 Race Replay (Tom Shaw and its contributors) and is kept as it was, with a
few fixes (an accurate leaderboard at the end of a race, rewind-safe insight windows).

**Features**

- Watch the race unfold with live driver positions on the track map
- Leaderboard with positions, tyre compounds and retired drivers marked "OUT"
- Simulated safety car, placed ahead of the leader whenever the official feed shows one
- Pause, rewind, fast forward and playback speed controls
- Driver telemetry (speed, gear, DRS) for the selected drivers
- Insights menu with live telemetry windows, and qualifying replays (in development)

**Controls**

- **Pause / resume:** SPACE
- **Rewind / fast forward:** ← / →
- **Playback speed:** ↑ / ↓ (0.5x, 1x, 2x, 4x), or keys 1-4
- **Restart:** R · **DRS zones:** D · **Progress bar:** B · **Driver names:** L
- **Select drivers:** click, or shift-click for several

**From the command line**

```bash
python main.py --cli                                      # menu in the terminal
python main.py --viewer --year 2025 --round 12            # a race directly
python main.py --viewer --year 2025 --round 12 --sprint   # the sprint
python main.py --viewer --year 2025 --round 12 --qualifying
python main.py --viewer --year 2025 --round 12 --refresh-data   # recompute the cached replay
```

**Custom insight windows:** subclass `PitWallWindow` to build a window fed by the live telemetry
stream. See [docs/PitWallWindow.md](./docs/PitWallWindow.md),
[docs/InsightsMenu.md](./docs/InsightsMenu.md) and [telemetry.md](./telemetry.md).

**Known issue:** in a `conda` environment you may see `Unable to create an OpenGL 3.3+ context`;
`conda install -c conda-forge libstdcxx-ng` fixes it (thanks to @el-mandaloriano, #12).

## Project layout

```
main.py                  entry point: session picker, replay or analyze
src/pipeline/            the data lake: extractor (Bronze), cleaner (Silver), feature store (Gold),
                         contracts, backfill, quality reports
src/analytics/           race overview, lap selection (laps.py), neutralised periods
src/gui/                 session picker, Analyze window and its charts, insights menu
src/interfaces/          race and qualifying replay (Arcade)
src/insights/            live telemetry windows
src/lib/                 running order, tyres, time, settings
tests/                   unit tests and golden tests on real races (fixtures in tests/fixtures/)
docs/                    guides, and a catalogue of F1 strategies
```

## Credits

- **[Tom Shaw](https://tomshaw.dev)** created [F1 Race Replay](https://github.com/IAmTomShaw/f1-race-replay),
  the replay, track map, insights menu and telemetry stream this project is built on.
- The F1 Race Replay contributors, listed in [contributors.md](./contributors.md).
- **[FastF1](https://github.com/theOehrly/Fast-F1)**, which makes the official timing data accessible.

## License

MIT License, as the original project.

## Disclaimer

Unofficial project, for education and non-commercial use. Formula 1 and related trademarks are the
property of their respective owners. All data comes from publicly available sources.
