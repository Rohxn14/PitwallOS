# Testing

The project uses `pytest`. The suite needs no data lake, no network and no display: GitHub runs it on Python 3.10, 3.11
and 3.12 on every push to `main` and on every pull request.

    python -m pip install -r requirements-dev.txt
    python -m pytest                       # everything, about 10 minutes
    python -m pytest tests/pipeline        # one part

## What is where

| Folder | Tests |
|---|---|
| `tests/pipeline/` | Bronze, Silver, Gold, contracts, the backfill, OpenF1, the telemetry clock and quality report: tiny hand-built tables (`tests/helpers.py`) and small real cuts |
| `tests/test_pipeline.py` | End to end on real Bronze fixtures, including the leakage test (Gold rebuilt from a cut race must be identical) |
| `tests/analytics/` | Each Phase 2 step: simulated races with known physics recovered exactly, and hand-checked facts of real races |
| `tests/models/` | The harness and each pace model: made-up races with the formula's physics (`simulated_race.py`), no leak, golden laps |
| `tests/simulator/` | Engine rules by hand (`hand_race.py`), real lap times reproducing real races exactly, what-ifs, Monte Carlo, events, the game |
| `tests/gui/` | Charts off-screen, and the windows with Qt's `offscreen` platform |
| `tests/lib/`, `tests/insights/` | The replay's running order, rewind behaviour of the live windows |

## The rules

- **A golden test for every number.** New analytics or model code is tested against a known fact from a source other than the code
  (Race Control's log, official lap times, a hand-worked gain), not just a type check.
- **Reference races** in `tests/fixtures/`: British 2024 (wet and dry), Qatar 2024 (safety cars, a VSC), Hungary 2024 (Norris's undercut
  on Piastri, 5.428 s).
- **Never the real lake.** Pass `root=` to anything that reads the lake. A test that falls back on `data/` passes on a machine that has
  the lake and fails on GitHub, which has none.
- **Never the network.** Downloads are always faked.
