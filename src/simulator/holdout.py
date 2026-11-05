"""
The test season for the simulator (src/simulator/holdout.py) - Phase 4, the end

2025 opens ONCE, here, and closes Phase 4. Everything the simulator uses is fitted ONCE on the development races (2023 +
2024): the pace rungs and their priors, the rule book, the cover model and the drawn world (replay.fit_fold, with every
race of the test season held out), and the misses the Monte Carlo draws (the game's saved models: the development races'
own misses, each measured with the fold that never saw it). Then the simulator is judged on the races of the test season,
which it never saw, by the same measures as on the development races:

    step 1   replays from lap 5, halfway and the lap before each SC / VSC / red flag (replay.replay_race): places off per
             car at the flag, against "nobody moves"; boosted+prior (the game's pace) and hindsight (the rules alone)
    step 2   the real undercut attempts (decisions.undercut_calls: right or wrong, against "it always fails" and the 1 s
             gap rule) and the neighbours at a safety car or VSC (decisions.neutralisation_pairs: who finishes ahead)
    step 3   the chances the game shows ('+ retirements': the neutralisations, the answers and the retirements drawn,
             nothing of the real race known) against what happened and against the history guess, from lap 5 and
             from halfway (calibration.summary)

Reports: sim_test_replay, sim_test_undercut_calls, sim_test_neutralisation_pairs, sim_test_calibration; the development
numbers are printed next to them when the lake has the development reports.

    python -m src.simulator.holdout          opens 2025 (about 20 minutes)

THE RULE AFTER THIS (as Phase 3's): the test season has been seen. Nothing in the simulator may be changed because of
these numbers; a changed simulator can only be judged on the development races again.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models import evaluate
from src.pipeline import lake
from src.simulator import calibration, decisions, game, montecarlo, race as race_module, replay
from src.simulator.race import Race
from src.simulator.replay import FoldModels

SOURCES = (calibration.BEST, replay.HINDSIGHT)      # the game's pace, and the real pace (the rules alone)
REPORTS = {"replay": "sim_test_replay", "calls": "sim_test_undercut_calls", "pairs": "sim_test_neutralisation_pairs",
           "chances": "sim_test_calibration"}
DEVELOPMENT_REPORTS = {"replay": "sim_replay", "calls": "sim_undercut_calls", "pairs": "sim_neutralisation_pairs",
                       "chances": "sim_calibration_drawn"}


def fit_test_fold(train_seasons: Sequence[int], test_seasons: Sequence[int],
                  root: str = lake.DEFAULT_LAKE_ROOT) -> Tuple[FoldModels, Dict[Tuple[int, int], Race], List[Tuple[int, int]]]:
    """The simulator fitted once on the training seasons, every race of the test seasons held out (THE OPENING: the
    test seasons are loaded here, open_test_set=True)."""
    laps = evaluate.load_laps(tuple(train_seasons) + tuple(test_seasons), root, open_test_set=True)
    races = {key: race_module.load(*key, root=root) for key in evaluate.races_of(laps)}
    test_keys = [key for key in races if key[0] in set(test_seasons)]
    return replay.fit_fold(laps, test_keys, races, root), races, test_keys


def test_rows(fold: FoldModels, races: Dict[Tuple[int, int], Race], test_keys: Sequence[Tuple[int, int]],
              misses: calibration.Misses, root: str = lake.DEFAULT_LAKE_ROOT, runs: int = calibration.RUNS,
              log: Callable[[str], None] = print) -> Dict[str, pd.DataFrame]:
    """Steps 1, 2 and 3 on every test race. misses: the development races' (what the Monte Carlo draws, and the history
    guess's places); none of them is a test race."""
    uncertainty = calibration.uncertainty_from(misses, [], montecarlo.KEPT_DRAWS)
    by_place = calibration.place_offsets_from(misses, [])
    full = calibration.DRAWN[calibration.FULL]
    replays, calls, pairs, chances = [], [], [], []
    for key in test_keys:
        race = races[key]
        replays += replay.replay_race(race, fold, sources=SOURCES)
        if race.mixed:                       # the checks of steps 2 and 3 are on dry races, as on the development races
            log(f"{race.season} {race.event_name}: replayed (rain: steps 2 and 3 left out)")
            continue
        calls += decisions.undercut_calls(race, fold, sources=SOURCES, root=root)
        pairs += decisions.neutralisation_pairs(race, fold, sources=SOURCES)
        for label, lap in replay.starts(race):
            if label not in calibration.STARTS:
                continue
            start = calibration.start_of(race, fold, label, lap, drawn=True)
            seed = calibration.SEED + 1000 * race.season + 10 * race.round + calibration.STARTS.index(label)
            chances.append(calibration.drawn_chances(start, uncertainty, full, fold.world, by_place, runs, seed)
                           .assign(config=calibration.FULL))
        log(f"{race.season} {race.event_name}")
    out = {"replay": pd.DataFrame(replays), "calls": pd.DataFrame(calls), "pairs": pd.DataFrame(pairs)}
    rows = pd.concat(chances, ignore_index=True) if chances else pd.DataFrame()
    if len(rows):
        history = calibration.history_chances(rows, misses.places_all)
        history[["moved_sim", "sim_passes"]] = np.nan                      # history moves nobody
        rows = pd.concat([rows, history.assign(config="history")], ignore_index=True)
    out["chances"] = rows
    return out


def describe(rows: Dict[str, pd.DataFrame]) -> str:
    """Each step's own summary of these rows."""
    lines = []
    replays = rows["replay"]
    if len(replays):
        for start in (replay.LAP_5, replay.HALFWAY, "neutralised"):
            table = replay.summary(replays, start)
            if len(table):
                lines += [f"\nStep 1, places off per car at the flag from {start} (dry races):", table.to_string(index=False)]
    if len(rows["calls"]) or len(rows["pairs"]):
        lines += ["\nStep 2:", decisions.summary(rows["calls"], rows["pairs"], pd.DataFrame(), pd.DataFrame())]
    if len(rows["chances"]):
        lines += ["\nStep 3, the chances the game shows ('" + calibration.FULL + "') against what happened:",
                  calibration.summary(rows["chances"], drawn=True)]
    return "\n".join(lines)


def development_rows(root: str = lake.DEFAULT_LAKE_ROOT) -> Optional[Dict[str, pd.DataFrame]]:
    """The same measures on the development races (out of fold), from their reports, for comparison; None without them."""
    if not all(lake.report_path(name, root).exists() for name in DEVELOPMENT_REPORTS.values()):
        return None
    out = {k: lake.read_report(name, root) for k, name in DEVELOPMENT_REPORTS.items()}
    out["replay"] = out["replay"][out["replay"]["source"].isin(SOURCES)]
    out["calls"] = out["calls"][out["calls"]["source"].isin(SOURCES)]
    out["chances"] = out["chances"][out["chances"]["config"].isin([calibration.FULL, "history"])]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4, the end: OPENS the test season and judges the simulator, "
                                                 "fitted once on the development races, on it")
    parser.add_argument("--train-season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS))
    parser.add_argument("--test-season", type=int, nargs="+", default=list(evaluate.TEST_SEASONS))
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--runs", type=int, default=calibration.RUNS)
    args = parser.parse_args(argv)

    models = game.load_models(args.lake, seasons=args.train_season)      # the development races' misses, out of fold
    fold, races, test_keys = fit_test_fold(args.train_season, args.test_season, args.lake)
    trained = [k for k in races if k not in set(test_keys)]
    print(f"Test season {', '.join(map(str, args.test_season))}: {len(test_keys)} races "
          f"({sum(not races[k].mixed for k in test_keys)} dry); the simulator fitted once on the {len(trained)} races of "
          f"{', '.join(map(str, args.train_season))}.")
    print(fold.book.describe())
    print(fold.world.describe())
    rows = test_rows(fold, races, test_keys, models.misses, args.lake, args.runs)
    print("\n=== THE TEST SEASON ===")
    print(describe(rows))
    for key, name in REPORTS.items():
        if len(rows[key]):
            print(f"Report: {lake.write_report(rows[key], name, args.lake)}")
    development = development_rows(args.lake)
    if development is not None:
        print("\n=== The development races (out of fold), for comparison ===")
        print(describe(development))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
