"""
The test season (src/models/holdout.py) - Phase 3 step 5

Every choice in the ladder was made on the development races (2023 + 2024), cross-validated by race.
That still flatters the rungs a little: the choices were kept because they scored well there (rung 4
alone tried about 15 settings). The test season answers the question that is left: how well does each
rung predict races that played no part in any choice, a year of car development later?

    every rung is fitted ONCE, on all the development races,
    then predicts every race of the test season (2025), 1, 3, 5 and 10 laps ahead,
    scored exactly like the development scorecards (the same rules for the scored laps, the same
    bootstrap over races, the same paired comparison between rungs).

The rule: the test season is opened once. No rung may be changed because of what it shows. A rung
changed afterwards has no untouched season left to be judged on until 2026 is in the lake (a new set
of regulations, so a harder test).

Also measured here: rung 3's bands on the test season (the spread the Phase 4 Monte Carlo will draw from).

Result, opened once on 2026-10-15 (24 races of 2025, 20,755 dry laps scored one lap ahead), dry MAE in s:

    laps ahead        1       3       5      10
    persistence    0.420   0.530   0.590   0.700
    linear         0.398   0.505   0.561   0.644
    state_space    0.386   0.494   0.550   0.633
    special_laps   0.386   0.492   0.550   0.634
    boosted        0.353   0.442   0.495   0.569

  * Every rung scores better than on the development races (persistence 0.479 -> 0.420), because the
    season itself is steadier: two clean dry laps in a row differ by 0.365 s on average in 2025, against
    0.443 in 2023 and 0.421 in 2024. Compare rungs with each other within a season, not across seasons.
  * The order holds at every horizon, and each rung beats the one above it in the table (paired
    bootstrap, "better"), except the hand-written baseline: no clear difference from rung 3 anywhere.
  * Rung 4 against rung 3: -0.033 s one lap ahead (95 % -0.044 to -0.023), -0.051 / -0.055 / -0.064 at 3 /
    5 / 10 laps: 8.5 % to 10.4 % better, against 9.6 % to 12.6 % on the development races. A little of the
    development gain was optimism; most of it held. Better in 22 of 24 races (not Qatar, Las Vegas).
  * Rung 3 against rung 2: better at every horizon (-0.011 to -0.012 s); on the development races only at 3
    laps ahead.
  * Rung 4 by what lap N was: better after a stop (1.26 -> 1.09) and on ordinary laps (0.347 -> 0.318),
    worse on restart laps (232 laps after a safety car, VSC or red flag: 0.99 -> 1.15).
  * Rung 3's bands, dry: 52.8 / 81.6 / 85.3 % inside the 50 / 90 / 95 % bands (development 49.7 / 79.5 /
    83.4): right in the middle, too narrow in the tails, as before.

Run it (it opens the test season):

    python -m src.models.holdout
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

import pandas as pd

from src.models import evaluate
from src.models.boosted import Boosted, SpecialLaps, by_lap_kind, lap_kind, rung3_inputs
from src.models.linear import LinearDecomposition
from src.models.persistence import Persistence
from src.models.state_space import StateSpace, band_distance, coverage_table
from src.pipeline import lake

RUNGS = (Persistence, LinearDecomposition, StateSpace, SpecialLaps, Boosted)
DEVELOPMENT_REPORT = "pace_model_horizons"      # the development scores, written by python -m src.models.boosted


def against_development(test_rows: pd.DataFrame, development_rows: pd.DataFrame) -> pd.DataFrame:
    """Dry MAE of every rung at every horizon: cross-validated on the development races, then on the
    test season, and the change (positive: worse on the test season)."""
    keys = ["laps_ahead", "rung"]
    table = test_rows[keys + ["dry_mae"]].rename(columns={"dry_mae": "test_mae"}).merge(
        development_rows[keys + ["dry_mae"]].rename(columns={"dry_mae": "development_mae"}), on=keys, how="left")
    table["change"] = table["test_mae"] - table["development_mae"]
    return table[keys + ["development_mae", "test_mae", "change"]].round(4)


def race_table(cards: Dict[str, Dict[int, evaluate.Scorecard]]) -> pd.DataFrame:
    """One row per test race: its dry laps and each rung's dry MAE one lap ahead."""
    keys = ["season", "round"]
    first = next(iter(cards.values()))[1].races
    table = first[keys + ["event_name", "dry_laps"]]
    for name, by_horizon in cards.items():
        table = table.merge(by_horizon[1].races[keys + ["dry_mae"]].rename(columns={"dry_mae": name}), on=keys)
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 step 5: every rung fitted once on the development races, "
                                                 "scored on the test season (opens it)")
    parser.add_argument("--train-season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to fit on (default: the development seasons, 2023 2024)")
    parser.add_argument("--test-season", type=int, nargs="+", default=list(evaluate.TEST_SEASONS),
                        help="seasons to score (default: the test season, 2025)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    train = evaluate.load_laps(args.train_season, args.lake)
    test = evaluate.load_laps(args.test_season, args.lake, open_test_set=True)
    print(f"Test season {', '.join(map(str, sorted(set(test['season']))))}: {len(evaluate.races_of(test))} races, "
          f"{len(test):,} rows. Every rung fitted once on the {len(evaluate.races_of(train))} races of "
          f"{', '.join(map(str, sorted(set(train['season']))))}.\n")

    cards = evaluate.held_out_scorecards(RUNGS, train, test)
    print(evaluate.describe(cards[Boosted.name][1]))

    rows = evaluate.horizon_rows(cards)
    print("\nLaps ahead on the test season (dry MAE, s; each rung against the one above it, paired bootstrap):")
    print(rows.to_string(index=False))
    print(f"\n{Boosted.name} against {StateSpace.name} on the test season, laps ahead:")
    for k in evaluate.HORIZONS:
        result = evaluate.compare(cards[StateSpace.name][k], cards[Boosted.name][k])
        print(f"  {k:2d}: {result['old_mae']:.3f} -> {result['new_mae']:.3f} s, {result['difference']:+.4f} "
              f"({result['low']:+.4f} to {result['high']:+.4f}): {result['verdict']}")

    if lake.report_path(DEVELOPMENT_REPORT, args.lake).exists():
        development = lake.read_report(DEVELOPMENT_REPORT, args.lake)
        print(f"\nDevelopment (cross-validated, from {lake.report_path(DEVELOPMENT_REPORT, args.lake)}) against the "
              "test season, dry MAE (s):")
        print(against_development(rows, development).to_string(index=False))

    races = race_table(cards)
    with pd.option_context("display.width", 200):
        print("\nDry MAE one lap ahead, race by race:")
        print(races.round(3).to_string(index=False))

    rung3 = StateSpace().fit(train)
    print(f"\n{StateSpace.name}'s bands on the test season (a fair band holds its share of laps):")
    print(coverage_table(band_distance(rung3, test), test).to_string(index=False))

    one_lap = {name: cards[name][1] for name in (StateSpace.name, SpecialLaps.name, Boosted.name)}
    print("\nDry MAE one lap ahead by what lap N was:")
    print(by_lap_kind(one_lap, lap_kind(test, rung3_inputs(rung3, test))).to_string())

    print(f"\nReports: {lake.write_report(rows, 'pace_model_test_horizons', args.lake)}, "
          f"{lake.write_report(races, 'pace_model_test_races', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
