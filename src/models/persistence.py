"""
Persistence (src/models/persistence.py) - Phase 3 rung 1

The simplest honest guess: the driver's next lap will be as fast as their last clean lap.
Nothing is learned from other races, so fit() does nothing.

When lap N ends, the prediction for lap N+1 is:
  1. lap N's own time, if lap N was a clean lap                  (Gold: lap_time_sec, is_clean_lap)
  2. otherwise the driver's most recent clean lap before lap N   (Gold: last_clean_lap_time_sec)
The harness only scores a lap once the driver has driven a clean lap, so one of the two always
exists. Rule 2 is used after a pit stop or a safety car, when lap N itself was not clean.

Why the last clean lap and not an average of several: measured on the development races
(2023 and 2024, dry laps, the same 40,325 laps for each):
    last clean lap                 MAE 0.479 s
    median of the last 3 clean     MAE 0.512 s
    mean of the last 3 laps        MAE 0.533 s   (Gold's roll3_clean_mean_sec)
    median of the last 5 clean     MAE 0.563 s
An average smooths out noise, but the pace keeps moving (fuel burning off, tyres wearing, the
track rubbering in), so an average of older laps lags behind it. The newest lap lags least.

What persistence cannot know: the car getting lighter, the tyres wearing, the car ahead. Those
are rung 2's terms. Persistence's bias (its mean error) shows how much they add up to.

Run it on the development races (2023 and 2024):

    python -m src.models.persistence
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import pandas as pd

from src.models import evaluate
from src.pipeline import lake

THIS_LAP = "lap N itself"
EARLIER_CLEAN_LAP = "an earlier clean lap"
NO_PREDICTION = "none"


class Persistence:
    """Rung 1: the next lap equals the driver's last clean lap."""

    name = "persistence"

    def fit(self, train_laps: pd.DataFrame) -> "Persistence":
        # Nothing to learn: the prediction only uses the race being predicted.
        return self

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        # The same guess for every lap ahead: persistence has no idea of anything changing.
        this_lap = laps["lap_time_sec"].where(laps["is_clean_lap"])   # lap N, if it was clean
        earlier = laps["last_clean_lap_time_sec"]                     # the last clean lap before N
        prediction = this_lap.fillna(earlier)                         # rule 1, else rule 2
        return prediction.rename("predicted_sec")


def prediction_source(laps: pd.DataFrame) -> pd.Series:
    """Which rule made each prediction (the later assignment wins, so rule 1 comes last)."""
    source = pd.Series(NO_PREDICTION, index=laps.index)
    source[laps["last_clean_lap_time_sec"].notna()] = EARLIER_CLEAN_LAP
    source[laps["is_clean_lap"]] = THIS_LAP
    return source


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 rung 1: persistence, scored on the development races")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to score (default: the development seasons, 2023 2024)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    predictions = evaluate.cross_validate(Persistence, laps)
    card = evaluate.score(laps, predictions, Persistence.name)
    print(evaluate.describe(card))

    # How each scored lap was predicted, and how good each rule was.
    source = prediction_source(laps)[card.errors.index]
    by_source = card.errors["error_sec"].abs().groupby(source).agg(["size", "mean"])
    by_source.columns = ["laps", "mae"]
    print("\nPredictions by rule:")
    print(by_source.round(3).to_string())

    report = evaluate.write_scorecard(card, args.lake)
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
