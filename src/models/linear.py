"""
Linear decomposition (src/models/linear.py) - Phase 3 rung 2

The lap-time formula, with its numbers fitted from data:

    lap time = base pace                            this driver and car, at this track, today
             + compound offset                      the soft or the hard against the medium
             + wear slope (per compound) x tyre age
             + fuel coefficient x fuel on board     fuel and track evolution together (see below)
             + traffic cost                         by the gap to the car ahead when the lap starts
             + noise

The numbers come from two places:

  1. THE TERMS are the same physics at every race, so fit() learns them from the training races:
     least squares over their scored laps, with one free base per driver per race (Phase 2's
     fixed_effects_fit), so only the changes within a driver's race count. Only dry races are used
     (nobody on intermediates or wets): with the slick laps of mixed races in, the drying track made
     the soft look slower than the medium (+0.24 +- 0.54 s instead of -0.39 +- 0.16 s).
  2. THE BASE PACE is this car and driver on this day, so it never comes from other races. Every
     clean lap gives one reading of it: the lap's time minus the lap's own terms. When lap N ends,
     the base is the latest reading, skipping odd ones (drop_odd_readings).

The prediction for lap N+1 is that base plus lap N+1's terms, all known when lap N ends:
    compound    the tyre on lap N (a scored lap N+1 is never an out-lap, so it is the same set)
    tyre age    lap N's age + 1
    fuel        on board when lap N+1 starts: FUEL_START_KG - N x burn per lap (Gold's fuel model)
    traffic     the gap when lap N ends, which is the gap when lap N+1 starts
Written out:  prediction = last good clean lap + (terms of lap N+1 - terms of that lap).
It is persistence, corrected by what the formula says changed in between.

Fuel and track evolution: both make the car faster lap after lap, in step with the lap number (the
fuel on board falls by the same amount every lap). Lap times cannot tell them apart, so one
coefficient holds both, in seconds per kg (Phase 2's fuel check: 0.031 s/kg).

Measured on the development races (2023 + 2024, dry MAE, cross-validated by race) before choosing:
  * The latest reading, not an average of several: median of the last 3 readings 0.477 s, of 5
    0.509, of 12 0.594, against 0.469 for the latest alone. The base wanders during a race (the
    driver pushing or saving tyres, traffic beyond the 3 s bins): what is left of one lap after its
    terms still carries into the next (correlation 0.57), and two readings differ more the further
    apart they are (0.38 s one lap apart, 0.58 s eight laps apart). Following a wandering base with
    the right weight is rung 3's job (a state-space model); plain exponential smoothing with weight
    0.7 on the newest reading already gives 0.460.
  * Skipping odd readings: 0.466 -> 0.442 (RMSE 1.42 -> 1.10). Without it, 168 laps that followed an
    odd clean lap (a standing restart, a lock-up) were predicted 11 s too slow on average: 0.4 % of
    the laps, 10 % of the error. The same guard on persistence: 0.479 -> 0.454. So of rung 2's gain
    over rung 1, about 0.025 s is the guard and 0.012 s the formula's terms. Skipping EVERY odd
    reading broke the wet laps (MAE 1.24 -> 2.02): when rain sets in, each lap is legitimately slower
    than the last, and the base stayed on the dry pace. Skipping only a single odd lap: wet 1.30.
  * Training on dry races only: the same accuracy, and the terms come out as Phase 2 measured them.
  * Not in: a "first flying lap on new tyres" term (-0.41 +- 0.12 s) and a "restart after a safety
    car or VSC" term (+0.61 +- 0.23 s). Both are real but too rare to change the score (-0.001 s
    each); the race simulator (Phase 4) should use them.
  * The traffic terms change nothing one lap ahead (0.464 without them): a car in traffic on lap N
    is usually still in traffic on lap N+1, and lap N's time already holds that cost. They stay in
    for the simulator, where traffic comes and goes.

Run it on the development races (2023 and 2024):

    python -m src.models.linear
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.analytics.degradation import SLICKS
from src.analytics.dirty_air import BIN_LABELS, fixed_effects_fit, gap_bin
from src.analytics.laps import OUTLIER_THRESHOLD_SEC
from src.models import evaluate
from src.models.persistence import EARLIER_CLEAN_LAP, THIS_LAP, Persistence
from src.pipeline import lake
from src.pipeline.feature_store import FUEL_END_KG, FUEL_START_KG

REFERENCE_COMPOUND = "MEDIUM"          # the compound offsets are measured against it
ODD_READING_SEC = OUTLIER_THRESHOLD_SEC  # step 1's rule: more than 3 s slower than it should be is odd
GUARD_READINGS = 3                     # ... than the median of the driver's previous 3 readings
ODD_LAP_SKIPPED = "lap N odd: an earlier clean lap"

# What Phase 2 measured, to check the fitted terms against (s per lap, s per kg):
PHASE_2 = {
    "wear SOFT": 0.064,            # step 3: pooled within-stint slope, fuel-corrected, dry races
    "wear MEDIUM": 0.049,
    "wear HARD": 0.044,
    "fuel": 0.031,                 # step 3: the fuel check, median over the races
    "traffic 0-0.5 s": 0.775,      # step 4: cost per gap bin against laps 3 s or more behind
    "traffic 0.5-1 s": 0.403,
    "traffic 1-1.5 s": 0.240,
    "traffic 1.5-2 s": 0.183,
    "traffic 2-3 s": 0.094,
}


# --- the terms ------------------------------------------------------------------------------------

def lap_terms(compound: pd.Series, tyre_age: pd.Series, fuel_kg: pd.Series, gap_sec: pd.Series) -> pd.DataFrame:
    """
    One column per term of the formula, one row per lap: the numbers the coefficients multiply.

      offset SOFT, offset HARD   1 on that compound, else 0 (the medium is the reference)
      wear SOFT/MEDIUM/HARD      the tyre's age on that compound, else 0
      fuel                       kg on board
      traffic <bin>              1 if the gap to the car ahead is in that bin (step 4's bins), else 0

    Intermediates and wets get no tyre terms (dry races cannot teach them), and neither does a lap
    with no compound or no tyre age. No car ahead counts as clear air: no traffic term.
    """
    terms = pd.DataFrame(index=compound.index)
    for c in SLICKS:
        if c != REFERENCE_COMPOUND:
            terms[f"offset {c}"] = (compound == c).astype(float)
    for c in SLICKS:
        terms[f"wear {c}"] = tyre_age.where(compound == c, 0.0)
    terms["fuel"] = fuel_kg
    bins = gap_bin(gap_sec)
    for label in BIN_LABELS:
        terms[f"traffic {label}"] = (bins == label).astype(float)
    return terms.astype(float).fillna(0.0)


def own_terms(laps: pd.DataFrame) -> pd.DataFrame:
    """Each lap's own terms: what the lap's own time contained (used to read the base off it)."""
    return lap_terms(laps["compound"], laps["tyre_age"], laps["fuel_mass_est"], laps["gap_on_track_start_sec"])


def next_lap_terms(laps: pd.DataFrame, laps_ahead: int = 1) -> pd.DataFrame:
    """
    The terms of lap N+k (k = laps_ahead), all known when lap N ends: the same compound, the tyre age
    + k, the fuel when lap N+k starts (FUEL_START_KG - (N + k - 1) x burn per lap, Gold's fuel model),
    and the gap to the car ahead as it was when lap N ended (for k = 1, exactly lap N+1's start gap).
    """
    burn_per_lap = (FUEL_START_KG - FUEL_END_KG) / laps["total_race_laps"]
    fuel_ahead = (FUEL_START_KG - (laps["lap_number"] + laps_ahead - 1) * burn_per_lap).clip(lower=0.0)
    return lap_terms(laps["compound"], laps["tyre_age"] + laps_ahead, fuel_ahead, laps["gap_on_track_end_sec"])


def dry_races(laps: pd.DataFrame) -> pd.Series:
    """True on the rows of races where nobody ran intermediates or wets."""
    any_wet = laps["is_wet_tyre"].astype(bool).groupby([laps["season"], laps["round"]]).transform("any")
    return ~any_wet


def driver_race(laps: pd.DataFrame) -> pd.Series:
    """One label per driver per race, e.g. '2024-23-VER': the groups that get their own base."""
    return laps["season"].astype(str) + "-" + laps["round"].astype(str) + "-" + laps["driver_code"].astype(str)


def drop_odd_readings(reading: pd.Series, key: pd.Series) -> pd.Series:
    """
    The readings without the odd ones. A reading is odd when it is more than ODD_READING_SEC slower
    than the median of the same driver's previous GUARD_READINGS readings (step 1's outlier rule,
    using earlier laps only). A single odd reading is skipped; the second odd reading in a row is
    kept, because two in a row are a new level (rain arriving, damage), not an incident. The first
    reading of a driver has nothing to compare with and is kept.
    """
    known = reading.dropna()
    by_driver = key[known.index]
    previous = known.groupby(by_driver).transform(
        lambda s: s.shift(1).rolling(GUARD_READINGS, min_periods=1).median())
    odd = known > previous + ODD_READING_SEC
    odd_before = odd.groupby(by_driver).shift(1, fill_value=False)
    skipped = odd & ~odd_before
    return known[~skipped].reindex(reading.index)


# --- the rung -------------------------------------------------------------------------------------

class LinearDecomposition:
    """Rung 2: the base from the driver's last good clean lap, plus the formula's terms for lap N+1."""

    name = "linear"

    def __init__(self) -> None:
        self.terms: Optional[pd.DataFrame] = None      # one row per term: coef, se
        self.training_races = 0

    def fit(self, train_laps: pd.DataFrame) -> "LinearDecomposition":
        rows = train_laps[train_laps["scored"] & dry_races(train_laps)]
        self.training_races = len(evaluate.races_of(rows))
        X = next_lap_terms(rows)
        if rows.empty:
            self.terms = pd.DataFrame({"coef": np.nan, "se": np.nan}, index=X.columns)
            return self
        race = rows["season"].astype(str) + "-" + rows["round"].astype(str)
        self.terms, _ = fixed_effects_fit(rows[evaluate.TARGET], X, driver_race(rows), race)
        return self

    def coefficients(self) -> pd.Series:
        """The fitted coefficients. A term the training races could not measure adds nothing (0)."""
        return self.terms["coef"].fillna(0.0)

    def base_readings(self, laps: pd.DataFrame) -> pd.Series:
        """One reading of the base per clean lap (its time minus its own terms); odd ones dropped."""
        coef = self.coefficients()
        reading = (laps["lap_time_sec"] - own_terms(laps)[coef.index] @ coef).where(laps["is_clean_lap"])
        return drop_odd_readings(reading, driver_race(laps))

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        coef = self.coefficients()
        # The latest reading of each driver, carried forward to the laps after it.
        base = self.base_readings(laps).groupby(driver_race(laps)).ffill()
        prediction = base + next_lap_terms(laps, laps_ahead)[coef.index] @ coef
        return prediction.rename("predicted_sec")


def base_source(rung: LinearDecomposition, laps: pd.DataFrame) -> pd.Series:
    """Where each prediction's base came from: lap N itself, an earlier clean lap, or an earlier one
    because lap N was odd."""
    kept = rung.base_readings(laps).notna()
    source = pd.Series(EARLIER_CLEAN_LAP, index=laps.index)
    source[laps["is_clean_lap"] & ~kept] = ODD_LAP_SKIPPED
    source[kept] = THIS_LAP
    return source


def terms_table(rung: LinearDecomposition) -> pd.DataFrame:
    """The fitted terms with their standard errors (clustered by race), next to Phase 2's numbers."""
    table = rung.terms.rename_axis("term").reset_index()
    table["phase_2"] = table["term"].map(PHASE_2)
    return table.round(4)


# --- command line ---------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 rung 2: linear decomposition, scored on the development "
                                                 "races and compared with rung 1")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to score (default: the development seasons, 2023 2024)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    card = evaluate.score(laps, evaluate.cross_validate(LinearDecomposition, laps), LinearDecomposition.name)
    print(evaluate.describe(card))

    # Against rung 1, on the same laps and the same bootstrap draws.
    baseline = evaluate.score(laps, evaluate.cross_validate(Persistence, laps), Persistence.name)
    result = evaluate.compare(baseline, card)
    print(f"\nAgainst {result['old']}: dry MAE {result['old_mae']:.3f} -> {result['new_mae']:.3f} s, "
          f"difference {result['difference']:+.3f} s (95 % interval {result['low']:+.3f} to {result['high']:+.3f}): "
          f"{result['verdict']}")

    # The terms, fitted on every race given (for reading: the scores above are cross-validated).
    rung = LinearDecomposition().fit(laps)
    terms = terms_table(rung)
    print(f"\nTerms fitted on the {rung.training_races} dry races (SE clustered by race), next to Phase 2:")
    print(terms.to_string(index=False))

    # Where the base came from, and how good each case was (bases read with the terms above).
    source = base_source(rung, laps)[card.errors.index]
    by_source = pd.DataFrame({"laps": card.errors.groupby(source).size(),
                              "persistence_mae": baseline.errors["error_sec"].abs().groupby(source).mean(),
                              "linear_mae": card.errors["error_sec"].abs().groupby(source).mean()})
    print("\nBy where the base came from:")
    print(by_source.round(3).to_string())

    print(f"\nReports: {evaluate.write_scorecard(card, args.lake)}, "
          f"{lake.write_report(terms, 'pace_model_linear_terms', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
