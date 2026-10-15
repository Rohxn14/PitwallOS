"""
State-space model (src/models/state_space.py) - Phase 3 rung 3

Rung 2 reads the driver's base pace off the latest good clean lap and trusts it completely. But one
lap is noisy, and the base also really moves during a race. A state-space model holds both ideas:

    hidden state     the level L: the driver's base pace (rung 2's base), which wanders
    every lap        L moves by a random step of variance q          (the driver's pace may have moved)
    every clean lap  gives one reading = L + noise of variance h      (rung 2's reading: the lap time
                                                                       minus its own terms)

The Kalman filter keeps an estimate of L and of how unsure it is (its variance P). Every lap:

    1.  P = P + q                      time passes: the level may have moved, so we are less sure
    and when the lap gives a reading:
    2.  K = P / (P + h)                the gain: how much to trust the new reading (0 = not at all, 1 = fully)
    3.  L = L + K x (reading - L)      move the estimate part of the way towards the reading
    4.  P = (1 - K) x P                and become surer

Laps without a reading (a stop, a safety car) still add q, so the first reading after them is trusted
more: what a fixed average cannot do. A driver's first reading starts L (with P = h). On intermediates
and wets every reading restarts L the same way: a wet track changes too fast to average over (measured:
the best ratio for wet laps was as large as the search allowed, i.e. no averaging).

The prediction for lap N+k:  L after lap N + lap N+k's terms (rung 2's: compound, wear, fuel, traffic),
with a spread (standard deviation) of  sqrt(h x (P + k x q / h + 1))  for the bands.

Only q / h matters for the prediction itself (call it rho). It is chosen to predict the training races
best (the one-lap MAE on their dry laps). h only sets the width of the bands: chosen so that half of
the training laps fall inside the 50 % band.

Measured on the development races (2023 + 2024, cross-validated by race) before choosing:
  * Maximum likelihood, the textbook way to fit q and h, failed: it chose h = 0 and q = (1.1 s)^2, i.e.
    trust every lap fully (= rung 2). It assumes bell-shaped errors; real laps have rare large jumps
    (restarts, a car ahead pitting, team orders), and a bell curve can only explain those by making the
    level jump all the time. Choosing rho by the prediction error itself does not have that problem.
  * A second hidden state for this race's wear rate (starting from rung 2's pooled slope, learned lap
    by lap, restarted at each tyre change) changed the MAE by less than 0.001 s at 1, 5 and 10 laps
    ahead: one driver's own laps cannot pin down the race's wear better than the pooled slope. Not in.
  * Capping how far one reading may move the level (a "robust" filter) was far worse (0.64 s at a 2 s
    cap): big jumps in the level are mostly real, and a filter that will not follow them stays behind.
  * The ratio chosen per fold: rho 2.2 to 3.8, so each new reading gets 0.75 to 0.83 of the weight.
  * The bands, out of fold: 49.7 % of dry laps inside the 50 % band, 79.5 % inside the 90 %, 83.4 %
    inside the 95 %: honest in the middle, too narrow in the tails (the rare large jumps again). Wet
    laps get their own noise scale (sqrt(h) 0.39 s against 0.16 s on slicks): 50.3 / 71.9 / 76.6 %.

Run it on the development races (2023 and 2024):

    python -m src.models.state_space
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from src.models import evaluate
from src.models.linear import LinearDecomposition, driver_race, next_lap_terms
from src.models.persistence import Persistence
from src.pipeline import lake

LOG_RHO_BOUNDS = (-3.0, 5.0)        # rho = q / h is searched between e^-3 (0.05) and e^5 (148)
NO_DRY_LAPS_RHO = 1e6               # nothing to choose rho from: trust every reading fully (= rung 2)
BANDS = (0.50, 0.90, 0.95)


def level_filter(reading: pd.Series, key: pd.Series, rho: float,
                 restart: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    The Kalman filter, run for every driver at once, in units of the noise of one reading (h = 1).

      reading   one reading of the base per clean lap (NaN on the other laps)
      key       which driver in which race each row belongs to (each driver's rows in lap order)
      rho       q / h: how far the level may move in one lap, against the noise of one reading
      restart   rows whose reading replaces the level outright (wet tyres)

    Returns the level and its variance (in units of h) after each row; NaN before a driver's first reading.
    """
    codes, drivers = pd.factorize(key)
    lap_index = key.groupby(key).cumcount().to_numpy()         # 0, 1, 2, ... down each driver's rows
    n_drivers, n_laps = len(drivers), int(lap_index.max()) + 1
    readings = np.full((n_drivers, n_laps), np.nan)
    readings[codes, lap_index] = reading.to_numpy(dtype=float)
    restarts = np.zeros((n_drivers, n_laps), dtype=bool)
    restarts[codes, lap_index] = restart.to_numpy(dtype=bool)

    level = np.full(n_drivers, np.nan)
    var = np.full(n_drivers, np.nan)
    level_after = np.full((n_drivers, n_laps), np.nan)
    var_after = np.full((n_drivers, n_laps), np.nan)
    for j in range(n_laps):                                     # one lap at a time, every driver at once
        var = var + rho                                         # 1. time passes: the level may have moved
        y = readings[:, j]
        seen = ~np.isnan(y)
        start = seen & (np.isnan(level) | restarts[:, j])       # a first reading, or one on wet tyres:
        level[start], var[start] = y[start], 1.0                #    the level is that lap, +- one reading's noise
        update = seen & ~start
        gain = var[update] / (var[update] + 1.0)                # 2. how much to trust the new reading
        level[update] += gain * (y[update] - level[update])     # 3. move part of the way towards it
        var[update] *= 1.0 - gain                               # 4. and become surer
        level_after[:, j], var_after[:, j] = level, var
    return (pd.Series(level_after[codes, lap_index], index=reading.index),
            pd.Series(var_after[codes, lap_index], index=reading.index))


def steady_gain(rho: float) -> float:
    """The weight a new reading gets once the filter has settled (lap after lap with a reading)."""
    return (-rho + np.sqrt(rho * rho + 4.0 * rho)) / 2.0


class StateSpace:
    """Rung 3: rung 2's terms, with the base followed by a Kalman filter instead of the latest lap."""

    name = "state_space"

    def __init__(self) -> None:
        self.rung2: Optional[LinearDecomposition] = None    # the terms and the readings
        self.rho = np.nan                                   # q / h
        self.h = np.nan                                     # the noise of one reading on slicks, s^2
        self.h_wet = np.nan                                 # ... on intermediates and wets

    def fit(self, train_laps: pd.DataFrame) -> "StateSpace":
        self.rung2 = LinearDecomposition().fit(train_laps)
        target = train_laps[evaluate.TARGET]
        scored = train_laps["scored"]
        dry = scored & (train_laps["conditions"] == evaluate.DRY)
        wet = scored & (train_laps["conditions"] == evaluate.WET)
        if not dry.any():
            self.rho = NO_DRY_LAPS_RHO
            return self

        def dry_mae(log_rho: float) -> float:
            self.rho = float(np.exp(log_rho))
            return float((self.predict(train_laps) - target)[dry].abs().mean())

        best = minimize_scalar(dry_mae, bounds=LOG_RHO_BOUNDS, method="bounded")
        self.rho = float(np.exp(best.x))
        # The noise scale: half the training laps inside the 50 % band (the median of a bell curve's
        # absolute values is 0.6745 standard deviations). A wet track is noisier: its own scale.
        _, var = self.filter(train_laps)
        z = ((self.predict(train_laps) - target) / np.sqrt(var + self.rho + 1.0)).abs()
        self.h = float((z[dry].median() / norm.ppf(0.75)) ** 2)
        self.h_wet = float((z[wet].median() / norm.ppf(0.75)) ** 2) if wet.any() else self.h
        return self

    def filter(self, laps: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """The level and its variance (units of h) after every lap, from rung 2's readings."""
        reading = self.rung2.base_readings(laps)
        return level_filter(reading, driver_race(laps), self.rho, laps["is_wet_tyre"].astype(bool))

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        level, _ = self.filter(laps)
        coef = self.rung2.coefficients()
        prediction = level + next_lap_terms(laps, laps_ahead)[coef.index] @ coef
        return prediction.rename("predicted_sec")

    def predict_sd(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        """The spread of each prediction (one standard deviation, s): what the bands are made of."""
        _, var = self.filter(laps)
        h = np.where(laps["is_wet_tyre"].astype(bool), self.h_wet, self.h)
        return np.sqrt(h * (var + laps_ahead * self.rho + 1.0)).rename("predicted_sd_sec")


def band_coverage(laps: pd.DataFrame, n_folds: int = evaluate.N_FOLDS) -> pd.DataFrame:
    """Out of fold: the share of scored laps (one lap ahead) inside the 50 / 90 / 95 % bands, dry and wet."""
    z = pd.Series(np.nan, index=laps.index)
    for held_out in evaluate.race_folds(laps, n_folds):
        is_held_out = evaluate.in_races(laps, held_out)
        rung = StateSpace().fit(laps[~is_held_out])
        z[is_held_out] = band_distance(rung, laps[is_held_out])
    return coverage_table(z, laps)


def band_distance(rung: StateSpace, laps: pd.DataFrame) -> pd.Series:
    """|error| / spread of each prediction one lap ahead: inside the 95 % band when below 1.96."""
    return ((rung.predict(laps) - laps[evaluate.TARGET]) / rung.predict_sd(laps)).abs()


def coverage_table(z: pd.Series, laps: pd.DataFrame) -> pd.DataFrame:
    """The share of scored laps (one lap ahead) inside the 50 / 90 / 95 % bands, dry and wet."""
    rows = []
    for conditions in (evaluate.DRY, evaluate.WET):
        zz = z[laps["scored"] & (laps["conditions"] == conditions)]
        rows.append({"conditions": conditions, "laps": len(zz),
                     **{f"inside {100 * b:.0f} % band": round(100 * float((zz < norm.ppf(0.5 + b / 2)).mean()), 1)
                        for b in BANDS}})
    return pd.DataFrame(rows)


# --- command line ---------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 rung 3: state-space model, scored on the development "
                                                 "races, compared with rungs 1 and 2, also several laps ahead")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to score (default: the development seasons, 2023 2024)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    card = evaluate.score(laps, evaluate.cross_validate(StateSpace, laps), StateSpace.name)
    print(evaluate.describe(card))

    # Against rung 2, on the same laps and the same bootstrap draws.
    rung2 = evaluate.score(laps, evaluate.cross_validate(LinearDecomposition, laps), LinearDecomposition.name)
    result = evaluate.compare(rung2, card)
    print(f"\nAgainst {result['old']}: dry MAE {result['old_mae']:.3f} -> {result['new_mae']:.3f} s, "
          f"difference {result['difference']:+.4f} s (95 % interval {result['low']:+.4f} to {result['high']:+.4f}): "
          f"{result['verdict']}")

    # What the filter learned, fitted on every race given (for reading: the scores are cross-validated).
    rung = StateSpace().fit(laps)
    print(f"\nFitted on all {len(evaluate.races_of(laps))} races: rho = q / h = {rung.rho:.2f}, so a new reading "
          f"gets {steady_gain(rung.rho):.2f} of the weight once settled; one reading's noise sqrt(h) = "
          f"{np.sqrt(rung.h):.3f} s on slicks, {np.sqrt(rung.h_wet):.3f} s on wet tyres; the level's step "
          f"per lap sqrt(q) = {np.sqrt(rung.h * rung.rho):.3f} s")

    print("\nThe bands, out of fold (a fair band holds its share of laps):")
    print(band_coverage(laps).to_string(index=False))

    horizons = evaluate.horizon_table([Persistence, LinearDecomposition, StateSpace], laps)
    print("\nLaps ahead (dry MAE, s; each rung against the one above it, paired bootstrap):")
    print(horizons.to_string(index=False))

    print(f"\nReports: {evaluate.write_scorecard(card, args.lake)}, "
          f"{lake.write_report(horizons, 'pace_model_horizons', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
