"""
Boosted corrections (src/models/boosted.py) - Phase 3 rung 4

Rung 3 is the lap-time formula with a base pace that wanders. What it still gets wrong is not all
noise: some of it follows from what is known when lap N ends. Rung 4 learns that part with LightGBM
(gradient-boosted decision trees) and adds it on:

    prediction for lap N+k = rung 3's prediction + correction(what is known when lap N ends, k)

WHY A CORRECTION AND NOT THE LAP TIME ITSELF
    A tree predicts the average of the training laps that land in the same leaf, so it cannot go
    beyond the values it was trained on. Lap times run from about 65 s to 110 s depending on the
    circuit; a tree predicting them directly would mostly be learning which circuit it is at. Rung 3
    already gets the lap time to within a few tenths, so the trees only learn the few tenths: rung 3's
    error, actual - rung 3's prediction. With nothing to learn, the correction is 0 and rung 4 is rung 3.

WHAT THE TREES SEE (all known when lap N ends; see features())
    lap N+k, known in advance   laps ahead (k), its lap number, race progress, compound, tyre age,
                                laps into the stint, whether the set was new
    traffic at the end of lap N gap to the car ahead on the road, gap to the car ahead in the order,
                                gap to the leader, position
    rung 3's own state          how unsure its level is (P), laps since its last reading, its error on
                                lap N itself, whether lap N's reading was skipped as odd, how fast its
                                level moved over the last DRIFT_LAPS laps
    the driver's noise          spread of their last 3 clean laps (Gold's roll3_clean_std_sec)
    what lap N was              clean, out-lap, yellow, safety car, VSC, red flag
    Left out on purpose: driver, team, circuit and the weather (see "Measured" below).

HOW IT IS FITTED (fit(), on the training races only)
    1. Rung 3 is fitted on the training races. Its errors on those same races are the targets: its
       in-sample dry MAE is 0.431 s against 0.434 s out of fold, so they are honest enough (rung 3 has
       only 12 numbers to fit).
    2. Two models. The next lap (k = 1) gets its own; laps 2 to AHEAD_TRAINED_LAPS ahead share one, with
       k as an input (each scored lap N appears once per k). A lap ahead beyond that is treated as
       AHEAD_TRAINED_LAPS ahead: no tree has a split on k above it.
    3. Only laps on slick tyres train the trees, and laps on intermediates or wets get no correction:
       the five development races with wet-tyre laps are too few to learn from (trained on them, the
       trees made the wet laps worse: 1.30 -> 1.43 s).
    4. The loss is the absolute error (L1), so each leaf predicts the MEDIAN error of its laps, the
       number that makes the MAE smallest. The squared error (L2) chased the rare big misses and
       stopped after about 10 trees with almost no gain (0.434 -> 0.433).
    5. How many trees: cross-validation by race INSIDE the training races (INNER_FOLDS folds of whole
       races). Trees are added until the MAE on the held-out inner races has not improved (by at least
       MIN_IMPROVEMENT_SEC) for STOP_AFTER_TREES trees; the model is then refitted on all training races
       with the number of trees that scored best.
       This is what keeps the trees from memorising the training races.

Measured on the development races (2023 + 2024, cross-validated by race) before choosing:
  * The weather was the strongest overfitting: air and track temperature were among the most used
    inputs (19 % of the gain), yet leaving them out made the score BETTER: 0.411 -> 0.403 one lap ahead,
    0.677 -> 0.648 ten laps ahead. A temperature names a race; the trees used it to recall training
    races, which does not carry over to a new one. The fuel and the laps remaining are fixed by the lap
    number and the race length, so they are left out too (no change).
  * One model for the next lap: 0.396 against 0.403 when the next lap is one more horizon in the shared
    model. Further ahead the shared model beats a model per horizon (3 / 5 / 10 laps ahead: 0.491 /
    0.547 / 0.641 against 0.494 / 0.553 / 0.644): those laps are alike and learn from each other.
  * The settings hardly matter: every variation tried (learning rate 0.02 or 0.05, 15 to 63 leaves,
    30 to 1,000 laps per leaf) lands within 0.003 s one lap ahead. The smallest leaf (100 laps) is best
    for the next-lap model, 400 for the shared model (0.640 against 0.647 ten laps ahead).
  * What the trees learned: (1) the base comes back. When rung 3's level fell fast over the last 5
    laps, the next laps are slower than rung 3 says (the fastest-falling tenth: +0.31 s ten laps
    ahead), and the other way round. Rung 3's level is a random walk; the real pace reverts. (2) A lap
    much slower than expected is mostly a one-off (the next lap is ~0.5 s faster than rung 3, which gave
    it 0.77 of the weight); a lap much faster than expected mostly stays. (3) The laps after a gap in
    the readings: after an out-lap (the first flying lap on new tyres) MAE 1.20 -> 0.96, after a safety
    car, VSC or red flag (the restart lap) 1.15 -> 0.90.
  * The whole field's movement (median of every driver's level drift and last error, a lap earlier)
    added 0.001 to 0.003 s; doing it without any look at later laps would need the time every lap ended,
    which Gold does not carry. Not in.
  * The hand-written baseline below (SpecialLaps: rung 3 + the median error of each kind of lap) gains
    0.006 to 0.010 s. About three quarters of rung 4's gain one lap ahead is on ordinary laps (lap N a
    clean lap, 0.397 -> 0.364), where no rule about special laps has anything to say.

Run it on the development races (2023 and 2024):

    python -m src.models.boosted
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models import evaluate
from src.models.linear import LinearDecomposition, driver_race
from src.models.persistence import Persistence
from src.models.state_space import StateSpace
from src.pipeline import lake

AHEAD_TRAINED_LAPS = 10        # the shared model learns laps 2 .. 10 ahead
DRIFT_LAPS = 5                 # level_drift: how far rung 3's level moved over this many laps, per lap
INNER_FOLDS = 5                # cross-validation inside the training races, to choose the number of trees
MAX_TREES = 3000
STOP_AFTER_TREES = 100         # stop adding trees when the inner MAE has not improved for this many
MIN_IMPROVEMENT_SEC = 1e-5     # ... by at least this much (0.01 ms): with nothing to learn, no endless trees
MIN_TRAINING_RACES = 2         # fewer than this: nothing to cross-validate, no correction (= rung 3)
HARDNESS = {"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 2.0}   # slicks only: wet tyres never get a correction

# The same for both models. seed + deterministic + a fixed number of threads: the same trees every run.
COMMON_PARAMS = {
    "objective": "l1",          # absolute error: each leaf predicts the median error of its laps
    "metric": "l1",             # and the number of trees is chosen by the MAE, the headline
    "learning_rate": 0.05,      # each tree's output is shrunk to 5 %: many small steps, none trusted alone
    "num_leaves": 31,           # how detailed one tree may be
    "feature_fraction": 0.8,    # each tree sees a random 80 % of the inputs ...
    "bagging_fraction": 0.8,    # ... and a random 80 % of the laps, so no tree relies on a few of them
    "bagging_freq": 1,
    "lambda_l2": 1.0,           # pulls each leaf's value a little towards 0
    "seed": 2026,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
NEXT_LAP_PARAMS = {**COMMON_PARAMS, "min_data_in_leaf": 100}   # ~35,000 training laps
AHEAD_PARAMS = {**COMMON_PARAMS, "min_data_in_leaf": 400}      # ~270,000 rows (laps x 9 horizons)

LAP_AHEAD_INPUTS = ["predicted_lap", "predicted_progress", "compound", "predicted_tyre_age",
                    "predicted_stint_lap", "fresh_tyre"]
TRAFFIC_INPUTS = ["gap_on_track_end_sec", "gap_ahead_sec", "gap_to_leader_sec", "position"]
RUNG3_INPUTS = ["level_variance", "laps_since_reading", "last_error", "odd_reading_skipped", "level_drift"]
LAP_N_INPUTS = ["roll3_clean_std_sec", "is_clean_lap", "is_out_lap", "is_yellow", "is_sc", "is_vsc", "is_red"]
NEXT_LAP_FEATURES = LAP_AHEAD_INPUTS + TRAFFIC_INPUTS + RUNG3_INPUTS + LAP_N_INPUTS
AHEAD_FEATURES = ["laps_ahead"] + NEXT_LAP_FEATURES

# Kinds of lap N, for the hand-written baseline and for reading the results (first that applies).
AFTER_STOP = "lap N an out-lap"                 # lap N+1 is the first flying lap on the new tyres
AFTER_NEUTRALISED = "lap N neutralised"         # safety car, VSC or red flag: lap N+1 is a restart
NO_READING = "lap N no reading"                 # not a clean lap for another reason, or skipped as odd
ORDINARY = "lap N a reading"
LAP_KINDS = (AFTER_STOP, AFTER_NEUTRALISED, NO_READING, ORDINARY)


# --- the inputs -----------------------------------------------------------------------------------

def rung3_inputs(rung3: StateSpace, laps: pd.DataFrame) -> pd.DataFrame:
    """
    What rung 3 knows about each driver when lap N ends, one row per lap:

      level_variance       P, how unsure the level is (units of one reading's noise; grows on laps
                           without a reading)
      laps_since_reading   0 if lap N gave a reading, 1 if lap N-1 did, ...; NaN before the first
      last_error           lap N's time - rung 3's prediction of lap N, made when lap N-1 ended
      odd_reading_skipped  1 if lap N was clean but its reading was skipped as odd (rung 2's guard)
      level_drift          (level after lap N - level after lap N-DRIFT_LAPS) / DRIFT_LAPS, s per lap

    Every value uses laps up to N only: the filter runs forward, and the shifts look back.
    """
    key = driver_race(laps)
    reading = rung3.rung2.base_readings(laps)
    level, variance = rung3.filter(laps)
    has_reading = reading.notna()
    readings_so_far = has_reading.groupby(key).cumsum()
    # Rows since the latest reading: number the rows of each run that starts at a reading (0 on the
    # reading itself, 1 on the next lap, ...); before a driver's first reading there is no run: NaN.
    since = has_reading.groupby([key, readings_so_far]).cumcount().where(readings_so_far > 0)
    lap_n_predicted = rung3.predict(laps, 1).groupby(key).shift(1)
    return pd.DataFrame({
        "level_variance": variance,
        "laps_since_reading": since.astype(float),
        "last_error": laps["lap_time_sec"] - lap_n_predicted,
        "odd_reading_skipped": (laps["is_clean_lap"].astype(bool) & ~has_reading).astype(float),
        "level_drift": (level - level.groupby(key).shift(DRIFT_LAPS)) / DRIFT_LAPS,
    }, index=laps.index)


def features(laps: pd.DataFrame, inputs: pd.DataFrame, laps_ahead: int) -> pd.DataFrame:
    """
    The trees' inputs for predicting lap N+k (k = laps_ahead) from the row of lap N.
    Lap N+k's lap number, progress, tyre age and laps into the stint are lap N's plus k: a scored
    lap N+k is on the same tyres (laps N+1 .. N+k are all pace laps, so none is a pit lap).
    inputs: rung3_inputs() of the same laps.
    """
    k = float(laps_ahead)
    f = pd.DataFrame(index=laps.index)
    f["laps_ahead"] = k
    f["predicted_lap"] = laps["lap_number"] + k
    f["predicted_progress"] = (laps["lap_number"] + k) / laps["total_race_laps"]
    f["compound"] = laps["compound"].map(HARDNESS)          # 0 soft, 1 medium, 2 hard; else NaN
    f["predicted_tyre_age"] = laps["tyre_age"] + k
    f["predicted_stint_lap"] = laps["stint_lap"] + k
    f["fresh_tyre"] = laps["fresh_tyre"]
    for column in TRAFFIC_INPUTS + LAP_N_INPUTS:
        f[column] = laps[column]
    for column in RUNG3_INPUTS:
        f[column] = inputs[column]
    return f.astype(float)


def lap_kind(laps: pd.DataFrame, inputs: pd.DataFrame) -> pd.Series:
    """What lap N was, for the row of lap N: the first of LAP_KINDS that applies."""
    neutralised = laps["is_sc"].astype(bool) | laps["is_vsc"].astype(bool) | laps["is_red"].astype(bool)
    kind = np.select([laps["is_out_lap"].astype(bool), neutralised, inputs["laps_since_reading"] != 0],
                     [AFTER_STOP, AFTER_NEUTRALISED, NO_READING], default=ORDINARY)
    return pd.Series(kind, index=laps.index, name="lap_kind")


def rung3_errors(rung3: StateSpace, laps: pd.DataFrame, laps_ahead: int) -> Tuple[pd.Series, pd.Series]:
    """Rung 3's error to learn (actual - predicted, s) on the laps that train a correction: scored k laps
    ahead and on slick tyres at lap N. Returns (which rows, error)."""
    answers = evaluate.ahead_answers(laps, laps_ahead)
    rows = answers["scored"] & ~laps["is_wet_tyre"].astype(bool)
    return rows, answers["actual_sec"] - rung3.predict(laps, laps_ahead)


def training_table(rung3: StateSpace, laps: pd.DataFrame, inputs: pd.DataFrame,
                   horizons: Sequence[int]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """The rows one model learns from: every training lap once per horizon. Returns the inputs, rung 3's
    errors (the target) and each row's (season, round) for the race-grouped folds."""
    X, y, where = [], [], []
    for k in horizons:
        rows, error = rung3_errors(rung3, laps, k)
        X.append(features(laps, inputs, k)[rows])
        y.append(error[rows])
        where.append(laps.loc[rows, ["season", "round"]])
    return (pd.concat(X, ignore_index=True), pd.concat(y, ignore_index=True),
            pd.concat(where, ignore_index=True))


def fit_correction(X: pd.DataFrame, y: pd.Series, where: pd.DataFrame,
                   params: Dict) -> Tuple[Optional[lgb.Booster], int]:
    """
    One LightGBM model of rung 3's error, with its number of trees chosen by cross-validation over
    whole training races. Returns (model, trees); (None, 0) when there are too few races to do that.
    """
    if len(evaluate.races_of(where)) < MIN_TRAINING_RACES:
        return None, 0
    folds = []
    for held_out in evaluate.race_folds(where, INNER_FOLDS):
        is_held_out = evaluate.in_races(where, held_out).to_numpy()
        folds.append((np.flatnonzero(~is_held_out), np.flatnonzero(is_held_out)))
    data = lgb.Dataset(X, y, free_raw_data=False)
    history = lgb.cv(params, data, num_boost_round=MAX_TREES, folds=folds,
                     callbacks=[lgb.early_stopping(STOP_AFTER_TREES, verbose=False, min_delta=MIN_IMPROVEMENT_SEC)])
    trees = len(history["valid l1-mean"])       # cut at the best number of trees
    return lgb.train(params, data, num_boost_round=trees), trees


# --- the rungs ------------------------------------------------------------------------------------

class Boosted:
    """Rung 4: rung 3, plus LightGBM's correction of rung 3's error."""

    name = "boosted"

    def __init__(self) -> None:
        self.rung3: Optional[StateSpace] = None
        self.next_lap: Optional[lgb.Booster] = None    # k = 1
        self.ahead: Optional[lgb.Booster] = None       # k = 2 .. AHEAD_TRAINED_LAPS
        self.trees = {"next lap": 0, "ahead": 0}

    def fit(self, train_laps: pd.DataFrame) -> "Boosted":
        self.rung3 = StateSpace().fit(train_laps)
        inputs = rung3_inputs(self.rung3, train_laps)
        X, y, where = training_table(self.rung3, train_laps, inputs, [1])
        self.next_lap, self.trees["next lap"] = fit_correction(X[NEXT_LAP_FEATURES], y, where, NEXT_LAP_PARAMS)
        X, y, where = training_table(self.rung3, train_laps, inputs, range(2, AHEAD_TRAINED_LAPS + 1))
        self.ahead, self.trees["ahead"] = fit_correction(X[AHEAD_FEATURES], y, where, AHEAD_PARAMS)
        return self

    def correction(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        """What the trees add to rung 3's prediction, s (0 on wet tyres, or with no model)."""
        model, columns = (self.next_lap, NEXT_LAP_FEATURES) if laps_ahead == 1 else (self.ahead, AHEAD_FEATURES)
        correction = pd.Series(0.0, index=laps.index, name="correction_sec")
        slick = ~laps["is_wet_tyre"].astype(bool)
        if model is None or not slick.any():
            return correction
        X = features(laps, rung3_inputs(self.rung3, laps), laps_ahead)
        correction[slick] = model.predict(X.loc[slick, columns])
        return correction

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        prediction = self.rung3.predict(laps, laps_ahead) + self.correction(laps, laps_ahead)
        return prediction.rename("predicted_sec")

    def importance(self) -> pd.DataFrame:
        """Each input's share of the trees' total gain (how much its splits reduced the error), in %."""
        table = pd.DataFrame(index=pd.Index(AHEAD_FEATURES, name="input"))
        for label, model in (("next lap %", self.next_lap), ("2-10 laps ahead %", self.ahead)):
            gain = pd.Series(np.nan, index=table.index) if model is None else \
                pd.Series(model.feature_importance("gain"), index=model.feature_name())
            table[label] = 100 * gain / gain.sum() if gain.sum() > 0 else 0.0
        return table.fillna(0.0).sort_values(list(table.columns), ascending=False).round(1)


class SpecialLaps:
    """
    The hand-written baseline for rung 4: rung 3, plus the median of rung 3's error over the training
    laps of the same kind (LAP_KINDS) and the same number of laps ahead. Slick tyres only, like rung 4.
    """

    name = "special_laps"

    def __init__(self) -> None:
        self.rung3: Optional[StateSpace] = None
        self.medians: Optional[pd.DataFrame] = None    # one row per kind of lap N, one column per k

    def fit(self, train_laps: pd.DataFrame) -> "SpecialLaps":
        self.rung3 = StateSpace().fit(train_laps)
        kind = lap_kind(train_laps, rung3_inputs(self.rung3, train_laps))
        columns = {}
        for k in range(1, AHEAD_TRAINED_LAPS + 1):
            rows, error = rung3_errors(self.rung3, train_laps, k)
            columns[k] = error[rows].groupby(kind[rows]).median()
        self.medians = pd.DataFrame(columns).reindex(list(LAP_KINDS))
        return self

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
        kind = lap_kind(laps, rung3_inputs(self.rung3, laps))
        k = min(laps_ahead, AHEAD_TRAINED_LAPS)
        correction = kind.map(self.medians[k]).fillna(0.0).where(~laps["is_wet_tyre"].astype(bool), 0.0)
        return (self.rung3.predict(laps, laps_ahead) + correction).rename("predicted_sec")


# --- command line ---------------------------------------------------------------------------------

def by_lap_kind(cards: Dict[str, evaluate.Scorecard], kind: pd.Series) -> pd.DataFrame:
    """Dry MAE one lap ahead for each kind of lap N, one column per rung."""
    table = {}
    for name, card in cards.items():
        dry = card.dry_errors()
        table["laps"] = dry.groupby(kind[dry.index]).size()      # the same laps for every rung
        table[name] = dry["error_sec"].abs().groupby(kind[dry.index]).mean()
    return pd.DataFrame(table).reindex(list(LAP_KINDS)).round(3)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 rung 4: LightGBM corrections of rung 3, scored on the "
                                                 "development races, against rungs 1-3 and a hand-written baseline")
    parser.add_argument("--season", type=int, nargs="+", default=list(evaluate.DEVELOPMENT_SEASONS),
                        help="seasons to score (default: the development seasons, 2023 2024)")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    laps = evaluate.load_laps(args.season, args.lake)
    rungs = [Persistence, LinearDecomposition, StateSpace, SpecialLaps, Boosted]
    cards = evaluate.horizon_scorecards(rungs, laps)       # every rung, cross-validated, at every horizon
    card = cards[Boosted.name][1]
    print(evaluate.describe(card))

    for old in (StateSpace.name, SpecialLaps.name):
        result = evaluate.compare(cards[old][1], card)
        print(f"\nAgainst {result['old']}: dry MAE {result['old_mae']:.3f} -> {result['new_mae']:.3f} s, "
              f"difference {result['difference']:+.4f} s (95 % interval {result['low']:+.4f} to "
              f"{result['high']:+.4f}): {result['verdict']}")

    horizons = evaluate.horizon_rows(cards)
    print("\nLaps ahead (dry MAE, s; each rung against the one above it, paired bootstrap):")
    print(horizons.to_string(index=False))
    print(f"\n{Boosted.name} against {StateSpace.name}, laps ahead:")
    for k in evaluate.HORIZONS:
        result = evaluate.compare(cards[StateSpace.name][k], cards[Boosted.name][k])
        print(f"  {k:2d}: {result['old_mae']:.3f} -> {result['new_mae']:.3f} s, {result['difference']:+.4f} "
              f"({result['low']:+.4f} to {result['high']:+.4f}): {result['verdict']}")

    # Fitted on every race given: the overfitting check and what the trees use (for reading only).
    rung = Boosted().fit(laps)
    in_sample = evaluate.score(laps, evaluate.predictions_of(rung, laps), Boosted.name)
    print(f"\nFitted on all {len(evaluate.races_of(laps))} races: {rung.trees['next lap']} trees for the next lap, "
          f"{rung.trees['ahead']} for 2-{AHEAD_TRAINED_LAPS} laps ahead (chosen by cross-validation inside the "
          f"training races). Dry MAE on the races it learned from {in_sample.dry_mae:.3f} s, on races it never "
          f"saw {card.dry_mae:.3f} s.")
    importance = rung.importance()
    print("\nWhat the trees use (share of the total gain):")
    print(importance.to_string())

    kind = lap_kind(laps, rung3_inputs(rung.rung3, laps))
    one_lap = {name: cards[name][1] for name in (StateSpace.name, SpecialLaps.name, Boosted.name)}
    print("\nDry MAE one lap ahead by what lap N was:")
    print(by_lap_kind(one_lap, kind).to_string())

    print(f"\nReports: {evaluate.write_scorecard(card, args.lake)}, "
          f"{lake.write_report(horizons, 'pace_model_horizons', args.lake)}, "
          f"{lake.write_report(importance.reset_index(), 'pace_model_boosted_inputs', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
