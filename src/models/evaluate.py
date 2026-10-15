"""
Evaluation harness (src/models/evaluate.py) - Phase 3 step 0

Every rung of the pace-model ladder is judged here, in exactly the same way, so the rungs can be
compared fairly. The ladder: persistence -> linear decomposition -> state-space tyre model -> LightGBM.
A rung must beat the one below it on races it has never seen, or the reason is written down.

WHAT IS PREDICTED
    When lap N ends, the driver's time for lap N+1, in seconds (Gold's target_next_lap_time_sec).
    Also, for the race simulator (Phase 4), lap N+k for k laps ahead: see HOW FAR AHEAD below.

WHAT A RUNG IS
    Any object with three things (see the Rung class below):
        name                        a short label, e.g. "persistence"
        fit(train_laps)             learn from complete races, answers included
        predict(laps, laps_ahead)   one predicted time per row: a pd.Series on laps.index
    predict() never sees the answers: model_inputs() removes ANSWER_COLUMNS first. It gets every row
    of one or more races, and for the row of lap N it may only use laps up to N of that race (rule 5
    of the pipeline, now for models). future_leaks() checks this by cutting races short.

WHICH LAPS ARE SCORED (the same laps for every rung)
    1. lap N+1 is a pace lap (Phase 2 step 1: green, has a lap time, not a pit lap, not an outlier).
       A safety-car lap or an in-lap is not pace; no lap-time model is meant to predict it.
    2. the driver has driven at least one clean lap by the end of lap N, so the model has seen their
       pace in this race. Before that, it could only guess from lap 1's standing start.
    Every row gets a score_status: "scored", or the first reason it is not. The ledger counts them.
    Measured on 2023 + 2024 before building: 51,024 rows, 43,258 with a pace lap next. 852 of those
    come before the driver's first clean lap (590 are lap 1 -> lap 2). Copying the current lap missed
    789 of them by 10.8 s on average, and the other 63 had no lap time at all: 2 % of the laps, but
    about a quarter of all the error. Rule 2 leaves them out.

HOW FAR AHEAD
    The simulator will ask for laps further ahead than the next one. Predicting lap N+k when lap N
    ends (k = laps_ahead) is scored when laps N+1, N+2, ..., N+k are ALL pace laps (the same tyres
    and green racing all the way) and rule 2 holds. What lap N+k will be driven on is known in
    advance: the same compound, the tyre age + k, the fuel after k more laps; the gap to the car
    ahead is taken as it was when lap N ended. For k = 1 this is exactly the scored set above.

WHICH RACES
    development  2023 + 2024 (46 races). Every rung is built, compared and tuned here, with
                 cross-validation BY RACE: a race is never split between training and scoring,
                 because the laps of one race share the track, the weather and the safety cars.
    test         2025 (24 races). Locked: load_laps() refuses it until the ladder is finished, so no
                 choice is ever tuned on it. A new season also tests whether the patterns survive a
                 year of car development.

HOW A PREDICTION IS SCORED
    error  = predicted - actual           seconds; positive = the model expected a slower lap
    MAE    = mean of |error|              the headline, on dry laps
    median = median of |error|            a typical lap
    p90    = 90th percentile of |error|   a bad lap: 1 lap in 10 misses by more
    RMSE   = square root of mean(error^2) punishes big misses, which the race simulator will feel
    bias   = mean of error                a model that is wrong in one direction
    dry = the predicted lap is on slicks; wet = on intermediates or wets (scored, reported apart).

UNCERTAINTY: THE BOOTSTRAP OVER RACES
    Laps of one race are not independent (same track, weather, safety cars), so we resample RACES:
    draw as many races as there are, with replacement (some twice, some not at all), and compute the
    MAE over their laps. Repeat BOOTSTRAP_SAMPLES times: the middle 95 % of those MAEs is the 95 %
    interval. To compare two rungs, both are scored on the same draws (a paired bootstrap): the new
    rung is "better" only if the whole 95 % interval of (new MAE - old MAE) is below zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.laps import COMPOUND_ORDER, PACE, classify_laps
from src.pipeline import lake

DEVELOPMENT_SEASONS = (2023, 2024)
TEST_SEASONS = (2025,)
N_FOLDS = 5                     # cross-validation: 46 races -> 5 folds of 9 or 10 races
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026           # fixed, so every run prints the same intervals
LEAK_CHECK_EVERY_LAPS = 10      # future_leaks() cuts the races after lap 10, 20, 30, ...
HORIZONS = (1, 3, 5, 10)        # laps ahead in horizon_table()

TARGET = "target_next_lap_time_sec"

# score_status: "scored", or why not (first reason that applies, in this order)
SCORED = "scored"
NO_NEXT_LAP = "no next lap"             # the driver's last lap: finished or retired
NEXT_LAP_NOT_PACE = "next lap: "        # + step 1's reason, e.g. "next lap: safety car"
NO_CLEAN_LAP_YET = "no clean lap yet"   # the driver's pace in this race not seen yet

# conditions of the predicted lap
DRY, WET, UNKNOWN_TYRES = "dry", "wet", "unknown tyres"
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

# Columns about lap N+1 or about the scoring. A rung may learn from them in fit(), never in predict().
ANSWER_COLUMNS = [TARGET, "target_next_is_clean", "next_lap_status", "next_lap_compound",
                  "conditions", "score_status", "scored"]


class Rung(Protocol):
    """What the harness needs from a rung of the ladder (a description: rungs do not inherit from it)."""
    name: str

    def fit(self, train_laps: pd.DataFrame) -> Any: ...

    def predict(self, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series: ...


class SeasonLocked(Exception):
    """A test season was asked for before the ladder is finished."""


# --- the laps -----------------------------------------------------------------------------------

def pace_seen(laps: pd.DataFrame) -> pd.Series:
    """True once the driver has driven a clean lap: this lap, or an earlier one (rule 2)."""
    return laps["is_clean_lap"].astype(bool) | laps["last_clean_lap_time_sec"].notna()


def attach_answers(gold: pd.DataFrame, classified: pd.DataFrame,
                   event_name: Optional[str] = None) -> pd.DataFrame:
    """
    One race's Gold rows (one per driver per lap), with the answer columns added:

      next_lap_status     step 1's pace_status of lap N+1 (null: there is no lap N+1)
      next_lap_compound   the tyre lap N+1 is driven on
      conditions          dry (slicks), wet (inters or wets), or unknown tyres
      score_status        "scored", or the first reason the row is not scored
      scored              score_status == "scored"

    classified: the same race's Silver laps after step 1 (laps.classify_laps). Pure function.
    """
    # Step 1's verdict on lap N+1, moved onto the row of lap N.
    next_lap = classified[["driver_code", "lap_number", "pace_status", "compound"]].rename(
        columns={"pace_status": "next_lap_status", "compound": "next_lap_compound"})
    next_lap = next_lap.assign(lap_number=next_lap["lap_number"] - 1,
                               driver_code=next_lap["driver_code"].astype(object))
    laps = gold.merge(next_lap, on=["driver_code", "lap_number"], how="left")
    for col in ("next_lap_status", "next_lap_compound"):
        laps[col] = laps[col].astype(object).where(laps[col].notna(), None)
    laps["event_name"] = event_name

    # Why each row is (not) scored. np.select picks the FIRST condition that is true.
    next_status = laps["next_lap_status"]
    laps["score_status"] = np.select(
        [next_status.isna(), next_status != PACE, ~pace_seen(laps)],
        [NO_NEXT_LAP, NEXT_LAP_NOT_PACE + next_status.astype(str), NO_CLEAN_LAP_YET],
        default=SCORED)
    laps["scored"] = laps["score_status"] == SCORED

    compound = laps["next_lap_compound"]
    laps["conditions"] = np.select([compound.isin(WET_COMPOUNDS), compound.notna()], [WET, DRY],
                                   default=UNKNOWN_TYRES)

    # A pace lap always has a lap time. If Gold's target is missing there, the layers are out of step.
    no_target = laps["scored"] & laps[TARGET].isna()
    if no_target.any():
        raise ValueError(f"{int(no_target.sum())} scored rows have no target: Gold and step 1 disagree.")
    return laps


def ahead_answers(laps: pd.DataFrame, laps_ahead: int = 1) -> pd.DataFrame:
    """
    What is scored when predicting lap N+k (k = laps_ahead) from the row of lap N, on laps.index:

      scored       laps N+1 ... N+k are all pace laps, and the driver's pace has been seen (rule 2)
      actual_sec   the time of lap N+k
      compound     the tyre lap N+k is driven on
      conditions   dry, wet or unknown tyres

    The row of lap N+j-1 holds what is known about lap N+j (next_lap_status, the target), so lap N+j
    is read from j-1 rows further down the same driver's laps (rows in lap order, as Gold keeps them).
    For k = 1 these are exactly the columns scored, TARGET, next_lap_compound and conditions.
    """
    if laps_ahead < 1:
        raise ValueError("laps_ahead must be 1 or more.")
    by_driver = laps.groupby(["season", "round", "driver_code"], sort=False)
    all_pace = pace_seen(laps)
    for j in range(laps_ahead):
        row = by_driver[["next_lap_status", "lap_number"]].shift(-j)          # the row of lap N+j
        all_pace &= row["next_lap_status"].eq(PACE) & (row["lap_number"] == laps["lap_number"] + j)
    last = by_driver[[TARGET, "next_lap_compound", "conditions"]].shift(-(laps_ahead - 1))
    return pd.DataFrame({"scored": all_pace.astype(bool), "actual_sec": last[TARGET],
                         "compound": last["next_lap_compound"], "conditions": last["conditions"]},
                        index=laps.index)


def load_race(season: int, round_number: int, root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """One race from the lake: Gold lap features, with the answers from its Silver laps (step 1)."""
    gold_path = lake.gold_features_path(season, round_number, "R", root)
    gold = lake.read_table(gold_path)
    silver = lake.read_table(lake.table_path(lake.SILVER, "laps", season, round_number, "R", root))
    event_name = lake.read_metadata(gold_path).get("event_name")
    return attach_answers(gold, classify_laps(silver), event_name)


def load_laps(seasons: Iterable[int] = DEVELOPMENT_SEASONS, root: str = lake.DEFAULT_LAKE_ROOT,
              open_test_set: bool = False) -> pd.DataFrame:
    """
    Every race of these seasons in the lake (main races only), with the answers attached.
    The test seasons are refused unless open_test_set=True, which is done once, at the very end.
    """
    seasons = sorted({int(s) for s in seasons})
    locked = [s for s in seasons if s in TEST_SEASONS]
    if locked and not open_test_set:
        raise SeasonLocked(f"Season {locked} holds the test races: locked until the ladder is finished. "
                           f"Develop on {list(DEVELOPMENT_SEASONS)}.")
    races = [load_race(season, round_number, root)
             for season, round_number in lake.list_races(lake.GOLD, lake.GOLD_TABLE, root)
             if season in seasons]
    if not races:
        raise FileNotFoundError(f"No Gold lap features in the lake '{root}' for seasons {seasons}.")
    return pd.concat(races, ignore_index=True)


def model_inputs(laps: pd.DataFrame) -> pd.DataFrame:
    """What predict() may see: the laps without the answer columns."""
    return laps.drop(columns=ANSWER_COLUMNS)


def races_of(laps: pd.DataFrame) -> List[Tuple[int, int]]:
    """The (season, round) of every race in the table, in calendar order."""
    return sorted(set(zip(laps["season"].tolist(), laps["round"].tolist())))


def in_races(laps: pd.DataFrame, races: Iterable[Tuple[int, int]]) -> pd.Series:
    """True on the rows that belong to one of these races."""
    race_of_row = pd.MultiIndex.from_arrays([laps["season"], laps["round"]])
    return pd.Series(race_of_row.isin(list(races)), index=laps.index)


# --- running a rung -----------------------------------------------------------------------------

def predictions_of(rung: Rung, laps: pd.DataFrame, laps_ahead: int = 1) -> pd.Series:
    """Ask a fitted rung to predict these laps (answers removed), and check what comes back."""
    predicted = rung.predict(model_inputs(laps), laps_ahead=laps_ahead)
    if not isinstance(predicted, pd.Series) or not predicted.index.equals(laps.index):
        raise ValueError(f"{rung.name}: predict() must return a pd.Series with one value per row, "
                         "on the laps' index.")
    predicted = predicted.astype(float)
    missing = ahead_answers(laps, laps_ahead)["scored"] & ~np.isfinite(predicted)
    if missing.any():
        raise ValueError(f"{rung.name}: no prediction for {int(missing.sum())} scored laps. "
                         "Every rung must predict every scored lap.")
    return predicted.rename("predicted_sec")


def fit_and_predict(make_rung: Callable[[], Rung], train_laps: pd.DataFrame, test_laps: pd.DataFrame,
                    laps_ahead: int = 1) -> pd.Series:
    """
    A fresh rung, fitted on train_laps, predicts test_laps. The two must not share a race.
    make_rung: the rung's class (or any function returning a new rung), so every fit starts clean.
    """
    return _fit_and_predict_all(make_rung, train_laps, test_laps, (laps_ahead,))[laps_ahead]


def _fit_and_predict_all(make_rung: Callable[[], Rung], train_laps: pd.DataFrame, test_laps: pd.DataFrame,
                         horizons: Sequence[int]) -> Dict[int, pd.Series]:
    """fit_and_predict for several horizons from one fit."""
    shared = set(races_of(train_laps)) & set(races_of(test_laps))
    if shared:
        raise ValueError(f"Races both trained on and predicted: {sorted(shared)}.")
    rung = make_rung()
    rung.fit(train_laps)
    return {k: predictions_of(rung, test_laps, k) for k in horizons}


def race_folds(laps: pd.DataFrame, n_folds: int = N_FOLDS) -> List[List[Tuple[int, int]]]:
    """
    The races split into folds of whole races. Race number i (in calendar order) goes to fold
    i mod n_folds, so every fold holds races from both seasons and from the whole calendar.
    A lake with fewer races than n_folds (the tests) gets one race per fold.
    """
    races = races_of(laps)
    if len(races) < 2:
        raise ValueError("Cross-validation needs at least 2 races.")
    n_folds = min(n_folds, len(races))
    return [races[i::n_folds] for i in range(n_folds)]


def horizon_predictions(make_rung: Callable[[], Rung], laps: pd.DataFrame, horizons: Sequence[int] = HORIZONS,
                        n_folds: int = N_FOLDS) -> Dict[int, pd.Series]:
    """
    Cross-validated predictions for several horizons: for each fold, a fresh rung is fitted on the
    races of the other folds and predicts the races of this fold, k laps ahead for every k.
    """
    predictions = {k: pd.Series(np.nan, index=laps.index, name="predicted_sec") for k in horizons}
    for held_out in race_folds(laps, n_folds):
        is_held_out = in_races(laps, held_out)
        fold = _fit_and_predict_all(make_rung, laps[~is_held_out], laps[is_held_out], horizons)
        for k in horizons:
            predictions[k].loc[is_held_out] = fold[k]
    return predictions


def cross_validate(make_rung: Callable[[], Rung], laps: pd.DataFrame, n_folds: int = N_FOLDS,
                   laps_ahead: int = 1) -> pd.Series:
    """
    One prediction for every row, each made by a rung that never saw that row's race: for each fold,
    a fresh rung is fitted on the races of the other folds and predicts the races of this fold.
    """
    return horizon_predictions(make_rung, laps, (laps_ahead,), n_folds)[laps_ahead]


def future_leaks(rung: Rung, laps: pd.DataFrame, every: int = LEAK_CHECK_EVERY_LAPS, laps_ahead: int = 1) -> int:
    """
    Rule 5 for models: how many predictions change when the races are cut short (0 = no leak).

    The prediction made at the end of lap N may only use laps up to N. So cut every race after lap
    `every`, 2 x every, ...: a rung that looks at later laps predicts differently on the cut race
    than on the full one. The rung must already be fitted.
    """
    full = predictions_of(rung, laps, laps_ahead)
    changed = 0
    for cut in range(every, int(laps["lap_number"].max()), every):
        kept = laps[laps["lap_number"] <= cut]
        short = predictions_of(rung, kept, laps_ahead)
        same = np.isclose(short, full[kept.index], equal_nan=True)
        changed += int((~same).sum())
    return changed


# --- scoring ------------------------------------------------------------------------------------

def scored_errors(laps: pd.DataFrame, predictions: pd.Series, laps_ahead: int = 1) -> pd.DataFrame:
    """The scored laps only, each with the lap predicted, its actual and predicted time, and the error."""
    answers = ahead_answers(laps, laps_ahead)
    scored = answers.index[answers["scored"]]
    errors = pd.DataFrame({
        "season": laps.loc[scored, "season"],
        "round": laps.loc[scored, "round"],
        "event_name": laps.loc[scored, "event_name"],
        "driver_code": laps.loc[scored, "driver_code"],
        "predicted_lap": laps.loc[scored, "lap_number"] + laps_ahead,
        "compound": answers.loc[scored, "compound"],
        "conditions": answers.loc[scored, "conditions"],
        "actual_sec": answers.loc[scored, "actual_sec"],
        "predicted_sec": predictions[scored],
    })
    errors["error_sec"] = errors["predicted_sec"] - errors["actual_sec"]
    return errors


def error_summary(error: pd.Series) -> Dict[str, float]:
    """The scores of a set of errors, in seconds (see the module docstring)."""
    size = error.abs()
    return {
        "laps": int(len(error)),
        "mae": size.mean(),
        "median": size.median(),
        "p90": size.quantile(0.9),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bias": error.mean(),
    }


def summary_table(errors: pd.DataFrame) -> pd.DataFrame:
    """error_summary for all scored laps, dry and wet laps, and each compound (soft to wet)."""
    groups = {"all scored laps": errors}
    for conditions in (DRY, WET, UNKNOWN_TYRES):
        groups[conditions] = errors[errors["conditions"] == conditions]
    for compound in COMPOUND_ORDER:
        groups[compound] = errors[errors["compound"] == compound]
    rows = [{"laps_group": name, **error_summary(group["error_sec"])}
            for name, group in groups.items() if len(group)]
    return pd.DataFrame(rows).round(3)


def race_table(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per race: scored laps and MAE, dry laps and dry MAE, bias."""
    rows = []
    for (season, round_number), race in errors.groupby(["season", "round"]):
        dry = race.loc[race["conditions"] == DRY, "error_sec"]
        rows.append({
            "season": season,
            "round": round_number,
            "event_name": race["event_name"].iloc[0],
            "laps": len(race),
            "mae": race["error_sec"].abs().mean(),
            "dry_laps": len(dry),
            "dry_mae": dry.abs().mean(),
            "bias": race["error_sec"].mean(),
        })
    return pd.DataFrame(rows).round(3)


def score_ledger(laps: pd.DataFrame) -> pd.DataFrame:
    """Every row of the table: how many are scored, and how many are not for each reason."""
    ledger = laps["score_status"].value_counts().rename_axis("score_status").reset_index(name="rows")
    ledger["pct"] = (100 * ledger["rows"] / len(laps)).round(1)
    return ledger


def draw_races(n_races: int, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap draws: `samples` rows, each n_races race numbers drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_races, size=(samples, n_races))


def bootstrap_mae(errors: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES,
                  seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """
    95 % interval of the MAE, resampling whole races. In each draw:
        MAE = (absolute error summed over the laps of the drawn races) / (number of those laps)
    """
    per_race = errors.assign(abs_error=errors["error_sec"].abs()).groupby(["season", "round"]).agg(
        total=("abs_error", "sum"), laps=("abs_error", "size"))
    if per_race.empty:
        return np.nan, np.nan
    total, laps = per_race["total"].to_numpy(), per_race["laps"].to_numpy()
    drawn = draw_races(len(per_race), samples, seed)
    mae = total[drawn].sum(axis=1) / laps[drawn].sum(axis=1)
    low, high = np.percentile(mae, [2.5, 97.5])
    return float(low), float(high)


@dataclass
class Scorecard:
    """How one rung did on a set of races, predicting laps_ahead laps ahead."""
    rung: str
    errors: pd.DataFrame                # every scored lap: actual, predicted, error
    summary: pd.DataFrame               # summary_table: all, dry, wet, each compound
    races: pd.DataFrame                 # race_table: one row per race
    ledger: pd.DataFrame                # score_ledger: every row, scored or why not (1 lap ahead)
    dry_mae: float                      # the headline
    dry_interval: Tuple[float, float]   # its 95 % interval (bootstrap over races)
    laps_ahead: int = 1

    def dry_errors(self) -> pd.DataFrame:
        return self.errors[self.errors["conditions"] == DRY]


def score(laps: pd.DataFrame, predictions: pd.Series, rung_name: str, laps_ahead: int = 1) -> Scorecard:
    """Score one rung's predictions on the scored laps."""
    errors = scored_errors(laps, predictions, laps_ahead)
    dry = errors[errors["conditions"] == DRY]
    return Scorecard(
        rung=rung_name,
        errors=errors,
        summary=summary_table(errors),
        races=race_table(errors),
        ledger=score_ledger(laps),
        dry_mae=float(dry["error_sec"].abs().mean()),
        dry_interval=bootstrap_mae(dry),
        laps_ahead=laps_ahead,
    )


def compare(old: Scorecard, new: Scorecard, samples: int = BOOTSTRAP_SAMPLES,
            seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """
    Does `new` beat `old` on dry laps? difference = new MAE - old MAE (negative: new is better),
    with a paired bootstrap over races (both rungs on the same draws). verdict: "better" if the
    whole 95 % interval is below zero, "worse" if above, otherwise "no clear difference".
    """
    a, b = old.dry_errors(), new.dry_errors()
    if not a.index.equals(b.index):
        raise ValueError("The two scorecards were not scored on the same laps.")
    if a.empty:
        return {"old": old.rung, "new": new.rung, "laps": 0, "old_mae": np.nan, "new_mae": np.nan,
                "difference": np.nan, "low": np.nan, "high": np.nan, "verdict": "no laps"}
    per_race = pd.DataFrame({"season": a["season"], "round": a["round"],
                             "old": a["error_sec"].abs(), "new": b["error_sec"].abs()})
    per_race = per_race.groupby(["season", "round"]).agg(old=("old", "sum"), new=("new", "sum"),
                                                         laps=("old", "size"))
    old_total, new_total, laps = (per_race[c].to_numpy() for c in ("old", "new", "laps"))
    drawn = draw_races(len(per_race), samples, seed)
    difference = (new_total[drawn].sum(axis=1) - old_total[drawn].sum(axis=1)) / laps[drawn].sum(axis=1)
    low, high = (float(x) for x in np.percentile(difference, [2.5, 97.5]))
    verdict = "better" if high < 0 else "worse" if low > 0 else "no clear difference"
    return {"old": old.rung, "new": new.rung, "laps": len(a), "old_mae": old.dry_mae, "new_mae": new.dry_mae,
            "difference": new.dry_mae - old.dry_mae, "low": low, "high": high, "verdict": verdict}


def horizon_scorecards(make_rungs: Sequence[Callable[[], Rung]], laps: pd.DataFrame,
                       horizons: Sequence[int] = HORIZONS,
                       n_folds: int = N_FOLDS) -> Dict[str, Dict[int, Scorecard]]:
    """Every rung's cross-validated scorecard at every horizon: {rung name: {laps ahead: Scorecard}}.
    One fit per fold serves all horizons, so a slow rung is fitted only n_folds times."""
    cards = {}
    for make in make_rungs:
        name = make().name
        predicted = horizon_predictions(make, laps, horizons, n_folds)
        cards[name] = {k: score(laps, predicted[k], name, k) for k in horizons}
    return cards


def held_out_scorecards(make_rungs: Sequence[Callable[[], Rung]], train_laps: pd.DataFrame,
                        test_laps: pd.DataFrame,
                        horizons: Sequence[int] = HORIZONS) -> Dict[str, Dict[int, Scorecard]]:
    """
    Every rung fitted ONCE on train_laps and scored on test_laps (no race in both) at every horizon:
    {rung name: {laps ahead: Scorecard}}, the same shape as horizon_scorecards. For the test season.
    """
    cards = {}
    for make in make_rungs:
        name = make().name
        predicted = _fit_and_predict_all(make, train_laps, test_laps, horizons)
        cards[name] = {k: score(test_laps, predicted[k], name, k) for k in horizons}
    return cards


def horizon_rows(cards: Dict[str, Dict[int, Scorecard]]) -> pd.DataFrame:
    """
    The horizon scorecards as a table (dry laps): MAE with its 95 % interval and bias at every horizon,
    and each rung against the one before it (paired bootstrap on the same laps).
    """
    horizons = next(iter(cards.values())).keys()
    rows = []
    for k in horizons:
        previous = None
        for name, by_horizon in cards.items():
            card = by_horizon[k]
            row = {"laps_ahead": k, "rung": name, "laps": len(card.dry_errors()), "dry_mae": card.dry_mae,
                   "low": card.dry_interval[0], "high": card.dry_interval[1],
                   "bias": card.dry_errors()["error_sec"].mean(), "vs_previous": np.nan, "verdict": ""}
            if previous is not None:
                result = compare(previous, card)
                row.update(vs_previous=result["difference"], verdict=result["verdict"])
            rows.append(row)
            previous = card
    return pd.DataFrame(rows).round(4)


def horizon_table(make_rungs: Sequence[Callable[[], Rung]], laps: pd.DataFrame,
                  horizons: Sequence[int] = HORIZONS, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """
    Every rung at every horizon (cross-validated, dry laps): MAE with its 95 % interval and bias, and
    each rung against the one before it in the list (paired bootstrap on the same laps).
    """
    return horizon_rows(horizon_scorecards(make_rungs, laps, horizons, n_folds))


# --- for people ---------------------------------------------------------------------------------

def describe(card: Scorecard) -> str:
    """The scorecard as text, for the command line."""
    races = card.races
    seasons = ", ".join(str(s) for s in sorted(races["season"].unique()))
    low, high = card.dry_interval
    worst = races.sort_values("dry_mae", ascending=False).head(5)
    spread = races["dry_mae"].quantile([0.1, 0.5, 0.9]).round(3).tolist()
    ahead = "" if card.laps_ahead == 1 else f", {card.laps_ahead} laps ahead"
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        lines = [
            f"{card.rung}: {len(races)} races of {seasons}, {len(card.errors):,} scored laps{ahead}",
            f"Headline: MAE on dry laps {card.dry_mae:.3f} s per lap "
            f"(95 % interval {low:.3f} to {high:.3f}, bootstrap over races)",
            "",
            "Errors in seconds (error = predicted - actual; positive bias = predicted too slow):",
            card.summary.to_string(index=False),
            "",
            f"Dry MAE per race: p10 {spread[0]:.3f}, median {spread[1]:.3f}, p90 {spread[2]:.3f}. Worst 5:",
            worst[["season", "round", "event_name", "dry_laps", "dry_mae", "bias"]].to_string(index=False),
        ]
        if card.laps_ahead == 1:
            lines += ["", "Rows, scored or why not:", card.ledger.to_string(index=False)]
    return "\n".join(lines)


def write_scorecard(card: Scorecard, root: str = lake.DEFAULT_LAKE_ROOT) -> Path:
    """The per-race table as a report: data/reports/pace_model_<rung>.csv."""
    return lake.write_report(card.races, f"pace_model_{card.rung}", root)
