"""
Phase 3 step 0: the evaluation harness (src/models/evaluate.py).

A simulated race with hand-picked lap times (tests/models/simulated_race.py) checks every rule by hand.
The two 2024 reference races check the scored laps against Race Control and the tyre record:
Qatar's safety car on the leader's laps 35-39 (Verstappen led), Hamilton's British stops on laps 27
(to inters) and 38 (to softs).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    ANSWER_COLUMNS, DRY, NO_CLEAN_LAP_YET, NO_NEXT_LAP, SCORED, TARGET, WET, SeasonLocked, ahead_answers,
    bootstrap_mae, compare, cross_validate, describe, error_summary, fit_and_predict, future_leaks,
    held_out_scorecards, horizon_table,
    in_races, load_laps, load_race, model_inputs, predictions_of, race_folds, races_of, score, score_ledger,
    write_scorecard,
)
from src.pipeline import lake
from tests.models.simulated_race import simulated_race

BRITISH, QATAR = (2024, 12), (2024, 23)


class ThisLap:
    """A toy rung: the next lap equals this lap."""
    name = "this lap"

    def fit(self, train_laps):
        pass

    def predict(self, laps, laps_ahead=1):
        return laps["lap_time_sec"]


def statuses(laps, driver):
    return laps[laps["driver_code"] == driver].set_index("lap_number")["score_status"]


@pytest.fixture(scope="module")
def race():
    return simulated_race()


@pytest.fixture(scope="module")
def four_races():
    return pd.concat([simulated_race(2023, 1), simulated_race(2023, 2), simulated_race(2024, 1),
                      simulated_race(2024, 2)], ignore_index=True)


# --- which laps are scored ------------------------------------------------------------------------

def test_each_row_gets_the_first_reason_it_is_not_scored(race):
    # Lap 1 predicts lap 2, but no clean lap has been driven yet; the last lap has nothing to predict.
    assert statuses(race, "AAA").to_dict() == {1: NO_CLEAN_LAP_YET, **{lap: SCORED for lap in range(2, 8)},
                                               8: NO_NEXT_LAP}
    bbb = statuses(race, "BBB")
    assert bbb[4] == "next lap: safety car"     # lap 5 is behind the safety car
    assert bbb[5] == SCORED                     # lap 5 itself was not clean, but laps 2-4 were


def test_the_ledger_adds_up_to_every_row(race):
    ledger = score_ledger(race).set_index("score_status")["rows"]
    assert ledger.sum() == len(race) == 24
    assert ledger.to_dict() == {SCORED: 17, NO_CLEAN_LAP_YET: 3, NO_NEXT_LAP: 3, "next lap: safety car": 1}


def test_conditions_follow_the_tyres_of_the_predicted_lap(race):
    scored = race[race["scored"]]
    assert set(scored.loc[scored["driver_code"] == "CCC", "conditions"]) == {WET}
    assert set(scored.loc[scored["driver_code"] != "CCC", "conditions"]) == {DRY}


def test_a_rung_never_sees_the_answers(race):
    assert not set(ANSWER_COLUMNS) & set(model_inputs(race).columns)

    class Peeker(ThisLap):
        def predict(self, laps, laps_ahead=1):
            return laps[TARGET]

    with pytest.raises(KeyError):
        predictions_of(Peeker(), race)


def test_every_rung_must_predict_every_scored_lap(race):
    class Silent(ThisLap):
        def predict(self, laps, laps_ahead=1):
            return pd.Series(np.nan, index=laps.index)

    class NotASeries(ThisLap):
        def predict(self, laps, laps_ahead=1):
            return laps["lap_time_sec"].to_numpy()

    with pytest.raises(ValueError, match="no prediction for 17 scored laps"):
        predictions_of(Silent(), race)
    with pytest.raises(ValueError, match="must return a pd.Series"):
        predictions_of(NotASeries(), race)


# --- scoring, worked by hand ----------------------------------------------------------------------

def test_errors_and_scores_worked_by_hand(race):
    card = score(race, predictions_of(ThisLap(), race), ThisLap.name)
    aaa = card.errors[card.errors["driver_code"] == "AAA"]
    # Lap N predicts lap N+1: 90.0 for 90.6, 90.6 for 91.2, 91.2 for 90.9, 90.9 for 91.5, 91.5 for 91.8, 91.8 for 92.0
    assert aaa["predicted_lap"].tolist() == [3, 4, 5, 6, 7, 8]
    assert aaa["error_sec"].round(3).tolist() == [-0.6, -0.6, 0.3, -0.6, -0.3, -0.2]
    s = error_summary(aaa["error_sec"])
    assert s["laps"] == 6
    assert s["mae"] == pytest.approx(2.6 / 6)               # (0.6 + 0.6 + 0.3 + 0.6 + 0.3 + 0.2) / 6
    assert s["median"] == pytest.approx(0.45)               # between 0.3 and 0.6
    assert s["rmse"] == pytest.approx(np.sqrt(1.30 / 6))    # (0.36 + 0.36 + 0.09 + 0.36 + 0.09 + 0.04) / 6
    assert s["bias"] == pytest.approx(-2.0 / 6)             # it expected faster laps than came


def test_describe_and_the_report(four_races, tmp_path):
    card = score(four_races, predictions_of(ThisLap(), four_races), ThisLap.name)
    text = describe(card)
    assert "this lap: 4 races of 2023, 2024" in text and "Headline: MAE on dry laps" in text
    assert NO_CLEAN_LAP_YET in text
    write_scorecard(card, str(tmp_path))
    report = lake.read_report("pace_model_this lap", str(tmp_path))
    assert report[["season", "round"]].values.tolist() == [[2023, 1], [2023, 2], [2024, 1], [2024, 2]]


# --- races: folds, training, the locked test set ------------------------------------------------

def test_folds_hold_whole_races_spread_over_the_calendar():
    races = [(2023, 1), (2023, 2), (2023, 3), (2023, 4), (2024, 1), (2024, 2), (2024, 3)]
    laps = pd.DataFrame([{"season": s, "round": r} for s, r in races for _ in range(3)])   # 3 rows per race
    folds = race_folds(laps, n_folds=3)
    assert folds == [[(2023, 1), (2023, 4), (2024, 3)], [(2023, 2), (2024, 1)], [(2023, 3), (2024, 2)]]
    assert sorted(r for fold in folds for r in fold) == races


def test_a_small_lake_gets_one_race_per_fold_and_one_race_is_refused():
    two = pd.DataFrame({"season": [2024, 2024], "round": [23, 12]})
    assert race_folds(two) == [[(2024, 12)], [(2024, 23)]]
    with pytest.raises(ValueError, match="at least 2 races"):
        race_folds(two.iloc[:1])


def test_cross_validation_never_predicts_a_race_the_rung_trained_on(four_races):
    made = []

    class Spy(ThisLap):
        def __init__(self):
            made.append(self)
            self.trained_on, self.predicted = set(), set()

        def fit(self, train_laps):
            self.trained_on = set(races_of(train_laps))

        def predict(self, laps, laps_ahead=1):
            self.predicted = set(races_of(laps))
            return laps["lap_time_sec"]

    predictions = cross_validate(Spy, four_races, n_folds=2)
    assert len(made) == 2                                   # a fresh rung for every fold
    for rung in made:
        assert rung.trained_on and rung.predicted
        assert not rung.trained_on & rung.predicted         # no race both trained on and predicted
    assert sorted(r for rung in made for r in rung.predicted) == races_of(four_races)   # each race once
    assert predictions[four_races["scored"]].notna().all()


def test_training_and_scoring_races_must_differ(four_races):
    one = four_races[in_races(four_races, [(2023, 1)])]
    with pytest.raises(ValueError, match="both trained on and predicted"):
        fit_and_predict(ThisLap, one, one)


def test_held_out_scorecards_fit_each_rung_once_and_score_only_the_other_races(four_races):
    made = []

    class Recorder(ThisLap):
        def __init__(self):
            self.trained_on, self.predicted = set(), set()
            made.append(self)

        def fit(self, train_laps):
            self.trained_on |= set(races_of(train_laps))

        def predict(self, laps, laps_ahead=1):
            self.predicted |= set(races_of(laps))
            return laps["lap_time_sec"]

    train, test = four_races[four_races["season"] == 2023], four_races[four_races["season"] == 2024]
    cards = held_out_scorecards([Recorder], train, test, horizons=(1, 3))
    fitted = [rung for rung in made if rung.trained_on]
    assert len(fitted) == 1                                     # one fit serves every horizon
    assert fitted[0].trained_on == {(2023, 1), (2023, 2)} and fitted[0].predicted == {(2024, 1), (2024, 2)}
    assert list(cards) == ["this lap"] and list(cards["this lap"]) == [1, 3]
    one_lap = cards["this lap"][1]
    assert set(one_lap.errors["season"]) == {2024}
    assert one_lap.dry_mae == score(test, test["lap_time_sec"], "this lap").dry_mae
    with pytest.raises(ValueError, match="both trained on and predicted"):
        held_out_scorecards([ThisLap], four_races, test)


def test_the_test_seasons_stay_locked_until_opened(tmp_path):
    with pytest.raises(SeasonLocked, match="2025"):
        load_laps([2024, 2025], root=str(tmp_path))
    with pytest.raises(FileNotFoundError):              # opened on purpose: then it is just an empty lake
        load_laps([2025], root=str(tmp_path), open_test_set=True)


def test_future_leaks_catches_a_rung_that_peeks_at_the_next_lap(race):
    class Cheat(ThisLap):
        def predict(self, laps, laps_ahead=1):
            next_lap = laps.groupby("driver_code")["lap_time_sec"].shift(-1)     # lap N+1: the answer
            return next_lap.fillna(laps["lap_time_sec"])

    assert future_leaks(ThisLap(), race, every=3) == 0
    assert future_leaks(Cheat(), race, every=3) > 0


# --- uncertainty ------------------------------------------------------------------------------

def test_races_with_equal_errors_give_an_interval_with_no_width():
    errors = pd.DataFrame({"season": 2024, "round": [1, 1, 2, 2, 3, 3], "error_sec": [0.5, -0.5] * 3})
    low, high = bootstrap_mae(errors)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_the_bootstrap_brackets_the_mae_and_repeats_exactly():
    rng = np.random.default_rng(1)
    spread = np.repeat(np.linspace(0.2, 2.0, 20), 30)          # 20 races, some far harder than others
    errors = pd.DataFrame({"season": 2024, "round": np.repeat(np.arange(1, 21), 30),
                           "error_sec": rng.normal(0.0, 1.0, 600) * spread})
    low, high = bootstrap_mae(errors)
    assert low < errors["error_sec"].abs().mean() < high
    assert bootstrap_mae(errors) == (low, high)                 # fixed seed: the same interval every run


def test_compare_needs_the_whole_interval_on_one_side(four_races):
    actual = four_races[TARGET]
    old = score(four_races, actual + 1.0, "old")                # misses every lap by 1.0 s
    new = score(four_races, actual + 0.5, "new")                # misses every lap by 0.5 s
    result = compare(old, new)
    assert result["difference"] == pytest.approx(-0.5)
    assert result["low"] == pytest.approx(-0.5) and result["high"] == pytest.approx(-0.5)
    assert result["verdict"] == "better"
    assert compare(new, old)["verdict"] == "worse"
    assert compare(old, old)["verdict"] == "no clear difference"


def test_compare_refuses_scorecards_on_different_laps(four_races):
    fewer = four_races[~in_races(four_races, [(2024, 2)])]
    with pytest.raises(ValueError, match="same laps"):
        compare(score(four_races, four_races[TARGET], "old"), score(fewer, fewer[TARGET], "new"))


# --- the reference races ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference(golden_gold_lake):
    return {race: load_race(*race, root=golden_gold_lake) for race in (BRITISH, QATAR)}


def test_load_laps_reads_every_race_of_the_season_with_its_name(golden_gold_lake):
    laps = load_laps([2024], root=golden_gold_lake)
    assert races_of(laps) == [BRITISH, QATAR]
    assert set(laps["event_name"]) == {"British Grand Prix", "Qatar Grand Prix"}


def test_every_row_of_the_reference_races_is_accounted_for(reference):
    for laps in reference.values():
        assert score_ledger(laps)["rows"].sum() == len(laps)
        assert laps.loc[laps["scored"], TARGET].notna().all()


def test_qatar_the_lap_before_the_safety_car_is_not_scored(reference):
    # Race Control: safety car on the leader's laps 35-39, 40-42 (VSC on 40). Verstappen led.
    ver = statuses(reference[QATAR], "VER")
    assert ver[33] == SCORED
    assert ver[34] == "next lap: safety car"
    assert ver[42] == SCORED          # lap 43, the restart, is green; his pace was seen before the safety car


def test_qatar_lap_1_predicts_a_safety_car_lap(reference):
    # Safety car on laps 1-4; two cars never started lap 2.
    lap_1 = reference[QATAR].query("lap_number == 1")["score_status"]
    assert lap_1.value_counts().to_dict() == {"next lap: safety car": 18, NO_NEXT_LAP: 2}


def test_british_lap_1_is_never_scored_no_clean_lap_yet(reference):
    assert set(reference[BRITISH].query("lap_number == 1")["score_status"]) == {NO_CLEAN_LAP_YET}


def test_british_hamilton_on_intermediates_is_scored_as_wet(reference):
    # The tyre record: Hamilton stopped on lap 27 for inters and on lap 38 for softs.
    ham = reference[BRITISH].query("driver_code == 'HAM'").set_index("lap_number")
    assert set(ham.loc[28:36, "conditions"]) == {WET} and ham.loc[28:36, "scored"].all()
    assert ham.loc[37, "score_status"] == "next lap: pit lap"
    assert ham.loc[39, "conditions"] == DRY and ham.loc[39, "scored"]


# --- several laps ahead -------------------------------------------------------------------------

def test_one_lap_ahead_is_exactly_the_scored_set(race, reference):
    for laps in (race, *reference.values()):
        answers = ahead_answers(laps, 1)
        assert answers["scored"].equals(laps["scored"])
        pd.testing.assert_series_equal(answers["actual_sec"], laps[TARGET], check_names=False)
        assert answers["conditions"].equals(laps["conditions"])


def test_three_laps_ahead_needs_three_pace_laps_in_a_row(race):
    answers = ahead_answers(race, 3)

    def scored(driver):
        return race.loc[answers["scored"] & (race["driver_code"] == driver), "lap_number"].tolist()

    assert scored("AAA") == [2, 3, 4, 5]     # laps 3-5 up to 6-8; at the end of lap 1 no clean lap yet
    assert scored("BBB") == [5]              # lap 5 is behind the safety car: only laps 6, 7, 8 follow it
    row_2 = race.index[(race["driver_code"] == "AAA") & (race["lap_number"] == 2)][0]
    assert answers.loc[row_2, "actual_sec"] == 90.9     # AAA's lap 5


def test_errors_three_laps_ahead_worked_by_hand(race):
    card = score(race, predictions_of(ThisLap(), race, laps_ahead=3), ThisLap.name, laps_ahead=3)
    aaa = card.errors[card.errors["driver_code"] == "AAA"]
    # Lap N predicts lap N+3: 90.0 for 90.9, 90.6 for 91.5, 91.2 for 91.8, 90.9 for 92.0
    assert aaa["predicted_lap"].tolist() == [5, 6, 7, 8]
    assert aaa["error_sec"].round(3).tolist() == [-0.9, -0.9, -0.6, -1.1]
    assert card.laps_ahead == 3 and "3 laps ahead" in describe(card)


def test_a_rung_must_predict_every_lap_scored_ahead(race):
    class Silent(ThisLap):
        def predict(self, laps, laps_ahead=1):
            return pd.Series(np.nan, index=laps.index)

    with pytest.raises(ValueError, match="no prediction for 9 scored laps"):     # AAA 4, BBB 1, CCC 4
        predictions_of(Silent(), race, laps_ahead=3)


def test_future_leaks_also_looks_further_ahead(race):
    class Cheat(ThisLap):
        def predict(self, laps, laps_ahead=1):
            later = laps.groupby("driver_code")["lap_time_sec"].shift(-laps_ahead)    # lap N+k: the answer
            return later.fillna(laps["lap_time_sec"])

    assert future_leaks(ThisLap(), race, every=3, laps_ahead=3) == 0
    assert future_leaks(Cheat(), race, every=3, laps_ahead=3) > 0


def test_horizon_table_scores_every_rung_at_every_horizon(four_races):
    class OneSecondSlower(ThisLap):
        name = "this lap + 1 s"

        def predict(self, laps, laps_ahead=1):
            return laps["lap_time_sec"] + 1.0

    table = horizon_table([ThisLap, OneSecondSlower], four_races, horizons=(1, 3), n_folds=2)
    assert table[["laps_ahead", "rung"]].values.tolist() == [
        [1, "this lap"], [1, "this lap + 1 s"], [3, "this lap"], [3, "this lap + 1 s"]]
    verdicts = table.set_index(["laps_ahead", "rung"])["verdict"]
    assert verdicts[(1, "this lap")] == "" and verdicts[(3, "this lap")] == ""     # nothing above them
    # Lap times rise by about 0.3 s a lap: one lap ahead, 1 s slower is worse; three laps ahead it helps.
    assert verdicts[(1, "this lap + 1 s")] == "worse"
    assert verdicts[(3, "this lap + 1 s")] == "better"
