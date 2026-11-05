"""
Phase 4 step 3: how true the chances are (src/simulator/calibration.py).

The scores worked by hand (Brier, log loss, reliability, how overconfident the sure calls were), the
history baseline by hand, a fold's own races left out of what it draws, and Qatar 2024's misses: the
whole-race misses are relative to the median driver, the lap noise only from clean laps in clear air, the
stop extras against the race's median green stop. The command line on a three-race lake.

Step 3b: a car that did not take the flag is a miss when the chances are judged on every car running at the start
(Qatar from lap 6: eighteen cars, fifteen finishers), and the history baseline counts it too. Step 3c: places moved
by hand (a run that is the real result moves the cars as much as reality did; nobody moving moves nobody), and the
place pools borrow from the rest when a group has too few misses.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.linear import LinearDecomposition
from src.simulator import calibration, montecarlo, race as R, replay

QATAR, HUNGARY = (2024, 23), (2024, 13)


def rows(p_podium, podium, part="early", start_place=1):
    return pd.DataFrame({"p_win": p_podium, "win": podium, "p_podium": p_podium, "podium": podium,
                         "p_points": p_podium, "points": podium, "part": part, "start_place": start_place})


def test_the_scores_by_hand():
    table = rows([0.9, 0.9, 0.1, 0.1, 0.5], [True, False, False, False, True])
    s = calibration.scores(table).set_index("event").loc["podium"]
    assert s["brier"] == pytest.approx((0.01 + 0.81 + 0.01 + 0.01 + 0.25) / 5)
    assert s["sure_yes_gap"] == pytest.approx(0.5 - 0.9)          # the 90 % calls came true half the time
    assert s["sure_no_gap"] == pytest.approx(0.0 - 0.1)
    bins = calibration.reliability(table, "podium")
    assert bins.loc["90%-100%", "n"] == 2 and bins.loc["90%-100%", "happened"] == pytest.approx(0.5)
    # calibration error: bins 10-20 % (|0 - 0.1| x 2), 50-60 % (|1 - 0.5| x 1), 90-100 % (|0.5 - 0.9| x 2), over 5
    assert s["calibration_error"] == pytest.approx((0.2 + 0.5 + 0.8) / 5)
    assert calibration.scores(rows([0.0], [True]))["log_loss"].iloc[0] == pytest.approx(-np.log(calibration.CLIP))


def test_luck_tells_a_true_forecaster_from_a_wrong_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, 700)
    true = calibration.scores(rows(p, rng.random(700) < p)).set_index("event").loc["podium"]      # outcomes drawn from p
    wrong = calibration.scores(rows(p, rng.random(700) < 0.5)).set_index("event").loc["podium"]   # outcomes a coin flip
    assert true["off_by_luck"] > 0.05 and true["calibration_error"] < true["luck_p95"]
    assert wrong["off_by_luck"] == 0.0 and wrong["calibration_error"] > 3 * wrong["luck_p95"]


def test_the_history_baseline_by_hand():
    places = pd.DataFrame({"part": "early", "start_place": [1, 1, 1, 2], "final_place": [1, 1, 4, 3]})
    out = calibration.history_chances(rows([0.0, 0.0], [True, False], start_place=[1, 3]), places)
    k = calibration.HISTORY_SMOOTHING
    overall = 3 / 4                                                # podiums in the early part: 3 of 4
    assert out["p_podium"].iloc[0] == pytest.approx((2 + k * overall) / (3 + k))
    assert out["p_podium"].iloc[1] == pytest.approx(overall)      # never seen 3rd: the overall rate


@pytest.fixture(scope="module")
def qatar(golden_gold_lake):
    race = R.load(*QATAR, root=golden_gold_lake)
    laps = evaluate.load_laps((2024,), golden_gold_lake)
    fold = replay.fit_fold(laps, [QATAR], {k: R.load(*k, root=golden_gold_lake) for k in evaluate.races_of(laps)},
                           golden_gold_lake)
    return race, fold, golden_gold_lake


def test_qatars_misses(qatar):
    race, fold, root = qatar
    misses = calibration.misses_of(race, fold, root)
    assert len(misses.offsets) and set(misses.offsets["part"]) <= set(montecarlo.PARTS)
    for _, group in misses.offsets.groupby("start_lap"):
        assert group["miss"].median() == pytest.approx(0.0, abs=1e-9)          # against the median driver
    noise = misses.lap_noise.merge(race.laps, on=["driver_code", "lap_number"])
    assert noise["is_clean_lap"].all() and (noise["gap_on_track_start_sec"].fillna(99) >= calibration.CLEAR_AIR_SEC).all()
    # Six green tyre changes with a pit-lane time; Stroll's (lap 7) was 11.4 s over the race's typical stop.
    extra = misses.stop_extra.set_index(["driver_code", "in_lap"])["extra"]
    assert len(extra) == 6 and extra.median() == pytest.approx(0.0) and extra[("STR", 7)] == pytest.approx(11.39, abs=0.01)
    british = R.load(2024, 12, root=root)
    assert calibration.misses_of(british, fold, root).offsets.empty            # a wet race: none


def test_a_fold_never_draws_its_own_races(qatar):
    race, fold, root = qatar
    misses = calibration.misses_of(race, fold, root)
    without = calibration.uncertainty_from(misses, [QATAR])
    assert all(len(v) == 0 for v in without.offsets.values()) and len(without.lap_noise) == 0
    with_it = calibration.uncertainty_from(misses, [HUNGARY], draws=("offset",))
    assert sum(len(v) for v in with_it.offsets.values()) > 0 and with_it.draws == ("offset",)


def three_race_lake(golden_gold_lake, tmp_path):
    """A copy of the reference lake (British, Qatar) with Hungary 2024 built to Gold: three races, two of them dry."""
    from src.pipeline.cleaner import build_silver
    from src.pipeline.feature_store import build_gold
    from tests.conftest import HUNGARY_FIXTURE_LAKE
    root = tmp_path / "lake"
    shutil.copytree(golden_gold_lake, root)
    shutil.copytree(HUNGARY_FIXTURE_LAKE / "bronze" / "season=2024" / "round=13", root / "bronze" / "season=2024" / "round=13")
    build_silver(*HUNGARY, "R", lake_root=str(root), telemetry=False)
    build_gold(*HUNGARY, "R", lake_root=str(root))
    return root


def test_command_line(golden_gold_lake, tmp_path, capsys):
    root = three_race_lake(golden_gold_lake, tmp_path)
    assert calibration.main(["--season", "2024", "--lake", str(root), "--runs", "20"]) == 0
    out = capsys.readouterr().out
    assert "From lap 5" in out and "Reliability" in out
    report = pd.read_csv(root / "reports" / f"{calibration.REPORT}.csv")
    assert set(report["config"]) == set(calibration.CONFIGS) | {"history"}
    assert set(zip(report["season"], report["round"])) == {QATAR, HUNGARY}          # dry races only


def test_a_car_that_did_not_finish_is_a_miss(qatar):
    race, fold, root = qatar
    start = calibration.start_of(race, fold, "lap 5", 6)
    drivers = list(start.state.index)
    outcomes = montecarlo.Outcomes(ranks=pd.DataFrame([{d: float(i + 1) for i, d in enumerate(drivers)}], columns=drivers))
    finishers, everyone = calibration._rows_of(start, outcomes), calibration._rows_of(start, outcomes, everyone=True)
    assert len(finishers) == 15 and len(everyone) == 18                      # Perez, Hulkenberg and Stroll retire
    gone = everyone[~everyone["finished"]]
    assert set(gone["driver_code"]) == {"PER", "HUL", "STR"} and gone["final_place"].isna().all()
    assert not gone[["win", "podium", "points"]].to_numpy().any()           # they did not win, podium or score
    assert set(finishers["driver_code"]) == set(everyone.loc[everyone["finished"], "driver_code"])


def test_the_history_of_every_car_running_has_the_retirements(qatar):
    race, fold, root = qatar
    misses = calibration.misses_of(race, fold, root)
    everyone, finishers = (t[t["start_lap"] == 6] for t in (misses.places_all, misses.places))
    assert len(everyone) == 18 and len(finishers) == 15 and finishers["final_place"].notna().all()
    assert everyone.loc[everyone["driver_code"].isin(["PER", "HUL", "STR"]), "final_place"].isna().all()


def test_command_line_with_the_events_drawn(golden_gold_lake, tmp_path, capsys):
    root = three_race_lake(golden_gold_lake, tmp_path)
    assert calibration.main(["--season", "2024", "--lake", str(root), "--runs", "20", "--drawn"]) == 0
    out = capsys.readouterr().out
    assert "Events drawn from" in out and "Retirements:" in out and "From lap 5" in out and "Podium Brier by what neutralised" in out
    report = pd.read_csv(root / "reports" / f"{calibration.DRAWN_REPORT}.csv")
    assert set(report["config"]) == {calibration.REAL, *calibration.DRAWN, "history"}
    assert set(zip(report["season"], report["round"])) == {QATAR, HUNGARY}          # dry races only
    full = report[report["config"] == calibration.FULL]
    assert (~full["finished"].astype(bool)).any() and not full.loc[~full["finished"].astype(bool), "podium"].any()


# --- step 3c ---------------------------------------------------------------------------------------------------

def test_places_moved_by_hand(qatar):
    race, fold, root = qatar
    real = R.finish_ranks(race, R.running_at(race, 6))
    as_real = montecarlo.Outcomes(ranks=pd.DataFrame([real]))                        # one run that is the real result
    sim, actual = calibration.places_moved(race, 6, real, as_real)
    assert sim == pytest.approx(actual) and sum(actual.values()) > 0
    held = R.hold_order_ranks(race, 6, list(real))
    nobody_moves = montecarlo.Outcomes(ranks=pd.DataFrame([held]))
    assert all(v == 0 for v in calibration.places_moved(race, 6, real, nobody_moves)[0].values())


def test_the_place_pools_borrow_when_they_are_small():
    drivers = [f"D{i}" for i in range(85)]
    offsets = pd.DataFrame({"season": 2024, "round": 1, "start_lap": 5, "part": "early", "driver_code": drivers,
                            "miss": [-0.1] * 40 + [0.1] * 40 + [0.5] * 5})
    places = pd.DataFrame({"season": 2024, "round": 1, "start_lap": 5, "part": "early", "driver_code": drivers,
                           "start_place": [1] * 40 + [20] * 40 + [8] * 5, "final_place": 1})
    misses = calibration.Misses(offsets, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), places, places)
    pools = calibration.place_offsets_from(misses, []).pools
    assert set(pools[("early", 0)]) == {-0.1} and set(pools[("early", 4)]) == {0.1}            # P1-3 and P16+
    assert len(pools[("early", 2)]) == 85               # P7-10: 5 misses, too few: the pool of the whole part
    assert len(pools[("late", 0)]) == 40                # nothing late: P1-3 over every part
