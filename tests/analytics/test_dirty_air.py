"""
The cost of following (src/analytics/dirty_air.py), Phase 2 step 4.

The central tests are races simulated with KNOWN physics: every lap costs fuel (0.03 s/kg), tyre wear
(medium 0.08, hard 0.05 s per lap of age), and a cost that depends only on the gap to the car ahead
on the road when the lap starts. The simulation finds that car its own way (the latest earlier line
crossing by another car, crossings in the pit lane left out, rounded to the ms as Gold does), so
the cars bunch up and spread out as they would; the code must recover the costs exactly.
Hand-built crossings test the backmarker rules. Golden tests on Qatar 2024: Piastri starts lap 8
1.004 s behind Russell (Silver's line crossings), the gaps are Gold's, and beta is clearly above 0;
the British GP is a wet race and says nothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.degradation import within_stint_slope
from src.analytics.dirty_air import (
    CLEAR_LABEL, EXCLUDED_BACKMARKER, EXCLUDED_BEING_LAPPED, EXCLUDED_NO_GAP, USED, dirty_air_laps,
    fixed_effects_fit, main, race_dirty_air,
)
from src.analytics.laps import classify_laps, race_pace
from src.pipeline.cleaner import clean_laps
from src.pipeline.feature_store import build_lap_features
from tests.helpers import make_bronze_laps, make_weather

BRITISH, QATAR = (2024, 12), (2024, 23)
RACE_LAPS = 40
TRUE_WEAR = {"MEDIUM": 0.08, "HARD": 0.05}
TRUE_FUEL = 0.03
DRIVERS = {          # base pace (s), first compound, stop lap (in-lap; out-lap the next)
    "AAA": (80.00, "MEDIUM", 14), "BBB": (79.85, "HARD", 22), "CCC": (80.10, "MEDIUM", 17),
    "DDD": (79.95, "HARD", 26), "EEE": (80.25, "MEDIUM", 11), "FFF": (80.05, "HARD", 20),
}
STEP_COSTS = ((0.5, 0.8), (1.0, 0.4), (1.5, 0.25), (2.0, 0.15), (3.0, 0.1))   # (gap below, cost), else 0


def fuel(lap):
    return 110.0 - (lap - 1) * (110.0 - 5.0) / RACE_LAPS


def flat_cost(gap):
    return 0.4 if gap < 1.5 else 0.0


def step_cost(gap):
    return next((cost for below, cost in STEP_COSTS if gap < below), 0.0)


def known_race(cost):
    """Classified laps of a 40-lap race where lap time = base + fuel + wear x age + cost(start gap), exactly."""
    ends, in_pits, rows = {}, {}, []
    for lap in range(1, RACE_LAPS + 1):
        for i, (code, (base, first, stop)) in enumerate(DRIVERS.items()):
            start = 0.0 if lap == 1 else ends[(code, lap - 1)]
            compound = first if lap <= stop else ("HARD" if first == "MEDIUM" else "MEDIUM")
            age = lap if lap <= stop else lap - stop
            if lap == 1:
                lap_time = 90.0 + 0.9 * i                       # the grid, 0.9 s apart
            else:
                earlier = [t for key, t in ends.items() if key[0] != code and t < start and not in_pits[key]]
                gap = round(start - max(earlier), 3) if earlier else np.inf
                lap_time = base + TRUE_FUEL * fuel(lap) + TRUE_WEAR[compound] * age + cost(gap)
            lap_time += 20.0 if lap in (stop, stop + 1) else 0.0
            row = {"driver": code, "lap": lap, "start": start, "end": start + lap_time, "lap_time": lap_time,
                   "compound": compound, "tyre_life": float(age)}
            if lap == stop:
                row["pit_in"] = row["end"] - 2.0
            if lap == stop + 1:
                row["pit_out"] = start + 20.0
            rows.append(row)
            ends[(code, lap)], in_pits[(code, lap)] = start + lap_time, lap == stop
    laps, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return classify_laps(laps)


@pytest.fixture(scope="module")
def flat():
    return race_dirty_air(known_race(flat_cost))


@pytest.fixture(scope="module")
def step():
    return race_dirty_air(known_race(step_cost))


def test_a_known_cost_of_following_is_recovered_exactly(flat):
    f = flat.following
    assert f["beta"] == pytest.approx(0.4, abs=1e-6)
    assert f["following_laps"] >= 50 and f["stints_both"] >= 5           # the race really mixes both
    bins = flat.bins.set_index("bin")["cost"]
    assert bins.tolist() == pytest.approx([0.4, 0.4, 0.4, 0.0, 0.0, 0.0], abs=1e-6)
    assert not flat.mixed


def test_known_costs_per_gap_bin_are_recovered_exactly(step):
    bins = step.bins.set_index("bin")
    assert (bins["laps"] >= 10).all()                                     # every bin has an estimate
    assert bins["cost"].tolist() == pytest.approx([c for _, c in STEP_COSTS] + [0.0], abs=1e-6)
    assert bins.loc[CLEAR_LABEL, "laps"] == int((step.laps["gap_start_sec"] >= 3.0).sum())


def test_each_laps_cost_averages_to_its_bins_cost(step):
    used = step.laps[step.laps["air_status"] == USED]
    means = used.groupby("gap_bin")["lap_cost_sec"].mean()
    bins = step.bins.set_index("bin")["cost"]
    assert (means - bins.reindex(means.index)).abs().max() < 1e-6


def test_wear_is_exact_only_with_following_in_the_fit(flat):
    w = flat.wear.set_index("compound")
    assert w.loc["MEDIUM", "slope_with_following"] == pytest.approx(0.08, abs=1e-6)
    assert w.loc["HARD", "slope_with_following"] == pytest.approx(0.05, abs=1e-6)
    assert (w["change"].abs() > 1e-3).all()            # without it, traffic leaks into the wear slopes


def test_fixed_effects_fit_matches_step_3s_within_stint_slope():
    rng = np.random.default_rng(4)
    stint = pd.Series(np.repeat(["a", "b", "c"], 8))
    age = pd.Series(np.tile(np.arange(1.0, 9.0), 3))
    y = pd.Series(90.0 + stint.map({"a": 0.0, "b": 3.0, "c": -2.0}) + 0.07 * age + rng.normal(0, 0.1, 24))
    coef, residual = fixed_effects_fit(y, pd.DataFrame({"age": age}), stint, stint)
    slope, se, _ = within_stint_slope(age, y, stint)
    assert coef.loc["age", "coef"] == pytest.approx(slope) and coef.loc["age", "se"] == pytest.approx(se)
    assert residual.groupby(stint).sum().abs().max() < 1e-9               # each stint keeps its own level


# --- backmarkers, by hand ---------------------------------------------------------------------

def crossings_race():
    """
    AAA leads, laps of 100 s. BBB runs 1.2 s behind it. CCC starts slowly and is lapped during lap 4:
    it crosses at 299 (its lap 2) just before AAA ends lap 3 at 300, then just after AAA ends lap 4.
    """
    rows = []
    for k in range(1, 11):
        rows.append({"driver": "AAA", "lap": k, "start": 100.0 * (k - 1), "end": 100.0 * k})
        rows.append({"driver": "BBB", "lap": k, "start": 100.0 * (k - 1) + (1.2 if k > 1 else 0.0),
                     "end": 100.0 * k + 1.2})
    ccc = [150.0, 299.0, 401.0] + [401.0 + 101.0 * n for n in range(1, 8)]
    for k, end in enumerate(ccc, start=1):
        rows.append({"driver": "CCC", "lap": k, "start": 0.0 if k == 1 else ccc[k - 2], "end": end})
    laps, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return dirty_air_laps(classify_laps(laps)).set_index(["driver_code", "lap_number"])


def test_laps_behind_a_backmarker_or_being_lapped_are_left_out():
    a = crossings_race()
    assert (a.loc[("AAA", 4), "car_ahead"], a.loc[("AAA", 4), "gap_start_sec"]) == ("CCC", 1.0)   # 300 - 299
    assert a.loc[("AAA", 4), "air_status"] == EXCLUDED_BACKMARKER         # CCC had done 2 laps, AAA 3
    assert (a.loc[("CCC", 4), "car_ahead"], a.loc[("CCC", 4), "gap_start_sec"]) == ("AAA", 1.0)   # 401 - 400
    assert a.loc[("CCC", 4), "air_status"] == EXCLUDED_BEING_LAPPED       # AAA had done 4 laps, CCC 3
    assert (a.loc[("BBB", 5), "car_ahead"], a.loc[("BBB", 5), "gap_start_sec"]) == ("CCC", 0.2)   # 401.2 - 401
    assert a.loc[("BBB", 5), "air_status"] == EXCLUDED_BACKMARKER
    assert (a.loc[("BBB", 4), "car_ahead"], a.loc[("BBB", 4), "gap_start_sec"]) == ("AAA", 1.2)   # same lap
    assert a.loc[("BBB", 4), "air_status"] == USED and bool(a.loc[("BBB", 4), "following"])
    assert a.loc[("AAA", 2), "air_status"] == EXCLUDED_NO_GAP              # nobody crossed before AAA at 100
    # AAA ends lap 4 at 400: the latest earlier crossing is BBB's lap 3 at 301.2, almost a lap up the road
    assert (a.loc[("AAA", 5), "car_ahead"], a.loc[("AAA", 5), "gap_start_sec"]) == ("BBB", pytest.approx(98.8))
    assert a.loc[("AAA", 5), "air_status"] == USED and not bool(a.loc[("AAA", 5), "following"])


# --- golden -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results(golden_silver_laps):
    return {r: race_dirty_air(race_pace(laps).laps) for r, laps in golden_silver_laps.items()}


def test_qatar_piastri_starts_lap_8_one_second_behind_russell(results):
    # Silver's line crossings at the end of lap 7: VER 4092.844, NOR 4094.045, RUS 4095.388, PIA 4096.392
    lap = results[QATAR].laps.set_index(["driver_code", "lap_number"]).loc[("PIA", 8)]
    assert lap["car_ahead"] == "RUS" and lap["car_ahead_laps_up"] == 0
    assert lap["gap_start_sec"] == pytest.approx(4096.392 - 4095.388)
    assert lap["air_status"] == USED and bool(lap["following"]) and lap["gap_bin"] == "1-1.5 s"


def test_qatar_gaps_are_golds(results, golden_silver_laps):
    gold = build_lap_features(golden_silver_laps[QATAR], 57).set_index(["driver_code", "lap_number"])
    a = results[QATAR].laps
    expected = [gold.loc[(d, n), "gap_on_track_start_sec"] for d, n in zip(a["driver_code"], a["lap_number"])]
    np.testing.assert_array_equal(a["gap_start_sec"].to_numpy(dtype=float), np.array(expected, dtype=float))


def test_qatar_following_costs_time_and_every_lap_is_accounted_for(results):
    r = results[QATAR]
    assert not r.mixed
    f = r.following
    assert f["beta"] - 2 * f["se"] > 0                                    # beta >= 0, clearly
    assert 0.1 < f["beta"] < 1.0
    ledger = r.ledger.set_index("reason")["laps"]
    assert ledger["pace laps"] == ledger.drop("pace laps").sum()
    assert ledger[EXCLUDED_BACKMARKER] + ledger[EXCLUDED_BEING_LAPPED] > 0
    assert r.drivers["following_laps"].sum() == f["following_laps"]


def test_british_gp_is_mixed_and_too_wet_for_an_estimate(results):
    r = results[BRITISH]
    assert r.mixed
    assert np.isnan(r.following["beta"])                                  # 10 dry slick laps: too few
    assert r.bins["cost"].isna().all()


def test_command_line_writes_a_dirty_air_report(golden_lake, capsys):
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    for race in (BRITISH, QATAR):
        build_silver(*race, "R", lake_root=golden_lake, telemetry=False)
    assert main(["--season", "2024", "--lake", golden_lake]) == 0
    report = lake.read_report("dirty_air", golden_lake).set_index("round")
    assert report.loc[12, "mixed"] and not report.loc[23, "mixed"]
    assert report.loc[23, "beta"] > 0 and report.loc[23, "laps 0-0.5 s"] > 0
    out = capsys.readouterr().out
    assert "1 dry races" in out and "pooled" in out
