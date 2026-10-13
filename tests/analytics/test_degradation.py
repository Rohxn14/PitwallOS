"""
Fuel-corrected tyre degradation (src/analytics/degradation.py), Phase 2 step 3.

The central test is a race built with KNOWN physics: medium tyres lose 0.08 s per lap of age, hard
0.05, and every kg of fuel costs 0.03 s. The code must recover all three exactly, and show the
raw slopes hiding the wear (negative here, because fuel burns off faster than the tyres wear).
Golden tests on the two reference races: Verstappen's Qatar medium stint gets faster raw (86.5 s
to 84.4 s, official lap times) and the correction adds exactly fuel effect x burn; the British GP
is a wet race and is marked as mixed conditions.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.degradation import (
    EXCLUDED_SHORT_STINT, EXCLUDED_WET_TRACK, EXCLUDED_WET_TYRE, FUEL_EFFECT_SEC_PER_KG, USED, degradation_laps,
    main, race_degradation, within_stint_slope,
)
from src.analytics.laps import classify_laps, race_pace
from src.pipeline.cleaner import clean_laps
from tests.helpers import make_bronze_laps, make_weather

BRITISH, QATAR = (2024, 12), (2024, 23)
TRUE_WEAR = {"MEDIUM": 0.08, "HARD": 0.05}
TRUE_FUEL = 0.03                      # s per kg
RACE_LAPS = 30
STRATEGIES = {"AAA": ("MEDIUM", "HARD"), "BBB": ("MEDIUM", "HARD"), "CCC": ("HARD", "MEDIUM"), "DDD": ("HARD", "MEDIUM")}


def fuel(lap):
    """110 kg at the start of lap 1, burned evenly to 5 kg (written out here, independent of the code)."""
    return 110.0 - (lap - 1) * (110.0 - 5.0) / RACE_LAPS


def known_race(strategies, compound_on_lap=None):
    """
    Classified laps where lap time = 80 + TRUE_FUEL x fuel + wear x tyre age, exactly. strategies:
    {driver: (first compound, second compound)}, everyone stopping on lap 15 (in-lap and out-lap
    +20 s). compound_on_lap overrides a driver's compound on given laps: {(driver, lap): compound}.
    """
    rows = []
    for code, (first, second) in strategies.items():
        t, age = 0.0, 0
        for lap in range(1, RACE_LAPS + 1):
            compound = (compound_on_lap or {}).get((code, lap), first if lap <= 15 else second)
            age = 1 if lap == 16 else age + 1
            lap_time = 80.0 + TRUE_FUEL * fuel(lap) + TRUE_WEAR.get(compound, 0.0) * age
            lap_time += 10.0 if lap == 1 else 0.0
            lap_time += 20.0 if lap in (15, 16) else 0.0
            row = {"driver": code, "lap": lap, "start": t, "end": t + lap_time, "lap_time": lap_time,
                   "compound": compound, "tyre_life": float(age)}
            if lap == 15:
                row["pit_in"] = row["end"] - 2.0
            if lap == 16:
                row["pit_out"] = t + 20.0
            rows.append(row)
            t += lap_time
    laps, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return classify_laps(laps)


def test_known_wear_and_fuel_are_recovered_exactly():
    result = race_degradation(known_race(STRATEGIES))
    c = result.compounds.set_index("compound")
    assert c.loc["MEDIUM", "slope"] == pytest.approx(0.08, abs=1e-6)
    assert c.loc["HARD", "slope"] == pytest.approx(0.05, abs=1e-6)
    assert c.loc["MEDIUM", "stints"] == 4 and c.loc["HARD", "stints"] == 4
    assert result.fuel["fuel_sec_per_kg"] == pytest.approx(TRUE_FUEL, abs=1e-6)
    assert (result.stints["slope"] - result.stints["compound"].map(TRUE_WEAR)).abs().max() < 1e-6
    assert not result.mixed


def test_raw_slopes_hide_the_wear_by_exactly_the_fuel_burn():
    c = race_degradation(known_race(STRATEGIES)).compounds.set_index("compound")
    burn_cost = FUEL_EFFECT_SEC_PER_KG * (110.0 - 5.0) / RACE_LAPS          # 0.105 s per lap
    assert c.loc["MEDIUM", "raw_slope"] == pytest.approx(0.08 - burn_cost, abs=1e-6)   # -0.025: "tyres improve"
    assert c.loc["HARD", "raw_slope"] == pytest.approx(0.05 - burn_cost, abs=1e-6)
    assert c["fuel_per_lap"].tolist() == pytest.approx([burn_cost, burn_cost])


def test_within_stint_slope_ignores_the_level_of_each_stint():
    stint = pd.Series(["a"] * 4 + ["b"] * 4)
    age = pd.Series([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    y = pd.Series([90.0, 90.1, 90.2, 90.3, 85.0, 85.1, 85.2, 85.3])        # 5 s apart, same wear
    slope, se, stints = within_stint_slope(age, y, stint)
    assert slope == pytest.approx(0.1) and se == pytest.approx(0.0, abs=1e-9) and stints == 2


def test_wet_tyres_and_a_wet_track_are_left_out_and_counted():
    # BBB runs inters on laps 3-6: its own laps are wet-tyre laps, and every first stint shares those laps
    wet = {("BBB", lap): "INTERMEDIATE" for lap in range(3, 7)}
    result = race_degradation(known_race(STRATEGIES, compound_on_lap=wet))
    assert result.mixed
    status = result.laps.groupby(["driver_code", "tyre_run"])["deg_status"].agg(set)
    assert EXCLUDED_WET_TYRE in status[("BBB", 1)]
    assert status[("AAA", 1)] == {EXCLUDED_WET_TRACK}                      # slicks while a car is on inters
    assert status[("AAA", 2)] == {USED}
    ledger = result.ledger.set_index("reason")["laps"]
    assert ledger["pace laps"] == ledger.drop(["pace laps"]).sum()        # every pace lap accounted for


def test_a_stint_with_fewer_than_six_usable_laps_gives_no_slope():
    classified = known_race(STRATEGIES)
    cut = classified[(classified["lap_number"] <= 6) | (classified["lap_number"] >= 15)]   # stint 1: laps 2-6 only
    d = degradation_laps(cut)
    assert (d.loc[d["lap_number"] <= 6, "deg_status"] == EXCLUDED_SHORT_STINT).all()
    assert (d.loc[d["lap_number"] >= 17, "deg_status"] == USED).all()


# --- golden ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results(golden_silver_laps):
    return {r: race_degradation(race_pace(laps).laps) for r, laps in golden_silver_laps.items()}


def test_qatar_verstappen_medium_stint_gets_faster_raw_and_the_correction_adds_the_burn(results):
    r = results[QATAR]
    assert not r.mixed
    ver = r.stints.query("driver_code == 'VER' and compound == 'MEDIUM'").iloc[0]
    assert (ver["first_lap"], ver["last_lap"]) == (6, 34)
    assert ver["raw_slope"] < 0                                            # 86.5 s on lap 6 -> 84.4 s on lap 34
    burn_cost = FUEL_EFFECT_SEC_PER_KG * (110.0 - 5.0) / 57
    assert ver["slope"] - ver["raw_slope"] == pytest.approx(burn_cost)


def test_qatar_compound_slopes_and_fuel_check(results):
    r = results[QATAR]
    c = r.compounds.set_index("compound")
    assert c.index.tolist() == ["SOFT", "MEDIUM", "HARD"]
    assert (c["slope"] - c["raw_slope"]).round(9).nunique() == 1          # the same burn cost for every compound
    assert np.isfinite(r.fuel["fuel_sec_per_kg"]) and r.fuel["fuel_se"] > 0


def test_british_gp_is_mixed_conditions(results):
    r = results[BRITISH]
    assert r.mixed
    ledger = r.ledger.set_index("reason")["laps"]
    assert ledger[EXCLUDED_WET_TYRE] == 185                                # the pace laps on inters (step 1)
    assert ledger["used for slopes"] < 0.05 * ledger["pace laps"]          # a wet race says little about dry wear


def test_command_line_writes_a_degradation_report(golden_lake, capsys):
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    for race in (BRITISH, QATAR):
        build_silver(*race, "R", lake_root=golden_lake, telemetry=False)
    assert main(["--season", "2024", "--lake", golden_lake]) == 0
    report = lake.read_report("degradation", golden_lake)
    assert set(report.loc[report["round"] == 23, "compound"]) == {"SOFT", "MEDIUM", "HARD"}
    assert report.loc[report["round"] == 12, "mixed"].all()
    assert "1 dry races" in capsys.readouterr().out
