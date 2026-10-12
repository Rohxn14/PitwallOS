"""
Pace laps (src/analytics/laps.py), Phase 2 step 1.

Unit tests build small races through the real Silver cleaner (so `is_clean_lap` is the real
definition) and pin the rules: one status per lap in a fixed order, the outlier rule (a fading
tyre is never odd, a sudden jump is), and the ledger adding up.

Golden tests use the two reference races. Facts checked against sources other than this code:
Race Control's log (Qatar safety cars 1-4 and 35-42), the official track status feed (Qatar
yellows on laps 5 and 30), the weather station (British GP rain from lap 18), and the official
lap times (Hamilton, Qatar laps 43-45: 89.65, 87.09, 85.28 s).
"""

import pandas as pd
import pytest

from src.analytics.laps import (
    EXCLUSION_ORDER, OUTLIER_THRESHOLD_SEC, PACE, REASON_GENERATED, REASON_INACCURATE, REASON_LAP_1,
    REASON_NO_LAP_TIME, REASON_OTHER_STATUS, REASON_OUTLIER, REASON_PIT_LAP, REASON_RED_FLAG,
    REASON_SAFETY_CAR, REASON_VSC, REASON_YELLOW, classify_laps, exclusion_ledger, main, pace_by_compound,
    pace_summary, race_pace,
)
from src.pipeline.cleaner import clean_laps
from tests.helpers import make_bronze_laps, make_weather

BRITISH, QATAR = (2024, 12), (2024, 23)


def bronze_run(driver, times, first_lap=1, start=0.0, compound="MEDIUM", tyre_start=1):
    """Bronze rows for consecutive laps of one driver, one lap time each."""
    rows, t = [], start
    for i, lap_time in enumerate(times):
        rows.append({"driver": driver, "lap": first_lap + i, "start": t, "end": t + lap_time, "lap_time": lap_time,
                     "compound": compound, "tyre_life": float(tyre_start + i)})
        t += lap_time
    return rows


def silver(rows, generated=()):
    """Bronze rows -> Silver laps through the real cleaner. `generated`: (driver, lap) FastF1 made up."""
    bronze = make_bronze_laps(rows)
    for driver, lap in generated:
        bronze.loc[(bronze["Driver"] == driver) & (bronze["LapNumber"] == lap), "FastF1Generated"] = True
    laps, _ = clean_laps(bronze, make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return laps


def statuses(classified, driver="AAA"):
    return classified[classified["driver_code"] == driver].set_index("lap_number")["pace_status"]


# --- the outlier rule ------------------------------------------------------------------

def test_a_steadily_fading_tyre_keeps_every_lap():
    times = [100.0] + [90.0 + 0.25 * i for i in range(30)]    # 30 laps, each 0.25 s slower: 7.25 s in all
    laps = classify_laps(silver(bronze_run("AAA", times)))
    racing = laps[laps["lap_number"] > 1]
    assert racing["pace_status"].eq(PACE).all()
    assert racing["over_reference_sec"].abs().max() <= 0.25 + 1e-9    # the reference follows the fade
    # One median for the whole stint would have thrown away the end of the stint: the tyre wear itself
    assert (racing["lap_time_sec"] - racing["lap_time_sec"].median() > OUTLIER_THRESHOLD_SEC).sum() > 0


def test_a_lap_much_slower_than_its_neighbours_is_an_outlier():
    times = [100.0] + [90.0] * 20
    times[10] = 100.0                                          # lap 11: a mistake, 10 s lost
    laps = classify_laps(silver(bronze_run("AAA", times))).set_index("lap_number")
    assert laps.loc[11, "pace_status"] == REASON_OUTLIER and laps.loc[11, "is_outlier"]
    assert laps.loc[11, "reference_sec"] == pytest.approx(90.0)
    assert laps.loc[11, "over_reference_sec"] == pytest.approx(10.0)
    assert (laps.drop(index=[1, 11])["pace_status"] == PACE).all()


@pytest.mark.parametrize("slower_by, expected", [(3.0, PACE), (3.01, REASON_OUTLIER)])
def test_an_outlier_is_more_than_three_seconds_slower(slower_by, expected):
    times = [100.0] + [90.0] * 10
    times[5] = 90.0 + slower_by                                # lap 6
    assert statuses(classify_laps(silver(bronze_run("AAA", times)))).loc[6] == expected


def test_a_fast_lap_is_never_an_outlier():
    times = [100.0] + [90.0] * 10
    times[5] = 84.0                                            # 6 s faster than its neighbours
    laps = classify_laps(silver(bronze_run("AAA", times))).set_index("lap_number")
    assert laps.loc[6, "pace_status"] == PACE and laps.loc[6, "over_reference_sec"] == pytest.approx(-6.0)


def test_a_run_too_short_to_check_stays_a_pace_lap_unchecked():
    laps = classify_laps(silver(bronze_run("AAA", [100.0, 90.0, 99.0])))    # two racing laps, then out
    assert statuses(laps).tolist() == [REASON_LAP_1, PACE, PACE]           # 9 s slower, but no reference
    assert not laps["outlier_checked"].any() and laps["reference_sec"].isna().all()
    assert pace_summary(laps)["unchecked"] == 2


def test_the_reference_never_reaches_across_a_pit_stop():
    # Laps 2-9 on worn tyres at 95 s, stop on lap 10, new tyres from lap 11 at 90 s, except lap 12 at 93.5 s.
    # Across the stop, the old 95 s laps would raise lap 12's reference and hide its 3.5 s.
    before = bronze_run("AAA", [100.0] + [95.0] * 9)
    before[-1]["pit_in"] = before[-1]["end"] - 3.0
    after = bronze_run("AAA", [115.0, 93.5] + [90.0] * 6, first_lap=11, start=before[-1]["end"])
    after[0]["pit_out"] = after[0]["start"] + 20.0
    laps = classify_laps(silver(before + after)).set_index("lap_number")
    assert laps.loc[[2, 9, 10], "tyre_run"].tolist() == [1, 1, 1]          # the in-lap ends run 1
    assert laps.loc[[11, 12, 18], "tyre_run"].tolist() == [2, 2, 2]
    assert laps.loc[12, "reference_sec"] == pytest.approx(90.0)
    assert laps.loc[12, "pace_status"] == REASON_OUTLIER
    assert laps.loc[[10, 11], "pace_status"].tolist() == [REASON_PIT_LAP, REASON_PIT_LAP]


# --- one status per lap, in order --------------------------------------------------------

def test_each_lap_gets_the_first_reason_that_applies():
    a = bronze_run("AAA", [100.0, 90.0, 120.0, 110.0, 91.0, 90.0, 90.0, 95.0, 105.0])
    a[2].update(status="14", pit_in=a[2]["end"] - 3.0)       # lap 3: stops as the safety car comes out
    a[3].update(status="41", pit_out=a[3]["start"] + 20.0)   # lap 4: out-lap as the safety car goes in
    a[4].update(status="12", lap_time=None)                  # lap 5: yellow, and no time either
    a[5].update(lap_time=None)                               # lap 6: green, no time
    a[6].update(accurate=False)                              # lap 7: FastF1 not sure of the time
    a[7].update(pit_in=a[7]["end"] - 3.0)                    # lap 8: a drive-through under green
    a[8].update(pit_out=a[8]["start"] + 20.0)                # lap 9: its out-lap
    b = bronze_run("BBB", [100.0, 95.0, 96.0, 97.0, 90.0])
    b[1].update(status="15")                                 # lap 2: red flag
    b[2].update(status="1267")                               # lap 3: yellow, VSC
    b[3].update(status="")                                   # lap 4: no status recorded
    laps = classify_laps(silver(a + b, generated=[("BBB", 5)]))   # lap 5: a lap FastF1 made up

    assert statuses(laps, "AAA").tolist() == [
        REASON_LAP_1, PACE, REASON_SAFETY_CAR, REASON_SAFETY_CAR, REASON_YELLOW, REASON_NO_LAP_TIME,
        REASON_INACCURATE, REASON_PIT_LAP, REASON_PIT_LAP]
    assert statuses(laps, "BBB").tolist() == [
        REASON_LAP_1, REASON_RED_FLAG, REASON_VSC, REASON_OTHER_STATUS, REASON_GENERATED]


def test_the_ledger_adds_up_to_every_lap():
    laps = classify_laps(silver(bronze_run("AAA", [100.0] + [90.0] * 9 + [100.0] + [90.0] * 5)))
    ledger = exclusion_ledger(laps).set_index("reason")
    assert ledger.index.tolist() == ["all laps", *EXCLUSION_ORDER, "pace laps"]
    assert ledger.loc["all laps", "laps"] == 16
    assert ledger.loc[REASON_LAP_1, "laps"] == 1 and ledger.loc[REASON_OUTLIER, "laps"] == 1
    assert ledger.loc["pace laps", "laps"] == 14
    assert ledger.loc[list(EXCLUSION_ORDER), "laps"].sum() + ledger.loc["pace laps", "laps"] == 16
    assert ledger["remaining"].is_monotonic_decreasing


def test_classification_refuses_flags_that_disagree_with_silver():
    laps = silver(bronze_run("AAA", [100.0, 90.0, 90.0]))
    laps.loc[laps["lap_number"] == 2, "is_clean_lap"] = False      # someone changed the definition
    with pytest.raises(ValueError, match="is_clean_lap"):
        classify_laps(laps)


def test_classification_needs_the_silver_flags():
    with pytest.raises(ValueError, match="missing columns"):
        classify_laps(pd.DataFrame({"driver_code": ["AAA"], "lap_number": [1]}))


# --- Rohan's note: nothing else excluded, pace grouped by compound -----------------------

def test_wet_tyre_and_deleted_laps_stay_pace_laps():
    laps = silver(bronze_run("AAA", [110.0] + [104.0] * 5, compound="INTERMEDIATE"))
    laps.loc[laps["lap_number"] == 3, "is_deleted"] = True         # track limits: keeps its own flag
    classified = classify_laps(laps)
    assert statuses(classified).loc[2:].eq(PACE).all()


def test_pace_by_compound_counts_every_pace_lap_soft_to_wet():
    rows = (bronze_run("AAA", [100.0] + [92.0] * 5, compound="HARD")
            + bronze_run("BBB", [100.0] + [90.0] * 5, compound="SOFT")
            + bronze_run("CCC", [110.0] + [104.0] * 5, compound="INTERMEDIATE")
            + bronze_run("DDD", [100.0] + [91.0] * 5, compound="None"))   # the feed had no compound
    classified = classify_laps(silver(rows))
    table = pace_by_compound(classified)
    assert table["compound"].tolist()[:3] == ["SOFT", "HARD", "INTERMEDIATE"]
    assert pd.isna(table["compound"].iloc[3])                        # unknown stays unknown, listed last
    assert table["laps"].sum() == int(classified["is_pace_lap"].sum()) == 20
    assert table.set_index("compound").loc["SOFT", "median_sec"] == pytest.approx(90.0)


# --- golden: Qatar GP 2024 ---------------------------------------------------------------

@pytest.fixture(scope="module")
def paces(golden_silver_laps):
    return {race: race_pace(laps) for race, laps in golden_silver_laps.items()}


def test_qatar_every_lap_is_accounted_for(paces, golden_silver_laps):
    p = paces[QATAR]
    ledger = p.ledger.set_index("reason")
    assert ledger.loc["all laps", "laps"] == len(golden_silver_laps[QATAR]) == len(p.laps)
    assert ledger.loc[list(EXCLUSION_ORDER), "laps"].sum() + p.summary["pace_laps"] == len(p.laps)
    assert p.summary["clean_laps"] == int(golden_silver_laps[QATAR]["is_clean_lap"].sum())
    assert ledger.loc[REASON_LAP_1, "laps"] == 20                     # 20 starters


def test_qatar_verstappen_lap_by_lap(paces):
    ver = statuses(paces[QATAR].laps, "VER")
    neutralised = [2, 3, 4] + list(range(35, 43))                     # Race Control: SC laps 1-4 and 35-42
    assert ver.loc[1] == REASON_LAP_1
    assert (ver.loc[neutralised] == REASON_SAFETY_CAR).all()             # his stop on lap 35 was under it
    assert (ver.loc[[5, 30]] == REASON_YELLOW).all()                      # yellows in the official feed
    assert (ver.drop([1, 5, 30] + neutralised) == PACE).all()
    assert (ver == PACE).sum() == 43


def test_qatar_restart_lap_is_slow_but_not_an_outlier(paces):
    # Known limit of the rolling reference: after the safety car the next laps are slow too (tyres coming
    # back to temperature), so the reference rises with them. Hamilton, laps 43-45: 89.65, 87.09, 85.28 s.
    ham = paces[QATAR].laps.query("driver_code == 'HAM'").set_index("lap_number")
    assert ham.loc[43, "lap_time_sec"] == pytest.approx(89.65, abs=0.01)
    assert ham.loc[43, "reference_sec"] == pytest.approx(87.09, abs=0.01)    # median of laps 43-45
    assert ham.loc[43, "pace_status"] == PACE
    assert paces[QATAR].summary["outliers"] == 0


def test_qatar_raw_compound_pace_shows_the_fuel_confound(paces):
    c = paces[QATAR].by_compound.set_index("compound")
    assert c.index.tolist() == ["SOFT", "MEDIUM", "HARD"]
    assert c.loc["HARD", "median_sec"] < c.loc["MEDIUM", "median_sec"]      # hards look faster...
    assert c.loc["HARD", "median_lap"] > c.loc["MEDIUM", "median_lap"] + 20  # ...because they ran ~30 laps later


# --- golden: British GP 2024 ----------------------------------------------------------------

def test_british_rain_shower_lap_is_flagged_for_drivers_on_slicks(paces, golden_silver_laps):
    silver_laps = golden_silver_laps[BRITISH]
    assert silver_laps.loc[silver_laps["lap_number"] == 18, "rainfall"].all()   # weather station: rain from lap 18
    laps = paces[BRITISH].laps
    shower = laps[(laps["lap_number"] == 19) & laps["is_outlier"]]           # lap 19: first full lap in it
    assert {"HAM", "VER", "NOR"} <= set(shower["driver_code"])
    assert len(shower) >= 12
    assert (shower["compound"] == "MEDIUM").all() and (shower["over_reference_sec"] > 6.0).all()


def test_british_intermediate_laps_are_pace_laps_of_their_own(paces):
    p = paces[BRITISH]
    c = p.by_compound.set_index("compound")
    assert c.index.tolist() == ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
    assert c.loc["INTERMEDIATE", "drivers"] == 19                            # all 19 starters ran inters
    assert c.loc["INTERMEDIATE", "median_sec"] > c.loc["MEDIUM", "median_sec"] + 8.0
    assert p.summary["pace_laps"] + p.summary["outliers"] == p.summary["clean_laps"]


# --- the command line, over a lake ------------------------------------------------------------

def test_command_line_summarises_every_race_and_writes_a_report(golden_lake, capsys):
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    for race in (BRITISH, QATAR):
        build_silver(*race, "R", lake_root=golden_lake, telemetry=False)
    assert main(["--season", "2024", "--lake", golden_lake]) == 0
    report = lake.read_report("pace_laps", golden_lake).set_index("round")
    assert report.index.tolist() == [12, 23]
    assert report.loc[23, "periods"] == "SC 1-4, SC 35-39, VSC 40, SC 40-42"
    assert report.loc[12, "periods"] == "none"
    assert "2 races" in capsys.readouterr().out
