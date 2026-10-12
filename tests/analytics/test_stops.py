"""
Pit stops and pit loss (src/analytics/stops.py), Phase 2 step 2.

Unit tests build small races through the real cleaner and the step 1 classification and pin the
arithmetic of each reference. Golden tests: the Phase 0 stop facts (British 45 tyre changes, HAM
on laps 27 and 38; Qatar 26, VER one stop on lap 35 under the safety car) and one stop worked by
hand from the official lap times.
"""

import pandas as pd
import pytest

from src.analytics.laps import classify_laps, race_pace
from src.analytics.stops import (
    DRIVER_REFERENCE, FIELD_REFERENCE, GREEN, LAP_1, RED_FLAG, SAFETY_CAR, main, pit_loss, race_stops, stop_counts,
    stops,
)
from src.pipeline.cleaner import clean_laps
from tests.helpers import make_bronze_laps, make_weather

BRITISH, QATAR = (2024, 12), (2024, 23)


def race(drivers, statuses=None, stop_lap=None, in_extra=8.0, out_extra=14.0, base=90.0, laps=15):
    """
    Bronze rows: each driver laps at `base` (lap 1 +10 s); drivers in `stop_lap` stop that lap
    (in-lap +in_extra, out-lap +out_extra, new tyres). `statuses`: {lap: status} for everyone.
    """
    rows = []
    for code in drivers:
        t, age, compound = 0.0, 0, "MEDIUM"
        for lap in range(1, laps + 1):
            lap_time = base + (10.0 if lap == 1 else 0.0)
            row = {"driver": code, "lap": lap, "status": (statuses or {}).get(lap, "1")}
            stop = (stop_lap or {}).get(code)
            if stop == lap:
                lap_time += in_extra
                row["pit_in"] = t + lap_time - 2.0
            if stop is not None and stop + 1 == lap:
                lap_time += out_extra
                row["pit_out"] = t + 20.0
                age, compound = 0, "HARD"
            age += 1
            row.update(start=t, end=t + lap_time, lap_time=lap_time, compound=compound, tyre_life=float(age))
            rows.append(row)
            t += lap_time
    laps_df, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return classify_laps(laps_df)


def one(table, driver="AAA"):
    return table[table["driver_code"] == driver].iloc[0]


def test_a_green_stop_costs_the_extra_time_against_the_drivers_own_pace():
    table = stops(race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 8}))
    s = one(table)
    assert (s["in_lap"], s["out_lap"], s["condition"]) == (8, 9, GREEN)
    assert s["reference_kind"] == DRIVER_REFERENCE and s["reference_sec"] == pytest.approx(180.0)
    assert s["reference_n"] == 10                                     # laps 3-7 and 10-14
    assert s["loss_sec"] == pytest.approx(22.0)                       # 8 + 14 s over two 90 s laps
    assert (s["compound_before"], s["compound_after"]) == ("MEDIUM", "HARD")


def test_a_safety_car_stop_is_measured_against_the_cars_that_stayed_out():
    # Everyone laps 30 s slower on laps 8-9 under the safety car; AAA also stops on lap 8.
    rows = race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 8}, statuses={8: "14", 9: "4"})
    rows.loc[rows["lap_number"].isin([8, 9]), "lap_time_sec"] += 30.0
    s = one(stops(rows))
    assert s["condition"] == SAFETY_CAR and s["reference_kind"] == FIELD_REFERENCE
    assert s["reference_n"] == 3 and s["reference_sec"] == pytest.approx(240.0)
    assert s["loss_sec"] == pytest.approx(22.0)        # against its own green pace it would "cost" 82 s


def test_a_red_flag_tyre_change_is_kept_with_no_loss():
    table = stops(race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 6}, statuses={6: "15"}))
    s = one(table)
    assert s["condition"] == RED_FLAG and pd.isna(s["loss_sec"]) and "red flag" in s["loss_note"]
    assert stop_counts(table).set_index("driver_code").loc["AAA", "strategic_stops"] == 0


def test_a_stop_on_lap_1_uses_the_field():
    s = one(stops(race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 1})))
    assert s["condition"] == LAP_1 and s["reference_kind"] == FIELD_REFERENCE
    assert s["loss_sec"] == pytest.approx(22.0)


def test_unknown_stays_unknown():
    no_time = race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 8})
    no_time.loc[(no_time["driver_code"] == "AAA") & (no_time["lap_number"] == 9), "lap_time_sec"] = float("nan")
    assert one(stops(no_time))["loss_note"] == "no lap time on the in-lap or out-lap"
    short = one(stops(race(["AAA", "BBB", "CCC", "DDD"], stop_lap={"AAA": 3}, laps=5)))   # pace laps 2 and 5 only
    assert pd.isna(short["loss_sec"]) and short["reference_n"] == 2 and "fewer than 3" in short["loss_note"]
    alone = one(stops(race(["AAA", "BBB"], stop_lap={"AAA": 1})))                # one car stayed out
    assert pd.isna(alone["loss_sec"]) and "cars stayed out" in alone["loss_note"]


def test_circuit_loss_is_the_median_of_green_stops_only():
    table = pd.DataFrame({"condition": [GREEN, GREEN, GREEN, SAFETY_CAR, RED_FLAG],
                          "loss_sec": [20.0, 22.0, 30.0, 12.0, float("nan")]})
    loss = pit_loss(table).set_index("condition")
    assert loss.loc[GREEN, "median_sec"] == 22.0 and loss.loc[GREEN, "measured"] == 3
    assert loss.loc[RED_FLAG, "stops"] == 1 and loss.loc[RED_FLAG, "measured"] == 0


# --- golden ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results(golden_silver_laps):
    return {r: race_stops(race_pace(laps).laps) for r, laps in golden_silver_laps.items()}


def test_british_stops(results):
    r = results[BRITISH]
    assert len(r.stops) == 45                                         # the 45 tyre changes of Phase 0
    ham = r.counts.set_index("driver_code").loc["HAM"]
    assert ham["in_laps"] == "27, 38" and ham["strategic_stops"] == 2
    assert "RUS" not in set(r.stops.loc[r.stops["in_lap"] == 33, "driver_code"])   # retired in the pits: no tyre change
    assert (r.stops["condition"] == GREEN).all()                      # no safety car in this race


def test_british_hamilton_first_stop_by_hand(results):
    # Official lap times: in-lap 27 105.183 s, out-lap 28 120.866 s. His pace laps around the stop
    # (22-25 on mediums, 29-33 on inters; lap 26 was an outlier) have a median of 100.420 s.
    s = results[BRITISH].stops.query("driver_code == 'HAM' and in_lap == 27").iloc[0]
    assert s["reference_n"] == 9 and s["reference_sec"] == pytest.approx(2 * 100.420)
    assert s["loss_sec"] == pytest.approx(105.183 + 120.866 - 200.840)   # 25.209 s


def test_qatar_stops(results):
    r = results[QATAR]
    assert len(r.stops) == 26
    ver = r.stops[r.stops["driver_code"] == "VER"]
    assert ver["in_lap"].tolist() == [35] and ver["condition"].tolist() == [SAFETY_CAR]
    assert pd.isna(ver["loss_sec"].iloc[0])                           # out-lap 36 has no time
    assert r.green_loss == pytest.approx(r.stops.loc[r.stops["condition"] == GREEN, "loss_sec"].median())


def test_command_line_writes_a_pit_loss_report(golden_lake, capsys):
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    for r in (BRITISH, QATAR):
        build_silver(*r, "R", lake_root=golden_lake, telemetry=False)
    assert main(["--season", "2024", "--lake", golden_lake]) == 0
    report = lake.read_report("pit_loss", golden_lake).set_index("round")
    assert report.loc[12, "tyre_changes"] == 45 and report.loc[23, "tyre_changes"] == 26
    assert report.loc[12, "green_stops"] == 45
    assert "2 races" in capsys.readouterr().out
