"""cleaner.clean_laps: Bronze laps -> Silver laps."""

import pandas as pd
import pytest

from src.pipeline.cleaner import clean_laps
from tests.helpers import make_bronze_laps, make_weather

NO_WEATHER = pd.DataFrame()


def silver(rows, weather=NO_WEATHER):
    laps, report = clean_laps(make_bronze_laps(rows), weather, 2024, 1, "R")
    return laps.set_index(["driver_code", "lap_number"]), report


def test_times_become_seconds_and_names_become_snake_case():
    laps, _ = silver([dict(driver="AAA", lap=2, start=100, end=191.5)])
    row = laps.loc[("AAA", 2)]
    assert row["lap_time_sec"] == 91.5
    assert row["lap_start_sec"] == 100.0 and row["lap_end_sec"] == 191.5
    assert row["season"] == 2024 and row["round"] == 1


def test_pit_flags():
    laps, _ = silver([
        dict(driver="AAA", lap=2, start=100, end=190),
        dict(driver="AAA", lap=3, start=190, end=285, pit_in=283),
        dict(driver="AAA", lap=4, start=285, end=395, pit_out=305),
    ])
    assert laps["is_in_lap"].tolist() == [False, True, False]
    assert laps["is_out_lap"].tolist() == [False, False, True]
    assert laps["is_pit_lap"].tolist() == [False, True, True]


@pytest.mark.parametrize("status, green, yellow, sc, vsc, red", [
    ("1", True, False, False, False, False),
    ("12", False, True, False, False, False),
    ("124", False, True, True, False, False),
    ("4", False, False, True, False, False),
    ("16", False, False, False, True, False),
    ("7", False, False, False, True, False),
    ("15", False, False, False, False, True),
])
def test_track_status_flags(status, green, yellow, sc, vsc, red):
    laps, _ = silver([dict(driver="AAA", lap=2, start=100, end=190, status=status)])
    row = laps.iloc[0]
    assert (row["is_green"], row["is_yellow"], row["is_sc"], row["is_vsc"], row["is_red"]) == (green, yellow, sc, vsc, red)


@pytest.mark.parametrize("overrides, expected", [
    (dict(), True),
    (dict(lap=1), False),               # standing start
    (dict(pit_in=180), False),          # in-lap
    (dict(pit_out=110), False),         # out-lap
    (dict(status="14"), False),         # safety car during the lap
    (dict(accurate=False), False),      # timing not trustworthy
    (dict(end=None), False),            # no lap time at all
    (dict(compound="INTERMEDIATE"), True),   # wet running is still a real lap
])
def test_clean_lap_definition(overrides, expected):
    row = {**dict(driver="AAA", lap=5, start=100, end=190), **overrides}
    laps, _ = silver([row])
    assert bool(laps.iloc[0]["is_clean_lap"]) is expected


def test_wet_tyre_flag_and_compound_uppercased():
    laps, _ = silver([dict(driver="AAA", lap=2, start=100, end=190, compound="intermediate"),
                      dict(driver="AAA", lap=3, start=190, end=280, compound="SOFT")])
    assert laps["compound"].tolist() == ["INTERMEDIATE", "SOFT"]
    assert laps["is_wet_tyre"].tolist() == [True, False]


def test_missing_values_stay_missing_instead_of_being_invented():
    laps, _ = silver([dict(driver="AAA", lap=7, start=100, end=190, tyre_life=None, position=None, compound=None)])
    row = laps.iloc[0]
    assert pd.isna(row["tyre_life"])      # the old pipeline wrote the lap number (7) here
    assert pd.isna(row["position"])
    assert pd.isna(row["compound"])
    assert not row["is_wet_tyre"]


def _stint_rows(driver, laps):
    """laps: [(lap, compound, tyre_life, pit)] with pit in (None, 'in', 'out'); 100 s per lap."""
    return [dict(driver=driver, lap=n, start=100 * n, end=100 * n + 95, compound=c, tyre_life=t,
                 pit_in=100 * n + 90 if pit == "in" else None, pit_out=100 * n + 10 if pit == "out" else None)
            for n, c, t, pit in laps]


def test_a_tyre_change_the_feed_shows_one_lap_late_is_still_a_tyre_change():
    """The out-lap still shows the old tyres (MEDIUM 12), the change appears a lap later (HARD 1).

    Seen at 2023 Canada (TSU lap 34), where the lap after had no tyre data at all (-> unknown,
    see the 'None' test below); here the late record is complete, so the change is known."""
    laps, _ = silver(_stint_rows("OCO", [(10, "MEDIUM", 10, None), (11, "MEDIUM", 11, "in"),
                                         (12, "MEDIUM", 12, "out"), (13, "HARD", 1, None),
                                         (14, "HARD", 2, None)]))
    assert laps.loc[("OCO", 11), "pit_kind"] == "tyre_change"
    assert laps["stint"].tolist() == [1, 1, 2, 2, 2]
    assert not laps["is_tyre_inconsistent"].any()


def test_a_real_pass_through_keeps_counting_after_the_out_lap():
    laps, _ = silver(_stint_rows("VER", [(35, "MEDIUM", 20, None), (36, "MEDIUM", 21, "in"),
                                         (37, "MEDIUM", 22, "out"), (38, "MEDIUM", 23, None)]))
    assert laps.loc[("VER", 36), "pit_kind"] == "pass_through"
    assert laps["stint"].tolist() == [1, 1, 1, 1]


def test_back_to_back_passages_are_judged_separately():
    """2024 Las Vegas, OCO: through the pit lane on lap 11 (tyres kept), then a stop on lap 12."""
    laps, _ = silver(_stint_rows("OCO", [(10, "MEDIUM", 10, None), (11, "MEDIUM", 11, "in"),
                                         (12, "MEDIUM", 12, "in"), (13, "HARD", 1, "out"),
                                         (14, "HARD", 2, None)]))
    assert laps.loc[("OCO", 11), "pit_kind"] == "pass_through"
    assert laps.loc[("OCO", 12), "pit_kind"] == "tyre_change"
    assert laps["stint"].tolist() == [1, 1, 1, 2, 2]


def test_a_tyre_record_that_changes_away_from_the_pits_is_flagged_and_not_trusted():
    """2025 Miami: the tyre age restarts four laps BEFORE the stop. The passage cannot be judged."""
    laps, _ = silver(_stint_rows("PIA", [(24, "HARD", 23, None), (25, "MEDIUM", 1, None),
                                         (26, "MEDIUM", 2, None), (27, "MEDIUM", 3, None),
                                         (28, "MEDIUM", 4, None), (29, "MEDIUM", 5, "in"),
                                         (30, "MEDIUM", 6, "out"), (31, "MEDIUM", 7, None)]))
    assert laps["is_tyre_inconsistent"].all()                # every lap of the driver
    assert laps.loc[("PIA", 29), "pit_kind"] == "unknown"     # not a confident pass-through
    assert laps.loc[("PIA", 30):, "stint"].isna().all()


def test_a_change_under_a_red_flag_is_not_inconsistent():
    """Under a red flag teams change tyres in the pit lane; the record may jump there."""
    rows = _stint_rows("NOR", [(1, "SOFT", 4, None), (2, "SOFT", 5, None), (3, "HARD", 1, None)])
    rows[1]["status"] = "15"
    laps, _ = silver(rows)
    assert not laps["is_tyre_inconsistent"].any()


def test_fastf1_none_text_is_a_missing_compound_not_a_tyre():
    """2023 Canadian GP: no compound for Tsunoda's last stint; FastF1 gives the text 'None'."""
    from src.pipeline import contracts

    laps, report = silver([
        dict(driver="TSU", lap=34, start=100, end=190, compound="HARD", tyre_life=20),
        dict(driver="TSU", lap=35, start=190, end=290, compound="HARD", tyre_life=21, pit_in=285),
        dict(driver="TSU", lap=36, start=290, end=390, compound="None", tyre_life=None, pit_out=305),
        dict(driver="TSU", lap=37, start=390, end=480, compound="None", tyre_life=None),
    ])
    assert laps["compound"].isna().tolist() == [False, False, True, True]
    assert laps.loc[("TSU", 35), "pit_kind"] == "unknown"     # which tyres went on is not known
    assert laps["stint"].isna().tolist() == [False, False, True, True]
    contracts.validate(laps.reset_index(), contracts.SILVER_LAPS, "Silver laps")


def test_duplicates_and_unidentifiable_rows_are_dropped_and_counted():
    _, report = silver([
        dict(driver="AAA", lap=2, start=100, end=190),
        dict(driver="AAA", lap=2, start=100, end=190),     # duplicate key
        dict(driver="AAA", lap=None, start=190, end=280),  # no lap number
    ])
    assert report == {"input_rows": 3, "dropped_null_key": 1, "dropped_duplicates": 1,
                      "output_rows": 1, "clean_laps": 1,
                      "pit_stops_published": 0, "pit_stops_matched": 0, "dropped_pit_stop_duplicates": 0,
                      "dropped_pit_stops_unmatched": 0, "pit_stop_values_not_numbers": 0,
                      "pit_tyre_change": 0, "pit_pass_through": 0, "pit_ended": 0, "pit_unknown": 0}


# --- F1's own pit-stop record --------------------------------------------------------

STOP_LAPS = [
    dict(driver="AAA", lap=2, start=100, end=190, tyre_life=2.0),
    dict(driver="AAA", lap=3, start=190, end=285, pit_in=283, tyre_life=3.0),
    dict(driver="AAA", lap=4, start=285, end=395, pit_out=305, compound="HARD", tyre_life=1.0),
    dict(driver="AAA", lap=5, start=395, end=485, compound="HARD", tyre_life=2.0),
]


def f1_record(*stops):
    """Bronze pit_stops rows, as the feed delivers them (text): (number, lap, stationary, pit lane)."""
    return pd.DataFrame([{"RacingNumber": n, "Lap": lap, "PitStopTime": still, "PitLaneTime": lane,
                          "Timestamp": "2024-01-01T14:30:00Z"} for n, lap, still, lane in stops], dtype="string")


def test_f1_pit_stop_times_land_on_the_in_lap_only():
    laps, report = clean_laps(make_bronze_laps(STOP_LAPS), NO_WEATHER, 2024, 1, "R",
                              pit_stops=f1_record(("1", "3", "2.4", "21.283")))
    laps = laps.set_index("lap_number")
    assert laps.loc[3, "pit_stop_time_sec"] == 2.4 and laps.loc[3, "pit_lane_time_sec"] == 21.283
    assert laps.drop(index=3)[["pit_stop_time_sec", "pit_lane_time_sec"]].isna().all().all()
    assert report["pit_stops_published"] == report["pit_stops_matched"] == 1


def test_no_f1_record_leaves_the_times_unknown_not_zero():
    laps, report = clean_laps(make_bronze_laps(STOP_LAPS), NO_WEATHER, 2024, 1, "R")
    assert laps["pit_stop_time_sec"].isna().all() and laps["pit_lane_time_sec"].isna().all()
    assert report["pit_stops_published"] == 0


def test_f1_records_that_do_not_fit_are_counted_never_silently_dropped():
    record = f1_record(("1", "3", "2.4", "21.283"), ("1", "3", "2.4", "21.283"),   # the same stop twice
                       ("1", "4", "2.6", "22.000"),                                 # lap 4 is not an in-lap
                       ("2", "3", "2.5", "21.900"))                                 # no such driver
    laps, report = clean_laps(make_bronze_laps(STOP_LAPS), NO_WEATHER, 2024, 1, "R", pit_stops=record)
    assert report["pit_stops_published"] == 4 and report["dropped_pit_stop_duplicates"] == 1
    assert report["pit_stops_matched"] == 1 and report["dropped_pit_stops_unmatched"] == 2
    assert int(laps["pit_stop_time_sec"].notna().sum()) == 1


def test_an_f1_value_that_is_not_a_number_stays_unknown_and_is_counted():
    laps, report = clean_laps(make_bronze_laps(STOP_LAPS), NO_WEATHER, 2024, 1, "R",
                              pit_stops=f1_record(("1", "3", "", "21.283")))
    in_lap = laps.set_index("lap_number").loc[3]
    assert pd.isna(in_lap["pit_stop_time_sec"]) and in_lap["pit_lane_time_sec"] == 21.283
    assert report["pit_stop_values_not_numbers"] == 1


def test_missing_required_column_raises_instead_of_filling_zero():
    bronze = make_bronze_laps([dict(driver="AAA", lap=2, start=100, end=190)]).drop(columns="TrackStatus")
    with pytest.raises(ValueError, match="TrackStatus"):
        clean_laps(bronze, NO_WEATHER, 2024, 1, "R")


def test_weather_takes_latest_reading_at_or_before_lap_end_never_a_future_one():
    weather = make_weather([(100, 30.0, False), (160, 31.0, True), (200, 99.0, False)])
    laps, _ = silver([dict(driver="AAA", lap=2, start=100, end=190)], weather)
    row = laps.iloc[0]
    assert row["track_temp_c"] == 31.0    # reading at t=160, not the closer-but-future t=200
    assert bool(row["rainfall"]) is True


def test_stale_weather_is_null_not_a_default():
    weather = make_weather([(100, 30.0, False)])
    laps, _ = silver([dict(driver="AAA", lap=9, start=900, end=990)], weather)   # 890 s after the reading
    assert pd.isna(laps.iloc[0]["track_temp_c"])


def test_no_weather_table_gives_null_weather_columns():
    laps, _ = silver([dict(driver="AAA", lap=2, start=100, end=190)])
    assert pd.isna(laps.iloc[0]["track_temp_c"]) and pd.isna(laps.iloc[0]["rainfall"])


def test_output_is_sorted_by_driver_then_lap():
    laps, _ = clean_laps(make_bronze_laps([
        dict(driver="BBB", lap=2, start=100, end=190), dict(driver="AAA", lap=3, start=190, end=280),
        dict(driver="AAA", lap=2, start=100, end=190)]), NO_WEATHER, 2024, 1, "R")
    assert list(zip(laps["driver_code"], laps["lap_number"])) == [("AAA", 2), ("AAA", 3), ("BBB", 2)]


# --- pit-lane passages: a stop is a TYRE CHANGE, not every trip through the pit lane ----------

def passage(next_compound="MEDIUM", next_tyre_life=6.0, in_tyre_life=5.0, with_next_lap=True):
    """Laps 4-5 of one driver, pitting at the end of lap 5 (tyres were 5 laps old on lap 5)."""
    rows = [
        dict(driver="AAA", lap=4, start=0, end=90, tyre_life=in_tyre_life - 1),
        dict(driver="AAA", lap=5, start=90, end=190, tyre_life=in_tyre_life, pit_in=185),
    ]
    if with_next_lap:
        rows.append(dict(driver="AAA", lap=6, start=190, end=300, pit_out=210,
                         compound=next_compound, tyre_life=next_tyre_life))
    return rows


@pytest.mark.parametrize("next_compound, next_tyre_life, expected", [
    ("HARD", 1.0, "tyre_change"),       # new compound
    ("MEDIUM", 1.0, "tyre_change"),     # fresh set of the same compound: tyre age restarts
    ("MEDIUM", 9.0, "tyre_change"),     # a used set of the same compound: age jumps
    ("MEDIUM", 6.0, "pass_through"),    # same tyres: SC through the pit lane, drive-through, stop-go
    ("MEDIUM", None, "unknown"),        # tyre age missing: cannot tell
])
def test_pit_passage_is_classified_from_the_tyres(next_compound, next_tyre_life, expected):
    laps, report = silver(passage(next_compound, next_tyre_life))
    assert laps.loc[("AAA", 5), "pit_kind"] == expected
    assert laps.loc[("AAA", 5), "is_tyre_change"] == (expected == "tyre_change")
    assert pd.isna(laps.loc[("AAA", 4), "pit_kind"])            # only in-laps carry a kind
    assert report[f"pit_{expected}"] == 1


def test_passage_with_no_next_lap_has_ended():
    laps, report = silver(passage(with_next_lap=False))
    assert laps.loc[("AAA", 5), "pit_kind"] == "ended"
    assert not laps.loc[("AAA", 5), "is_tyre_change"]
    assert report["pit_ended"] == 1


def test_stint_counts_tyre_changes_and_ignores_pass_throughs():
    # Real stop at lap 2, then two pass-throughs (laps 3 and 4) behind the safety car
    rows = [
        dict(driver="AAA", lap=1, start=0, end=90, tyre_life=1.0, stint=1.0),
        dict(driver="AAA", lap=2, start=90, end=190, tyre_life=2.0, pit_in=185, stint=1.0),
        dict(driver="AAA", lap=3, start=190, end=300, tyre_life=1.0, compound="HARD", pit_out=210, pit_in=295, stint=2.0),
        dict(driver="AAA", lap=4, start=300, end=410, tyre_life=2.0, compound="HARD", pit_out=320, pit_in=405, stint=3.0),
        dict(driver="AAA", lap=5, start=410, end=500, tyre_life=3.0, compound="HARD", pit_out=430, stint=4.0),
    ]
    laps, _ = silver(rows)
    assert laps["pit_kind"].tolist()[1:4] == ["tyre_change", "pass_through", "pass_through"]
    assert laps["stint"].tolist() == [1, 1, 2, 2, 2]       # FastF1 said 1, 1, 2, 3, 4


def test_stint_is_unknown_after_a_passage_that_cannot_be_classified():
    laps, _ = silver(passage("MEDIUM", None))
    assert laps["stint"].tolist()[:2] == [1, 1]
    assert pd.isna(laps.loc[("AAA", 6), "stint"])
