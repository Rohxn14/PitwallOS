"""
Race overview (src/analytics/race_overview.py): what the Analyze window shows today.

Golden tests run on the real Bronze fixtures (2024 British GP, 2024 Qatar GP) and compare with
facts about those races known independently of this code. Unit tests pin the stint rules.
"""

import pandas as pd
import pytest

from src.analytics.race_overview import (
    END_RACE, END_RETIRED_IN_PITS, END_TYRE_CHANGE, classification, lake_status, load_overview,
    strategy_label, stints,
)
from src.pipeline.cleaner import build_silver

BRITISH, QATAR = (2024, 12), (2024, 23)


def overview(golden_lake, race):
    build_silver(*race, "R", lake_root=golden_lake, telemetry=False)
    return load_overview(*race, root=golden_lake)


def strategy_of(ov, code):
    return ov.strategies.set_index("driver_code").loc[code]


# --- golden: British GP 2024 (wet/dry, no safety car) -------------------------

def test_british_classification(golden_lake):
    ov = overview(golden_lake, BRITISH)
    c = ov.classification
    assert c["driver_code"].head(3).tolist() == ["HAM", "VER", "NOR"]
    assert c.loc[0, "result"] == "1:22:27.059"            # Hamilton's race time
    assert c.loc[1, "result"] == "+1.465s"
    lapped = c.set_index("driver_code")
    assert lapped.loc["RIC", "result"] == "+1 lap"         # results Time is not a gap for him
    assert lapped.loc["OCO", "result"] == "+2 laps"
    assert lapped.loc["GAS", "result"] == "Did not start"


def test_british_strategies(golden_lake):
    ov = overview(golden_lake, BRITISH)
    ham, ver, rus = (strategy_of(ov, c) for c in ("HAM", "VER", "RUS"))
    assert (ham["strategy"], ham["stops"], ham["pit_laps"]) == ("M-I-S", 2, "27, 38")
    assert ver["strategy"] == "M-I-H"                       # Verstappen took hards at the end
    assert (rus["strategy"], rus["stops"]) == ("M-I", 1)    # retired in the pits on lap 33
    rus_last = ov.stints[ov.stints["driver_code"] == "RUS"].iloc[-1]
    assert (rus_last["last_lap"], rus_last["ended_by"]) == (33, END_RETIRED_IN_PITS)
    assert ov.strategies["stops"].sum() == 45               # the 45 tyre changes of Phase 0
    assert ov.quality["tyre_changes"] == 45


def test_a_retirement_does_not_count_towards_its_strategy(golden_lake):
    ov = overview(golden_lake, BRITISH)
    groups = ov.strategy_groups.set_index("strategy")
    assert groups.loc["M-I", "drivers"] == "RUS†"           # Russell retired on lap 33
    assert groups.loc["M-I", "finishers"] == 0 and pd.isna(groups.loc["M-I", "best_result"])
    assert groups.loc["M-I-S", "best_result"] == 1          # Hamilton's strategy won
    assert ov.strategy_groups.iloc[0]["strategy"] == "M-I-S"  # the most finishers


def test_british_data_quality_counts_match_the_flags(golden_lake):
    ov = overview(golden_lake, BRITISH)
    q = ov.quality
    assert q["drivers"] == 19 and q["laps"] == 960
    assert q["safety_car_laps"] == 0 and q["vsc_laps"] == 0
    assert q["wet_tyre_laps"] == 234
    assert q["ended_in_pits"] == 1


# --- golden: Qatar GP 2024 (safety cars, pass-throughs) -------------------------

def test_qatar_strategies(golden_lake):
    ov = overview(golden_lake, QATAR)
    ver = strategy_of(ov, "VER")
    assert (ver["strategy"], ver["stops"], ver["pit_laps"]) == ("M-H", 1, "35")
    assert ov.quality["tyre_changes"] == 26 and ov.quality["pass_throughs"] == 34
    # Pass-throughs (the safety car led the field down the pit lane) do not start a stint
    ver_stints = ov.stints[ov.stints["driver_code"] == "VER"]
    assert len(ver_stints) == 2 and ver_stints["pass_throughs"].sum() == 2
    top = ov.strategy_groups.iloc[0]
    assert (top["strategy"], top["stops"], top["best_result"]) == ("M-H", 1, 1)


def test_lake_status(golden_lake):
    before = lake_status(*QATAR, root=golden_lake)
    assert before.bronze and not before.silver and not before.ready
    assert before.event_name == "Qatar Grand Prix" and before.total_laps == 57
    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=False)
    assert lake_status(*QATAR, root=golden_lake).ready


def test_overview_refuses_a_race_that_is_not_in_the_lake(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_overview(2024, 1, root=str(tmp_path))


# --- unit: stint rules -------------------------------------------------------------

def silver(rows):
    """rows: (driver, lap, compound, stint, pit_kind)."""
    return pd.DataFrame({
        "driver_code": [r[0] for r in rows],
        "lap_number": [r[1] for r in rows],
        "compound": [r[2] for r in rows],
        "stint": pd.array([r[3] for r in rows], dtype="Int64"),
        "pit_kind": pd.array([r[4] for r in rows], dtype="string"),
        "tyre_life": pd.array([r[1] for r in rows], dtype="Int64"),
        "fresh_tyre": True,
    })


def test_a_pass_through_does_not_start_a_new_stint():
    table = stints(silver([("AAA", 1, "MEDIUM", 1, None), ("AAA", 2, "MEDIUM", 1, "pass_through"),
                           ("AAA", 3, "MEDIUM", 1, None)]))
    assert len(table) == 1 and table.loc[0, "pass_throughs"] == 1 and table.loc[0, "ended_by"] == END_RACE


def test_after_an_unknown_passage_a_compound_change_still_splits_but_the_number_stays_unknown():
    table = stints(silver([("AAA", 1, "MEDIUM", 1, "unknown"), ("AAA", 2, "HARD", None, None),
                           ("AAA", 3, "HARD", None, None), ("AAA", 4, "SOFT", None, None)]))
    assert table["compound"].tolist() == ["MEDIUM", "HARD", "SOFT"]
    assert table["stint_number"].isna().tolist() == [False, True, True]


def test_stints_never_run_across_drivers():
    table = stints(silver([("AAA", 1, "MEDIUM", 1, None), ("BBB", 1, "MEDIUM", 1, None)]))
    assert table["driver_code"].tolist() == ["AAA", "BBB"]


def test_tyre_change_ends_a_stint_on_the_in_lap():
    table = stints(silver([("AAA", 1, "MEDIUM", 1, None), ("AAA", 2, "MEDIUM", 1, "tyre_change"),
                           ("AAA", 3, "HARD", 2, None)]))
    assert table[["first_lap", "last_lap", "ended_by"]].values.tolist() == [[1, 2, END_TYRE_CHANGE], [3, 3, END_RACE]]


def test_strategy_label():
    assert strategy_label(["MEDIUM", "HARD"]) == "M-H"
    assert strategy_label(["SOFT", None, "INTERMEDIATE", "WET"]) == "S-?-I-W"


def test_classification_puts_unclassified_rows_last():
    results = pd.DataFrame({
        "Abbreviation": ["BBB", "AAA"], "Position": [None, 1.0], "Status": ["Did not start", "Finished"],
        "Laps": [0.0, 10.0], "Time": pd.to_timedelta([None, 3600.5], unit="s"), "GridPosition": [0.0, 1.0],
        "ClassifiedPosition": ["W", "1"],
    })
    out = classification(results)
    assert out["driver_code"].tolist() == ["AAA", "BBB"]
    assert out.loc[0, "result"] == "1:00:00.500"
