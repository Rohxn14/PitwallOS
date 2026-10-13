"""
End-to-end pipeline tests against two real races.

tests/fixtures/lake holds the real Bronze lap, weather and status tables for the
2024 British GP and 2024 Qatar GP (no telemetry, to keep the repo small). These
tests run Silver and Gold on them, offline, and compare the output with facts
about those races that are known independently of this code.

A "golden" test answers the question unit tests cannot: is the number RIGHT,
not merely the right type?
"""

import pandas as pd
import pytest

from src.analytics.report import UNEXPLAINED, load_race_report, sanity_checks
from src.pipeline import contracts, lake
from src.pipeline.cleaner import build_silver
from src.pipeline.feature_store import build_gold

BRITISH, QATAR = (2024, 12), (2024, 23)


def run(golden_lake, race):
    """Silver (laps only) + Gold for one race. Returns (silver_laps, gold_features, gold_path)."""
    year, rnd = race
    silver_res = build_silver(year, rnd, "R", lake_root=golden_lake, telemetry=False)
    gold_res = build_gold(year, rnd, "R", lake_root=golden_lake)
    return lake.read_table(silver_res["file_path"]), lake.read_table(gold_res["file_path"]), gold_res["file_path"]


def laps_of(df, driver, flag):
    return df[(df["driver_code"] == driver) & df[flag]]["lap_number"].tolist()


# --- British GP 2024: a wet/dry race with no safety car --------------------

def test_british_gp_shape(golden_lake):
    silver, gold, _ = run(golden_lake, BRITISH)
    assert gold["total_race_laps"].iloc[0] == 52
    assert silver["driver_code"].nunique() == 19          # Gasly retired before the start
    assert len(silver) == len(gold) == 960


def test_british_gp_had_no_safety_car_or_vsc(golden_lake):
    silver, _, _ = run(golden_lake, BRITISH)
    assert not silver["is_sc"].any()
    assert not silver["is_vsc"].any()
    assert silver["is_green"].all()


def test_british_gp_podium(golden_lake):
    _, gold, _ = run(golden_lake, BRITISH)
    final = gold[gold["lap_number"] == 52].sort_values("position")
    assert final["driver_code"].head(3).tolist() == ["HAM", "VER", "NOR"]
    assert final["gap_to_leader_sec"].iloc[0] == 0.0


def test_british_gp_rain_and_intermediates(golden_lake):
    silver, _, _ = run(golden_lake, BRITISH)
    assert silver["is_wet_tyre"].sum() == 234
    assert silver["rainfall"].any()
    # Hamilton: mediums -> intermediates (in on lap 27) -> softs (in on lap 38)
    assert laps_of(silver, "HAM", "is_in_lap") == [27, 38]
    ham = silver[silver["driver_code"] == "HAM"].set_index("lap_number")
    assert ham.loc[26, "compound"] == "MEDIUM"
    assert ham.loc[30, "compound"] == "INTERMEDIATE"
    assert ham.loc[45, "compound"] == "SOFT"


# --- Qatar GP 2024: safety cars and a VSC ----------------------------------

def test_qatar_gp_shape_and_podium(golden_lake):
    silver, gold, _ = run(golden_lake, QATAR)
    assert gold["total_race_laps"].iloc[0] == 57
    assert silver["driver_code"].nunique() == 20
    final = gold[gold["lap_number"] == 57].sort_values("position")
    assert final["driver_code"].head(3).tolist() == ["VER", "LEC", "PIA"]


def test_qatar_gp_safety_car_laps_match_the_official_track_status(golden_lake):
    silver, _, _ = run(golden_lake, QATAR)
    sc_laps = sorted(silver[silver["is_sc"]]["lap_number"].unique())
    vsc_laps = sorted(silver[silver["is_vsc"]]["lap_number"].unique())
    assert sc_laps == [1, 2, 3, 4] + list(range(34, 43))
    assert vsc_laps == [40]
    assert not silver["is_wet_tyre"].any()


def test_no_lap_under_safety_car_is_ever_clean(golden_lake):
    silver, _, _ = run(golden_lake, QATAR)
    assert not (silver["is_clean_lap"] & (silver["is_sc"] | silver["is_vsc"] | silver["is_yellow"])).any()


# --- properties that must hold for every race ------------------------------

@pytest.mark.parametrize("race, expected_clean", [(BRITISH, 850), (QATAR, 679)])
def test_clean_lap_count_is_stable(golden_lake, race, expected_clean):
    """Regression pin: if this number moves, the definition of a clean lap changed."""
    silver, _, _ = run(golden_lake, race)
    assert int(silver["is_clean_lap"].sum()) == expected_clean


@pytest.mark.parametrize("race", [BRITISH, QATAR])
def test_gold_invariants(golden_lake, race):
    _, gold, _ = run(golden_lake, race)
    contracts.validate(gold, contracts.GOLD_LAP_FEATURES, "Gold")

    assert not gold.duplicated(["driver_code", "lap_number"]).any()
    assert (gold.loc[gold["lap_number"] == 1, "fuel_mass_est"] == 110.0).all()
    assert gold.loc[gold["position"] == 1, "gap_ahead_sec"].isna().all()

    # The target really is the same driver's next lap time.
    nxt = gold.set_index(["driver_code", "lap_number"])["lap_time_sec"]
    sample = gold.dropna(subset=["target_next_lap_time_sec"])
    looked_up = [nxt[(d, n + 1)] for d, n in zip(sample["driver_code"], sample["lap_number"])]
    assert sample["target_next_lap_time_sec"].tolist() == looked_up

    # Following cars are usually close: the median on-track gap is a few seconds, never minutes.
    assert 0.5 < gold["gap_on_track_end_sec"].median() < 10.0


@pytest.mark.parametrize("race", [BRITISH, QATAR])
def test_gold_does_not_depend_on_the_future(golden_lake, race):
    """Leakage test on real data: cut the race at half distance and rebuild."""
    from src.pipeline.feature_store import build_lap_features

    silver, _, _ = run(golden_lake, race)
    total = int(silver["lap_number"].max())
    cutoff = silver.loc[silver["lap_number"] == total // 2, "lap_end_sec"].min()
    past = silver[silver["lap_end_sec"] <= cutoff]

    key = ["driver_code", "lap_number"]
    full_f = build_lap_features(silver, total).set_index(key)
    past_f = build_lap_features(past, total).set_index(key)
    features = [c for c in past_f.columns if not c.startswith("target_")]
    pd.testing.assert_frame_equal(full_f.loc[past_f.index, features], past_f[features])


def test_gold_metadata_identifies_the_race(golden_lake):
    _, _, gold_path = run(golden_lake, QATAR)
    meta = lake.read_metadata(gold_path)
    assert (meta["year"], meta["round_number"], meta["event_name"]) == ("2024", "23", "Qatar Grand Prix")


def test_silver_and_gold_never_import_fastf1(golden_lake, monkeypatch):
    """Silver and Gold must rebuild offline. Make any FastF1 import explode and run them."""
    import sys
    monkeypatch.setitem(sys.modules, "fastf1", None)      # 'import fastf1' now raises ImportError
    run(golden_lake, BRITISH)


@pytest.mark.parametrize("race, event", [(BRITISH, "British Grand Prix"), (QATAR, "Qatar Grand Prix")])
def test_race_report_builds_on_a_freshly_built_lake(golden_lake, race, event):
    """Bronze -> Silver here, then the whole of Phase 2 (steps 1-5) reads it and checks its own numbers."""
    run(golden_lake, race)
    report = load_race_report(*race, root=golden_lake)
    assert (report.season, report.round_number, report.event_name) == (*race, event)
    assert not (sanity_checks(report)["status"] == UNEXPLAINED).any()


def test_missing_bronze_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run Bronze extraction first"):
        build_silver(2030, 1, "R", lake_root=str(tmp_path))


# --- pit stops: tyre changes vs other trips through the pit lane ----------------------------------

def stops(silver, kind="tyre_change"):
    return silver[silver["pit_kind"] == kind]


def test_british_gp_pit_stops(golden_lake):
    silver, gold, _ = run(golden_lake, BRITISH)
    assert len(stops(silver)) == 45
    assert stops(silver, "pass_through").empty
    assert laps_of(silver, "HAM", "is_tyre_change") == [27, 38]
    # Russell retired into the pits on lap 33: a pit-lane passage, not a stop
    assert stops(silver, "ended")[["driver_code", "lap_number"]].values.tolist() == [["RUS", 33]]
    assert gold["is_tyre_change"].sum() == 45


def test_qatar_gp_safety_car_through_the_pit_lane_is_not_a_stop(golden_lake):
    # Race control, lap 36: "SAFETY CAR THROUGH THE PIT LANE". The field drove through the pits
    # twice on the same tyres; FastF1's Stint counted both as new stints.
    silver, gold, _ = run(golden_lake, QATAR)
    assert len(stops(silver)) == 26
    assert len(stops(silver, "pass_through")) == 34
    assert laps_of(silver, "VER", "is_tyre_change") == [35]
    assert silver.loc[silver["driver_code"] == "VER", "stint"].max() == 2
    # Penalties served in the pit lane are passages without new tyres
    pass_throughs = stops(silver, "pass_through")[["driver_code", "lap_number"]].values.tolist()
    assert ["NOR", 45] in pass_throughs          # 10 s stop/go
    assert ["HAM", 47] in pass_throughs          # drive-through for pit-lane speeding
    assert stops(silver, "ended")[["driver_code", "lap_number"]].values.tolist() == [["STR", 8]]
    assert gold.loc[gold["driver_code"] == "VER", "stint_number"].max() == 2.0


# --- F1's own pit-stop record (published since 2024 round 19) -------------------------------------

def test_qatar_gp_f1_pit_stop_times(golden_lake):
    silver, _, _ = run(golden_lake, QATAR)
    assert lake.read_manifest(*QATAR, root=golden_lake)["pit_stops"] == "published"
    timed = silver[silver["pit_stop_time_sec"].notna()].set_index(["driver_code", "lap_number"])
    # Every tyre change has a time, plus NOR's 10 s stop/go (Race Control): 26 + 1.
    assert len(timed) == 27 and silver.loc[silver["is_tyre_change"], "pit_stop_time_sec"].notna().all()
    assert timed.loc[("NOR", 45), "pit_kind"] == "pass_through" and timed.loc[("NOR", 45), "pit_stop_time_sec"] == 10.8
    assert timed.loc[("VER", 35), "pit_stop_time_sec"] == 2.5
    # HAM's drive-through (lap 47) never stopped: no stationary time.
    ham_47 = silver[(silver["driver_code"] == "HAM") & (silver["lap_number"] == 47)].iloc[0]
    assert ham_47["pit_kind"] == "pass_through" and pd.isna(ham_47["pit_stop_time_sec"])
    # F1's pit-lane time and the timing's (pit entry to exit) are two measurements of the same thing.
    by = silver.set_index(["driver_code", "lap_number"])
    timing = [by.loc[(d, n + 1), "pit_out_sec"] - by.loc[(d, n), "pit_in_sec"] for d, n in timed.index]
    differences = (timed["pit_lane_time_sec"] - pd.Series(timing, index=timed.index)).abs()
    assert differences.max() < 0.75 and differences.median() < 0.1


def test_british_gp_has_no_f1_pit_stop_record(golden_lake):
    silver, _, _ = run(golden_lake, BRITISH)
    assert lake.read_manifest(*BRITISH, root=golden_lake)["pit_stops"] == "not published"
    assert silver["pit_stop_time_sec"].isna().all() and silver["pit_lane_time_sec"].isna().all()
