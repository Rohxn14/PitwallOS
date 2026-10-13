"""contracts.py: a broken table must be stopped before it is written."""

import pandas as pd
import pytest

from src.pipeline import contracts, lake
from src.pipeline.cleaner import build_silver, clean_laps
from src.pipeline.feature_store import build_lap_features
from tests.helpers import make_bronze_laps

ROWS = [dict(driver="AAA", lap=n, start=90.0 * (n - 1), end=90.0 * n) for n in range(1, 6)]


def silver_laps():
    laps, _ = clean_laps(make_bronze_laps(ROWS), pd.DataFrame(), 2024, 1, "R")
    return laps


def test_valid_tables_pass():
    laps = silver_laps()
    contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")
    contracts.validate(build_lap_features(laps, 5), contracts.GOLD_LAP_FEATURES, "Gold")


def test_duplicate_key_is_rejected():
    laps = pd.concat([silver_laps(), silver_laps().head(1)], ignore_index=True)
    with pytest.raises(contracts.ContractError, match="Silver laps broke its data contract"):
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")


def test_implausible_clean_lap_is_rejected():
    laps = silver_laps()
    laps.loc[2, "lap_time_sec"] = 12.0          # a 12 second lap flagged as clean
    with pytest.raises(contracts.ContractError, match="clean lap time outside"):
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")


def test_unknown_compound_is_rejected():
    laps = silver_laps()
    laps.loc[0, "compound"] = "BANANA"
    with pytest.raises(contracts.ContractError):
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")


def test_clean_flag_on_a_pit_lap_is_rejected():
    laps = silver_laps()
    laps.loc[3, ["is_pit_lap", "is_in_lap"]] = True     # still flagged clean: contradiction
    with pytest.raises(contracts.ContractError, match="flagged clean"):
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")


def test_pit_stop_times_off_an_in_lap_are_rejected():
    laps = silver_laps()
    laps.loc[2, "pit_stop_time_sec"] = 2.4                # lap 3 is not an in-lap
    with pytest.raises(contracts.ContractError, match="not an in-lap"):
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")


def test_all_failures_are_reported_together():
    laps = silver_laps()
    laps.loc[0, "compound"] = "BANANA"
    laps.loc[1, "lap_number"] = 0
    with pytest.raises(contracts.ContractError) as err:
        contracts.validate(laps, contracts.SILVER_LAPS, "Silver laps")
    assert "compound" in str(err.value) and "lap_number" in str(err.value)


def test_negative_gap_is_rejected_in_gold():
    features = build_lap_features(silver_laps(), 5)
    features.loc[1, "gap_ahead_sec"] = -0.5
    with pytest.raises(contracts.ContractError):
        contracts.validate(features, contracts.GOLD_LAP_FEATURES, "Gold")


def test_a_contract_failure_writes_nothing(tmp_path):
    bad = make_bronze_laps(ROWS)
    bad["Compound"] = "BANANA"
    lake.write_table(bad, lake.table_path(lake.BRONZE, "laps", 2024, 1, root=tmp_path))
    lake.write_table(pd.DataFrame(), lake.table_path(lake.BRONZE, "weather", 2024, 1, root=tmp_path))
    lake.write_manifest({"season": 2024, "round": 1, "session": "R", "event_name": "T", "location": "",
                         "country": "", "total_laps": 5}, 2024, 1, root=tmp_path)
    with pytest.raises(contracts.ContractError):
        build_silver(2024, 1, "R", lake_root=str(tmp_path), telemetry=False)
    assert not (tmp_path / "silver").exists()
