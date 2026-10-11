"""lake.py: paths, round-trips, atomic writes."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import lake


def test_table_path_follows_hive_layout(tmp_path):
    path = lake.table_path(lake.SILVER, "laps", 2024, 12, "R", root=tmp_path)
    assert path == tmp_path / "silver" / "season=2024" / "round=12" / "session=R" / "laps.parquet"


def test_unknown_layer_is_rejected():
    with pytest.raises(ValueError):
        lake.session_dir("platinum", 2024, 12)


def test_write_then_read_round_trips_data_and_metadata(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "t": pd.to_timedelta([1.5, None], unit="s")})
    path = lake.write_table(df, tmp_path / "x" / "t.parquet", metadata={"season": 2024, "event_name": "Test GP"})

    pd.testing.assert_frame_equal(lake.read_table(path), df)
    assert lake.read_metadata(path) == {"season": "2024", "event_name": "Test GP"}


def test_write_leaves_no_temp_file(tmp_path):
    lake.write_table(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet")
    assert [p.name for p in tmp_path.iterdir()] == ["t.parquet"]


def test_reading_a_missing_table_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the earlier pipeline stage"):
        lake.read_table(tmp_path / "nope.parquet")


def test_read_layer_concatenates_every_session(tmp_path):
    for rnd in (1, 2):
        lake.write_table(pd.DataFrame({"round": [rnd, rnd]}),
                         lake.table_path(lake.GOLD, "lap_features", 2024, rnd, root=tmp_path))
    combined = lake.read_layer(lake.GOLD, "lap_features", root=tmp_path)
    assert combined["round"].tolist() == [1, 1, 2, 2]


def test_manifest_round_trip(tmp_path):
    lake.write_manifest({"event_name": "Test GP", "total_laps": 50}, 2024, 3, root=tmp_path)
    assert lake.read_manifest(2024, 3, root=tmp_path)["total_laps"] == 50
    with pytest.raises(FileNotFoundError):
        lake.read_manifest(2024, 4, root=tmp_path)
