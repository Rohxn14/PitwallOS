"""runner.py: one race end to end, offline from the golden lake's laps-only Bronze."""

import pytest

from src.pipeline.runner import run_pipeline


def test_laps_only_run_from_bronze(golden_lake):
    result = run_pipeline(2024, 12, lake_root=golden_lake, skip_extract=True, telemetry=False)
    assert result["gold"]["feature_rows"] == 960
    assert result["silver"]["telemetry_path"] is None


def test_telemetry_cannot_be_built_from_a_laps_only_bronze(golden_lake):
    with pytest.raises(ValueError, match="ingested laps only"):
        run_pipeline(2024, 12, lake_root=golden_lake, skip_extract=True, telemetry=True)
