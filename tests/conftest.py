"""Shared pytest fixtures for the data pipeline tests."""

import shutil
from pathlib import Path

import pytest

FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake"
GOLDEN_RACES = ((2024, 12), (2024, 23))   # British GP (wet/dry, no safety car), Qatar GP (safety cars)


@pytest.fixture
def golden_lake(tmp_path):
    """A temporary lake pre-loaded with real Bronze lap data for two 2024 races."""
    root = tmp_path / "lake"
    shutil.copytree(FIXTURE_LAKE, root)
    return str(root)


@pytest.fixture(scope="session")
def golden_silver_laps(tmp_path_factory):
    """
    Silver laps of both reference races, built once per test session from the Bronze fixtures:
    {(2024, 12): laps, (2024, 23): laps}. Shared between tests, so treat them as read-only.
    """
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    root = tmp_path_factory.mktemp("golden") / "lake"
    shutil.copytree(FIXTURE_LAKE, root)
    out = {}
    for race in GOLDEN_RACES:
        build_silver(*race, "R", lake_root=str(root), telemetry=False)
        out[race] = lake.read_table(lake.table_path(lake.SILVER, "laps", *race, "R", str(root)))
    return out
