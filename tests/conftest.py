"""Shared pytest fixtures for the data pipeline tests."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake"
GOLDEN_RACES = ((2024, 12), (2024, 23))   # British GP (wet/dry, no safety car), Qatar GP (safety cars)
# Kept in its own lake so the tests above that count the races of 2024 are not affected.
HUNGARY_FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake_hungary"
HUNGARY = (2024, 13)                      # Hungarian GP: Norris undercuts Piastri at the second stops
# Real OpenF1 data, as delivered: Qatar 2024's small tables, and the telemetry of VER and NOR for laps 1-6.
OPENF1_FIXTURES = Path(__file__).parent / "fixtures" / "openf1"
QATAR = (2024, 23)
LUSAIL_M = 5419.0                         # the official length of the Lusail circuit


def openf1_qatar_bronze(root) -> None:
    """Put Qatar 2024 into a lake's Bronze as openf1_bronze would: OpenF1's tables + two cars' telemetry."""
    from src.pipeline import lake
    bronze = lake.session_dir(lake.BRONZE, *QATAR, "R", root)
    shutil.copytree(OPENF1_FIXTURES / "qatar_2024", bronze)
    manifest = json.loads((bronze / lake.MANIFEST_NAME).read_text(encoding="utf-8"))
    for table in ("openf1_car_data", "openf1_location"):
        shutil.copy(OPENF1_FIXTURES / f"qatar_{table}.parquet", bronze / f"{table}.parquet")
        frame = pd.read_parquet(bronze / f"{table}.parquet")
        manifest["tables"][table] = {"rows": len(frame), "bytes": (bronze / f"{table}.parquet").stat().st_size,
                                     "columns": list(frame.columns)}       # as openf1_bronze writes every entry
    manifest["telemetry"] = True
    (bronze / lake.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="session")
def openf1_qatar_lake(tmp_path_factory):
    """A lake built from OpenF1 alone: Qatar 2024 in Bronze, Silver (laps, telemetry of two cars, results) and Gold.
    Built once per test session and shared: treat it as read-only."""
    from src.pipeline import cleaner, feature_store
    root = str(tmp_path_factory.mktemp("openf1_lake"))
    openf1_qatar_bronze(root)
    cleaner.build_silver(*QATAR, "R", root, telemetry=True)
    feature_store.build_gold(*QATAR, "R", root)
    return root


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


@pytest.fixture(scope="session")
def golden_gold_lake(tmp_path_factory):
    """
    A lake with both reference races built to Gold (laps only), once per test session: what the
    Phase 3 models read. Shared between tests, so treat it as read-only (copy it to write into it).
    """
    from src.pipeline.cleaner import build_silver
    from src.pipeline.feature_store import build_gold

    root = tmp_path_factory.mktemp("gold") / "lake"
    shutil.copytree(FIXTURE_LAKE, root)
    for race in GOLDEN_RACES:
        build_silver(*race, "R", lake_root=str(root), telemetry=False)
        build_gold(*race, "R", lake_root=str(root))
    return str(root)


@pytest.fixture
def hungary_lake(tmp_path):
    """A temporary lake pre-loaded with real Bronze lap data for the 2024 Hungarian GP."""
    root = tmp_path / "lake_hungary"
    shutil.copytree(HUNGARY_FIXTURE_LAKE, root)
    return str(root)


@pytest.fixture(scope="session")
def hungary_silver_laps(tmp_path_factory):
    """Silver laps of the 2024 Hungarian GP, built once per test session. Treat as read-only."""
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver

    root = tmp_path_factory.mktemp("hungary") / "lake"
    shutil.copytree(HUNGARY_FIXTURE_LAKE, root)
    build_silver(*HUNGARY, "R", lake_root=str(root), telemetry=False)
    return lake.read_table(lake.table_path(lake.SILVER, "laps", *HUNGARY, "R", str(root)))
