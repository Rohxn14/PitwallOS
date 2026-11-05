"""Shared pytest fixtures for the data pipeline tests."""

import shutil
from pathlib import Path

import pytest

FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake"
GOLDEN_RACES = ((2024, 12), (2024, 23))   # British GP (wet/dry, no safety car), Qatar GP (safety cars)
# Kept in its own lake so the tests above that count the races of 2024 are not affected.
HUNGARY_FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake_hungary"
HUNGARY = (2024, 13)                      # Hungarian GP: Norris undercuts Piastri at the second stops


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


@pytest.fixture(scope="session")
def game_lake(golden_gold_lake, tmp_path_factory):
    """
    The strategist game's lake (Phase 4 step 4): the three reference races built to Gold (British, Qatar and Hungary
    2024), with the game's models fitted on them (each race played with the fold that never saw it) and saved in the
    lake, once per test session: (models, root). Tests may add reports to it; nothing else changes it.
    """
    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver
    from src.pipeline.feature_store import build_gold
    from src.simulator import game

    root = tmp_path_factory.mktemp("game") / "lake"
    shutil.copytree(golden_gold_lake, root)
    shutil.copytree(HUNGARY_FIXTURE_LAKE / "bronze" / "season=2024" / "round=13",
                    root / "bronze" / "season=2024" / "round=13")
    build_silver(*HUNGARY, "R", lake_root=str(root), telemetry=False)
    build_gold(*HUNGARY, "R", lake_root=str(root))
    models = game.fit_models(str(root), seasons=(2024,))
    lake.write_model({"version": game.MODELS_VERSION, "folds": models.folds, "misses": models.misses,
                      "dry": models.dry}, game.MODELS_NAME, str(root))
    return models, str(root)


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
