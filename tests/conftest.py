"""Shared pytest fixtures for the data pipeline tests."""

import shutil
from pathlib import Path

import pytest

FIXTURE_LAKE = Path(__file__).parent / "fixtures" / "lake"


@pytest.fixture
def golden_lake(tmp_path):
    """A temporary lake pre-loaded with real Bronze lap data for two 2024 races."""
    root = tmp_path / "lake"
    shutil.copytree(FIXTURE_LAKE, root)
    return str(root)
