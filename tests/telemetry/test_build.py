"""
build.py and the pipeline's Phase 5 step: a race that gets telemetry also gets its track model and corner tables.

On a copy of the OpenF1 test lake (Qatar 2024, two cars). Lusail's official corner list is put in the copy's reference
folder, where the lake keeps MultiViewer's answers, so nothing touches the network.
"""

import json
import shutil
from pathlib import Path

import pytest

from src.lib import race_calendar
from src.pipeline import backfill, lake
from src.pipeline.runner import run_pipeline
from src.telemetry import build, corner_features as cf, track_model as tm
from tests.conftest import QATAR

LUSAIL = Path(__file__).parent.parent / "fixtures" / "multiviewer" / "circuit_150_2024.json"
TABLES = (tm.LINE_TABLE, tm.CORNERS_TABLE, cf.CORNER_TABLE, cf.LAP_TABLE)


@pytest.fixture
def qatar_lake(openf1_qatar_lake, tmp_path):
    root = str(tmp_path / "lake")
    shutil.copytree(openf1_qatar_lake, root)
    lake.write_json(json.loads(LUSAIL.read_text(encoding="utf-8")), lake.reference_path("circuit_150_2024", root))
    return root


def gold(root, table):
    return lake.table_path(lake.GOLD, table, *QATAR, "R", root)


def test_one_call_builds_the_track_model_then_the_corner_tables(qatar_lake):
    out = build.build_race_telemetry(*QATAR, root=qatar_lake)
    assert out["track_built"] and out["telemetry_error"] == "" and out["no_official_corners"] == ""
    assert out["corner_source"] == "official"
    laps = lake.read_table(gold(qatar_lake, cf.LAP_TABLE))
    assert out["corner_rows"] == 16 * len(laps) > 0
    assert all(gold(qatar_lake, t).exists() for t in TABLES)


def test_without_the_official_list_the_maps_bends_are_numbered_and_the_reason_is_kept(qatar_lake, monkeypatch):
    def offline(*args, **kwargs):
        raise race_calendar.CalendarError("MultiViewer could not be reached")
    monkeypatch.setattr(race_calendar, "circuit", offline)
    out = build.build_race_telemetry(*QATAR, root=qatar_lake)
    assert out["corner_source"] == "map" and out["corner_rows"] > 0
    assert "MultiViewer could not be reached" in out["no_official_corners"]


def test_a_track_model_that_fails_leaves_no_old_corner_tables(qatar_lake, monkeypatch):
    build.build_race_telemetry(*QATAR, root=qatar_lake)
    assert all(gold(qatar_lake, t).exists() for t in TABLES)

    def faulty(*args, **kwargs):
        raise ValueError("positions stand still")
    monkeypatch.setattr(tm, "build_track", faulty)
    out = build.build_race_telemetry(*QATAR, root=qatar_lake)              # never raises
    assert not out["track_built"] and out["corner_rows"] == 0
    assert "positions stand still" in out["telemetry_error"]
    assert not any(gold(qatar_lake, t).exists() for t in TABLES)          # nothing older than the telemetry


def test_the_runner_builds_corner_tables_with_telemetry_and_not_without(qatar_lake):
    result = run_pipeline(*QATAR, lake_root=qatar_lake, skip_extract=True, telemetry=True)
    assert result["telemetry_gold"]["corner_rows"] > 0 and result["telemetry_gold"]["corner_source"] == "official"
    assert run_pipeline(*QATAR, lake_root=qatar_lake, skip_extract=True, telemetry=False)["telemetry_gold"] is None


def test_backfill_adds_missing_corner_tables_and_leaves_built_ones_alone(qatar_lake):
    row = backfill.build_race(*QATAR, root=qatar_lake, telemetry=True)   # Silver and Gold are there: no download
    assert row["status"] == backfill.OK and row["source"] == backfill.ALREADY_BUILT
    assert row["corner_rows"] > 0 and row["corner_error"] == ""
    again = backfill.build_race(*QATAR, root=qatar_lake, telemetry=True)
    assert "corner_rows" not in again                                    # built and current: not rebuilt
    laps_only = backfill.build_race(*QATAR, root=qatar_lake, telemetry=False)
    assert "corner_rows" not in laps_only
