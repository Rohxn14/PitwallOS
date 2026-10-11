"""
Replay cache file names (src/f1_data.py).

The old name contained the session's ':' ("2026 Season Round 1: Australian Grand Prix - Race").
On Windows that writes the pickle into a hidden NTFS alternate data stream, so ~1.6 GB of caches
were invisible. These tests pin the safe name and the one-time move from the old name.

Race and sprint caches carry a version in their name (CACHE_VERSIONS): version 2 holds the
running order from src/lib/running_order.py. An older race cache holds the old, wrong order, so it
is never loaded: the data is rebuilt, and the outdated copies are removed once the new one is saved.
"""

import os
import pickle

import pytest

f1_data = pytest.importorskip("src.f1_data")


class FakeSession:
    def __str__(self):
        return "2026 Season Round 1: Australian Grand Prix - Race"


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(f1_data.sys, "argv", ["main.py"])
    return tmp_path


def _write(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_race_cache_name_is_safe_and_versioned(in_tmp):
    name = os.path.basename(f1_data._cache_path(FakeSession(), "race"))
    assert ":" not in name and " " not in name
    assert name == "2026_Season_Round_1_Australian_Grand_Prix_-_Race_race_telemetry_v2.pkl"


def test_quali_cache_name_is_unchanged(in_tmp):
    name = os.path.basename(f1_data._cache_path(FakeSession(), "quali"))
    assert name == "2026_Season_Round_1_Australian_Grand_Prix_-_Race_quali_telemetry.pkl"


def test_save_then_load_round_trips(in_tmp):
    f1_data._save_cached(FakeSession(), "race", {"frames": [1, 2, 3]})
    assert f1_data._load_cached(FakeSession(), "race") == {"frames": [1, 2, 3]}
    assert not any(p.endswith(".tmp") for p in os.listdir("computed_data"))


def test_missing_cache_returns_none(in_tmp):
    assert f1_data._load_cached(FakeSession(), "race") is None


def test_old_quali_cache_is_moved_to_the_new_name(in_tmp):
    legacy = f1_data._legacy_cache_path(FakeSession(), "quali")
    _write(legacy, {"results": ["old"]})

    assert f1_data._load_cached(FakeSession(), "quali") == {"results": ["old"]}
    assert os.path.exists(f1_data._cache_path(FakeSession(), "quali"))
    with pytest.raises(FileNotFoundError):
        open(legacy, "rb")


def test_old_race_caches_are_not_loaded(in_tmp):
    """Both older race caches (hidden name, unversioned safe name) hold the old running order."""
    for stale in f1_data._stale_cache_paths(FakeSession(), "race"):
        _write(stale, {"frames": ["old order"]})
    assert f1_data._load_cached(FakeSession(), "race") is None


def test_saving_a_rebuilt_race_cache_removes_the_outdated_ones(in_tmp):
    stale = f1_data._stale_cache_paths(FakeSession(), "race")
    for path in stale:
        _write(path, {"frames": ["old order"]})

    f1_data._save_cached(FakeSession(), "race", {"frames": ["new order"]})

    assert f1_data._load_cached(FakeSession(), "race") == {"frames": ["new order"]}
    assert not any(os.path.exists(p) for p in stale)


def test_refresh_flag_ignores_the_cache(in_tmp, monkeypatch):
    f1_data._save_cached(FakeSession(), "race", {"frames": []})
    monkeypatch.setattr(f1_data.sys, "argv", ["main.py", "--refresh-data"])
    assert f1_data._load_cached(FakeSession(), "race") is None
