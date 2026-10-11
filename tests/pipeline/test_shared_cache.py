"""
One FastF1 download cache for the whole app.

The replay (src/f1_data.enable_cache) and the Bronze extractor ask FastF1 for the same data,
so they must use the same cache folder: the "FastF1 Cache Location" setting. Two folders meant
every race was downloaded and stored twice.
"""

from src.lib.settings import get_settings
from src.pipeline import extractor


def test_extractor_defaults_to_the_settings_cache():
    assert extractor.fastf1_cache_dir() == get_settings().cache_location


def test_extract_session_loads_through_the_shared_cache(tmp_path, monkeypatch):
    seen = {}

    def fake_load(year, round_number, session_type, cache_dir):
        seen["cache_dir"] = cache_dir
        raise RuntimeError("stop here: only the cache folder is under test")

    monkeypatch.setattr(extractor, "_load_fastf1_session", fake_load)
    try:
        extractor.extract_session(2024, 1, "R", lake_root=str(tmp_path))
    except RuntimeError:
        pass
    assert seen["cache_dir"] == get_settings().cache_location


def test_an_explicit_cache_dir_still_wins(tmp_path, monkeypatch):
    seen = {}

    def fake_load(year, round_number, session_type, cache_dir):
        seen["cache_dir"] = cache_dir
        raise RuntimeError("stop")

    monkeypatch.setattr(extractor, "_load_fastf1_session", fake_load)
    try:
        extractor.extract_session(2024, 1, "R", lake_root=str(tmp_path), cache_dir="elsewhere")
    except RuntimeError:
        pass
    assert seen["cache_dir"] == "elsewhere"
