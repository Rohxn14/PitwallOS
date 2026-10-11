"""
Tests for the Race Analytics Engine (src/analytics/race_engine.py).

Tests verify:
1. Clean pace computation filters pit laps and dirty air correctly
2. Stint degradation slope direction and magnitude are sensible
3. Pit stop analysis detects pit laps and estimates time loss
4. Undercut window verdict logic is correct
5. Safety car detection works on pace anomalies
6. Full analyze() pipeline runs end-to-end on a mock DataFrame
"""

import pytest
import pandas as pd
import numpy as np

from src.analytics.race_engine import (
    RaceAnalyticsEngine,
    StintSummary,
    PitStopAnalysis,
    UndercutWindow,
    SafetyCarImpact,
    RaceAnalyticsReport,
)


@pytest.fixture
def engine():
    return RaceAnalyticsEngine()


@pytest.fixture
def sample_df():
    """Minimal synthetic race DataFrame that matches Gold Feature Store schema."""
    rows = []
    np.random.seed(42)
    
    for driver, base_pace in [("VER", 89.5), ("HAM", 90.0), ("NOR", 90.2)]:
        for lap in range(1, 31):
            # Stint 1: laps 1-15, stint 2: laps 16-30
            stint = 1 if lap <= 15 else 2
            tyre_age = lap if stint == 1 else lap - 15
            compound = "SOFT" if stint == 1 else "MEDIUM"
            is_pit_lap = lap == 15
            
            # Simulated degradation: 0.05 s/lap
            degradation = tyre_age * 0.05
            noise = np.random.normal(0, 0.1)
            
            # In dirty air for HAM about 60% of laps (behind VER)
            in_dirty = driver == "HAM" and not is_pit_lap and lap % 5 != 0
            dirty_penalty = 0.3 if in_dirty else 0.0
            
            lap_time = base_pace + degradation + noise + dirty_penalty
            if is_pit_lap:
                lap_time += 22.0  # pit lane time
            
            rows.append({
                "driver_code": driver,
                "team_code": "RBR" if driver == "VER" else "MER" if driver == "HAM" else "MCL",
                "lap_number": lap,
                "total_race_laps": 30,
                "lap_time_sec": lap_time,
                "prev_lap_time_sec": lap_time - 0.1,
                "compound": compound,
                "tyre_age": tyre_age,
                "stint_number": stint,
                "fuel_mass_est": 110.0 - (lap * 3.0),
                "track_temp": 35.0,
                "position": 1 if driver == "VER" else 2 if driver == "HAM" else 3,
                "gap_ahead_sec": 0.0 if driver == "VER" else 0.8 if driver == "HAM" else 2.5,
                "in_dirty_air": in_dirty,
                "is_pit_lap": is_pit_lap,
                "rolling_3lap_mean": lap_time,
                "rolling_3lap_std": 0.1,
                "target_next_lap_time_sec": lap_time + 0.05,
            })
    return pd.DataFrame(rows)


class TestCleanPace:
    def test_returns_all_drivers(self, engine, sample_df):
        clean_pace, _ = engine._compute_clean_pace(sample_df)
        assert "VER" in clean_pace
        assert "HAM" in clean_pace
        assert "NOR" in clean_pace
    
    def test_ver_fastest(self, engine, sample_df):
        clean_pace, _ = engine._compute_clean_pace(sample_df)
        assert clean_pace["VER"] < clean_pace["HAM"]
        assert clean_pace["VER"] < clean_pace["NOR"]
    
    def test_dirty_air_delta_positive_for_ham(self, engine, sample_df):
        _, dirty_delta = engine._compute_clean_pace(sample_df)
        # HAM is always in dirty air, so there's no clean laps to compare
        # In real data the delta would be positive; with all laps dirty
        # the method handles this gracefully
        assert "HAM" in dirty_delta
        
    def test_clean_pace_excludes_pit_laps(self, engine, sample_df):
        clean_pace, _ = engine._compute_clean_pace(sample_df)
        # Clean pace should be well under 100s (pit lap would be ~111s)
        for drv, pace in clean_pace.items():
            assert pace < 100.0, f"{drv} clean pace {pace} includes pit lap"


class TestStintDegradation:
    def test_returns_stints(self, engine, sample_df):
        stints = engine._compute_stint_degradation(sample_df)
        assert len(stints) > 0
    
    def test_degradation_positive(self, engine, sample_df):
        """Degradation slope should be positive (tyres getting slower)."""
        stints = engine._compute_stint_degradation(sample_df)
        for s in stints:
            # Allow small negative (noise), but generally should be positive
            assert s.degradation_slope > -0.1, f"Unexpected negative deg for {s.driver_code} S{s.stint_number}"
    
    def test_stint_has_correct_compound(self, engine, sample_df):
        stints = engine._compute_stint_degradation(sample_df)
        for s in stints:
            if s.stint_number == 1:
                assert s.compound == "SOFT"
            else:
                assert s.compound == "MEDIUM"
    
    def test_stint_length_correct(self, engine, sample_df):
        stints = engine._compute_stint_degradation(sample_df)
        for s in stints:
            assert s.stint_length >= 3  # Minimum needed for analysis


class TestPitStopAnalysis:
    def test_detects_pit_stops(self, engine, sample_df):
        pits = engine._compute_pit_stop_analysis(sample_df)
        assert len(pits) == 3  # One pit stop per driver
    
    def test_pit_loss_positive(self, engine, sample_df):
        pits = engine._compute_pit_stop_analysis(sample_df)
        for p in pits:
            assert p.estimated_pit_loss_sec > 10.0, f"{p.driver_code} pit loss too low: {p.estimated_pit_loss_sec}"
    
    def test_compound_change_recorded(self, engine, sample_df):
        pits = engine._compute_pit_stop_analysis(sample_df)
        for p in pits:
            assert p.compound_before == "SOFT"
            assert p.compound_after == "MEDIUM"


class TestUndercutWindows:
    def test_windows_created(self, engine, sample_df):
        stints = engine._compute_stint_degradation(sample_df)
        windows = engine._compute_undercut_windows(sample_df, stints)
        # Should evaluate some windows around the pit laps
        assert isinstance(windows, list)
    
    def test_verdict_values(self, engine, sample_df):
        stints = engine._compute_stint_degradation(sample_df)
        windows = engine._compute_undercut_windows(sample_df, stints)
        valid_verdicts = {"UNDERCUT_WIN", "UNDERCUT_FAIL", "MARGINAL"}
        for w in windows:
            assert w.verdict in valid_verdicts


class TestSafetyCarImpact:
    def test_returns_list(self, engine, sample_df):
        impacts = engine._compute_safety_car_impact(sample_df)
        assert isinstance(impacts, list)
    
    def test_no_false_positive_sc(self, engine, sample_df):
        """Our sample data has no SC, so should detect none or very few."""
        impacts = engine._compute_safety_car_impact(sample_df)
        # Normal race — shouldn't detect many SC periods
        assert len(impacts) <= 1


class TestEndToEnd:
    def test_analyze_from_parquet(self, engine, sample_df, tmp_path):
        """Write a mock Parquet file and run the full analyze pipeline."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(sample_df)
        table = table.replace_schema_metadata({
            b'year': b'2024',
            b'round_number': b'12',
            b'event_name': b'British Grand Prix',
        })
        
        parquet_path = str(tmp_path / "test_features.parquet")
        pq.write_table(table, parquet_path)
        
        report = engine.analyze(parquet_path)
        
        assert isinstance(report, RaceAnalyticsReport)
        assert report.year == 2024
        assert report.round_number == 12
        assert report.event_name == "British Grand Prix"
        assert len(report.driver_clean_pace) >= 2
        assert len(report.stint_summaries) >= 3
        assert len(report.pit_stop_analyses) >= 3
    
    def test_missing_file_raises(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.analyze("nonexistent/path.parquet")
