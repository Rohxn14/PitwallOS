"""
Race Analytics Engine (src/analytics/race_engine.py)

Consumes Gold Feature Store Parquet data and computes:
1. True clean pace per driver (filtered for dirty air & pit laps)
2. Stint degradation slope (seconds lost per lap on each compound)
3. Pit stop time loss & delta analysis
4. Undercut / overcut opportunity windows
5. Safety Car & VSC "cheap pit stop" delta gains

This is a pure computation module — no UI, no side effects.
All methods accept a DataFrame and return structured results.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Classes — typed containers for analytics results
# ---------------------------------------------------------------------------

@dataclass
class StintSummary:
    """Summary of a single tyre stint for one driver."""
    driver_code: str
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int
    stint_length: int
    avg_pace_sec: float
    degradation_slope: float  # seconds lost per lap (positive = getting slower)
    best_lap_sec: float
    worst_lap_sec: float


@dataclass
class PitStopAnalysis:
    """Analysis of a pit stop event."""
    driver_code: str
    pit_lap: int
    compound_before: str
    compound_after: str
    estimated_pit_loss_sec: float  # time lost due to pitting vs staying out
    position_before: Optional[int] = None
    position_after: Optional[int] = None


@dataclass
class UndercutWindow:
    """Evaluation of an undercut opportunity between two drivers."""
    attacking_driver: str
    defending_driver: str
    evaluation_lap: int
    gap_sec: float
    pit_loss_sec: float
    fresh_tyre_advantage_per_lap: float
    laps_of_advantage: int
    undercut_margin_sec: float  # positive = undercut succeeds
    verdict: str  # "UNDERCUT_WIN", "UNDERCUT_FAIL", "MARGINAL"


@dataclass
class SafetyCarImpact:
    """Impact of a safety car period on pit strategy."""
    sc_type: str  # "SC" or "VSC"
    start_lap: int
    end_lap: int
    normal_pit_loss_sec: float
    sc_pit_loss_sec: float
    time_saved_sec: float
    drivers_who_pitted: List[str] = field(default_factory=list)


@dataclass
class RaceAnalyticsReport:
    """Complete analytics report for a race."""
    event_name: str
    year: int
    round_number: int
    driver_clean_pace: Dict[str, float]       # driver -> avg clean pace (sec)
    driver_dirty_air_delta: Dict[str, float]  # driver -> avg time lost in dirty air (sec)
    stint_summaries: List[StintSummary]
    pit_stop_analyses: List[PitStopAnalysis]
    undercut_windows: List[UndercutWindow]
    safety_car_impacts: List[SafetyCarImpact]


# ---------------------------------------------------------------------------
# Core Analytics Engine
# ---------------------------------------------------------------------------

class RaceAnalyticsEngine:
    """
    Stateless analytics engine that operates on Gold Feature Store DataFrames.
    
    Usage:
        engine = RaceAnalyticsEngine()
        report = engine.analyze("data/features/race_2024_12_features.parquet")
    """
    
    # Typical pit lane time loss under green flag conditions (seconds)
    DEFAULT_PIT_LOSS_SEC = 22.0
    # Reduced pit loss under VSC (~40% cheaper)
    VSC_PIT_LOSS_SEC = 10.0
    # Reduced pit loss under full SC (~55% cheaper)
    SC_PIT_LOSS_SEC = 8.0
    # Dirty air threshold — within this gap, downforce is reduced
    DIRTY_AIR_THRESHOLD_SEC = 1.5
    
    def analyze(self, features_path: str) -> RaceAnalyticsReport:
        """
        Run full race analytics on a Gold Feature Store Parquet file.
        
        Args:
            features_path: Path to data/features/race_{year}_{round}_features.parquet
            
        Returns:
            RaceAnalyticsReport with all computed metrics.
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature store not found: {features_path}")
            
        table = pq.read_table(features_path)
        df = table.to_pandas()
        metadata = table.schema.metadata or {}
        
        year = int(metadata.get(b'year', b'2024').decode())
        round_number = int(metadata.get(b'round_number', b'0').decode())
        event_name = metadata.get(b'event_name', b'Race').decode()
        
        # Run each analysis module
        clean_pace, dirty_air_delta = self._compute_clean_pace(df)
        stint_summaries = self._compute_stint_degradation(df)
        pit_analyses = self._compute_pit_stop_analysis(df)
        undercut_windows = self._compute_undercut_windows(df, stint_summaries)
        sc_impacts = self._compute_safety_car_impact(df)
        
        return RaceAnalyticsReport(
            event_name=event_name,
            year=year,
            round_number=round_number,
            driver_clean_pace=clean_pace,
            driver_dirty_air_delta=dirty_air_delta,
            stint_summaries=stint_summaries,
            pit_stop_analyses=pit_analyses,
            undercut_windows=undercut_windows,
            safety_car_impacts=sc_impacts
        )
    
    # ------------------------------------------------------------------
    # 1. True Clean Pace vs Dirty Air Delta
    # ------------------------------------------------------------------
    
    def _compute_clean_pace(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Separates clean-air laps from dirty-air laps to calculate:
        - True intrinsic pace per driver (clean air only)
        - Average time penalty from running in dirty air
        
        Clean air lap = not a pit lap AND gap_ahead > 1.5s AND valid lap time.
        """
        clean_pace = {}
        dirty_air_delta = {}
        
        for driver in df['driver_code'].unique():
            drv = df[(df['driver_code'] == driver) & (df['lap_time_sec'].notna()) & (~df['is_pit_lap'])]
            
            if drv.empty:
                continue
                
            clean_laps = drv[~drv['in_dirty_air']]
            dirty_laps = drv[drv['in_dirty_air']]
            
            if not clean_laps.empty:
                # Use median to be robust against outliers (SC restarts, etc.)
                clean_pace[driver] = round(float(clean_laps['lap_time_sec'].median()), 3)
            
            if not clean_laps.empty and not dirty_laps.empty:
                delta = dirty_laps['lap_time_sec'].median() - clean_laps['lap_time_sec'].median()
                dirty_air_delta[driver] = round(float(delta), 3)
            else:
                dirty_air_delta[driver] = 0.0
                
        return clean_pace, dirty_air_delta
    
    # ------------------------------------------------------------------
    # 2. Stint Degradation Analysis
    # ------------------------------------------------------------------
    
    def _compute_stint_degradation(self, df: pd.DataFrame) -> List[StintSummary]:
        """
        For each driver stint, fits a linear slope to lap times to measure
        tyre degradation rate (seconds lost per lap).
        
        degradation_slope > 0 means the driver is getting slower each lap
        (tyres are wearing out). A typical value is +0.05 to +0.15 s/lap.
        """
        summaries = []
        
        for driver in df['driver_code'].unique():
            drv = df[df['driver_code'] == driver].copy()
            
            for stint_num in drv['stint_number'].unique():
                stint = drv[(drv['stint_number'] == stint_num) & (drv['lap_time_sec'].notna()) & (~drv['is_pit_lap'])]
                
                if len(stint) < 3:
                    continue
                    
                compound = stint['compound'].mode().iloc[0] if not stint['compound'].mode().empty else 'UNKNOWN'
                
                # Fit linear regression: lap_time = slope * tyre_age + intercept
                x = stint['tyre_age'].values.astype(float)
                y = stint['lap_time_sec'].values.astype(float)
                
                if len(x) >= 2:
                    slope, _ = np.polyfit(x, y, 1)
                else:
                    slope = 0.0
                    
                summaries.append(StintSummary(
                    driver_code=driver,
                    stint_number=int(stint_num),
                    compound=compound,
                    start_lap=int(stint['lap_number'].min()),
                    end_lap=int(stint['lap_number'].max()),
                    stint_length=int(stint['lap_number'].max() - stint['lap_number'].min() + 1),
                    avg_pace_sec=round(float(y.mean()), 3),
                    degradation_slope=round(float(slope), 4),
                    best_lap_sec=round(float(y.min()), 3),
                    worst_lap_sec=round(float(y.max()), 3)
                ))
                
        return summaries
    
    # ------------------------------------------------------------------
    # 3. Pit Stop Time Loss Analysis
    # ------------------------------------------------------------------
    
    def _compute_pit_stop_analysis(self, df: pd.DataFrame) -> List[PitStopAnalysis]:
        """
        Identifies pit stop laps and estimates the time lost by comparing
        pit lap time to the driver's rolling clean pace.
        """
        analyses = []
        
        for driver in df['driver_code'].unique():
            drv = df[df['driver_code'] == driver].sort_values('lap_number')
            pit_laps = drv[drv['is_pit_lap']]
            
            for _, pit_row in pit_laps.iterrows():
                lap_num = int(pit_row['lap_number'])
                
                # Find compound before and after
                before = drv[drv['lap_number'] < lap_num]
                after = drv[drv['lap_number'] > lap_num]
                
                compound_before = before['compound'].iloc[-1] if not before.empty else 'UNKNOWN'
                compound_after = after['compound'].iloc[0] if not after.empty else 'UNKNOWN'
                
                # Estimate pit loss as: pit_lap_time - rolling_clean_pace
                clean_ref = drv[(~drv['is_pit_lap']) & (drv['lap_time_sec'].notna())]
                nearby_clean = clean_ref[(clean_ref['lap_number'] >= lap_num - 5) & (clean_ref['lap_number'] <= lap_num + 5)]
                
                if not nearby_clean.empty:
                    baseline_pace = nearby_clean['lap_time_sec'].median()
                    pit_time = pit_row['lap_time_sec']
                    if pd.notna(pit_time):
                        pit_loss = float(pit_time - baseline_pace)
                    else:
                        pit_loss = self.DEFAULT_PIT_LOSS_SEC
                else:
                    pit_loss = self.DEFAULT_PIT_LOSS_SEC
                    
                pos_before = int(pit_row['position']) if pd.notna(pit_row.get('position')) else None
                # Look for position on the lap right after
                after_row = drv[drv['lap_number'] == lap_num + 1]
                pos_after = int(after_row.iloc[0]['position']) if not after_row.empty and pd.notna(after_row.iloc[0].get('position')) else None
                    
                analyses.append(PitStopAnalysis(
                    driver_code=driver,
                    pit_lap=lap_num,
                    compound_before=compound_before,
                    compound_after=compound_after,
                    estimated_pit_loss_sec=round(pit_loss, 2),
                    position_before=pos_before,
                    position_after=pos_after
                ))
                
        return analyses
    
    # ------------------------------------------------------------------
    # 4. Undercut / Overcut Window Calculator
    # ------------------------------------------------------------------
    
    def _compute_undercut_windows(self, df: pd.DataFrame, stint_summaries: List[StintSummary]) -> List[UndercutWindow]:
        """
        Evaluates undercut opportunities between drivers on different strategies.
        
        An undercut succeeds when:
            pit_loss < (pace_delta + fresh_tyre_advantage) * laps_of_advantage
            
        We evaluate windows around actual pit stops to show what the optimal
        strategy decision was.
        """
        windows = []
        
        # Build a degradation lookup: (driver, stint) -> slope
        deg_lookup = {}
        for s in stint_summaries:
            deg_lookup[(s.driver_code, s.stint_number)] = s.degradation_slope
            
        pit_laps_df = df[df['is_pit_lap']].copy()
        
        for driver in pit_laps_df['driver_code'].unique():
            drv_pits = pit_laps_df[pit_laps_df['driver_code'] == driver]
            
            for _, pit_row in drv_pits.iterrows():
                eval_lap = int(pit_row['lap_number'])
                
                # Find rivals within 5 seconds at this lap
                lap_df = df[(df['lap_number'] == eval_lap) & (df['driver_code'] != driver)]
                
                for _, rival_row in lap_df.iterrows():
                    rival = rival_row['driver_code']
                    
                    # Only evaluate if gap data is available
                    drv_at_lap = df[(df['driver_code'] == driver) & (df['lap_number'] == eval_lap)]
                    if drv_at_lap.empty:
                        continue
                        
                    drv_row = drv_at_lap.iloc[0]
                    drv_pos = drv_row.get('position')
                    rival_pos = rival_row.get('position')
                    
                    if pd.isna(drv_pos) or pd.isna(rival_pos):
                        continue
                    if abs(int(drv_pos) - int(rival_pos)) > 2:
                        continue  # Only analyze close battles
                        
                    # Estimate fresh tyre pace advantage (typical: 0.8 - 1.5 s/lap)
                    drv_stint = int(drv_row.get('stint_number', 1))
                    rival_stint = int(rival_row.get('stint_number', 1))
                    drv_deg = deg_lookup.get((driver, drv_stint), 0.08)
                    fresh_tyre_adv = max(0.3, abs(drv_deg) * 8)  # estimate from degradation
                    
                    gap = 3.0  # default gap estimate
                    if pd.notna(drv_row.get('gap_ahead_sec')):
                        gap = abs(float(drv_row['gap_ahead_sec']))
                        
                    pit_loss = self.DEFAULT_PIT_LOSS_SEC
                    laps_adv = max(1, min(5, int(rival_row.get('tyre_age', 10)) - 1))
                    
                    total_gain = fresh_tyre_adv * laps_adv
                    margin = total_gain - pit_loss
                    
                    if margin > 1.0:
                        verdict = "UNDERCUT_WIN"
                    elif margin > -1.0:
                        verdict = "MARGINAL"
                    else:
                        verdict = "UNDERCUT_FAIL"
                        
                    windows.append(UndercutWindow(
                        attacking_driver=driver,
                        defending_driver=rival,
                        evaluation_lap=eval_lap,
                        gap_sec=round(gap, 2),
                        pit_loss_sec=pit_loss,
                        fresh_tyre_advantage_per_lap=round(fresh_tyre_adv, 3),
                        laps_of_advantage=laps_adv,
                        undercut_margin_sec=round(margin, 2),
                        verdict=verdict
                    ))
                    
        return windows
    
    # ------------------------------------------------------------------
    # 5. Safety Car / VSC Impact Analysis
    # ------------------------------------------------------------------
    
    def _compute_safety_car_impact(self, df: pd.DataFrame) -> List[SafetyCarImpact]:
        """
        Identifies laps where drivers pitted during safety car / VSC periods
        and calculates the time saved vs. a normal green-flag pit stop.
        
        Note: This uses a heuristic based on the feature store data.
        A future enhancement could cross-reference race_control_messages.
        """
        impacts = []
        
        # Detect potential SC/VSC laps: laps where many drivers have unusually slow times
        if 'lap_time_sec' not in df.columns:
            return impacts
            
        lap_medians = df.groupby('lap_number')['lap_time_sec'].median()
        overall_median = df[~df['is_pit_lap']]['lap_time_sec'].median()
        
        if pd.isna(overall_median):
            return impacts
            
        # SC/VSC laps are typically 15-40% slower than normal race pace
        slow_laps = lap_medians[lap_medians > overall_median * 1.15]
        
        if slow_laps.empty:
            return impacts
            
        # Group consecutive slow laps into SC periods
        slow_lap_nums = sorted(slow_laps.index.tolist())
        sc_periods = []
        current_start = slow_lap_nums[0]
        current_end = slow_lap_nums[0]
        
        for lap in slow_lap_nums[1:]:
            if lap <= current_end + 2:  # allow 1-lap gap
                current_end = lap
            else:
                sc_periods.append((current_start, current_end))
                current_start = lap
                current_end = lap
        sc_periods.append((current_start, current_end))
        
        for start_lap, end_lap in sc_periods:
            # Check who pitted during this window
            sc_pitters = df[
                (df['is_pit_lap']) &
                (df['lap_number'] >= start_lap) &
                (df['lap_number'] <= end_lap)
            ]['driver_code'].unique().tolist()
            
            median_slow = float(lap_medians.loc[start_lap:end_lap].median())
            slowdown_pct = (median_slow - overall_median) / overall_median
            
            # Determine SC type from slowdown magnitude
            sc_type = "SC" if slowdown_pct > 0.30 else "VSC"
            sc_pit_loss = self.SC_PIT_LOSS_SEC if sc_type == "SC" else self.VSC_PIT_LOSS_SEC
            
            impacts.append(SafetyCarImpact(
                sc_type=sc_type,
                start_lap=int(start_lap),
                end_lap=int(end_lap),
                normal_pit_loss_sec=self.DEFAULT_PIT_LOSS_SEC,
                sc_pit_loss_sec=sc_pit_loss,
                time_saved_sec=round(self.DEFAULT_PIT_LOSS_SEC - sc_pit_loss, 1),
                drivers_who_pitted=sc_pitters
            ))
            
        return impacts


# ---------------------------------------------------------------------------
# Convenience function for quick CLI usage
# ---------------------------------------------------------------------------

def analyze_race(features_path: str) -> RaceAnalyticsReport:
    """Shorthand to run the analytics engine on a feature store file."""
    engine = RaceAnalyticsEngine()
    return engine.analyze(features_path)


if __name__ == "__main__":
    import json
    
    path = "data/features/race_2024_12_features.parquet"
    if os.path.exists(path):
        report = analyze_race(path)
        print(f"\n=== {report.event_name} {report.year} Round {report.round_number} ===")
        print(f"\n--- Clean Pace (median, seconds) ---")
        for drv, pace in sorted(report.driver_clean_pace.items(), key=lambda x: x[1]):
            delta = report.driver_dirty_air_delta.get(drv, 0.0)
            print(f"  {drv}: {pace:.3f}s  (dirty air penalty: +{delta:.3f}s)")
        print(f"\n--- Stint Degradation ---")
        for s in report.stint_summaries[:10]:
            print(f"  {s.driver_code} Stint {s.stint_number} ({s.compound}): "
                  f"Laps {s.start_lap}-{s.end_lap}, Deg: {s.degradation_slope:+.4f} s/lap, "
                  f"Avg: {s.avg_pace_sec:.3f}s")
        print(f"\n--- Pit Stops ({len(report.pit_stop_analyses)} total) ---")
        for p in report.pit_stop_analyses[:5]:
            print(f"  {p.driver_code} Lap {p.pit_lap}: {p.compound_before} -> {p.compound_after}, "
                  f"Loss: {p.estimated_pit_loss_sec:.1f}s, Pos: P{p.position_before} -> P{p.position_after}")
        print(f"\n--- Undercut Windows ({len(report.undercut_windows)} evaluated) ---")
        for u in report.undercut_windows[:5]:
            print(f"  {u.attacking_driver} vs {u.defending_driver} Lap {u.evaluation_lap}: "
                  f"Margin {u.undercut_margin_sec:+.2f}s -> {u.verdict}")
        print(f"\n--- Safety Car Impacts ({len(report.safety_car_impacts)}) ---")
        for sc in report.safety_car_impacts:
            print(f"  {sc.sc_type} Laps {sc.start_lap}-{sc.end_lap}: "
                  f"Saved {sc.time_saved_sec:.1f}s, Pitters: {sc.drivers_who_pitted}")
    else:
        print(f"Feature store not found at {path}. Run pipeline first.")
