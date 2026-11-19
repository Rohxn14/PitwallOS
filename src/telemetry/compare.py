"""
Two drivers compared, corner by corner (src/telemetry/compare.py) - Phase 5 step 5.3

Where does driver A gain time on driver B, and how: braking later, carrying more speed through the corner, leaving it
faster, or getting back on the throttle sooner? Reads step 5.2's Gold tables (corner_features, lap_telemetry); the
Analyze window's Compare tab draws the result.

PAIRED LAPS. Only laps that BOTH drivers ran clean and complete, compared lap N with lap N: the same fuel load, the same
track, the same point of the race. Tyre ages can still differ (the tables show by how much). Every number below is
first a difference on one lap (A minus B), then summarised over the paired laps.

SIGNS. A gap is A's time minus B's: NEGATIVE = A FASTER. Every other difference is said from the corner's winner's side
in words ("brakes 6 m later").

TWO SUMMARIES, ON PURPOSE.
- The median of a corner's per-lap gaps is what a typical lap looks like, and is what the chart shows. Medians do not
  add up: the 16 corners' medians do not make the median lap gap.
- The mean does add up, exactly: a lap's segment times add up to the lap (step 5.2), so the corners' mean gaps add up
  to the mean lap gap. "How much of the gap comes from slow corners" uses the means.
- The build-up over the lap is the per-lap running total at each corner, then its median: it ends at the median lap gap.

    python -m src.telemetry.compare --season 2023 2024 2025      # every teammate pair: data/reports/teammates.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.dirty_air import FOLLOW_THRESHOLD_SEC
from src.pipeline import lake
from src.telemetry import corner_features as cf

# metric: (which way is better, words for the winner, unit)
METRICS: Dict[str, Tuple[str, str, str]] = {
    "brake_point_m": ("lower", "brakes {v:.0f} m later", "m"),
    "corner_speed_kmh": ("higher", "{v:.0f} km/h faster at the corner", "km/h"),
    "exit_speed_kmh": ("higher", "{v:.0f} km/h faster on the exit", "km/h"),
    "throttle_on_m": ("lower", "on the throttle {v:.0f} m sooner", "m"),
}
CONSISTENT = 2 / 3            # a difference is part of "how" if the winner has it on at least this share of paired laps
SLOW_KMH = 150.0              # a slow corner, for the share of the gap
MIN_PAIRED_LAPS = 10          # the season report leaves out pairs with fewer


class NoPairedLaps(ValueError):
    """The two drivers have no lap that both ran clean and complete."""


@dataclass
class Comparison:
    a: str
    b: str
    laps: List[int]                       # the paired lap numbers
    lap_gap_median_sec: float             # A minus B, per paired lap, median (negative: A faster)
    lap_gap_mean_sec: float
    corners: pd.DataFrame                 # one row per corner (see compare)
    build_up: pd.DataFrame                # corner_order, distance_m (where the corner's segment ends), median, q25, q75
    tyre_age_gap: float = np.nan          # A's tyre age minus B's, median over paired laps (if the laps table is given)
    following: Dict[str, int] = field(default_factory=dict)   # paired laps each driver started following a car
    notes: List[str] = field(default_factory=list)

    @property
    def corners_won_by_a(self) -> int:
        return int((self.corners["gap_median_sec"] < 0).sum())

    def slow_gap_sec(self) -> float:
        """The part of the mean lap gap made in corners slower than SLOW_KMH (seconds, A minus B)."""
        return float(self.corners.loc[self.corners["corner_speed_kmh"] < SLOW_KMH, "gap_mean_sec"].sum())

    def slow_share(self) -> float:
        """The share of the mean lap gap that comes from corners slower than SLOW_KMH (nan if there is no gap)."""
        if not self.lap_gap_mean_sec:
            return np.nan
        return self.slow_gap_sec() / self.lap_gap_mean_sec


def paired_laps(lap_table: pd.DataFrame, a: str, b: str) -> List[int]:
    """The lap numbers both drivers ran clean and complete (step 5.2's lap_telemetry)."""
    ok = lap_table[lap_table["is_clean_lap"].astype(bool) & lap_table["is_complete"].astype(bool)]
    of = lambda driver: set(ok.loc[ok["driver_code"] == driver, "lap_number"].astype(int))
    return sorted(of(a) & of(b))


def _how(row: pd.Series, winner: str) -> str:
    """The differences the corner's winner has on at least CONSISTENT of the paired laps, in words."""
    side = "a" if winner == "A" else "b"
    parts = []
    for metric, (better, words, _) in METRICS.items():
        share = row.get(f"{metric}_{side}_better")
        if pd.isna(share) or share < CONSISTENT:
            continue
        value = abs(row[f"{metric}_diff"])
        parts.append(words.format(v=value))
    return ", ".join(parts) if parts else "no single difference"


def compare(features: pd.DataFrame, lap_table: pd.DataFrame, a: str, b: str) -> Comparison:
    """A against B on one race. `features`, `lap_table`: step 5.2's Gold tables of that race."""
    if a == b:
        raise ValueError("pick two different drivers")
    laps = paired_laps(lap_table, a, b)
    if not laps:
        raise NoPairedLaps(f"{a} and {b} have no lap that both ran clean and complete")
    columns = ["segment_time_sec"] + list(METRICS)
    f = features[features["lap_number"].isin(laps) & features["driver_code"].isin([a, b])]
    wide = f.set_index(["lap_number", "corner_order", "driver_code"])[columns].unstack("driver_code")
    diff = wide.xs(a, level="driver_code", axis=1) - wide.xs(b, level="driver_code", axis=1)

    by_corner = diff.groupby(level="corner_order")
    corners = (f.drop_duplicates("corner_order").set_index("corner_order")[["corner", "corner_distance_m"]]
               .sort_index())
    corners["corner_speed_kmh"] = f.groupby("corner_order")["corner_speed_kmh"].median()
    corners["gap_median_sec"] = by_corner["segment_time_sec"].median()
    corners["gap_mean_sec"] = by_corner["segment_time_sec"].mean()
    corners["a_faster_share"] = by_corner["segment_time_sec"].apply(lambda s: float((s < 0).mean()))
    for metric, (better, _, _) in METRICS.items():
        corners[f"{metric}_diff"] = by_corner[metric].median()
        a_wins = (lambda s: s < 0) if better == "lower" else (lambda s: s > 0)
        b_wins = (lambda s: s > 0) if better == "lower" else (lambda s: s < 0)
        known = lambda s: s.dropna()
        corners[f"{metric}_a_better"] = by_corner[metric].apply(
            lambda s: float(a_wins(known(s)).mean()) if known(s).size else np.nan)
        corners[f"{metric}_b_better"] = by_corner[metric].apply(
            lambda s: float(b_wins(known(s)).mean()) if known(s).size else np.nan)
    corners["winner"] = np.where(corners["gap_median_sec"] < 0, a, np.where(corners["gap_median_sec"] > 0, b, ""))
    corners["how"] = [_how(row, "A" if row["winner"] == a else "B") if row["winner"] else "level"
                      for _, row in corners.iterrows()]

    running = diff["segment_time_sec"].unstack("corner_order").sort_index(axis=1).cumsum(axis=1)
    edges = cf.segment_edges(corners["corner_distance_m"].to_numpy(), np.nan)[1:-1]
    ends = np.append(edges, np.nan)                  # the last segment ends at the line: its distance is the lap's
    build_up = pd.DataFrame({"corner_order": running.columns, "distance_m": ends,
                             "median_sec": running.median().to_numpy(), "q25_sec": running.quantile(0.25).to_numpy(),
                             "q75_sec": running.quantile(0.75).to_numpy()})

    lap_a = lap_table.set_index(["driver_code", "lap_number"])
    timing = lap_a["lap_time_sec"].where(lap_a["lap_time_sec"].notna(), lap_a["segments_time_sum_sec"])
    gaps = np.array([timing.loc[(a, n)] - timing.loc[(b, n)] for n in laps], dtype=float)   # official, to the ms
    out = Comparison(a=a, b=b, laps=laps, lap_gap_median_sec=float(np.median(gaps)),
                     lap_gap_mean_sec=float(gaps.mean()), corners=corners.reset_index(), build_up=build_up)
    if len(laps) < MIN_PAIRED_LAPS:
        out.notes.append(f"only {len(laps)} paired laps: read with care")
    return out


def tyre_age_gap(silver_laps: pd.DataFrame, a: str, b: str, laps: Sequence[int]) -> float:
    """A's tyre age minus B's on the paired laps (median), from Silver: what pairing by lap number does not equalise."""
    t = silver_laps[silver_laps["lap_number"].isin(laps)].set_index(["driver_code", "lap_number"])["tyre_life"]
    gaps = [t.get((a, n)) - t.get((b, n)) for n in laps
            if pd.notna(t.get((a, n))) and pd.notna(t.get((b, n)))]
    return float(np.median(gaps)) if gaps else np.nan


class NotComparable(FileNotFoundError):
    """The race has no corner tables; the message says why and what builds them."""


@dataclass
class RaceCorners:
    """What a comparison needs from one race, read once: step 5.2's two Gold tables and the tyre ages from Silver."""
    year: int
    round_number: int
    features: pd.DataFrame
    lap_table: pd.DataFrame
    tyres: pd.DataFrame                   # driver_code, lap_number, tyre_life
    traffic: Optional[pd.DataFrame] = None   # driver_code, lap_number, gap_on_track_start_sec (Gold lap_features)

    @property
    def drivers(self) -> List[str]:
        return sorted(self.lap_table["driver_code"].astype(str).unique())


def load_race(year: int, round_number: int, session: str = "R",
              root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> RaceCorners:
    """One race's corner tables. Raises NotComparable saying why they are missing and what builds them."""
    from src.telemetry import track_model

    path = lambda layer, table: lake.table_path(layer, table, year, round_number, session, root)
    runner = f"python -m src.pipeline.runner --year {year} --round {round_number}"
    if not path(lake.SILVER, "telemetry").exists():
        raise NotComparable(f"this race is in the lake without telemetry (laps only). To compare drivers corner by "
                            f"corner, build it with telemetry (about 3 minutes): {runner}")
    if not path(lake.GOLD, track_model.LINE_TABLE).exists():
        raise NotComparable("this race has no track model: step 5.1 refuses a race whose car positions are faulty "
                            f"(Hungary 2026 is one), or it has not run. To try: {runner} --skip-extract")
    if not path(lake.GOLD, cf.CORNER_TABLE).exists() or not path(lake.GOLD, cf.LAP_TABLE).exists():
        raise NotComparable(f"this race's corner features are not built. To build them: {runner} --skip-extract")
    tyres = lake.read_table(path(lake.SILVER, "laps"), columns=["driver_code", "lap_number", "tyre_life"])
    traffic = lake.read_table(lake.gold_features_path(year, round_number, session, root),
                              columns=["driver_code", "lap_number", "gap_on_track_start_sec"])
    return RaceCorners(year=int(year), round_number=int(round_number), features=lake.read_table(path(lake.GOLD, cf.CORNER_TABLE)),
                       lap_table=lake.read_table(path(lake.GOLD, cf.LAP_TABLE)), tyres=tyres, traffic=traffic)


def following_laps(traffic: pd.DataFrame, driver: str, laps: Sequence[int]) -> int:
    """On how many of `laps` the driver started the lap less than FOLLOW_THRESHOLD_SEC behind a car on the road (Gold's
    on-track gap; step 2's definition of following). Pairing by lap number does not take traffic out; measured on
    2023-2024, leaving those laps out does not make teammate gaps agree any better from one race to the next (see the
    guide), so they stay in and are counted for the reader."""
    mine = traffic[(traffic["driver_code"] == driver) & traffic["lap_number"].isin(laps)]
    return int((mine["gap_on_track_start_sec"] < FOLLOW_THRESHOLD_SEC).sum())


def compare_race(race: RaceCorners, a: str, b: str) -> Comparison:
    """compare() on a loaded race, with the tyre-age gap."""
    result = compare(race.features, race.lap_table, a, b)
    result.tyre_age_gap = tyre_age_gap(race.tyres, a, b, result.laps)
    if race.traffic is not None:
        result.following = {d: following_laps(race.traffic, d, result.laps) for d in (a, b)}
    return result


def load_comparison(year: int, round_number: int, a: str, b: str, session: str = "R",
                    root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> Comparison:
    """compare() on a race in the lake, with the tyre-age gap from Silver."""
    return compare_race(load_race(year, round_number, session, root), a, b)


def teammates(silver_laps: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """(team, driver, driver) for every team with exactly two drivers in the race, in alphabetical order."""
    pairs = []
    for team, drivers in silver_laps.groupby("team")["driver_code"].unique().items():
        if len(drivers) == 2:
            pairs.append((str(team), *sorted(str(d) for d in drivers)))
    return pairs


def race_teammates(year: int, round_number: int, root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> List[dict]:
    """One row per teammate pair of one race, from the FASTER driver's side (on the mean paired lap)."""
    silver = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, "R", root),
                             columns=["driver_code", "team", "lap_number", "tyre_life"])
    gold = lambda table: lake.read_table(lake.table_path(lake.GOLD, table, year, round_number, "R", root))
    features, lap_table = gold(cf.CORNER_TABLE), gold(cf.LAP_TABLE)
    rows = []
    for team, d1, d2 in teammates(silver):
        try:
            c = compare(features, lap_table, d1, d2)
        except NoPairedLaps:
            continue
        if c.lap_gap_mean_sec > 0:                  # say it from the faster driver's side
            c = compare(features, lap_table, d2, d1)
        corners = c.corners
        rows.append({"season": year, "round": round_number, "team": team, "faster": c.a, "slower": c.b,
                     "paired_laps": len(c.laps), "lap_gap_mean_sec": c.lap_gap_mean_sec,
                     "lap_gap_median_sec": c.lap_gap_median_sec, "corners": len(corners),
                     "corners_won": int((corners["gap_mean_sec"] < 0).sum()),
                     "corners_won_median": c.corners_won_by_a,
                     "slow_corners": int((corners["corner_speed_kmh"] < SLOW_KMH).sum()),
                     "slow_share_of_gap": c.slow_share(),
                     "tyre_age_gap": tyre_age_gap(silver, c.a, c.b, c.laps)})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Every teammate pair, corner by corner")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    rows = []
    for year, round_number in lake.list_races(lake.GOLD, cf.CORNER_TABLE, args.lake):
        if year in args.season:
            rows += race_teammates(year, round_number, args.lake)
    if not rows:
        print("No race with corner features in these seasons: run src.telemetry.corner_features first.")
        return 1
    report = pd.DataFrame(rows)
    path = lake.write_report(report, "teammates", args.lake)
    used = report[report["paired_laps"] >= MIN_PAIRED_LAPS]
    print(f"{len(report)} teammate pairs in {report[['season', 'round']].drop_duplicates().shape[0]} races; "
          f"{len(used)} with {MIN_PAIRED_LAPS}+ paired laps. Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
