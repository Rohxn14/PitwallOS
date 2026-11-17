"""
What each car did in each corner of each lap (src/telemetry/corner_features.py) - Phase 5 step 5.2

Step 5.1 gave every race a reference line and its corners. This step reads a race's telemetry lap by lap against that
line and writes, for every car, lap and corner, the numbers a race engineer looks at:

    segment_time_sec     the time spent in the corner's segment of the lap
    corner_speed_kmh     the speed at the corner itself (its marker on the line)
    min_speed_kmh        the slowest point in the segment, and where (min_speed_distance_m). In a corner the cars
                         slow for, this is the apex speed; in one taken flat out while accelerating, the slowest
                         point is the segment's start, which is why corner_speed_kmh exists
    entry_speed_kmh      the speed where braking for the corner began; without braking, ENTRY_M before the slowest point
                         (or at the segment's start if that comes first)
    brake_point_m        how far before the slowest point the braking began (null: the corner was taken without brakes)
    exit_speed_kmh       the speed EXIT_M after the slowest point (or at the segment's end if that comes first)
    throttle_on_m        how far after the slowest point the throttle was back to THROTTLE_ON_PCT
    full_throttle_share  the share of the segment's time at full throttle

and for every car and lap (lap_telemetry): top speed, the shares of the lap at full throttle, on the brakes and
coasting (neither: "lift and coast"), and the sum of its segment times.

THE CORNERS are the official ones where the race has them (MultiViewer's numbers: "turn 4" means what everyone means
by it); where it has none, the bends step 5.1 found on the map, numbered in order.

SEGMENTS. The lap is cut at the midpoints between consecutive corners (by distance along the line): corner k's
segment runs from halfway after corner k-1 to halfway before corner k+1; the first starts at the start/finish line and
the last ends there. Every metre of the lap belongs to exactly one segment, so A LAP'S SEGMENT TIMES ADD UP TO ITS LAP
TIME: that is the check on everything here, against the official lap times (which this code never reads).

WHERE A SAMPLE IS. Each sample's place on the line is the nearest point of the line that is also near where the car's
own distance says it should be (track_model.project_lap): where a circuit passes close to itself (Suzuka's crossover,
a hairpin), the nearest point alone can be on the wrong stretch.

RULES KEPT. Nothing invented: a segment with a telemetry hole in it (or no position), or with fewer than MIN_TICKS
samples (a car that stopped before the corner), gets its row with is_complete False and every feature null, its time
too. Every feature of lap N is known when lap N ends (rule 5: no future).
Brake and throttle readings that are unknown (OpenF1's "no reading") count as not braking / no throttle only for
the shares' denominators' sake: they are left out of both.

Reads Silver (laps, telemetry) and Gold (track_line, track_corners); writes Gold (corner_features, lap_telemetry).

    python -m src.telemetry.corner_features --season 2023 2024 2025 2026       # data/reports/corner_features.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline import contracts, lake
from src.telemetry import track_model

CORNER_TABLE, LAP_TABLE = "corner_features", "lap_telemetry"
OFFICIAL = "official"
ENTRY_M = 100.0              # without braking, "entry" is this far before the slowest point
EXIT_M = 100.0               # "exit" is this far after it
THROTTLE_ON_PCT = 50.0       # the throttle counts as picked up from here
FULL_THROTTLE_PCT = 99.0
COAST_THROTTLE_PCT = 5.0     # under this, and no brake: coasting
MIN_TICKS = 3                # a segment needs this many samples (10 Hz: a 20 m segment at 300 km/h has 2)
TICK_SEC = 0.1


def corner_list(corners: pd.DataFrame) -> pd.DataFrame:
    """The corners features are measured at: the official ones if the race has them, else the map's bends numbered
    in order. Columns: order (1, 2, ... along the lap), label (what the corner is called: "4", or "1A" where the
    official list has a letter, as the Hungaroring's turns 1A and 12A), distance_m, source (which list)."""
    official = corners[corners["method"] == OFFICIAL]
    if len(official):
        labels = official["label"] if "label" in official else official["number"].astype(int).astype(str)
        out = pd.DataFrame({"label": labels.astype(str).to_numpy(), "distance_m": official["distance_m"].to_numpy(float),
                            "source": OFFICIAL})
    else:
        bends = corners[corners["method"] == track_model.MAP].sort_values("distance_m")
        out = pd.DataFrame({"label": [str(i) for i in range(1, len(bends) + 1)],
                            "distance_m": bends["distance_m"].to_numpy(float), "source": track_model.MAP})
    out = out.sort_values("distance_m", kind="stable").reset_index(drop=True)
    return out.assign(order=np.arange(1, len(out) + 1))


def segment_edges(corner_distances: Sequence[float], length: float) -> np.ndarray:
    """The distances where the lap is cut: the line (0), the midpoint between each two consecutive corners, the line
    again (length). One more edge than corners."""
    d = np.asarray(corner_distances, dtype=float)
    return np.concatenate([[0.0], (d[:-1] + d[1:]) / 2.0, [float(length)]])


def lap_on_line(line: pd.DataFrame, lap: pd.DataFrame, length: float) -> np.ndarray:
    """The distance along the line of each sample of one lap, never going backwards, running from about 0 to about
    `length` (a sample just before the line at the lap's start is slightly negative, not `length`)."""
    own = lap["lap_distance_m"].to_numpy(dtype=float)
    share = np.clip(np.nan_to_num(own / length), 0.0, 1.0)     # of the LINE's length: a lap that stops early stays early
    d = track_model.project_lap(line, lap["x"].to_numpy(dtype=float), lap["y"].to_numpy(dtype=float), share)
    d = np.where((share < 0.25) & (d > length / 2), d - length, d)        # across the line at the start ...
    d = np.where((share > 0.75) & (d < length / 2), d + length, d)        # ... and at the end
    return np.maximum.accumulate(d)


def _first_true(mask: np.ndarray) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def segment_features(d: np.ndarray, t: np.ndarray, speed: np.ndarray, throttle: np.ndarray, brake: np.ndarray,
                     start: float, end: float) -> Dict[str, float]:
    """The features of one segment of one lap, from the samples whose distance lies in [start, end). `brake` is 1.0
    braking, 0.0 not, nan unknown; `throttle` in percent, nan unknown. Assumes no hole (the caller checked)."""
    inside = (d >= start) & (d < end)
    i = np.flatnonzero(inside)
    lo, hi = i[0], i[-1] + 1
    sd, ss, st, sb = d[lo:hi], speed[lo:hi], throttle[lo:hi], brake[lo:hi]
    apex = int(np.argmin(ss))
    apex_d = float(sd[apex])
    out = {"min_speed_kmh": float(ss[apex]), "min_speed_distance_m": apex_d}

    # braking for this corner: the last unbroken stretch on the brakes that ends at or before the slowest point
    braking = sb[:apex + 1] > 0.5
    began = None
    if braking.any():
        last = int(np.flatnonzero(braking)[-1])
        began = last
        while began > 0 and braking[began - 1]:
            began -= 1
    if began is not None:
        out["brake_point_m"] = apex_d - float(sd[began])
        out["entry_speed_kmh"] = float(ss[began])
    else:
        out["brake_point_m"] = np.nan
        out["entry_speed_kmh"] = float(np.interp(apex_d - ENTRY_M, sd, ss))        # never before the segment starts

    out["exit_speed_kmh"] = float(np.interp(apex_d + EXIT_M, sd, ss))              # never after it ends
    on = _first_true(st[apex:] >= THROTTLE_ON_PCT)
    out["throttle_on_m"] = float(sd[apex + on] - apex_d) if on is not None else np.nan
    known = ~np.isnan(st)
    out["full_throttle_share"] = float((st[known] >= FULL_THROTTLE_PCT).mean()) if known.any() else np.nan
    return out


def lap_features(lap: pd.DataFrame, line: pd.DataFrame, corners: pd.DataFrame, length: float
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(one row per corner, one row for the lap) for one car's lap. `lap`: its Silver telemetry, in time order."""
    nowhere = (lap["x"].isna() | lap["y"].isna()).to_numpy()        # no position: a hole, whatever the car data says
    lap = lap.assign(x=lap["x"].interpolate(limit_direction="both"), y=lap["y"].interpolate(limit_direction="both"))
    d = lap_on_line(line, lap, length)
    t = lap["session_time_sec"].to_numpy(dtype=float)
    speed = lap["speed_kmh"].to_numpy(dtype=float)
    throttle = lap["throttle_pct"].astype("float64").to_numpy()
    brake = lap["brake"].astype("float64").to_numpy()
    hole = lap["is_data_gap"].to_numpy(dtype=bool) | np.isnan(speed) | nowhere

    edges = segment_edges(corners["distance_m"], length)
    edge_times = np.interp(edges, d, t)
    edge_times[0], edge_times[-1] = t[0], t[-1] + TICK_SEC          # the lap's own start and end: the sum is the lap
    rows = []
    for k, corner in enumerate(corners.itertuples()):
        start, end = edges[k], edges[k + 1]
        inside = (d >= start) & (d < end)
        row: Dict[str, Any] = {"corner_order": int(corner.order), "corner": str(corner.label),
                               "corner_distance_m": float(corner.distance_m),
                               "segment_time_sec": float(edge_times[k + 1] - edge_times[k]),
                               "is_complete": bool(inside.sum() >= MIN_TICKS and not hole[inside].any())}
        if row["is_complete"]:
            row.update(segment_features(d, t, speed, throttle, brake, start, end))
            row["corner_speed_kmh"] = float(np.interp(corner.distance_m, d, speed))
        else:
            row["segment_time_sec"] = np.nan         # a car that stopped before this corner spent no time in it
        rows.append(row)

    whole = not hole.any()
    known_throttle, known_brake = ~np.isnan(throttle), ~np.isnan(brake)
    both = known_throttle & known_brake
    lap_row = {
        "is_complete": bool(whole),
        "top_speed_kmh": float(np.nanmax(speed)) if whole else np.nan,
        "full_throttle_share": float((throttle[known_throttle] >= FULL_THROTTLE_PCT).mean()) if whole and known_throttle.any() else np.nan,
        "brake_share": float((brake[known_brake] > 0.5).mean()) if whole and known_brake.any() else np.nan,
        "coast_share": float(((throttle[both] < COAST_THROTTLE_PCT) & (brake[both] < 0.5)).mean()) if whole and both.any() else np.nan,
        "segments_time_sum_sec": float(edge_times[-1] - edge_times[0]) if whole else np.nan,
        "telemetry_ticks": int(len(lap)),
    }
    return rows, lap_row


# --- one race ------------------------------------------------------------------------------------------------------------

COLUMNS = ["season", "round", "driver_number", "driver_code", "lap_number", "corner_order", "corner", "corner_source",
           "corner_distance_m", "is_complete", "segment_time_sec", "corner_speed_kmh", "min_speed_kmh", "min_speed_distance_m",
           "entry_speed_kmh", "brake_point_m", "exit_speed_kmh", "throttle_on_m", "full_throttle_share"]


def build_corner_features(year: int, round_number: int, session: str = "R",
                          root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> Dict[str, Any]:
    """One race's corner_features and lap_telemetry into Gold, and its report row. Needs step 5.1's track model:
    without one (5.1 refused the race, or never ran) the race fails, and its old corner tables are removed."""
    if not lake.table_path(lake.GOLD, track_model.LINE_TABLE, year, round_number, session, root).exists():
        for table in (CORNER_TABLE, LAP_TABLE):
            lake.remove_table(lake.table_path(lake.GOLD, table, year, round_number, session, root))
        raise FileNotFoundError(f"{year} round {round_number}: no track model (step 5.1 refused this race or never ran)")
    line = lake.read_table(lake.table_path(lake.GOLD, track_model.LINE_TABLE, year, round_number, session, root))
    corners = corner_list(lake.read_table(lake.table_path(lake.GOLD, track_model.CORNERS_TABLE, year, round_number,
                                                          session, root)))
    if corners.empty:
        raise ValueError(f"{year} round {round_number}: the track model has no corner")
    length = track_model.lap_length(line)
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, session, root))
    telemetry = lake.read_table(lake.table_path(lake.SILVER, "telemetry", year, round_number, session, root),
                                columns=["driver_number", "driver_code", "lap_number", "session_time_sec", "x", "y",
                                         "speed_kmh", "throttle_pct", "brake", "lap_distance_m", "is_data_gap"])
    timed = laps.assign(driver_number=laps["driver_number"].astype(str)).set_index(["driver_number", "lap_number"])
    source = corners["source"].iloc[0]

    corner_rows, lap_rows = [], []
    for (driver, lap_number), lap in telemetry.groupby(["driver_number", "lap_number"], sort=False):
        if (driver, lap_number) not in timed.index or len(lap) < 10 * MIN_TICKS:
            continue
        if lap["x"].isna().all() or not lap["lap_distance_m"].max() > 0:
            continue
        rows, lap_row = lap_features(lap, line, corners, length)
        identity = {"season": int(year), "round": int(round_number), "driver_number": str(driver),
                    "driver_code": str(lap["driver_code"].iloc[0]), "lap_number": int(lap_number)}
        corner_rows += [{**identity, "corner_source": source, **row} for row in rows]
        silver = timed.loc[(driver, lap_number)]
        lap_rows.append({**identity, **lap_row, "lap_time_sec": float(silver["lap_time_sec"]),
                         "is_clean_lap": bool(silver["is_clean_lap"])})

    features = pd.DataFrame(corner_rows).reindex(columns=COLUMNS)
    lap_table = pd.DataFrame(lap_rows)
    contracts.validate(features, contracts.GOLD_CORNER_FEATURES, "Gold corner features")
    contracts.validate(lap_table, contracts.GOLD_LAP_TELEMETRY, "Gold lap telemetry")
    manifest = lake.read_manifest(year, round_number, session, root)
    meta = {"season": int(year), "round": int(round_number), "session": session,
            "event_name": manifest.get("event_name"), "corner_source": source}
    lake.write_table(features, lake.table_path(lake.GOLD, CORNER_TABLE, year, round_number, session, root), metadata=meta)
    lake.write_table(lap_table, lake.table_path(lake.GOLD, LAP_TABLE, year, round_number, session, root), metadata=meta)

    clean = lap_table[lap_table["is_clean_lap"] & lap_table["is_complete"] & lap_table["lap_time_sec"].notna()]
    off = (clean["segments_time_sum_sec"] - clean["lap_time_sec"]).abs()
    clean_keys = lap_table.loc[lap_table["is_clean_lap"], ["driver_number", "lap_number"]]
    on_clean = features.merge(clean_keys, on=["driver_number", "lap_number"])
    return {**meta, "location": manifest.get("location"), "corners": int(len(corners)), "laps": int(len(lap_table)),
            "rows": int(len(features)), "clean_laps": int(lap_table["is_clean_lap"].sum()),
            "clean_laps_complete": int(len(clean)),
            "complete_share_clean": float(on_clean["is_complete"].mean()) if len(on_clean) else np.nan,
            "sum_vs_lap_time_median_sec": float(off.median()) if len(off) else np.nan,
            "sum_vs_lap_time_p95_sec": float(off.quantile(0.95)) if len(off) else np.nan,
            "braked_share": float(on_clean["brake_point_m"].notna().mean()) if len(on_clean) else np.nan}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per car, lap and corner: what the telemetry says")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    rows = []
    for year, round_number in lake.list_races(lake.GOLD, track_model.LINE_TABLE, args.lake):
        if year not in args.season:
            continue
        try:
            row = build_corner_features(year, round_number, root=args.lake)
        except Exception as err:
            print(f"{year} round {round_number}: FAILED {type(err).__name__}: {str(err)[:200]}")
            continue
        rows.append(row)
        print(f"{year} round {round_number:>2} {str(row['location']):<18} {row['corners']:>2} corners ({row['corner_source']}), "
              f"{row['rows']:>6,} rows, complete on clean laps {100 * row['complete_share_clean']:.1f} %, segment sums vs "
              f"lap times: median {row['sum_vs_lap_time_median_sec']:.3f} s, p95 {row['sum_vs_lap_time_p95_sec']:.3f} s")
    if not rows:
        print("No race with a track model in these seasons: run src.telemetry.track_model first.")
        return 1
    report = pd.DataFrame(rows)
    path = lake.write_report(report, "corner_features", args.lake)
    print(f"{len(report)} races, {int(report['rows'].sum()):,} rows. Segment sums against the official lap times: median of "
          f"races {report['sum_vs_lap_time_median_sec'].median():.3f} s, worst race {report['sum_vs_lap_time_median_sec'].max():.3f} s. "
          f"Complete on clean laps: {100 * report['complete_share_clean'].median():.1f} % (median race). Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
