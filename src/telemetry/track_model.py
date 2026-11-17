"""
The track model (src/telemetry/track_model.py) - Phase 5 step 5.1

Every later telemetry step asks "where on the track?": corner speeds, braking points, where a tyre gives up first.
Silver's lap_distance_m cannot answer it: it is each car's own speed added up, about 1 % short, drifting, different
for every car. This module builds, for one race, the map every car shares:

    the reference line   the median path of the race's good laps: x, y every ~5 m, with the distance along it, the
                         median speed there and how sharply the path bends (curvature)
    the corners          found on that line, two ways (below), and scored against the official corner list

GOOD LAPS: clean laps (Silver's is_clean_lap) whose telemetry has no hole and whose length is within 2 % of the
race's median. Each is stretched onto the same grid by its own share of the lap done (0 to 1), so a car's 1 % short
distance does not matter; the median over laps, point by point, is the line.

FINDING CORNERS
  1. from speed alone (the baseline): a corner is where the median speed has a local minimum, by at least
     MIN_SPEED_DROP_KMH below the speed on both sides. Cars slow for it, so it is a corner. It cannot see a corner
     taken flat out.
  2. from the map: a bend is a stretch where the line curves one way with a radius under MAX_RADIUS_M, turning at
     least MIN_TURN_DEG in all; its corner is where it is tightest. Two corners that turn the same way in a row (a
     double apex) make one such stretch: where its curvature has two peaks with a dip below SPLIT_DIP of the smaller
     one between them, and each part turns MIN_TURN_DEG on its own, it is two corners. It sees flat-out kinks, and
     nothing the cars do.
  3. both: every bend of the map, plus the speed minima no bend explains.

THE ANSWER KEY: MultiViewer's corner list for the circuit (src/lib/race_calendar.circuit): each corner's number and
position, in the telemetry's own coordinates. A found corner matches an official one within MATCH_TOLERANCE_M along
the line, each official corner matched at most once.

Reads Silver (laps, telemetry) and the lake's reference data; writes Gold (track_line, track_corners, each checked
against its contract first) and one report row per race. Pure functions do the work; build_track does the reading and writing.

    python -m src.telemetry.track_model --season 2023 2024 2025        # data/reports/track_model.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pipeline import contracts, lake

GRID_STEP_M = 5.0              # spacing of the reference line
LENGTH_TOLERANCE = 0.02        # a good lap is this close to the race's median length
MAX_LAPS = 150                 # enough for a steady median; more only costs time
SMOOTH_M = 30.0                # the speed profile and the path are smoothed over this distance
MIN_SPEED_DROP_KMH = 10.0      # method 1: a minimum this far below both of its neighbouring maxima is a corner
MAX_RADIUS_M = 600.0           # method 2: a bend curves tighter than this ...
MIN_TURN_DEG = 12.0            # ... and turns at least this much
APEX_SHARE = 0.1               # a corner's apex: the middle of the stretch within 10 % of its tightest curvature
SPLIT_DIP = 0.5                # a bend whose curvature dips below this share of its two peaks is two corners
MERGE_M = 40.0                 # two found corners closer than this are one
MATCH_TOLERANCE_M = 75.0       # a found corner this close to an official one is that corner

LINE_TABLE, CORNERS_TABLE = "track_line", "track_corners"
SPEED, MAP, BOTH = "speed", "map", "both"


# --- the reference line ------------------------------------------------------------------------------------------

def good_laps(laps: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    """(driver_number, lap_number, length_m) of the laps the line is built from."""
    clean = laps.loc[laps["is_clean_lap"].fillna(False).astype(bool), ["driver_number", "lap_number"]]
    clean = clean.assign(driver_number=clean["driver_number"].astype(str))
    per_lap = telemetry.groupby(["driver_number", "lap_number"]).agg(
        length_m=("lap_distance_m", "max"), holes=("is_data_gap", "sum"),
        missing=("x", lambda s: int(s.isna().sum()))).reset_index()
    full = per_lap.merge(clean, on=["driver_number", "lap_number"])
    full = full[(full["holes"] == 0) & (full["missing"] == 0) & (full["length_m"] > 0)]
    if full.empty:
        return full[["driver_number", "lap_number", "length_m"]]
    median = full["length_m"].median()
    full = full[(full["length_m"] / median - 1).abs() <= LENGTH_TOLERANCE]
    return full[["driver_number", "lap_number", "length_m"]].reset_index(drop=True)


def _smooth_loop(values: np.ndarray, window: int) -> np.ndarray:
    """A moving average around a closed loop (the lap ends where it began)."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def reference_line(telemetry: pd.DataFrame, chosen: pd.DataFrame, step_m: float = GRID_STEP_M) -> pd.DataFrame:
    """
    The median path of the chosen laps: one row every ~step_m with distance_m (along the line, from the start/finish
    line), x, y (tenths of a metre, as the telemetry), speed_kmh (median), curvature (1/m, + = turning left) and
    laps (how many laps the point is the median of).
    """
    if chosen.empty:
        raise ValueError("no good lap to build the reference line from")
    if len(chosen) > MAX_LAPS:
        chosen = chosen.iloc[np.linspace(0, len(chosen) - 1, MAX_LAPS).round().astype(int)]
    points = max(int(round(chosen["length_m"].median() / step_m)), 50)
    share = np.arange(points) / points                       # of the lap done: 0 at the line, never quite 1
    wanted = set(zip(chosen["driver_number"], chosen["lap_number"]))
    xs, ys, speeds = [], [], []
    for key, lap in telemetry.groupby(["driver_number", "lap_number"], sort=False):
        if key not in wanted:
            continue
        done = lap["lap_distance_m"].to_numpy(dtype=float)
        done = done / done.max()
        xs.append(np.interp(share, done, lap["x"].to_numpy(dtype=float)))
        ys.append(np.interp(share, done, lap["y"].to_numpy(dtype=float)))
        speeds.append(np.interp(share, done, lap["speed_kmh"].to_numpy(dtype=float)))
    x, y = np.median(xs, axis=0), np.median(ys, axis=0)
    window = max(int(round(SMOOTH_M / step_m)), 1)
    step = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0]))) / 10.0      # metres, closing the loop
    distance = np.concatenate([[0.0], np.cumsum(step[:-1])])
    sx, sy = _smooth_loop(x, window), _smooth_loop(y, window)
    heading = np.unwrap(np.arctan2(np.roll(sy, -1) - np.roll(sy, 1), np.roll(sx, -1) - np.roll(sx, 1)))
    turn = np.roll(heading, -1) - np.roll(heading, 1)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi                                             # the seam of the loop
    spacing = np.roll(step, 1) + step
    curvature = np.divide(turn, spacing, out=np.zeros_like(turn), where=spacing > 0)
    return pd.DataFrame({"distance_m": distance, "x": x, "y": y,
                         "speed_kmh": _smooth_loop(np.median(speeds, axis=0), window),
                         "curvature": _smooth_loop(curvature, window), "laps": len(xs)})


def project(line: pd.DataFrame, x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """The shared lap distance of any points (x, y in the telemetry's tenths of a metre): the distance along the line
    of the nearest point of the line. Every car, every lap, gets the same distance at the same place, unlike Silver's
    lap_distance_m (each car's own speed added up)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([line["x"].to_numpy(dtype=float), line["y"].to_numpy(dtype=float)]))
    _, nearest = tree.query(np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)]))
    return line["distance_m"].to_numpy(dtype=float)[nearest]


PROJECT_CANDIDATES = 24        # project_lap looks at this many nearest points of the line ...
PROJECT_REACH_M = 30.0         # ... no further than this beyond the nearest one ...
PROJECT_STRETCH_M = 60.0       # ... and takes the nearest on the same stretch as the one the car's own distance points at


def project_lap(line: pd.DataFrame, x: Sequence[float], y: Sequence[float], share: Sequence[float]) -> np.ndarray:
    """As project, for the samples of one lap, where `share` says roughly how much of the lap each sample has done
    (0 to 1, from the car's own distance). Where the circuit passes close to itself (a crossover, the two sides of a
    hairpin, the pit straight next to the back straight) the nearest point of the line can be on the wrong stretch:
    so, in two steps: of the points of the line near the sample, the one whose distance is closest to share * length
    says WHICH STRETCH the car is on; the answer is the nearest point of that stretch (within PROJECT_STRETCH_M of it
    along the line). The car's own distance only chooses the stretch: it is 1 % short and drifts, so it must not
    choose the point."""
    from scipy.spatial import cKDTree
    distance = line["distance_m"].to_numpy(dtype=float)
    length = lap_length(line)
    tree = cKDTree(np.column_stack([line["x"].to_numpy(dtype=float), line["y"].to_numpy(dtype=float)]))
    k = min(PROJECT_CANDIDATES, len(line))
    away, index = tree.query(np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)]), k=k)
    candidate = distance[index]                                              # nearest first, per sample
    expected = np.asarray(share, dtype=float)[:, None] * length
    gap = np.abs(candidate - expected)
    gap = np.minimum(gap, length - gap)                                      # the short way round the loop
    near = away <= away[:, :1] + PROJECT_REACH_M * 10
    rows = np.arange(len(candidate))
    pointed_at = candidate[rows, np.where(near, gap, np.inf).argmin(axis=1)][:, None]
    along = np.abs(candidate - pointed_at)
    same_stretch = near & (np.minimum(along, length - along) <= PROJECT_STRETCH_M)
    return candidate[rows, same_stretch.argmax(axis=1)]                      # nearest first: the first on the stretch


def lap_length(line: pd.DataFrame) -> float:
    """The length of the line, back to where it starts (metres)."""
    back = np.hypot(line["x"].iloc[0] - line["x"].iloc[-1], line["y"].iloc[0] - line["y"].iloc[-1]) / 10.0
    return float(line["distance_m"].iloc[-1] + back)


# --- corners ---------------------------------------------------------------------------------------------------------

def _loop_minima(values: np.ndarray, min_drop: float) -> List[int]:
    """Indices of the local minima of a closed loop that lie at least min_drop below the highest value met on each
    side before a lower minimum is reached (their "prominence"). Hand-written: the rule fits in a few lines."""
    n = len(values)
    out = []
    for i in range(n):
        if not (values[i] <= values[i - 1] and values[i] < values[(i + 1) % n]):
            continue
        rises = []
        for direction in (-1, 1):
            highest, j = values[i], i
            for _ in range(n - 1):
                j = (j + direction) % n
                if values[j] < values[i]:
                    break
                highest = max(highest, values[j])
            rises.append(highest - values[i])
        if min(rises) >= min_drop:
            out.append(_middle_of_flat(values, i))
    return out


def _middle_of_flat(values: np.ndarray, i: int, share: float = 0.0) -> int:
    """The middle of the stretch around i whose values stay within `share` of values[i] (a flat-bottomed minimum, or
    with share > 0 a flat-topped peak of |values|): its middle, not its edge, is where the corner is."""
    n, target = len(values), values[i]
    close = (lambda v: v == target) if share == 0 else (lambda v: abs(v) >= (1 - share) * abs(target))
    left = right = 0
    while left < n // 2 and close(values[(i - left - 1) % n]):
        left += 1
    while right < n // 2 and close(values[(i + right + 1) % n]):
        right += 1
    return (i + (right - left) // 2) % n


def corners_from_speed(line: pd.DataFrame, min_drop_kmh: float = MIN_SPEED_DROP_KMH) -> pd.DataFrame:
    """Method 1: where the median speed has a minimum cars really slow for."""
    at = _loop_minima(line["speed_kmh"].to_numpy(dtype=float), min_drop_kmh)
    return _corner_rows(line, at, SPEED)


def corners_from_map(line: pd.DataFrame, max_radius_m: float = MAX_RADIUS_M,
                     min_turn_deg: float = MIN_TURN_DEG) -> pd.DataFrame:
    """Method 2: each stretch where the line curves one way tighter than max_radius_m and turns at least
    min_turn_deg in all; the corner is where it is tightest."""
    curvature = line["curvature"].to_numpy(dtype=float)
    n = len(curvature)
    side = np.where(np.abs(curvature) >= 1.0 / max_radius_m, np.sign(curvature), 0.0)
    step = lap_length(line) / n
    start = next((i for i in range(n) if side[i] != side[i - 1]), None)      # begin at a boundary, not mid-bend
    if start is None:
        return _corner_rows(line, [], MAP)
    order = [(start + k) % n for k in range(n)]
    at, run = [], [order[0]]
    for i in order[1:] + [None]:
        if i is not None and side[i] == side[run[0]]:
            run.append(i)
            continue
        if side[run[0]] != 0 and abs(np.degrees(curvature[run].sum() * step)) >= min_turn_deg:
            at += [run[k] for k in _apexes(curvature[run], step, min_turn_deg)]
        run = [i] if i is not None else []
    return _corner_rows(line, sorted(at), MAP)


def _apexes(curvature: np.ndarray, step: float, min_turn_deg: float, dip: float = SPLIT_DIP) -> List[int]:
    """Where in one bend its corners are: the tightest point, or several if the bend holds separate corners (peaks
    of curvature with a dip below `dip` of the smaller one between every two, each part turning min_turn_deg)."""
    a = np.abs(curvature)
    tightest = _middle_of_flat(a, int(np.argmax(a)), share=APEX_SHARE)
    peaks = [int(np.argmax(a))]
    candidates = [i for i in range(1, len(a) - 1) if a[i] >= a[i - 1] and a[i] > a[i + 1]]
    for i in sorted(candidates, key=lambda i: -a[i]):
        if all(a[min(i, j):max(i, j) + 1].min() < dip * min(a[i], a[j]) for j in peaks):
            peaks.append(i)
    peaks.sort()
    if len(peaks) == 1:
        return [tightest]
    cuts = [0] + [(peaks[q] + peaks[q + 1]) // 2 for q in range(len(peaks) - 1)] + [len(a)]
    parts = [curvature[cuts[q]:cuts[q + 1]] for q in range(len(peaks))]
    if all(abs(np.degrees(part.sum() * step)) >= min_turn_deg for part in parts):
        return [_middle_of_flat(a, k, share=APEX_SHARE) for k in peaks]
    return [tightest]


def corners_from_both(line: pd.DataFrame) -> pd.DataFrame:
    """Method 3: every bend of the map, plus the speed minima no bend is within MERGE_M of."""
    bends, slow = corners_from_map(line), corners_from_speed(line)
    length = lap_length(line)
    unexplained = [all(loop_gap(d, b, length) > MERGE_M for b in bends["distance_m"]) for d in slow["distance_m"]]
    out = pd.concat([bends, slow[unexplained]], ignore_index=True)
    return out.sort_values("distance_m").assign(method=BOTH).reset_index(drop=True)


def _corner_rows(line: pd.DataFrame, at: Sequence[int], method: str) -> pd.DataFrame:
    rows = line.iloc[list(at)]
    return pd.DataFrame({"distance_m": rows["distance_m"].to_numpy(), "x": rows["x"].to_numpy(),
                         "y": rows["y"].to_numpy(), "speed_kmh": rows["speed_kmh"].to_numpy(),
                         "curvature": rows["curvature"].to_numpy(), "method": method})


def loop_gap(a: float, b: float, length: float) -> float:
    """The distance between two points of the lap, the short way round."""
    gap = abs(a - b) % length
    return min(gap, length - gap)


# --- the answer key ---------------------------------------------------------------------------------------------------

def official_corners(circuit: Dict[str, Any], line: pd.DataFrame) -> pd.DataFrame:
    """MultiViewer's corners placed on the line: number, label (the number and MultiViewer's letter, if any: the
    Hungaroring has a turn 1 and a turn 1A), distance_m (of the nearest point of the line), off_line_m (how far the
    official position is from it), x, y."""
    lx, ly = line["x"].to_numpy(dtype=float), line["y"].to_numpy(dtype=float)
    rows = []
    for corner in circuit.get("corners", []):
        x, y = float(corner["trackPosition"]["x"]), float(corner["trackPosition"]["y"])
        away = np.hypot(lx - x, ly - y)
        nearest = int(np.argmin(away))
        rows.append({"number": int(corner["number"]), "label": f"{int(corner['number'])}{corner.get('letter') or ''}",
                     "distance_m": float(line["distance_m"].iloc[nearest]),
                     "off_line_m": float(away[nearest]) / 10.0, "x": x, "y": y,
                     "speed_kmh": float(line["speed_kmh"].iloc[nearest])})
    return pd.DataFrame(rows, columns=["number", "label", "distance_m", "off_line_m", "x", "y", "speed_kmh"])


def score(found: pd.DataFrame, official: pd.DataFrame, length: float,
          tolerance_m: float = MATCH_TOLERANCE_M) -> Dict[str, Any]:
    """How the found corners compare with the official ones: closest pairs first, each corner used once."""
    pairs = sorted((loop_gap(f, o, length), i, j) for i, f in enumerate(found["distance_m"])
                   for j, o in enumerate(official["distance_m"]))
    used_found, used_official, errors = set(), set(), []
    for gap, i, j in pairs:
        if gap > tolerance_m:
            break
        if i in used_found or j in used_official:
            continue
        used_found.add(i)
        used_official.add(j)
        errors.append(gap)
    labels = official["label"] if "label" in official else official["number"].astype(int).astype(str)
    missed = sorted(((int(official["number"].iloc[j]), str(labels.iloc[j])) for j in range(len(official))
                     if j not in used_official))
    missed = [label for _, label in missed]
    return {"official": int(len(official)), "found": int(len(found)), "matched": len(errors),
            "missed": len(missed), "extra": int(len(found) - len(errors)),
            "exact_count": bool(len(found) == len(official)),
            "all_right": bool(not missed and len(found) == len(official)),
            "missed_numbers": " ".join(map(str, missed)),
            "median_error_m": float(np.median(errors)) if errors else np.nan}


# --- one race ------------------------------------------------------------------------------------------------------------

def build_track(year: int, round_number: int, session: str = "R", root: lake.PathLike = lake.DEFAULT_LAKE_ROOT,
                circuit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    One race's track model into Gold (track_line, track_corners), and its report row. `circuit`: MultiViewer's
    description of the circuit (the answer key); without it the corners are found but not scored.
    """
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, session, root))
    telemetry = lake.read_table(lake.table_path(lake.SILVER, "telemetry", year, round_number, session, root),
                                columns=["driver_number", "lap_number", "x", "y", "speed_kmh", "lap_distance_m",
                                         "is_data_gap"])
    chosen = good_laps(laps, telemetry)
    line = reference_line(telemetry, chosen)
    length = lap_length(line)
    found = {SPEED: corners_from_speed(line), MAP: corners_from_map(line), BOTH: corners_from_both(line)}
    manifest = lake.read_manifest(year, round_number, session, root)
    identity = {"season": int(year), "round": int(round_number), "session": session,
                "event_name": manifest.get("event_name"), "location": manifest.get("location")}
    row: Dict[str, Any] = {**identity, "good_laps": int(len(chosen)), "line_points": int(len(line)),
                           "lap_length_m": round(length, 1)}
    corners = pd.concat(found.values(), ignore_index=True)
    if circuit is not None:
        key = official_corners(circuit, line)
        corners = pd.concat([corners, key.assign(method="official")], ignore_index=True)
        row["official_off_line_m"] = round(float(key["off_line_m"].median()), 1) if len(key) else np.nan
        for method, table in found.items():
            row.update({f"{method}_{name}": value for name, value in score(table, key, length).items()})
    try:
        contracts.validate(line, contracts.GOLD_TRACK_LINE, "Gold track line")
        contracts.validate(corners, contracts.GOLD_TRACK_CORNERS, "Gold track corners")
    except contracts.ContractError:
        for table in (LINE_TABLE, CORNERS_TABLE):          # an older map of this race must not outlive the refusal
            lake.remove_table(lake.table_path(lake.GOLD, table, year, round_number, session, root))
        raise
    lake.write_table(line, lake.table_path(lake.GOLD, LINE_TABLE, year, round_number, session, root), metadata=identity)
    lake.write_table(corners, lake.table_path(lake.GOLD, CORNERS_TABLE, year, round_number, session, root),
                     metadata=identity)
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.lib import race_calendar

    parser = argparse.ArgumentParser(description="The track model of every race in the lake, scored against the "
                                                 "official corners")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    rows = []
    for year, round_number in lake.list_races(lake.SILVER, "telemetry", args.lake):
        if year not in args.season:
            continue
        try:
            sessions = lake.read_table(lake.table_path(lake.BRONZE, lake.openf1_table("sessions"), year, round_number,
                                                       "R", args.lake))
            circuit = race_calendar.circuit(year, int(sessions["circuit_key"].iloc[0]), args.lake)
        except Exception as err:
            print(f"{year} round {round_number}: no answer key ({type(err).__name__}: {err})")
            circuit = None
        try:
            rows.append(build_track(year, round_number, root=args.lake, circuit=circuit))
        except Exception as err:
            print(f"{year} round {round_number}: FAILED {type(err).__name__}: {err}")
            continue
        r = rows[-1]
        print(f"{year} round {round_number:>2} {str(r['location']):<18} {r['lap_length_m']:>7.0f} m  official "
              f"{r.get('official', r.get('both_official', '-'))}  speed {r.get('speed_found', '-')}  map "
              f"{r.get('map_found', '-')}  both {r.get('both_found', '-')}  missed [{r.get('both_missed_numbers', '')}]")
    if not rows:
        print("No race with Silver telemetry in these seasons.")
        return 1
    report = pd.DataFrame(rows)
    path = lake.write_report(report, "track_model", args.lake)
    for method in (SPEED, MAP, BOTH):
        if f"{method}_exact_count" in report:
            print(f"{method:>5}: exact count in {int(report[f'{method}_exact_count'].sum())} of {len(report)} races; "
                  f"every corner right in {int(report[f'{method}_all_right'].sum())}; matched "
                  f"{int(report[f'{method}_matched'].sum())} of {int(report[f'{method}_official'].sum())} official "
                  f"corners, {int(report[f'{method}_extra'].sum())} extra")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
