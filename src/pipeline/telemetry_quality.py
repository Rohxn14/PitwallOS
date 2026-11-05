"""
How complete and how trustworthy each race's telemetry is (src/pipeline/telemetry_quality.py) - Phase 5 step 5.0

One row per race, from Silver telemetry (the 10 Hz grid), Silver laps and Bronze (the manifest, F1's timing loop
speeds, OpenF1's lap table). Every Phase 5 step builds on these samples, so before any of them is trusted we measure:

    coverage    drivers who completed a lap / drivers with telemetry (who is missing, by car number)
    clock       where the samples came from (FastF1 or OpenF1); for OpenF1 the clock's report, recomputed from Bronze
                (telemetry_clock.session_offset): which estimate (lap 1 or the later laps), how far apart they were
    holes       ticks inside a hole (source samples more than 2 s apart: Silver's is_data_gap) as a share of the RACING
                ticks: outside red-flag laps (while the race is stopped the cars sit in the pit lane and send nothing)
                and outside the lap a car stopped on (a driver's last lap without a lap time: Russell, Australia 2023,
                engine fire on lap 18, sends nothing for the 38 minutes his unfinished lap stays open). Both are real
                and expected; the share over all ticks is reported apart. Also how many holes, the longest
    full laps   clean laps (Silver's is_clean_lap) with telemetry and no hole: the laps Phase 5 measures
    lap length  the distance driven on each full lap (speed integrated over time): the race's median, and the share of
                full laps more than 2 % away from it (a lap is a lap: they should agree)
    speed vs    the same lap measured a second, independent way: the path drawn by the position stream (its own
    position    receiver). Speed integrated / path length, median over full laps: ~1.00 if both streams are right
    line speed  the car's own speed at the tick nearest the end of each lap minus the speed F1's timing loop measured at
                the line (Bronze SpeedFL). Its median is a steady OFFSET per circuit (-1 to +12 km/h in 2023-2024: the
                loop is not exactly on the line); what is checked is how STEADY it is lap to lap (interquartile range):
                0.9-2.8 km/h in the 44 sound races of 2023-2024, over 100 km/h where the clock was minutes out
    finish line where the cars are, by the map, when the lap table says they cross the line (median x, y of the
                lap ends, pit laps left out), measured FROM A LANDMARK the clock cannot move: the slowest point of the
                lap (median over clean laps of where each lap's speed is lowest: a hairpin or chicane). The line does
                not move from year to year, so each race is compared with every race at that circuit (season_report):
                a clock error of 0.45 s put Las Vegas 2024's line 36 m from 2023's, 2.1 s put Zandvoort 2023 174 m from
                2024 (with the right clock 4 and 7 m). From the landmark, not on the bare map: in 2025 the map of
                Suzuka itself moved by about 30 m (the slowest corner by (-8.8, +30.1) m, the line by (-6.8, +27.3) m)
    no reading  ticks whose brake is unknown (held from a sample that carried 104, "no reading"); the throttle has no
                such column: Silver interpolates it across a 104 like across any missing sample
    DRS open    share of ticks with DRS open (0 for 2026 cars, which have none)

Each check has a named limit (below); a race outside one gets the reason in `problems` ("" = every check passed).
Claude's expectation, written before measuring: holes under 0.1 % of ticks, 95 % of clean laps complete, lap lengths
within 2 % of the race's median, speed and position within 1 % of each other, line speed within 5 km/h. The line speed
as an absolute difference was WRONG (an offset per circuit, measured on 2023-2024): the limit is now on its steadiness,
5 km/h, set after looking (twice the worst sound race); the finish line, 15 m (about 0.2 s at racing speed; sound
races sit 4-7 m from the other years), was added after the clock errors were found.

    python -m src.pipeline.telemetry_quality --season 2023 2024 2025      data/reports/telemetry_quality.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.pipeline import lake

REPORT = "telemetry_quality"
HZ = 10                      # Silver's grid (cleaner.TELEMETRY_HZ)
MAX_HOLE_SHARE = 0.001       # of the ticks outside red-flag laps
MIN_FULL_LAP_SHARE = 0.95    # of clean laps
LENGTH_TOLERANCE = 0.02      # a full lap this far from the race's median length counts as off
MAX_OFF_LENGTH_SHARE = 0.01  # of full laps
SPEED_POSITION_BAND = (0.99, 1.01)
MAX_LINE_SPEED_IQR_KMH = 5.0
MAX_LINE_OFFSET_M = 15.0


def runs(flags: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in a boolean array, in order (e.g. [F, T, T, F, T] -> [2, 1])."""
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def hole_summary(tel: pd.DataFrame) -> Dict[str, Any]:
    """Ticks inside holes, how many holes (per driver), and the longest, in seconds."""
    lengths = np.concatenate([runs(g.to_numpy()) for _, g in tel.groupby("driver_number")["is_data_gap"]] or [[]])
    return {"hole_share": float(tel["is_data_gap"].mean()) if len(tel) else np.nan,
            "holes": int(len(lengths)),
            "longest_hole_sec": float(lengths.max()) / HZ if len(lengths) else 0.0,
            "drivers_with_holes": int(tel.groupby("driver_number")["is_data_gap"].any().sum())}


def per_lap(tel: pd.DataFrame) -> pd.DataFrame:
    """One row per driver-lap with telemetry: ticks, hole ticks, distance from speed, path length from x / y (m)."""
    tel = tel.sort_values(["driver_number", "session_time_sec"])
    same_lap = (tel["driver_number"].eq(tel["driver_number"].shift())
                & tel["lap_number"].eq(tel["lap_number"].shift()))
    step = np.hypot(tel["x"].diff(), tel["y"].diff()) / 10.0                  # x, y are in tenths of a metre
    tel = tel.assign(path_m=step.where(same_lap))
    return (tel.groupby(["driver_number", "lap_number"])
            .agg(ticks=("session_time_sec", "size"), hole_ticks=("is_data_gap", "sum"),
                 length_m=("lap_distance_m", "max"), path_m=("path_m", "sum"))
            .reset_index())


def at_lap_ends(tel: pd.DataFrame, ends: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """For each lap end (driver_number, lap_end in session seconds), the values of the tick nearest it (within a tick)."""
    ticks = tel[["driver_number", "session_time_sec", *columns]].dropna().sort_values("session_time_sec")
    return pd.merge_asof(ends.sort_values("lap_end"), ticks, left_on="lap_end", right_on="session_time_sec",
                         by="driver_number", direction="nearest", tolerance=1.0 / HZ)


def line_speeds(tel: pd.DataFrame, bronze_laps: pd.DataFrame) -> pd.Series:
    """Telemetry speed minus F1's timing-loop speed at the line (km/h), one value per lap the loop measured."""
    loop = bronze_laps[["DriverNumber", "LapNumber", "Time", "SpeedFL"]].dropna()
    loop = loop.assign(driver_number=loop["DriverNumber"].astype(str), lap_end=loop["Time"].dt.total_seconds())
    both = at_lap_ends(tel, loop, ["speed_kmh"])
    return (both["speed_kmh"] - both["SpeedFL"]).dropna()


def finish_line(tel: pd.DataFrame, laps: pd.DataFrame) -> Dict[str, float]:
    """Where the cars are (m) at the lap ends of laps that are not pit laps: the median point and the median distance
    of the lap ends from it."""
    ends = laps[~laps["is_pit_lap"].fillna(False).astype(bool)].dropna(subset=["lap_end_sec"])
    ends = ends.assign(lap_end=ends["lap_end_sec"])[["driver_number", "lap_end"]]
    points = at_lap_ends(tel, ends, ["x", "y"]).dropna(subset=["x", "y"])
    if points.empty:
        return {"line_x_m": np.nan, "line_y_m": np.nan, "line_spread_m": np.nan}
    x, y = points["x"].median(), points["y"].median()
    return {"line_x_m": x / 10, "line_y_m": y / 10,
            "line_spread_m": float(np.median(np.hypot(points["x"] - x, points["y"] - y))) / 10}


def not_racing(laps: pd.DataFrame) -> pd.DataFrame:
    """(driver_number, lap_number, out=True) of the laps whose holes are expected: red-flag laps, and the lap each car
    stopped on (its last lap, when that lap has no lap time: a finisher's last lap always has one)."""
    red = laps.loc[laps["is_red"].fillna(False).astype(bool), ["driver_number", "lap_number"]]
    last = laps.sort_values("lap_number").groupby("driver_number").tail(1)
    stopped = last.loc[last["lap_time_sec"].isna(), ["driver_number", "lap_number"]]
    return pd.concat([red, stopped]).drop_duplicates().assign(out=True)


def slowest_point(tel: pd.DataFrame, laps: pd.DataFrame) -> Dict[str, float]:
    """The landmark: where each clean lap without a hole is slowest (median x, y, m) and the median distance of the
    laps' slowest points from it. A place on the track, found without the clock."""
    clean = laps.loc[laps["is_clean_lap"].fillna(False).astype(bool), ["driver_number", "lap_number"]]
    ticks = tel.merge(clean, on=["driver_number", "lap_number"])
    ticks = ticks[~ticks["is_data_gap"]].dropna(subset=["speed_kmh", "x", "y"])
    if ticks.empty:
        return {"slowest_x_m": np.nan, "slowest_y_m": np.nan, "slowest_spread_m": np.nan}
    lows = ticks.loc[ticks.groupby(["driver_number", "lap_number"])["speed_kmh"].idxmin()]
    x, y = lows["x"].median(), lows["y"].median()
    return {"slowest_x_m": x / 10, "slowest_y_m": y / 10,
            "slowest_spread_m": float(np.median(np.hypot(lows["x"] - x, lows["y"] - y))) / 10}


def openf1_clock(year: int, round_number: int, session: str, root: str) -> Dict[str, Any]:
    """The clock report for an OpenF1 race, recomputed from Bronze (telemetry_clock, as Silver did)."""
    from src.pipeline import telemetry_clock

    def bronze(table: str) -> pd.DataFrame:
        return lake.read_table(lake.table_path(lake.BRONZE, table, year, round_number, session, root))
    return telemetry_clock.session_offset(bronze(lake.BRONZE_OPENF1_TABLES[2]), bronze("laps"))[1]


def race_quality(year: int, round_number: int, session: str = "R", root: str = lake.DEFAULT_LAKE_ROOT) -> Dict[str, Any]:
    """One report row for a race whose Silver telemetry exists (the finish line is compared across races later)."""
    manifest = lake.read_manifest(year, round_number, session, root)
    source = lake.bronze_telemetry_source(year, round_number, session, root)
    row: Dict[str, Any] = {"season": int(year), "round": int(round_number), "event_name": manifest.get("event_name"),
                           "location": manifest.get("location"), "source": source}
    if source == lake.OPENF1:
        row["openf1_session_key"] = manifest[lake.OPENF1]["session_key"]
        row.update(openf1_clock(year, round_number, session, root))

    tel = lake.read_table(lake.table_path(lake.SILVER, "telemetry", year, round_number, session, root))
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, session, root))
    laps = laps.assign(driver_number=laps["driver_number"].astype(str))
    raced = set(laps.loc[laps["lap_end_sec"].notna(), "driver_number"])
    have = set(tel["driver_number"])
    row.update(drivers=len(raced), drivers_with_telemetry=len(have & raced),
               drivers_missing=" ".join(sorted(raced - have, key=int)), ticks=int(len(tel)))

    row.update(hole_summary(tel))
    racing = ~tel[["driver_number", "lap_number"]].merge(not_racing(laps), how="left")["out"].notna().to_numpy()
    row["racing_hole_share"] = float(tel["is_data_gap"].to_numpy()[racing].mean()) if racing.any() else np.nan

    lap_rows = per_lap(tel).merge(laps[["driver_number", "lap_number", "is_clean_lap"]],
                                  on=["driver_number", "lap_number"], how="left")
    clean = laps["is_clean_lap"].fillna(False).astype(bool)
    full = lap_rows[lap_rows["is_clean_lap"].fillna(False).astype(bool) & (lap_rows["hole_ticks"] == 0)]
    median_m = float(full["length_m"].median()) if len(full) else np.nan
    off = (full["length_m"] / median_m - 1).abs() > LENGTH_TOLERANCE
    row.update(clean_laps=int(clean.sum()), full_laps=int(len(full)),
               full_lap_share=len(full) / clean.sum() if clean.sum() else np.nan,
               lap_length_m=median_m, lap_length_p1_m=float(full["length_m"].quantile(0.01)) if len(full) else np.nan,
               lap_length_p99_m=float(full["length_m"].quantile(0.99)) if len(full) else np.nan,
               off_length_share=float(off.mean()) if len(full) else np.nan,
               speed_to_position=float((full["length_m"] / full["path_m"]).median()) if len(full) else np.nan)

    bronze_laps = lake.read_table(lake.table_path(lake.BRONZE, "laps", year, round_number, session, root),
                                  columns=["DriverNumber", "LapNumber", "Time", "SpeedFL"])
    diff = line_speeds(tel, bronze_laps)
    row.update(line_speed_laps=int(len(diff)),
               line_speed_offset_kmh=float(diff.median()) if len(diff) else np.nan,
               line_speed_iqr_kmh=float(diff.quantile(0.75) - diff.quantile(0.25)) if len(diff) else np.nan)
    row.update(finish_line(tel, laps))
    row.update(slowest_point(tel, laps))

    row.update(brake_unknown_share=float(tel["brake"].isna().mean()),
               drs_open_share=float(tel["drs_open"].fillna(False).astype(bool).mean()))
    row["problems"] = "; ".join(problems(row))
    return row


def problems(row: Dict[str, Any]) -> List[str]:
    """Every check of one race the race fails, in words (empty = all passed). The finish line is judged across races."""
    out = []
    if row["drivers_missing"]:
        out.append(f"no telemetry for car {row['drivers_missing']}")
    if not row["racing_hole_share"] <= MAX_HOLE_SHARE:
        out.append(f"holes {100 * row['racing_hole_share']:.2f} % of racing ticks (longest {row['longest_hole_sec']:.0f} s)")
    if not row["full_lap_share"] >= MIN_FULL_LAP_SHARE:
        out.append(f"only {100 * row['full_lap_share']:.1f} % of clean laps complete")
    if not row["off_length_share"] <= MAX_OFF_LENGTH_SHARE:
        out.append(f"{100 * row['off_length_share']:.1f} % of full laps off the median length by > 2 %")
    low, high = SPEED_POSITION_BAND
    if not low <= row["speed_to_position"] <= high:
        out.append(f"speed and position disagree ({row['speed_to_position']:.3f})")
    if not row["line_speed_iqr_kmh"] <= MAX_LINE_SPEED_IQR_KMH:
        out.append(f"line speed unsteady (IQR {row['line_speed_iqr_kmh']:.1f} km/h)")
    return out


def line_offsets(report: pd.DataFrame) -> pd.Series:
    """Each race's finish line, seen from its slowest point, against the median of every race at its circuit (m);
    NaN alone at a circuit."""
    rel = pd.DataFrame({"location": report["location"], "dx": report["line_x_m"] - report["slowest_x_m"],
                        "dy": report["line_y_m"] - report["slowest_y_m"]})
    med = rel.groupby("location")[["dx", "dy"]].transform("median")
    count = rel.groupby("location")["dx"].transform("count")
    return np.hypot(rel["dx"] - med["dx"], rel["dy"] - med["dy"]).where(count >= 2)


def season_report(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT, log=print) -> pd.DataFrame:
    """Every race of these seasons with Silver telemetry; a race that cannot be measured is reported, not skipped."""
    rows = []
    for year, rnd in lake.list_races(lake.SILVER, "telemetry", root=root):
        if year not in set(seasons):
            continue
        try:
            rows.append(race_quality(year, rnd, root=root))
        except Exception as err:
            rows.append({"season": year, "round": rnd, "problems": f"not measured: {type(err).__name__}: {err}"})
        log(f"{year} round {rnd}: {rows[-1].get('problems') or 'ok'}")
    report = pd.DataFrame(rows)
    if "line_x_m" in report:
        report["line_offset_m"] = line_offsets(report)
        far = report["line_offset_m"] > MAX_LINE_OFFSET_M
        report.loc[far, "problems"] = [
            "; ".join(filter(None, [p, f"finish line {d:.0f} m from the circuit's other races"]))
            for p, d in zip(report.loc[far, "problems"], report.loc[far, "line_offset_m"])]
    return report


def describe(report: pd.DataFrame) -> str:
    """The season summary in a few lines."""
    ok = report["problems"].fillna("") == ""
    lines = [f"{len(report)} races, {int(ok.sum())} pass every check, {int((~ok).sum())} do not."]
    if "racing_hole_share" in report:
        def pct(col, q):
            return 100 * report[col].quantile(q)
        lines += [f"Clock: {report.get('clock_method', pd.Series(dtype=str)).value_counts().to_dict()}.",
                  f"Holes outside red flags: median {pct('racing_hole_share', 0.5):.3f} % of ticks, max "
                  f"{pct('racing_hole_share', 1):.3f} %; with red flags max {pct('hole_share', 1):.2f} %.",
                  f"Clean laps complete: median {pct('full_lap_share', 0.5):.1f} %, min {pct('full_lap_share', 0):.1f} %.",
                  f"Speed / position length: median {report['speed_to_position'].median():.4f} "
                  f"({report['speed_to_position'].min():.4f}-{report['speed_to_position'].max():.4f}).",
                  f"Line speed: offset {report['line_speed_offset_kmh'].min():.1f} to "
                  f"{report['line_speed_offset_kmh'].max():.1f} km/h, steadiness (IQR) median "
                  f"{report['line_speed_iqr_kmh'].median():.1f}, max {report['line_speed_iqr_kmh'].max():.1f}.",
                  ("Finish line from the circuit's other races: " + (
                      f"median {report['line_offset_m'].median():.1f} m, max {report['line_offset_m'].max():.1f} m."
                      if report["line_offset_m"].notna().any() else "no circuit has two races to compare.")),
                  f"Brake 'no reading': median {pct('brake_unknown_share', 0.5):.3f} % of ticks."]
    lines += [f"  {r.season} round {r['round']} {r.get('event_name', '')}: {r.problems}"
              for _, r in report[~ok].iterrows()]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="How complete and trustworthy each race's telemetry is")
    parser.add_argument("--season", type=int, nargs="+", required=True)
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    args = parser.parse_args(argv)
    report = season_report(args.season, args.lake)
    if report.empty:
        print("No race of these seasons has Silver telemetry yet.")
        return 1
    print(describe(report))
    print(f"Report: {lake.write_report(report, REPORT, args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
