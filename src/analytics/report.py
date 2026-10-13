"""
Race report (src/analytics/report.py) - Phase 2 step 6

Everything Phase 2 measures about one race, in one place. Each number comes with how many laps or
stops it rests on (n), how uncertain it is, and a note on what was left out:

    step 1  pace laps and neutralised periods    src/analytics/laps.py, neutralisations.py
    step 2  pit stops and pit loss               src/analytics/stops.py
    step 3  fuel-corrected tyre wear             src/analytics/degradation.py
    step 4  the cost of following (dirty air)    src/analytics/dirty_air.py
    step 5  undercuts                            src/analytics/undercuts.py

Then every number is checked against a sanity range (RANGES), written down before the 2023-2025
races were measured: what is physically plausible, not what is typical. A number outside its range
must be explained (EXPLAINED, each with its evidence), or it is a problem to find. Two exceptions:
  - tyre wear, the fuel check and the cost of following are not checked in mixed conditions (someone
    ran inters or wets, so the track kept changing; steps 3 and 4 already set these races aside);
  - a safety-car or VSC stop loss outside its range is low confidence, not a mystery: step 2 compares
    laps by their number, and a safety car that starts or ends in the middle of a lap makes those
    laps not comparable (measured on Spain 2025 and Sao Paulo 2024).

Run it over every race in the lake (exit code 1 if any number is outside its range unexplained):

    python -m src.analytics.report --season 2023 2024 2025
    python -m src.analytics.report --season 2024 --round 13      # one race, printed in full
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analytics.degradation import RaceDegradation, race_degradation
from src.analytics.dirty_air import RaceDirtyAir, race_dirty_air
from src.analytics.laps import RacePace, race_pace
from src.analytics.neutralisations import describe_periods
from src.analytics.stops import GREEN, RED_FLAG, SAFETY_CAR, VSC, RaceStops, race_stops
from src.analytics.undercuts import RaceUndercuts, race_undercuts
from src.pipeline import lake

# Written down before measuring (2026-10-13): physically plausible, wide enough for real extremes.
RANGES = {
    "pace laps, % of laps": (40.0, 100.0),
    "outliers, % of clean laps": (0.0, 5.0),
    "green pit loss, s": (15.0, 35.0),
    "SC/VSC stop loss, s": (0.0, 30.0),
    "tyre wear, s per lap": (-0.05, 0.30),
    "fuel check, s per kg": (0.0, 0.08),
    "cost of following, s per lap": (-0.3, 2.0),
    "undercut gain, s": (-10.0, 12.0),
}
MIN_WEAR_STINTS = 3     # as step 3's season figures: fewer stints give too noisy a slope to check
NOT_IN_MIXED = ("tyre wear, s per lap", "fuel check, s per kg", "cost of following, s per lap")
LOW_CONFIDENCE = {
    "SC/VSC stop loss, s": "step 2 compares laps by number, so a safety car or VSC that starts or ends mid-lap "
                           "(or a slow lap before the stop) distorts the reference",
}

# Every number found outside its range in 2023-2025, with its evidence. Key: (season, round, check, item).
EXPLAINED = {
    (2024, 23, "tyre wear, s per lap", "HARD"):
        "Qatar: laps got faster as the track gained rubber (step 3)",
    (2023, 21, "undercut gain, s", "SAR on HAM, laps 15/17"):
        "HAM collided with PIA (Race Control); HAM's in-lap was about 17 s slow",
    (2024, 3, "undercut gain, s", "MAG on BOT, laps 7/8"):
        "BOT spent 46.9 s in the pit lane, MAG 18.2 s: a slow stop",
    (2024, 11, "undercut gain, s", "ALO on RIC, laps 35/37"):
        "ALO served a 10 s penalty at this stop (Race Control, lap 23)",
    (2024, 18, "undercut gain, s", "RIC on MAG, laps 46/49"):
        "MAG's in-lap took 2:50.660 (Race Control): a forced stop; MAG retired on lap 57",
}

OK = "ok"
EXPLAINED_OUTSIDE = "outside, explained"
LOW = "outside, low confidence"
UNEXPLAINED = "OUTSIDE, UNEXPLAINED"
NOT_CHECKED = "not checked (mixed conditions)"
NOT_MEASURED = "not measured"


@dataclass
class RaceReport:
    """Steps 1-5 for one race."""
    season: int
    round_number: int
    event_name: Optional[str]
    pace: RacePace
    stops: RaceStops
    degradation: RaceDegradation
    dirty_air: RaceDirtyAir
    undercuts: RaceUndercuts


def build_report(silver_laps: pd.DataFrame, event_name: Optional[str] = None) -> RaceReport:
    """Run steps 1-5 on one race's Silver laps. Pure: reads and writes nothing."""
    pace = race_pace(silver_laps)
    laps = pace.laps                           # every lap, with step 1's pace_status
    return RaceReport(season=int(silver_laps["season"].iloc[0]), round_number=int(silver_laps["round"].iloc[0]),
                      event_name=event_name, pace=pace, stops=race_stops(laps), degradation=race_degradation(laps),
                      dirty_air=race_dirty_air(laps), undercuts=race_undercuts(laps))


# --- the numbers --------------------------------------------------------------------------------

def _number(rows: list, step: int, quantity: str, value, n=None, uncertainty: str = "", note: str = ""):
    rows.append({"step": step, "quantity": quantity, "value": value, "n": n, "uncertainty": uncertainty,
                 "note": note})


def headline_numbers(report: RaceReport) -> pd.DataFrame:
    """One row per number the report shows: step, quantity, value, n, uncertainty, note."""
    rows: List[dict] = []
    s = report.pace.summary
    _number(rows, 1, "pace laps, % of laps", s["pace_pct"], s["laps"],
            note=f"{s['pace_laps']} pace laps; {s['outliers']} outliers ({s['outlier_pct_of_clean']} % of clean laps)")
    _number(rows, 1, "neutralised periods", describe_periods(report.pace.periods), len(report.pace.periods))

    stops = report.stops.stops
    for condition, quantity in ((GREEN, "green pit loss, s"), (SAFETY_CAR, "SC stop loss, s (low confidence)"),
                                (VSC, "VSC stop loss, s (low confidence)")):
        loss = stops.loc[stops["condition"] == condition, "loss_sec"].dropna()
        if len(loss):
            _number(rows, 2, quantity, loss.median(), len(loss),
                    f"IQR {loss.quantile(0.25):.1f} to {loss.quantile(0.75):.1f}")
    _number(rows, 2, "tyre changes", len(stops), note=f"{int((stops['condition'] == RED_FLAG).sum())} under a red "
            f"flag; {int(stops['loss_sec'].isna().sum())} without a measured loss")

    deg = report.degradation
    mixed = "; mixed conditions" if deg.mixed else ""
    for c in deg.compounds.dropna(subset=["slope"]).to_dict("records"):
        _number(rows, 3, f"tyre wear {c['compound'].lower()}, s per lap", c["slope"], c["stints"],
                f"+- {c['se']:.3f} (SE)", f"{c['laps']} laps; raw slope {c['raw_slope']:+.3f}{mixed}")
    if not np.isnan(deg.fuel["fuel_sec_per_kg"]):
        _number(rows, 3, "fuel check, s per kg", deg.fuel["fuel_sec_per_kg"], deg.fuel["drivers"],
                f"+- {deg.fuel['fuel_se']:.4f} (SE)", f"assumed 0.03{mixed}")

    f = report.dirty_air.following
    if not np.isnan(f["beta"]):
        _number(rows, 4, "cost of following, s per lap", f["beta"], f["following_laps"], f"+- {f['se']:.3f} (SE)",
                f"{f['laps']} laps; {f['stints_both']} stints with both{mixed}")

    u = report.undercuts.summary
    if u["attempts"]:
        _number(rows, 5, "undercuts that worked, %", 100 * u["share"], u["attempts"],
                f"95 % CI {100 * u['share_lo']:.0f} to {100 * u['share_hi']:.0f}",
                f"median gain {u['gain_median']:+.2f} s")
    else:
        _number(rows, 5, "undercuts that worked, %", np.nan, 0, note="no undercut attempt")
    return pd.DataFrame(rows, columns=["step", "quantity", "value", "n", "uncertainty", "note"])


# --- the sanity checks --------------------------------------------------------------------------

def judge(season: int, round_number: int, check: str, item: str, value, mixed: bool):
    """(status, explanation) of one number against its sanity range."""
    if mixed and check in NOT_IN_MIXED:
        return NOT_CHECKED, "someone ran inters or wets, so the track kept changing"
    if value is None or pd.isna(value):
        return NOT_MEASURED, ""
    low, high = RANGES[check]
    if low <= value <= high:
        return OK, ""
    if (season, round_number, check, item) in EXPLAINED:
        return EXPLAINED_OUTSIDE, EXPLAINED[(season, round_number, check, item)]
    if check in LOW_CONFIDENCE:
        return LOW, LOW_CONFIDENCE[check]
    return UNEXPLAINED, "outside its range: find out why"


def sanity_checks(report: RaceReport) -> pd.DataFrame:
    """One row per number checked: check, item, value, low, high, status, explanation."""
    values = []                                # (check, item, value)
    s = report.pace.summary
    values.append(("pace laps, % of laps", "", s["pace_pct"]))
    values.append(("outliers, % of clean laps", "", s["outlier_pct_of_clean"]))
    stops = report.stops.stops
    green = stops.loc[stops["condition"] == GREEN, "loss_sec"].dropna()
    values.append(("green pit loss, s", "", green.median() if len(green) else np.nan))
    for condition, item in ((SAFETY_CAR, "SC"), (VSC, "VSC")):
        loss = stops.loc[stops["condition"] == condition, "loss_sec"].dropna()
        if len(loss):
            values.append(("SC/VSC stop loss, s", item, loss.median()))
    for c in report.degradation.compounds.to_dict("records"):
        if c["stints"] >= MIN_WEAR_STINTS:
            values.append(("tyre wear, s per lap", c["compound"], c["slope"]))
    values.append(("fuel check, s per kg", "", report.degradation.fuel["fuel_sec_per_kg"]))
    values.append(("cost of following, s per lap", "", report.dirty_air.following["beta"]))
    for a in report.undercuts.attempts.to_dict("records"):
        item = f"{a['driver_code']} on {a['car_ahead']}, laps {a['in_lap']}/{a['car_ahead_in_lap']}"
        values.append(("undercut gain, s", item, a["gain_sec"]))

    rows = []
    for check, item, value in values:
        status, why = judge(report.season, report.round_number, check, item, value, report.degradation.mixed)
        low, high = RANGES[check]
        rows.append({"check": check, "item": item, "value": value, "low": low, "high": high, "status": status,
                     "explanation": why})
    return pd.DataFrame(rows, columns=["check", "item", "value", "low", "high", "status", "explanation"])


# --- one race, every race -----------------------------------------------------------------------

def load_race_report(year: int, round_number: int, session: str = "R",
                     root: str = lake.DEFAULT_LAKE_ROOT) -> RaceReport:
    path = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    if not path.exists():
        raise FileNotFoundError(f"{year} round {round_number}: no Silver laps in the lake ({path}).")
    try:
        event = lake.read_manifest(year, round_number, session, root).get("event_name")
    except FileNotFoundError:
        event = None
    return build_report(lake.read_table(path), event)


def season_reports(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT) -> List[RaceReport]:
    """The report of every race of these seasons in the lake, in race order."""
    wanted = {int(s) for s in seasons}
    reports = []
    for path in lake.list_sessions(lake.SILVER, "laps", root):
        parts = dict(p.split("=") for p in path.parts if "=" in p)     # "season=2024" -> {"season": "2024"}
        if parts.get("session") == "R" and int(parts["season"]) in wanted:
            reports.append(load_race_report(int(parts["season"]), int(parts["round"]), "R", root))
    return sorted(reports, key=lambda r: (r.season, r.round_number))


def print_report(report: RaceReport):
    print(f"\n{report.event_name or 'Race'} {report.season} (round {report.round_number})")
    for r in headline_numbers(report).to_dict("records"):
        value = r["value"] if isinstance(r["value"], str) else f"{r['value']:.3f}"
        print(f"  step {r['step']}  {r['quantity']:36s} {value:>10s}  n={r['n']}  {r['uncertainty']}  {r['note']}")
    for r in sanity_checks(report).to_dict("records"):
        print(f"  check  {r['check']:30s} {r['item']:24s} {r['status']}  {r['explanation']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 step 6: the race report and its sanity checks")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2023 2024")
    parser.add_argument("--round", type=int, help="only this round, printed in full")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    reports = season_reports(args.season, args.lake)
    if args.round is not None:
        reports = [r for r in reports if r.round_number == args.round]
        for r in reports:
            print_report(r)
    if not reports:
        print("No Silver laps in the lake for these races.")
        return 1
    checks = pd.concat([sanity_checks(r).assign(season=r.season, round=r.round_number, event_name=r.event_name)
                        for r in reports], ignore_index=True)
    print(f"\n{len(checks)} numbers checked in {len(reports)} races:")
    for status, count in checks["status"].value_counts().items():
        print(f"  {status:32s} {count:5d}")
    outside = checks[checks["status"].isin([EXPLAINED_OUTSIDE, LOW, UNEXPLAINED])]
    for r in outside.to_dict("records"):
        print(f"  {r['season']} {r['event_name']}: {r['check']} {r['item']} = {r['value']:.3f} "
              f"(range {r['low']:g} to {r['high']:g}) -> {r['status']}: {r['explanation']}")
    if args.round is None:
        numbers = pd.concat([headline_numbers(r).assign(season=r.season, round=r.round_number,
                                                        event_name=r.event_name) for r in reports], ignore_index=True)
        print(f"Reports: {lake.write_report(numbers, 'race_report', args.lake)}, "
              f"{lake.write_report(checks, 'sanity', args.lake)}")
    return 1 if (checks["status"] == UNEXPLAINED).any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
