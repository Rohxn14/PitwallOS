"""
Undercuts, measured (src/analytics/undercuts.py) — Phase 2 step 5

Rohan's definition (2026-10-13): an undercut is when a trailing driver pits early to gain time with
fresher tyres; when the car ahead next pits, the trailing driver hopes to come out ahead.

Here, car B makes a tyre change and car A is the car one position ahead of B in the race at the end
of the lap before B's in-lap. It is an undercut ATTEMPT when
  - B's stop is green (in-lap and out-lap, step 2's condition) and on dry tyres,
  - B is less than MAX_GAP_SEC behind A (closer than 3 s, a car is held up by the one ahead: step 4),
  - A responds: its next tyre change is 1 to MAX_RESPONSE_LAPS laps after B's. Later than that, A is
    on another plan (an offset strategy), not responding; in 2023-2025 the laps between the two stops
    spike at 1 and 2 and are a flat tail from 3 on,
  - A's stop is green and on dry tyres too, and no safety car, VSC or red flag touches either car on
    any lap of the attempt (B's in-lap to A's out-lap).
It WORKED if B is ahead of A at the line at the end of A's out-lap, the first time both cars are back
on track on their new tyres (measured before building: 88 % of the attempts that worked are still
ahead 3 laps later, and only 5 % of the failures get ahead anyway).

    gap before  = B's line crossing - A's, at the end of the lap before B's in-lap   (> 0: B behind)
    gap after   = the same at the end of A's out-lap                              (< 0: B ahead)
    gain        = gap before - gap after                                          (> 0: B gained)

The gain splits exactly into three parts (they add up to it):
    stop          A's pit-lane time - B's (pit entry to pit exit: a quicker stop is a gain)
    laps between  what B gained on the full racing laps between its out-lap and A's in-lap, on new
                  tyres against old ones. There are such laps only when A waits 3 laps; otherwise 0.
    track around the stops   the rest: the on-track part of both cars' in-laps and out-laps, where the
                  classic undercut lives (a quick out-lap on new tyres against an in-lap on old ones).
Lap by lap, an in-lap or out-lap mostly says where the pit box sits relative to the timing line (an
out-lap "costs" ~14 s), so attempt_laps shows one attempt lap by lap for reading, not for the split.

Every tyre change gets one status: ATTEMPT, or the first reason in LEDGER_ORDER that it is not one
(first what B's stop was, then what A did). Only the car directly ahead counts: an undercut on a
car two places up is not measured. Team-mates are kept and flagged.

Run it over every race in the lake:

    python -m src.analytics.undercuts --season 2023 2024 2025
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.degradation import WET_COMPOUNDS
from src.analytics.dirty_air import CLEAR_AIR_SEC
from src.analytics.laps import classify_laps
from src.analytics.stops import GREEN, RED_FLAG, stops
from src.pipeline import lake

MAX_GAP_SEC = CLEAR_AIR_SEC      # 3 s: closer than this, B is held up by A (step 4), so stopping first is the way past
MAX_RESPONSE_LAPS = 3            # A's next stop at most this many laps after B's, or it is another plan
GAP_BANDS = (0.0, 1.0, 2.0, MAX_GAP_SEC)

ATTEMPT = "attempt"
RED_FLAG_CHANGE = "red-flag tyre change (free)"
NOT_GREEN = "B's stop not green (SC, VSC, yellow, lap 1)"
WET = "wet tyres (inters or wets)"
LEADING = "B was leading"
NO_GAP = "no gap to the car ahead"
TOO_FAR = f"car ahead {MAX_GAP_SEC:g} s or more away"
A_FIRST = f"car ahead had stopped first (up to {MAX_RESPONSE_LAPS} laps before)"
SAME_LAP = "car ahead stopped on the same lap"
A_NEVER = "car ahead did not stop again"
A_LATE = f"car ahead stopped {MAX_RESPONSE_LAPS + 1}+ laps later (another plan)"
A_NOT_GREEN = "car ahead's stop not green"
NEUTRALISED = "safety car, VSC or red flag during the attempt"
NO_GAP_AFTER = "no gap after (a lap is missing)"
LEDGER_ORDER = (RED_FLAG_CHANGE, NOT_GREEN, WET, LEADING, NO_GAP, TOO_FAR, A_FIRST, SAME_LAP, A_NEVER, A_LATE,
                A_NOT_GREEN, NEUTRALISED, NO_GAP_AFTER)

EVENT_COLUMNS = [
    "driver_code", "team", "in_lap", "condition", "compound_before", "compound_after", "tyre_age_before",
    "position_before", "car_ahead", "teammates", "gap_before_sec", "car_ahead_in_lap", "lag",
    "car_ahead_condition", "car_ahead_compound_before", "car_ahead_compound_after", "car_ahead_tyre_age_before",
    "gap_after_sec", "gain_sec", "stop_gain_sec", "between_gain_sec", "track_gain_sec", "worked", "status",
]


def _wet(*compounds) -> bool:
    return any(isinstance(c, str) and c in WET_COMPOUNDS for c in compounds)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % interval of a share k / n (Wilson score: stays inside 0-1 and behaves for small n)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - half, centre + half


# --- every tyre change: an attempt, or why not ---------------------------------------------------

def undercut_events(classified: pd.DataFrame, stop_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    One row per tyre change (step 2's stops), in EVENT_COLUMNS: the car ahead, the gaps, the car ahead's
    response, the gain and its split (attempts only), worked, and status (ATTEMPT or the first reason
    in LEDGER_ORDER). `classified` = laps.classify_laps(silver); stop_table defaults to stops(classified).
    """
    st = stops(classified) if stop_table is None else stop_table
    by = classified.set_index(["driver_code", "lap_number"])
    if not by.index.is_unique:
        raise ValueError("Silver laps have more than one row for a driver and lap; cannot pair cars")
    end, pos = by["lap_end_sec"], by["position"]
    pit_in, pit_out, team = by["pit_in_sec"], by["pit_out_sec"], by["team"]
    neutral = by["is_sc"].astype(bool) | by["is_vsc"].astype(bool) | by["is_red"].astype(bool)
    at_position = {}                              # (lap, position) -> driver, only where exactly one car has it
    placed = classified.dropna(subset=["position"])
    for (lap, p), group in placed.groupby(["lap_number", "position"]):
        if len(group) == 1:
            at_position[(int(lap), int(p))] = str(group["driver_code"].iloc[0])
    strategic = st[st["condition"] != RED_FLAG]
    stops_of = {str(d): g.sort_values("in_lap") for d, g in strategic.groupby("driver_code")}

    def gap(lap: int, b: str, a: str) -> float:
        return float(end.get((b, lap), np.nan)) - float(end.get((a, lap), np.nan))

    rows: List[Dict[str, Any]] = []
    for s in st.to_dict("records"):
        b, n = str(s["driver_code"]), int(s["in_lap"])
        row = {c: np.nan for c in EVENT_COLUMNS}
        row.update(driver_code=b, team=s["team"], in_lap=n, condition=s["condition"],
                   compound_before=s["compound_before"], compound_after=s["compound_after"],
                   tyre_age_before=s["tyre_age_before"], car_ahead=None, teammates=pd.NA, worked=pd.NA,
                   car_ahead_condition=None, car_ahead_compound_before=None, car_ahead_compound_after=None)
        rows.append(row)
        if s["condition"] == RED_FLAG:
            row["status"] = RED_FLAG_CHANGE
            continue
        if s["condition"] != GREEN:
            row["status"] = NOT_GREEN
            continue
        if _wet(s["compound_before"], s["compound_after"]):
            row["status"] = WET
            continue
        p = pos.get((b, n - 1), pd.NA)
        if pd.isna(p):
            row["status"] = NO_GAP
            continue
        row["position_before"] = int(p)
        if int(p) == 1:
            row["status"] = LEADING
            continue
        a = at_position.get((n - 1, int(p) - 1))
        g0 = gap(n - 1, b, a) if a is not None else np.nan
        if a is None or np.isnan(g0) or g0 < 0:   # unknown, or timing and positions disagree: unknown, not 0
            row["status"] = NO_GAP
            continue
        team_a = team.get((a, n - 1))
        same_team = pd.NA if pd.isna(team_a) or pd.isna(s["team"]) else str(team_a) == str(s["team"])
        row.update(car_ahead=a, gap_before_sec=g0, teammates=same_team)
        if g0 >= MAX_GAP_SEC:
            row["status"] = TOO_FAR
            continue
        a_stops = stops_of.get(a, strategic.iloc[0:0])
        if a_stops["in_lap"].between(n - MAX_RESPONSE_LAPS, n - 1).any():
            row["status"] = A_FIRST
            continue
        if (a_stops["in_lap"] == n).any():
            row["status"] = SAME_LAP
            continue
        later = a_stops[a_stops["in_lap"] > n]
        if later.empty:
            row["status"] = A_NEVER
            continue
        r = later.iloc[0]
        m = int(r["in_lap"])
        row.update(car_ahead_in_lap=m, lag=m - n, car_ahead_condition=r["condition"],
                   car_ahead_compound_before=r["compound_before"], car_ahead_compound_after=r["compound_after"],
                   car_ahead_tyre_age_before=r["tyre_age_before"])
        if m - n > MAX_RESPONSE_LAPS:
            row["status"] = A_LATE
            continue
        if r["condition"] != GREEN:
            row["status"] = A_NOT_GREEN
            continue
        if _wet(r["compound_before"], r["compound_after"]):
            row["status"] = WET
            continue
        if any(bool(neutral.get((d, k), False)) for d in (a, b) for k in range(n, m + 2)):
            row["status"] = NEUTRALISED
            continue
        g1 = gap(m + 1, b, a)
        if np.isnan(g1):
            row["status"] = NO_GAP_AFTER
            continue
        gain = g0 - g1
        stop_gain = ((float(pit_out.get((a, m + 1), np.nan)) - float(pit_in.get((a, m), np.nan)))
                     - (float(pit_out.get((b, n + 1), np.nan)) - float(pit_in.get((b, n), np.nan))))
        between = gap(n + 1, b, a) - gap(m - 1, b, a) if m - n >= 3 else 0.0
        row.update(gap_after_sec=g1, gain_sec=gain, stop_gain_sec=stop_gain, between_gain_sec=between,
                   track_gain_sec=gain - stop_gain - between, worked=bool(g1 < 0), status=ATTEMPT)
    out = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for col in ("tyre_age_before", "position_before", "car_ahead_in_lap", "lag", "car_ahead_tyre_age_before"):
        out[col] = pd.array(out[col], dtype="Int64")
    for col in ("teammates", "worked"):
        out[col] = out[col].astype("boolean")
    return out.sort_values(["in_lap", "driver_code"]).reset_index(drop=True)


def undercut_ledger(events: pd.DataFrame) -> pd.DataFrame:
    """Tyre changes -> undercut attempts, one row per reason (listed even at 0)."""
    counts = events["status"].value_counts()
    rows = [{"reason": "tyre changes", "count": int(len(events))}]
    rows += [{"reason": r, "count": int(counts.get(r, 0))} for r in LEDGER_ORDER]
    rows.append({"reason": "undercut attempts", "count": int(counts.get(ATTEMPT, 0))})
    return pd.DataFrame(rows, columns=["reason", "count"])


def attempt_summary(attempts: pd.DataFrame) -> Dict[str, Any]:
    """n, how many worked, the share with its 95 % interval, the median gain (IQR) and the median split."""
    n = int(len(attempts))
    worked = int(attempts["worked"].astype(bool).sum()) if n else 0
    lo, hi = wilson(worked, n)
    gain = attempts["gain_sec"]
    q = lambda s, x: float(s.quantile(x)) if s.notna().any() else np.nan
    return {"attempts": n, "worked": worked, "share": worked / n if n else np.nan, "share_lo": lo, "share_hi": hi,
            "gain_median": q(gain, 0.5), "gain_q25": q(gain, 0.25), "gain_q75": q(gain, 0.75),
            "stop_median": q(attempts["stop_gain_sec"], 0.5), "track_median": q(attempts["track_gain_sec"], 0.5),
            "between_median": q(attempts["between_gain_sec"], 0.5)}


def attempt_laps(classified: pd.DataFrame, attempt: Dict[str, Any]) -> pd.DataFrame:
    """
    One attempt lap by lap, from the lap before B's in-lap to A's out-lap: each car's lap time and what
    the lap was (old tyres, in-lap, out-lap, new tyres), the gap at the end of the lap (B's crossing -
    A's) and what B gained on that lap.
    """
    b, a = attempt["driver_code"], attempt["car_ahead"]
    n, m = int(attempt["in_lap"]), int(attempt["car_ahead_in_lap"])
    by = classified.set_index(["driver_code", "lap_number"])

    def kind(driver: str, lap: int, own_in_lap: int) -> str:
        if lap == own_in_lap:
            return "in-lap"
        if lap == own_in_lap + 1:
            return "out-lap"
        return "old tyres" if lap < own_in_lap else "new tyres"

    rows = []
    for k in range(n - 1, m + 2):
        tb, ta = by["lap_end_sec"].get((b, k), np.nan), by["lap_end_sec"].get((a, k), np.nan)
        rows.append({"lap": k, "b_lap_sec": by["lap_time_sec"].get((b, k), np.nan), "b_kind": kind(b, k, n),
                     "a_lap_sec": by["lap_time_sec"].get((a, k), np.nan), "a_kind": kind(a, k, m),
                     "gap_sec": float(tb) - float(ta)})
    out = pd.DataFrame(rows)
    out["b_gained_sec"] = -out["gap_sec"].diff()
    return out


# --- one race, every race -----------------------------------------------------------------------

@dataclass
class RaceUndercuts:
    events: pd.DataFrame        # undercut_events: every tyre change with its status
    attempts: pd.DataFrame      # the events that are attempts
    ledger: pd.DataFrame        # undercut_ledger
    summary: Dict[str, Any]     # attempt_summary of this race's attempts


def race_undercuts(classified: pd.DataFrame) -> RaceUndercuts:
    """Everything step 5 says about one race, from its classified laps (laps.classify_laps). Pure."""
    events = undercut_events(classified)
    attempts = events[events["status"] == ATTEMPT].reset_index(drop=True)
    return RaceUndercuts(events=events, attempts=attempts, ledger=undercut_ledger(events),
                         summary=attempt_summary(attempts))


def load_race_undercuts(year: int, round_number: int, session: str = "R",
                        root: str = lake.DEFAULT_LAKE_ROOT) -> RaceUndercuts:
    path = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    if not path.exists():
        raise FileNotFoundError(f"{year} round {round_number}: no Silver laps in the lake ({path}).")
    return race_undercuts(classify_laps(lake.read_table(path)))


def season_table(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """Every tyre change of these seasons' races in the lake, with season, round and event name in front."""
    wanted = {int(s) for s in seasons}
    frames = []
    for path in lake.list_sessions(lake.SILVER, "laps", root):
        if "session=R" not in path.parts:
            continue
        laps = lake.read_table(path)
        if laps.empty or int(laps["season"].iloc[0]) not in wanted:
            continue
        season, round_number = int(laps["season"].iloc[0]), int(laps["round"].iloc[0])
        try:
            event = lake.read_manifest(season, round_number, "R", root).get("event_name")
        except FileNotFoundError:
            event = None
        events = undercut_events(classify_laps(laps))
        events.insert(0, "event_name", event)
        events.insert(0, "round", round_number)
        events.insert(0, "season", season)
        frames.append(events)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["season", "round", "in_lap"]).reset_index(drop=True)


def _share(group: pd.DataFrame) -> str:
    s = attempt_summary(group)
    if not s["attempts"]:
        return "no attempts"
    return (f"{100 * s['share']:.0f} % of {s['attempts']} (95 % CI {100 * s['share_lo']:.0f}-{100 * s['share_hi']:.0f} %), "
            f"median gain {s['gain_median']:+.2f} s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 step 5 over every race in the lake: undercuts, measured")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2023 2024")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    table = season_table(args.season, args.lake)
    if table.empty:
        print("No Silver laps in the lake for these seasons.")
        return 1
    races = table[["season", "round"]].drop_duplicates().shape[0]
    print(f"{len(table):,} tyre changes in {races} races. Undercut status (first reason that applies):")
    for r in undercut_ledger(table).to_dict("records")[1:]:
        print(f"  {r['reason']:52s} {r['count']:6,}")
    attempts = table[table["status"] == ATTEMPT]
    s = attempt_summary(attempts)
    if s["attempts"]:
        print(f"\nUndercuts that worked (B ahead of A at the end of A's out-lap): {s['worked']} of {s['attempts']} = "
              f"{100 * s['share']:.0f} % (95 % CI {100 * s['share_lo']:.0f}-{100 * s['share_hi']:.0f} %)")
        bands = pd.cut(attempts["gap_before_sec"], list(GAP_BANDS), right=False)
        print("  by gap before:")
        for band, group in attempts.groupby(bands, observed=True):
            print(f"    {band.left:g}-{band.right:g} s   {_share(group)}")
        print("  by when the car ahead stopped:")
        for lag, group in attempts.groupby("lag"):
            print(f"    {int(lag)} lap{'s' if lag > 1 else ''} later   {_share(group)}")
        print(f"  gain: median {s['gain_median']:+.2f} s (IQR {s['gain_q25']:+.2f} to {s['gain_q75']:+.2f}); medians of "
              f"its parts: stop {s['stop_median']:+.2f}, track around the stops {s['track_median']:+.2f}, laps between "
              f"{s['between_median']:+.2f} s")
        print(f"  team-mates: {int(attempts['teammates'].astype(bool).sum())} of the attempts")
    print(f"Report: {lake.write_report(table, 'undercuts', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
