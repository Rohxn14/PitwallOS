"""
Running order for the race replay (src/lib/running_order.py)

Who is P1..PN at every replay frame, and whose race is already over.

The order comes from official timing, never from where the cars happen to be drawn:

1. Running cars: the official timing-tower position (FastF1 timing stream, updated at the
   timing loops, about 25 times a lap at Baku), held until its next update. Where that stream
   has a hole, the frame falls back to race progress from telemetry (laps done + fraction of
   the current lap), with the grid order breaking the tie at the start.
2. A car whose race is over leaves the running order at the end of its last lap:
   - it took the chequered flag      -> FINISHED: it keeps its place, shown with a flag;
   - it stopped or was black-flagged -> RETIRED / DISQUALIFIED: it drops to the bottom
     (OUT / DSQ), retirees in the order of the official classification.
3. Once every car's race is over, the board IS the official classification (results Position).
   That is the only way post-race penalties reach the board: at Baku 2026 Bortoleto crossed
   the line 13th and was classified 15th after a 10 s penalty.

Why this module exists: telemetry stops at the finish line, so the replay froze every finished
car at almost the same spot and sorted them by a few metres of measurement noise (Baku 2026:
the winner was shown 10th). The order used to be computed three times (frame builder, replay
window, leaderboard widget), each differently; now it is computed once, here.

Times: every time in this module is in replay seconds (t = 0 at the first replay frame),
except where a name says `session`. `t0` converts: replay t = session time - t0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RUNNING, FINISHED, RETIRED, DISQUALIFIED = 0, 1, 2, 3
STATE_NAMES = ("running", "finished", "retired", "disqualified")

# Result statuses that mean "took the chequered flag": Ergast style ("+1 Lap") and the F1 API's.
_FLAG_STATUSES = {"finished", "lapped"}
# Statuses that do not say whether the car took the flag; the lap data decides.
_UNDECIDED_STATUSES = {"", "nan", "none", "<na>", "not classified"}

# Sort key for an unknown position: after every known one.
_UNKNOWN = 999


@dataclass(frozen=True)
class RaceOutcome:
    """How one driver's race ended. Times in replay seconds."""
    code: str
    ended_at: Optional[float]           # end of their last lap: the flag, or where they stopped
    took_flag: bool                     # finished the race, on the lead lap or lapped
    disqualified: bool
    laps_completed: Optional[int]
    classified_position: Optional[int]  # official result order 1..N, retirees included
    grid_position: Optional[int]        # None for a pit-lane start or when unknown
    official_gap_s: Optional[float]     # official gap to the winner, lead-lap finishers only

    def to_dict(self) -> dict:
        return asdict(self)


def outcome_from_dict(data: Mapping) -> RaceOutcome:
    """Inverse of RaceOutcome.to_dict (outcomes are cached with the replay frames as dicts)."""
    return RaceOutcome(**{name: data.get(name) for name in RaceOutcome.__dataclass_fields__})


def _seconds(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value.total_seconds()) if hasattr(value, "total_seconds") else float(value)


def _int_or_none(value, minimum: int = 1) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return number if number >= minimum else None


def took_the_flag(status: str, finished_by_laps: bool) -> bool:
    """
    Did the driver take the chequered flag?

    The official status decides when it can ("Finished", "Lapped", "+1 Lap": yes; "Retired",
    "Accident", "Engine"...: no). When it cannot (no status, "Not classified", "Disqualified":
    a disqualified car may well have crossed the line), the lap data decides.
    """
    s = (status or "").strip().lower()
    if s in _FLAG_STATUSES or s.startswith("+"):
        return True
    if s in _UNDECIDED_STATUSES or s.startswith("disq"):
        return finished_by_laps
    return False


def race_outcomes(laps: pd.DataFrame, results: Optional[pd.DataFrame], t0: float,
                  fallback_end: Optional[Mapping[str, float]] = None) -> Dict[str, RaceOutcome]:
    """
    One RaceOutcome per driver who has laps.

    laps:    FastF1 laps: Driver, LapNumber, LapTime, Time (session time at the end of the lap).
    results: FastF1 results: Abbreviation, Status, Position, GridPosition, Laps, Time. May be None.
    t0:      session seconds at replay t = 0.
    fallback_end: replay seconds where each driver's telemetry ends, used only when the last
        lap has no end time.

    The chequered flag falls when the leader completes the final lap: the earliest end of a
    timed lap with the highest lap number. A car "finished by laps" if its last lap is timed and
    ends at or after the flag.

    Official gaps: results `Time` is the gap to the winner only for cars that finished on the
    lead lap. For lapped cars it holds something else (British GP 2024: RIC, a lap down, shows
    9.9 s), so they get no gap rather than a wrong one.
    """
    laps = laps.dropna(subset=["Driver", "LapNumber"])
    timed = laps[laps["LapTime"].notna() & laps["Time"].notna()]
    flag_session = None
    if not timed.empty:
        final_lap = timed["LapNumber"].max()
        flag_session = _seconds(timed.loc[timed["LapNumber"] == final_lap, "Time"].min())

    by_code = {}
    if results is not None and len(results) and "Abbreviation" in results.columns:
        by_code = {str(row["Abbreviation"]): row for _, row in results.iterrows()}

    drafts = {}
    for code, driver_laps in laps.groupby("Driver"):
        code = str(code)
        last = driver_laps.sort_values("LapNumber").iloc[-1]
        last_lap = int(last["LapNumber"])
        last_lap_timed = pd.notna(last["LapTime"])
        ended_session = _seconds(last["Time"])
        finished_by_laps = bool(last_lap_timed and flag_session is not None
                                and ended_session is not None and ended_session >= flag_session - 1e-6)

        row = by_code.get(code)
        status = "" if row is None else str(row.get("Status", ""))
        took_flag = took_the_flag(status, finished_by_laps)

        laps_from_results = None if row is None else _int_or_none(row.get("Laps"), minimum=0)
        if took_flag or laps_from_results is None:
            laps_completed = last_lap if last_lap_timed else last_lap - 1
        else:
            laps_completed = laps_from_results

        if ended_session is not None:
            ended_at = ended_session - t0
        else:
            ended_at = None if fallback_end is None else fallback_end.get(code)

        drafts[code] = dict(
            code=code,
            ended_at=ended_at,
            took_flag=took_flag,
            disqualified=status.strip().lower().startswith("disq"),
            laps_completed=laps_completed,
            classified_position=None if row is None else _int_or_none(row.get("Position")),
            grid_position=None if row is None else _int_or_none(row.get("GridPosition")),
            status=status.strip().lower(),
            result_time=None if row is None else _seconds(row.get("Time")),
        )

    winner = next((d for d in drafts.values() if d["classified_position"] == 1 and d["took_flag"]), None)
    outcomes: Dict[str, RaceOutcome] = {}
    for code, d in drafts.items():
        gap = None
        if winner is not None and d["took_flag"] and d["laps_completed"] == winner["laps_completed"]:
            if d is winner:
                gap = 0.0
            elif d["status"] == "finished":
                gap = d["result_time"]
        outcomes[code] = RaceOutcome(
            code=code, ended_at=d["ended_at"], took_flag=d["took_flag"],
            disqualified=d["disqualified"], laps_completed=d["laps_completed"],
            classified_position=d["classified_position"], grid_position=d["grid_position"],
            official_gap_s=gap,
        )
    return outcomes


def official_positions(stream: Optional[pd.DataFrame], number_to_code: Mapping[str, str],
                       codes: Sequence[str], session_times: np.ndarray) -> np.ndarray:
    """
    Official timing-tower position of every driver at every instant, shape [len(codes), len(times)].

    stream: FastF1 timing stream rows: Time (session time), Driver (car number), Position.
    Each value is the latest update at or before the instant (held); NaN before the first one.
    """
    session_times = np.asarray(session_times, dtype=float)
    out = np.full((len(codes), len(session_times)), np.nan)
    if stream is None or len(stream) == 0:
        return out
    rows = stream.dropna(subset=["Time", "Driver", "Position"])
    rows = rows.assign(_code=rows["Driver"].astype(str).map(number_to_code),
                       _t=rows["Time"].dt.total_seconds())
    for i, code in enumerate(codes):
        mine = rows[rows["_code"] == code].sort_values("_t", kind="stable")
        if mine.empty:
            continue
        t = mine["_t"].to_numpy(dtype=float)
        p = mine["Position"].to_numpy(dtype=float)
        idx = np.searchsorted(t, session_times, side="right") - 1     # last update at or before
        known = idx >= 0
        out[i, known] = p[idx[known]]
    return out


def running_order(times: np.ndarray, codes: Sequence[str], outcomes: Mapping[str, RaceOutcome],
                  progress: np.ndarray, official: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Position (1..N) and race state of every driver at every frame.

    times:    replay seconds of each frame, shape [T]
    codes:    driver codes, the row order of `progress` and `official`
    outcomes: RaceOutcome per code; a driver without one is running throughout
    progress: race progress in laps from telemetry, [D, T]; used where `official` has a hole
    official: official timing-tower positions, [D, T], NaN where unknown; None if unavailable

    Returns (positions int16 [D, T], states int8 [D, T]) with states indexing STATE_NAMES.
    """
    times = np.asarray(times, dtype=float)
    progress = np.asarray(progress, dtype=float)
    n_drivers, n_frames = progress.shape

    def column(field: str, missing: float) -> np.ndarray:
        values = []
        for code in codes:
            value = getattr(outcomes[code], field) if code in outcomes else None
            values.append(missing if value is None else float(value))
        return np.array(values, dtype=float)[:, None]

    ended = column("ended_at", np.inf)                    # no known end: never over
    took_flag = column("took_flag", 0.0) > 0
    disqualified = column("disqualified", 0.0) > 0
    classified = column("classified_position", _UNKNOWN)
    grid = column("grid_position", _UNKNOWN)
    laps_done = column("laps_completed", np.nan)

    over = times[None, :] >= ended                        # [D, T]
    states = np.where(~over, RUNNING,
                      np.where(disqualified, DISQUALIFIED,
                               np.where(took_flag, FINISHED, RETIRED))).astype(np.int8)
    dropped = over & ~took_flag                           # retired or black-flagged: the bottom
    finished = over & took_flag
    everyone_done = over.all(axis=0)                      # [T]

    # A finished car has covered exactly the laps it completed; its frozen telemetry says less.
    prog = np.where(finished & ~np.isnan(laps_done), laps_done, progress)

    if official is not None:
        official = np.asarray(official, dtype=float)
        # A frame uses official positions only if every car still in the order has one.
        use_official = (~np.isnan(official) | dropped).all(axis=0)
        official_key = np.where(np.isnan(official), _UNKNOWN, official)
    else:
        use_official = np.zeros(n_frames, dtype=bool)
        official_key = np.full((n_drivers, n_frames), float(_UNKNOWN))

    in_order_key = np.where(use_official[None, :], official_key, -prog)
    main = np.where(everyone_done[None, :] | dropped, classified, in_order_key)
    group = np.where(everyone_done[None, :], 0, dropped.astype(int))
    retired_later_first = -np.where(np.isinf(ended), 0.0, ended)
    tie = np.where(dropped, retired_later_first, np.where(finished, classified, grid))

    # lexsort: the LAST key is the primary one; equal keys keep the `codes` order (stable)
    order = np.lexsort((tie, main, group), axis=0)        # order[k, t] = driver in place k + 1
    positions = (np.argsort(order, axis=0) + 1).astype(np.int16)
    return positions, states


def official_gap_between(ahead: Optional[RaceOutcome], behind: Optional[RaceOutcome]) -> Optional[float]:
    """Official time gap between two lead-lap finishers (penalties included), or None."""
    if ahead is None or behind is None:
        return None
    if ahead.official_gap_s is None or behind.official_gap_s is None:
        return None
    return behind.official_gap_s - ahead.official_gap_s
