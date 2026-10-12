"""
Neutralised periods (src/analytics/neutralisations.py) — Phase 2 step 1

When the race was neutralised: safety car, virtual safety car, red flag. Read only from the
official track status that Silver keeps on every lap (`track_status`: the codes seen during the
lap, in the order they first appeared, e.g. "1264"), never guessed from laps that look slow.

Race laps are counted the way Race Control counts them: by the leader. For each lap number the
status that counts is the one on the lap of the first car to complete it. Pooling every driver's
lap numbers instead (what the backfill report's `sc_lap_numbers` does) shifts periods by a lap:
when the safety car came out in Qatar 2024 the leader was on lap 35 but a lapped car was still
on its lap 34, so the pooled count says 34.

A period is a run of one kind in the leader's sequence of codes, so the order inside a lap
matters: Qatar 2024 lap 40 reads "1264" (green, yellow, VSC, safety car), which is a VSC on
lap 40 followed by a safety car from lap 40.

Golden facts (Race Control's log, 2024 Qatar GP): safety car on laps 1-4, 35-39 and 40-42; a VSC
on lap 40 that became the third safety car 20 s later. 2024 British GP: none.

Limitation: the feed lists each code once per lap, so the same kind of neutralisation ending
and starting again inside ONE lap reads as a single period.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src.pipeline.cleaner import STATUS_RED, STATUS_SC, STATUS_VSC

SAFETY_CAR = "SC"
VIRTUAL_SAFETY_CAR = "VSC"
RED_FLAG = "red flag"
KINDS = (SAFETY_CAR, VIRTUAL_SAFETY_CAR, RED_FLAG)

# Track status code -> kind of neutralisation. Green (1) and yellow (2) are not neutralisations:
# yellow is a local flag, the race goes on.
KIND_OF_CODE = {STATUS_SC: SAFETY_CAR, STATUS_RED: RED_FLAG, **{code: VIRTUAL_SAFETY_CAR for code in STATUS_VSC}}

PERIOD_COLUMNS = ["kind", "first_lap", "last_lap", "laps"]


def leader_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    One Silver row per race lap: the lap of the car that completed it first (the leader).

    Laps FastF1 generated are skipped: their end times are made up (2023 Monza lap 51). Where no
    lap of a number has an end time, the car in P1 at the end of that lap stands in. A lap number
    with neither is left out: its status is unknown, not guessed.
    """
    usable = laps[~laps["is_generated"].astype(bool)] if "is_generated" in laps.columns else laps
    first = (usable.dropna(subset=["lap_end_sec"])
             .sort_values(["lap_number", "lap_end_sec"])
             .drop_duplicates("lap_number", keep="first"))
    rest = usable[~usable["lap_number"].isin(first["lap_number"])]
    in_p1 = rest[rest["position"].eq(1).fillna(False).astype(bool)].drop_duplicates("lap_number", keep="first")
    return pd.concat([first, in_p1]).sort_values("lap_number").reset_index(drop=True)


def _code_sequence(leaders: pd.DataFrame) -> List[Tuple[int, str]]:
    """The leader's track status codes in race order, each with the lap it was seen on."""
    sequence = []
    for lap, status in zip(leaders["lap_number"], leaders["track_status"]):
        codes = "" if pd.isna(status) else str(status)
        sequence.extend((int(lap), code) for code in codes)
    return sequence


def neutralised_periods(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Every safety car, VSC and red-flag period of the race, in race order.

    Silver laps in; one row per period out: kind (SC / VSC / red flag), first_lap and last_lap in
    the leader's lap numbers (a period that starts or ends during a lap includes that lap), and
    laps = how many race laps it touched.
    """
    periods: List[list] = []
    current = None                     # [kind, first_lap, last_lap] of the period being read
    for lap, code in _code_sequence(leader_laps(laps)):
        kind = KIND_OF_CODE.get(code)
        if current is not None and kind == current[0]:
            current[2] = lap           # the same neutralisation goes on
            continue
        if current is not None:
            periods.append(current)    # anything else ends it: green, yellow, or another kind
            current = None
        if kind is not None:
            current = [kind, lap, lap]
    if current is not None:
        periods.append(current)

    out = pd.DataFrame(periods, columns=PERIOD_COLUMNS[:3])
    out["laps"] = out["last_lap"] - out["first_lap"] + 1
    return out.astype({"first_lap": "int64", "last_lap": "int64", "laps": "int64"})


def describe_periods(periods: pd.DataFrame) -> str:
    """Periods as text: 'SC 1-4, SC 35-39, VSC 40, SC 40-42', or 'none'."""
    if periods.empty:
        return "none"
    return ", ".join(f"{kind} {first}" if first == last else f"{kind} {first}-{last}"
                     for kind, first, last in zip(periods["kind"], periods["first_lap"], periods["last_lap"]))
