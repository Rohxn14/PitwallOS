"""
The telemetry clock (src/pipeline/telemetry_clock.py) - Phase 5 step 5.0

OpenF1 stamps every telemetry sample with the time of day in UTC ("2024-12-01T16:03:12.345000+00:00"). Our lake
counts time from the start of the timing session instead ("session time": Bronze laps' LapStartTime). The two
clocks differ by ONE number per race, t0 = the UTC moment when session time was zero:

    UTC = t0 + session time        so        session time = UTC - t0

A laps-only Bronze never learned t0 (FastF1 only works it out when it loads telemetry), so we measure it from the
moments both sources record for every driver: the starts of the laps, in OpenF1's lap table (UTC) and Bronze's (session
time). Two estimates, one checking the other:

1. LAP 1 (exact): t0 = OpenF1's lap-1 date_start - Bronze's lap-1 LapStartTime, one value per driver, their median.
   On the four 2026 races where FastF1 knows t0 (Australia, Barcelona, Baku, Bahrain): 1 ms off, for every driver.
   THE VOTE: a few drivers can carry a wrong lap 1 (Sao Paulo 2023: the four cars caught in the lap-1 crash get the
   restart, 27 minutes late; Mexico 2023: Perez, out at the first corner, 70 minutes), so drivers more than 5 ms from
   the median are outliers (named), and lap 1 counts only if MORE THAN HALF of the drivers, and at least 3, agree.

2. THE LATER LAPS (about +-0.05 s, but never fooled by one lap): OpenF1 stamps laps 2 onwards later than they started,
   by 0.173 s (median over the 56 races of 2023-2025 where both estimates agree; race-to-race sd 0.025 s). So
   t0 = median over every lap from 2 on of (date_start - LapStartTime) - 0.173 s.

THE RULE: lap 1 when the two agree within 0.25 s (10 sd of the later laps' own scatter); the later laps when they do
not, or when lap 1 cannot vote. Why: in some races EVERY car's lap 1 is wrong in the same way, so the vote cannot see
it: Japan 2024 and Monaco 2024 (red flag on lap 1: lap 1 dated from the restart, 28 and 41 minutes away), Zandvoort
2023 (2.1 s) and Las Vegas 2024 (0.45 s); Miami 2025 has no lap-1 start at all. Checked against a source neither
estimate uses, the finish line (its place on the map does not move from year to year): at Las Vegas 2024 the cars
cross it 35.6 m away from where they did in 2023 on the lap-1 clock, 4.3 m away on the later laps' clock; Zandvoort
2023 against 2024: 174 m against 7.2 m. The clock refuses (ClockError) only when neither estimate is possible.

Then each OpenF1 table is put in the shape the Silver resampler reads (FastF1's column names):

    car_data -> DriverNumber, Date, SessionTime, Speed, RPM, nGear, Throttle, Brake, DRS
    location -> DriverNumber, Date, SessionTime, X, Y, Z

and one raw value is decoded: 104 on the brake means NO READING (it comes on the same samples as throttle 104),
so the brake becomes unknown there, not "pressed" (FastF1 turned it into "pressed"). Gear and DRS stay the raw codes
(gear can carry codes like 128 in the pit lane: Silver's resampler decides what is a gear).
"""

from typing import Dict, Optional, Tuple

import pandas as pd

NO_READING = 104                # the value the brake and the throttle carry when the car sent nothing
MAX_SPREAD_MS = 5.0             # a driver this far from the median lap-1 t0 is an outlier
MIN_DRIVERS = 3                 # fewer agreeing drivers on lap 1: too little to trust
LATER_LAP_LAG_SEC = 0.173       # OpenF1 stamps laps 2+ this much after they start (measured, sd 0.025 s)
MAX_DISAGREEMENT_SEC = 0.25     # lap 1 further than this from the later laps: lap 1 is not trusted


class ClockError(ValueError):
    """The two clocks could not be lined up with confidence."""


def utc(stamps: pd.Series) -> pd.Series:
    """OpenF1's ISO time stamps as UTC times without a time zone (like FastF1's Date column)."""
    return pd.to_datetime(stamps, format="ISO8601", utc=True).dt.tz_localize(None)


def lap_offsets(openf1_laps: pd.DataFrame, bronze_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (driver, lap) known to both sources: OpenF1's start (UTC) minus Bronze's start (session time)."""
    theirs = openf1_laps[["driver_number", "lap_number", "date_start"]].dropna()
    ours = bronze_laps[["DriverNumber", "LapNumber", "LapStartTime"]].dropna()
    ours = ours.assign(driver_number=ours["DriverNumber"].astype(int), lap_number=ours["LapNumber"].astype(int))
    both = theirs.astype({"driver_number": int, "lap_number": int}).merge(ours, on=["driver_number", "lap_number"])
    return both.assign(offset=utc(both["date_start"]) - pd.to_timedelta(both["LapStartTime"]))


def lap_one_vote(pairs: pd.DataFrame) -> Tuple[pd.Timestamp, Dict]:
    """t0 from lap 1 (pairs of lap 1 only): the median, if more than half of the drivers agree with it within 5 ms."""
    t0 = pairs["offset"].median()
    agree = (pairs["offset"] - t0).dt.total_seconds().abs() * 1000 <= MAX_SPREAD_MS
    if agree.sum() < MIN_DRIVERS or agree.sum() * 2 <= len(pairs):
        raise ClockError(f"only {int(agree.sum())} of {len(pairs)} drivers' lap-1 clocks agree within "
                         f"{MAX_SPREAD_MS:.0f} ms (need more than half, and at least {MIN_DRIVERS})")
    kept = pairs.loc[agree, "offset"]
    outliers = sorted(pairs.loc[~agree, "driver_number"].astype(int))
    return t0, {"clock_drivers": int(agree.sum()),
                "clock_spread_ms": round((kept.max() - kept.min()).total_seconds() * 1000, 3),
                "clock_outliers": " ".join(map(str, outliers))}


def later_laps(pairs: pd.DataFrame) -> Optional[pd.Timestamp]:
    """t0 from every lap from 2 on (pairs of those laps), corrected for OpenF1's usual lag; None without such laps."""
    if pairs.empty:
        return None
    return pairs["offset"].median() - pd.Timedelta(seconds=LATER_LAP_LAG_SEC)


def session_offset(openf1_laps: pd.DataFrame, bronze_laps: pd.DataFrame) -> Tuple[pd.Timestamp, Dict]:
    """t0, and a short report: which estimate was used and why, how many drivers agreed, who did not."""
    pairs = lap_offsets(openf1_laps, bronze_laps)
    later = later_laps(pairs[pairs["lap_number"] >= 2])
    try:
        t0, report = lap_one_vote(pairs[pairs["lap_number"] == 1])
    except ClockError as err:
        if later is None:
            raise
        t0, report = later, {"clock_method": "later laps", "clock_note": f"lap 1 cannot vote: {err}"}
    else:
        report["clock_method"] = "lap 1"
        if later is not None:
            gap = (t0 - later).total_seconds()
            report["clock_check_sec"] = round(gap, 3)              # lap 1 against the later laps (~0 when both right)
            if abs(gap) > MAX_DISAGREEMENT_SEC:
                report.update(clock_method="later laps",
                              clock_note=f"lap 1 is {gap:+.3f} s from the later laps: every car's lap 1 is off")
                t0 = later
    report["t0_utc"] = str(t0)
    report["clock_laps"] = int(len(pairs))
    return t0, report


def _on_session_clock(table: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """The columns every sample has: whose it is, and when (UTC and session time)."""
    out = pd.DataFrame({"DriverNumber": table["driver_number"].astype(int).astype(str)})
    out["Date"] = utc(table["date"])
    out["SessionTime"] = out["Date"] - t0
    return out


def car_samples(car_data: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """OpenF1's car data in FastF1's shape, on the session clock."""
    out = _on_session_clock(car_data, t0)
    out["Speed"] = car_data["speed"].astype(float)
    out["RPM"] = car_data["rpm"].astype(float)
    out["nGear"] = car_data["n_gear"].astype("Int16")       # raw codes, up to 255: wider than a gear needs
    out["Throttle"] = car_data["throttle"].astype(float)    # 104 is left in: the resampler blanks it
    pressed = (car_data["brake"] > 0).astype("boolean")      # FastF1's rule: anything but 0 is pressed...
    out["Brake"] = pressed.mask(car_data["brake"] == NO_READING)   # ...except "no reading", which is unknown
    out["DRS"] = car_data["drs"].astype("Int16")
    return out.reset_index(drop=True)


def position_samples(location: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """OpenF1's location in FastF1's shape, on the session clock (x, y, z in tenths of a metre, as delivered)."""
    out = _on_session_clock(location, t0)
    for raw, name in (("x", "X"), ("y", "Y"), ("z", "Z")):
        out[name] = location[raw].astype(float)
    return out.reset_index(drop=True)
