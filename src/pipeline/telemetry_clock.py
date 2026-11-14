"""
OpenF1's telemetry on the session clock (src/pipeline/telemetry_clock.py) - Phase 5 step 5.0

OpenF1 stamps every telemetry sample with the time of day in UTC ("2024-12-01T16:03:12.345000+00:00"). The lake counts
time from one instant per session, t0 (for a race: lights out, found by openf1_translate.lights_out):

    session time = UTC - t0

This module puts each OpenF1 table in the shape the Silver resampler reads:

    car_data -> DriverNumber, Date, SessionTime, Speed, RPM, nGear, Throttle, Brake, DRS
    location -> DriverNumber, Date, SessionTime, X, Y, Z

and decodes one raw value: 104 on the brake means NO READING (it comes on the same samples as throttle 104), so the
brake becomes unknown there, not "pressed". Gear and DRS stay the raw codes (gear can carry codes like 128 in the pit
lane: Silver's resampler decides what is a gear).

Until November 2026 this module also worked out t0 by matching OpenF1's lap starts against FastF1's lap table, with a
vote among the drivers. With every table coming from OpenF1 there is one clock and nothing to match (git history has it).
LATER_LAP_LAG_SEC is what that work measured and what remains in use: OpenF1 stamps the start of laps 2 onwards
0.173 s late (median over 56 races, race-to-race sd 0.025 s).
"""

import pandas as pd

NO_READING = 104                # the value the brake and the throttle carry when the car sent nothing
LATER_LAP_LAG_SEC = 0.173       # OpenF1 stamps laps 2+ this much after they start (measured, sd 0.025 s)


def utc(stamps: pd.Series) -> pd.Series:
    """OpenF1's ISO time stamps as UTC times without a time zone (like FastF1's Date column)."""
    return pd.to_datetime(stamps, format="ISO8601", utc=True).dt.tz_localize(None)


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
