"""
A qualifying session for the replay, read from the lake (src/lib/lake_qualifying.py) - the OpenF1 overhaul, piece 7

The qualifying replay (f1_data.get_quali_telemetry, src/interfaces/qualifying.py; Tom Shaw's code) shows, for each
driver and each part of qualifying (Q1, Q2, Q3), the fastest lap as a trace: position, speed, gear, throttle, brake,
DRS. It asked FastF1 for the results, for the laps split into the three parts, and for one lap's telemetry.
LakeQualifying answers the same three questions from Bronze's OpenF1 tables.

It reads BRONZE, not Silver: Silver's cleaner is about races (a clock from lights out, laps chained from lap 2, clean
laps, stints). A qualifying session has none of that, and nothing but this replay consumes it, so it gets no Silver
table: what the replay needs is a handful of laps, cut straight from the raw tables.

THE THREE PARTS. Race Control (OpenF1 merges F1's session status into it) has one "SESSION STARTED" and one
"SESSION FINISHED" per part, in order. A lap belongs to the part during which it STARTED: a lap begun before the
chequered flag counts, as in the rules; the lap back to the pits after it does not.

A DRIVER'S FASTEST LAP of a part: the lap of that part whose time is the driver's official best in OpenF1's result
table (a faster lap deleted for track limits is not it); failing that, the shortest timed lap that is not a pit-out
lap.

ONE LAP'S TELEMETRY: the car_data and location samples between the lap's start and its end, on a clock that starts
at 0 with the lap. The start is OpenF1's date_start less its usual lag (0.173 s, measured on races); the two streams
(speed and position, ~4 samples a second each, on their own clocks) are put on the speed stream's clock by
interpolating the position. Distance is speed added up over time, scaled to nothing: the replay only needs it to
grow along the lap.

Sessions: "Q" (Qualifying) and "SQ" (Sprint Qualifying; "Sprint Shootout" in 2023).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline import lake, openf1_translate
from src.pipeline.openf1_translate import SESSION_STARTED
from src.pipeline.telemetry_clock import LATER_LAP_LAG_SEC, NO_READING, utc

SESSION_TITLES = {"Q": "Qualifying", "SQ": "Sprint Qualifying"}
SESSION_FINISHED = "SESSION FINISHED"
SEGMENTS = ("Q1", "Q2", "Q3")
TABLES = ("sessions", "drivers", "laps", "stints", "race_control", "weather", "session_result")
MIN_SAMPLES = 20                      # fewer telemetry samples than this in a lap: no trace


def available(year: int, round_number: int, session_type: str = "Q",
              root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> bool:
    """True if the lake can replay this qualifying session: Bronze from OpenF1, with telemetry."""
    if session_type not in SESSION_TITLES or not lake.has_bronze(year, round_number, session_type, root):
        return False
    manifest = lake.read_manifest(year, round_number, session_type, root)
    return manifest.get("source") == lake.OPENF1 and bool(manifest.get("telemetry"))


def segment_windows(race_control: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) in UTC of each part of qualifying, in order: every SESSION STARTED with the SESSION FINISHED
    that follows it. A part stopped by a red flag and started again is one part: Race Control numbers the parts
    (qualifying_phase), and the windows of one number are joined."""
    rc = race_control.assign(t=utc(race_control["date"])).sort_values("t", kind="stable")
    marks = rc[rc["message"].isin([SESSION_STARTED, SESSION_FINISHED])]
    windows: Dict[Any, List[pd.Timestamp]] = {}
    open_at, phase = None, None
    for row in marks.itertuples():
        if row.message == SESSION_STARTED and open_at is None:
            open_at = row.t
            phase = getattr(row, "qualifying_phase", None)
            phase = len(windows) + 1 if phase is None or pd.isna(phase) else int(phase)
        elif row.message == SESSION_FINISHED and open_at is not None:
            start, end = windows.get(phase, [open_at, row.t])
            windows[phase] = [min(start, open_at), max(end, row.t)]
            open_at = None
    return [(start, end) for _, (start, end) in sorted(windows.items())]


class LakeQualifying:
    """One qualifying session from the lake, with what the qualifying replay reads from a FastF1 Session."""

    def __init__(self, year: int, round_number: int, session_type: str = "Q",
                 root: lake.PathLike = lake.DEFAULT_LAKE_ROOT):
        if not available(year, round_number, session_type, root):
            raise FileNotFoundError(f"{year} round {round_number} '{session_type}' is not in the lake with telemetry")
        self.year, self.round_number, self.session_type, self.root = int(year), int(round_number), session_type, root
        self.manifest = lake.read_manifest(year, round_number, session_type, root)
        tables = {name: self._bronze(name) for name in TABLES}
        self._tables = tables
        rc = tables["race_control"]
        self.windows = segment_windows(rc)
        if not self.windows:
            raise ValueError(f"{year} round {round_number} '{session_type}': Race Control has no SESSION STARTED / "
                             "FINISHED pair, so the parts of qualifying cannot be told apart")
        self.t0_date: pd.Timestamp = self.windows[0][0]

        who = tables["drivers"].drop_duplicates("driver_number")
        self._by_number = {str(int(r.driver_number)): r for r in who.itertuples()}
        self._number_of = {str(r.name_acronym): str(int(r.driver_number)) for r in who.itertuples()}
        laps = tables["laps"]
        self._laps = laps.assign(start=utc(laps["date_start"]))
        self.drivers: List[str] = [n for n in self._by_number if int(n) in set(laps["driver_number"])]
        self.results = self._results(tables["session_result"])
        timeline = openf1_translate.status_timeline(rc, self.t0_date)
        self.track_status = pd.DataFrame({"Time": pd.to_timedelta(timeline["t_sec"], unit="s"),
                                          "Status": timeline["code"].astype(str)})
        self.weather_data = openf1_translate.weather_table(tables["weather"], self.t0_date)
        start = pd.Timestamp(self.manifest.get("event_date")) if self.manifest.get("event_date") else None
        self.event: Dict[str, Any] = {"EventName": self.manifest.get("event_name", ""), "RoundNumber": self.round_number,
                                      "Location": self.manifest.get("location", ""),
                                      "Country": self.manifest.get("country", ""), "EventDate": start}
        self._car: Optional[pd.DataFrame] = None
        self._location: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        return (f"{self.year} Season Round {self.round_number} {self.event['EventName']} - "
                f"{SESSION_TITLES[self.session_type]} - lake")

    def _bronze(self, endpoint: str) -> pd.DataFrame:
        return lake.read_table(lake.table_path(lake.BRONZE, lake.openf1_table(endpoint), self.year, self.round_number,
                                               self.session_type, self.root))

    # --- who -----------------------------------------------------------------------------------------------------

    def get_driver(self, who: Any) -> Dict[str, str]:
        """A driver by car number or by three-letter code (FastF1 takes both)."""
        number = str(who) if str(who) in self._by_number else self._number_of[str(who)]
        row = self._by_number[number]
        first, last = getattr(row, "first_name", None), getattr(row, "last_name", None)
        full = f"{first} {last}" if first and last else str(getattr(row, "full_name", row.name_acronym))
        return {"Abbreviation": str(row.name_acronym), "DriverNumber": number, "FullName": full}

    def driver_colors(self) -> Dict[str, tuple]:
        """{driver code: (r, g, b)} from the team colour OpenF1 gives each driver."""
        out = {}
        for row in self._by_number.values():
            colour = str(getattr(row, "team_colour", "") or "").lstrip("#")
            if len(colour) == 6:
                out[str(row.name_acronym)] = tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))
        return out

    def _results(self, session_result: pd.DataFrame) -> pd.DataFrame:
        """FastF1's results for a qualifying session: Position, Abbreviation, FullName, Q1, Q2, Q3 (timedeltas)."""
        rows = []
        for r in session_result.itertuples():
            number = str(int(r.driver_number))
            if number not in self._by_number:
                continue
            best = r.duration if isinstance(r.duration, (list, tuple, np.ndarray)) else [r.duration]
            best = [None if b is None or pd.isna(b) else float(b) for b in best] + [None] * 3
            driver = self.get_driver(number)
            rows.append({"DriverNumber": number, "Abbreviation": driver["Abbreviation"], "FullName": driver["FullName"],
                         "Position": float(r.position) if pd.notna(r.position) else np.nan,
                         **{seg: pd.to_timedelta(best[i], unit="s") if best[i] is not None else pd.NaT
                            for i, seg in enumerate(SEGMENTS)}})
        out = pd.DataFrame(rows).sort_values("Position", na_position="last").reset_index(drop=True)
        for seg in SEGMENTS:
            out[seg] = pd.to_timedelta(out[seg])
        return out

    # --- laps and telemetry ------------------------------------------------------------------------------------------

    def segment_laps(self, segment: str) -> pd.DataFrame:
        """The laps started during one part of qualifying. Raises ValueError if the session has no such part."""
        index = SEGMENTS.index(segment)
        if index >= len(self.windows):
            raise ValueError(f"{segment} does not exist for this session.")
        start, end = self.windows[index]
        laps = self._laps
        return laps[(laps["start"] >= start) & (laps["start"] <= end)]

    def fastest_lap(self, driver: Any, segment: str) -> Optional[pd.Series]:
        """A driver's fastest timed lap of one part (not a pit-out lap), in FastF1's column names, or None."""
        number = int(self.get_driver(driver)["DriverNumber"])
        laps = self.segment_laps(segment)
        out_lap = laps["is_pit_out_lap"].fillna(False).astype(bool) if "is_pit_out_lap" in laps else False
        mine = laps[(laps["driver_number"] == number) & laps["lap_duration"].notna() & ~out_lap]
        if mine.empty:
            return None
        # The result table has the driver's official best of this part: a faster lap whose time was deleted (track
        # limits) is not it. Take the lap with that time; without one, the shortest.
        official = self.results.loc[self.results["DriverNumber"] == str(number), segment]
        official = official.iloc[0].total_seconds() if len(official) and pd.notna(official.iloc[0]) else None
        same = mine[(mine["lap_duration"] - official).abs() < 0.0015] if official is not None else mine.iloc[0:0]
        lap = (same if len(same) else mine).sort_values("lap_duration").iloc[0]
        stints = self._tables["stints"]
        tyre = stints[(stints["driver_number"] == number) & (stints["lap_start"] <= lap["lap_number"])
                      & (stints["lap_end"] >= lap["lap_number"])]
        return pd.Series({
            "Driver": self.get_driver(number)["Abbreviation"], "DriverNumber": str(number),
            "LapNumber": int(lap["lap_number"]), "LapTime": pd.to_timedelta(float(lap["lap_duration"]), unit="s"),
            "Sector1Time": pd.to_timedelta(lap.get("duration_sector_1"), unit="s"),
            "Sector2Time": pd.to_timedelta(lap.get("duration_sector_2"), unit="s"),
            "Sector3Time": pd.to_timedelta(lap.get("duration_sector_3"), unit="s"),
            "Compound": tyre["compound"].iloc[-1] if len(tyre) else None,
            "start_utc": lap["start"] - pd.Timedelta(seconds=LATER_LAP_LAG_SEC),
        })

    def _samples(self, kind: str) -> pd.DataFrame:
        """A telemetry table of the session, read once, with its stamps as UTC times."""
        if kind == "car":
            if self._car is None:
                table = self._bronze("car_data")
                self._car = table.assign(t=utc(table["date"])).sort_values("t", kind="stable")
            return self._car
        if self._location is None:
            table = self._bronze("location")
            self._location = table.assign(t=utc(table["date"])).sort_values("t", kind="stable")
        return self._location

    def lap_telemetry(self, lap: pd.Series) -> pd.DataFrame:
        """One lap's telemetry as FastF1's lap.get_telemetry(): Time (from 0), X, Y, Distance, RelativeDistance,
        Speed, nGear, Throttle, Brake, DRS. Empty if OpenF1 has too few samples for the lap."""
        number = int(lap["DriverNumber"])
        start, end = lap["start_utc"], lap["start_utc"] + lap["LapTime"]
        car = self._samples("car")
        car = car[(car["driver_number"] == number) & (car["t"] >= start) & (car["t"] <= end)]
        where = self._samples("location")
        where = where[(where["driver_number"] == number) & (where["t"] >= start - pd.Timedelta(seconds=1))
                      & (where["t"] <= end + pd.Timedelta(seconds=1)) & ~((where["x"] == 0) & (where["y"] == 0))]
        def number_of(table: pd.DataFrame, column: str) -> np.ndarray:        # a missing reading becomes nan
            return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)

        car = car[pd.to_numeric(car["speed"], errors="coerce").notna()]
        where = where[pd.to_numeric(where["x"], errors="coerce").notna() & pd.to_numeric(where["y"], errors="coerce").notna()]
        if len(car) < MIN_SAMPLES or len(where) < MIN_SAMPLES:
            return pd.DataFrame()
        t = (car["t"] - start).dt.total_seconds().to_numpy(dtype=float)
        t_where = (where["t"] - start).dt.total_seconds().to_numpy(dtype=float)
        speed = number_of(car, "speed")
        distance = np.concatenate([[0.0], np.cumsum(speed[1:] / 3.6 * np.diff(t))])
        throttle, brake = np.nan_to_num(number_of(car, "throttle")), np.nan_to_num(number_of(car, "brake"))
        gear, drs = np.nan_to_num(number_of(car, "n_gear")), np.nan_to_num(number_of(car, "drs"))
        return pd.DataFrame({
            "Time": pd.to_timedelta(t, unit="s"),
            "X": np.interp(t, t_where, number_of(where, "x")),
            "Y": np.interp(t, t_where, number_of(where, "y")),
            "Distance": distance,
            "RelativeDistance": distance / distance[-1] if distance[-1] > 0 else 0.0,
            "Speed": speed,
            "nGear": np.where((gear >= 0) & (gear <= 8), gear, 0).astype(int),
            "Throttle": np.where(throttle > 100, 0.0, throttle),            # 104 = no reading
            "Brake": (brake > 0) & (brake != NO_READING),
            "DRS": drs.astype(int),                                         # no reading: 0, as closed
        })

    def segment_fastest_lap(self, driver: Any, segment: str) -> Tuple[Optional[pd.Series], pd.DataFrame]:
        """(the driver's fastest lap of a part, its telemetry): what the qualifying replay draws. (None, empty) if
        the driver set no time in that part."""
        lap = self.fastest_lap(driver, segment)
        if lap is None:
            return None, pd.DataFrame()
        return lap, self.lap_telemetry(lap)

    def example_lap(self) -> Optional[pd.DataFrame]:
        """The fastest lap of the session with telemetry: the track's outline and DRS zones."""
        for row in self.results.itertuples():
            for segment in reversed(SEGMENTS[:len(self.windows)]):
                lap, telemetry = self.segment_fastest_lap(row.DriverNumber, segment)
                if lap is not None and len(telemetry):
                    return telemetry
        return None

    def circuit_rotation(self) -> float:
        try:
            from src.lib import race_calendar
            return race_calendar.rotation(self.year, int(self._tables["sessions"]["circuit_key"].iloc[0]), self.root)
        except Exception as err:
            print(f"Circuit rotation not available ({type(err).__name__}: {err}); the map is not turned")
            return 0.0
