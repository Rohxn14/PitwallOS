"""
A race for the replay, read from the lake (src/lib/lake_session.py) - the OpenF1 overhaul, piece 5

The replay (src/f1_data.py, main.py, src/interfaces/race_replay.py; Tom Shaw's code) was written against FastF1's
Session object: session.laps, session.results, session.track_status, lap.get_telemetry() ... Since F1 locked its
archive that object can only be built for sessions left in FastF1's cache. The lake holds the same facts for every
race from 2023 on, from OpenF1. LakeSession offers them under the names the replay already uses, so the replay's
frame builder and its window keep their code; only the few places that asked FastF1 for something FastF1-specific
(a lap's telemetry, the timing stream, team colours, the circuit's rotation) ask this object instead.

Where each thing comes from:
    laps, results, weather_data         the translation of Bronze's OpenF1 tables (openf1_translate), FastF1's shape
    track_status                        the same status timeline Silver's laps were flagged with
    race_control_messages               Bronze's openf1_race_control, on the session clock
    driver_samples()                    Silver telemetry (10 Hz, every driver on one clock), per driver
    timing_stream()                     OpenF1's position and intervals tables: the official order and gaps
    driver_colors()                     OpenF1's drivers table (team_colour)
    example_lap()                       Silver telemetry of the fastest clean lap: the track's outline, DRS zones
    circuit_rotation()                  MultiViewer, kept in the lake (src/lib/race_calendar.py)

ONE CLOCK: every time is seconds since lights out (openf1_translate), the clock of Silver. The replay subtracts
the first telemetry sample's time anyway, so it never sees the difference from FastF1's clock.

The lake must hold the race's Bronze (from OpenF1) and Silver laps + telemetry: available() says so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.lib.tyres import get_tyre_compound_int
from src.pipeline import lake, openf1_translate
from src.pipeline.openf1_translate import seconds_since

RACE = "R"
SESSION_TITLES = {RACE: "Race", "S": "Sprint"}        # the sessions the lake can replay, and what to call them
DEFAULT_ROTATION_DEG = 0.0


def available(year: int, round_number: int, session_type: str = RACE,
              root: lake.PathLike = lake.DEFAULT_LAKE_ROOT) -> bool:
    """True if the lake can replay this session: a race or a sprint, Bronze from OpenF1, Silver laps and telemetry."""
    if session_type not in SESSION_TITLES or not lake.has_bronze(year, round_number, session_type, root):
        return False
    if lake.read_manifest(year, round_number, session_type, root).get("source") != lake.OPENF1:
        return False
    return all(lake.table_path(lake.SILVER, table, year, round_number, session_type, root).exists()
               for table in ("laps", "telemetry"))


class LakeSession:
    """One race from the lake, with the attributes of FastF1's Session that the replay reads."""

    def __init__(self, year: int, round_number: int, session_type: str = RACE,
                 root: lake.PathLike = lake.DEFAULT_LAKE_ROOT):
        if not available(year, round_number, session_type, root):
            raise FileNotFoundError(f"{year} round {round_number} '{session_type}' is not in the lake with telemetry")
        self.year, self.round_number, self.session_type, self.root = int(year), int(round_number), session_type, root
        self.manifest = lake.read_manifest(year, round_number, session_type, root)
        self._tables = openf1_translate.read_bronze(year, round_number, session_type, root)
        translated = openf1_translate.translate(self._tables)
        self.t0_date: pd.Timestamp = translated.lights_out_utc
        self.laps: pd.DataFrame = translated.laps
        self.results: pd.DataFrame = translated.results
        self.weather_data: pd.DataFrame = translated.weather
        self.total_laps: Optional[int] = translated.total_laps

        timeline = openf1_translate.status_timeline(self._tables["race_control"], self.t0_date)
        self.track_status = pd.DataFrame({"Time": pd.to_timedelta(timeline["t_sec"], unit="s"),
                                          "Status": timeline["code"].astype(str)})
        rc = self._tables["race_control"]
        number = rc["driver_number"] if "driver_number" in rc else pd.Series(np.nan, index=rc.index)
        self.race_control_messages = pd.DataFrame({
            "Time": pd.to_timedelta(seconds_since(rc["date"], self.t0_date), unit="s"),
            "Category": rc.get("category"), "Message": rc.get("message"), "Flag": rc.get("flag"),
            "Scope": rc.get("scope"), "Sector": rc.get("sector"), "RacingNumber": number,
        }).sort_values("Time", kind="stable").reset_index(drop=True)

        who = self._tables["drivers"].drop_duplicates("driver_number")
        self._codes = {str(int(n)): str(c) for n, c in zip(who["driver_number"], who["name_acronym"])}
        self._colours = {str(c): str(h) for c, h in zip(who["name_acronym"], who.get("team_colour", "")) if h}
        in_laps = set(self.laps["DriverNumber"].astype(str))
        self.drivers: List[str] = [n for n in self._codes if n in in_laps]      # as FastF1: the cars that have laps
        start = pd.Timestamp(self.manifest.get("event_date")) if self.manifest.get("event_date") else None
        self.event: Dict[str, Any] = {"EventName": self.manifest.get("event_name", ""), "RoundNumber": self.round_number,
                                      "Location": self.manifest.get("location", ""),
                                      "Country": self.manifest.get("country", ""), "EventDate": start}
        self._telemetry: Optional[pd.DataFrame] = None
        self._silver_laps: Optional[pd.DataFrame] = None

    def __str__(self) -> str:      # the replay names its cache file after this: never the name FastF1's session has
        return (f"{self.year} Season Round {self.round_number} {self.event['EventName']} - "
                f"{SESSION_TITLES[self.session_type]} - lake")

    def get_driver(self, number: Any) -> Dict[str, str]:
        return {"Abbreviation": self._codes[str(number)], "DriverNumber": str(number)}

    # --- Silver, read once -------------------------------------------------------------------------------------

    def _silver(self, table: str) -> pd.DataFrame:
        return lake.read_table(lake.table_path(lake.SILVER, table, self.year, self.round_number, self.session_type,
                                               self.root))

    @property
    def telemetry(self) -> pd.DataFrame:
        if self._telemetry is None:
            self._telemetry = self._silver("telemetry")
        return self._telemetry

    @property
    def silver_laps(self) -> pd.DataFrame:
        if self._silver_laps is None:
            self._silver_laps = self._silver("laps")
        return self._silver_laps

    # --- what the replay asked FastF1 for ------------------------------------------------------------------------

    def driver_samples(self, number: Any) -> Optional[Dict[str, Any]]:
        """
        One driver's whole race as arrays over time, the shape f1_data._process_single_driver returns: t, x, y, dist
        (metres since lights out), rel_dist (share of the lap done), lap, tyre, tyre_life, speed, gear, drs, throttle,
        brake, progress (laps done + share of this one). None if the driver has no usable telemetry.

        Ticks inside a telemetry hole (no position or speed: is_data_gap) are left out: the replay interpolates
        between the samples it gets, so a hole becomes a straight glide, as it did with FastF1's samples.
        """
        number = str(number)
        tel = self.telemetry
        mine = tel[tel["driver_number"] == number]
        mine = mine[mine["x"].notna() & mine["y"].notna() & mine["speed_kmh"].notna()].sort_values("session_time_sec")
        if len(mine) < 2:
            return None
        lap = mine["lap_number"].to_numpy(dtype=float)
        in_lap_m = mine["lap_distance_m"].to_numpy(dtype=float)
        lap_length = mine.groupby("lap_number")["lap_distance_m"].transform("max").to_numpy(dtype=float)
        rel = np.clip(np.divide(in_lap_m, lap_length, out=np.zeros_like(in_lap_m), where=lap_length > 0), 0.0, 1.0)
        # distance since lights out: the laps before this one, plus the distance into this lap
        per_lap = mine.groupby("lap_number")["lap_distance_m"].max().sort_index()
        before = per_lap.cumsum().shift(1, fill_value=0.0)
        dist = mine["lap_number"].map(before).to_numpy(dtype=float) + in_lap_m

        tyres = self.silver_laps[self.silver_laps["driver_number"].astype(str) == number].set_index("lap_number")
        compound = mine["lap_number"].map(tyres["compound"]) if len(tyres) else pd.Series(pd.NA, index=mine.index)
        life = mine["lap_number"].map(tyres["tyre_life"]) if len(tyres) else pd.Series(np.nan, index=mine.index)
        t = mine["session_time_sec"].to_numpy(dtype=float)
        return {
            "code": self._codes[number],
            "data": {
                "t": t, "x": mine["x"].to_numpy(dtype=float), "y": mine["y"].to_numpy(dtype=float),
                "dist": dist, "rel_dist": rel, "lap": lap,
                "tyre": np.array([get_tyre_compound_int(c) if pd.notna(c) else -1 for c in compound], dtype=float),
                "tyre_life": pd.to_numeric(life, errors="coerce").fillna(0).to_numpy(dtype=float),
                "speed": mine["speed_kmh"].to_numpy(dtype=float),
                "gear": mine["gear"].astype("float64").fillna(0).to_numpy(),
                "drs": mine["drs"].astype("float64").fillna(0).to_numpy(),
                "throttle": mine["throttle_pct"].astype("float64").fillna(0).to_numpy(),
                "brake": mine["brake"].astype("float64").fillna(0).to_numpy(),
                "progress": (lap - 1) + rel,
            },
            "t_min": float(t.min()), "t_max": float(t.max()), "max_lap": int(lap.max()),
        }

    def timing_stream(self) -> pd.DataFrame:
        """
        The official order and gaps over time, in the columns of FastF1's timing stream: Time, Driver (car number),
        Position, GapToLeader, IntervalToPositionAhead. Position changes come from OpenF1's position table, the gaps
        from its intervals table (about every 4 s per car); each row carries what that table said at that moment.
        """
        columns = ["Time", "Driver", "Position", "GapToLeader", "IntervalToPositionAhead"]
        parts = []
        position = self._tables.get("position")
        if position is not None and len(position):
            parts.append(pd.DataFrame({"t": seconds_since(position["date"], self.t0_date),
                                       "Driver": position["driver_number"].astype(int).astype(str),
                                       "Position": position["position"].astype(float)}))
        gaps_path = lake.table_path(lake.BRONZE, lake.openf1_table("intervals"), self.year, self.round_number,
                                    self.session_type, self.root)
        if gaps_path.exists():
            gaps = lake.read_table(gaps_path)
            parts.append(pd.DataFrame({"t": seconds_since(gaps["date"], self.t0_date),
                                       "Driver": gaps["driver_number"].astype(int).astype(str),
                                       "GapToLeader": gaps["gap_to_leader"].map(_gap_text),
                                       "IntervalToPositionAhead": gaps["interval"].map(_gap_text)}))
        if not parts:
            return pd.DataFrame(columns=columns)
        rows = pd.concat(parts, ignore_index=True).sort_values("t", kind="stable")
        for column in ("Position", "GapToLeader", "IntervalToPositionAhead"):
            if column not in rows:
                rows[column] = np.nan
        held = rows.groupby("Driver")[["Position"]].ffill()       # a gap row carries the position held at that time
        rows["Position"] = held["Position"]
        rows["Time"] = pd.to_timedelta(rows["t"], unit="s")
        return rows.dropna(subset=["Position"])[columns].reset_index(drop=True)

    def driver_colors(self) -> Dict[str, tuple]:
        """{driver code: (r, g, b)} from the team colour OpenF1 gives each driver."""
        out = {}
        for code, colour in self._colours.items():
            colour = colour.lstrip("#")
            if len(colour) == 6:
                out[code] = tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))
        return out

    def example_lap(self) -> Optional[pd.DataFrame]:
        """
        The fastest clean lap without a telemetry hole (failing that, the fastest timed lap that is whole), as
        FastF1's lap.get_telemetry() gives one: X, Y, Distance,
        RelativeDistance, Speed, DRS, nGear, Throttle, Brake, SessionTime. The replay draws the track from it.
        (FastF1's replay preferred a qualifying lap, for the DRS zones; the lake has races only so far.)
        """
        laps = self.silver_laps
        tel = self.telemetry
        timed = laps[laps["lap_time_sec"].notna() & ~laps["is_pit_lap"] & ~laps["is_lap1"]
                     & laps["driver_number"].astype(str).isin(set(tel["driver_number"]))].sort_values("lap_time_sec")
        # clean laps first; if none is whole (a race run mostly behind the safety car), any timed lap that is
        candidates = pd.concat([timed[timed["is_clean_lap"]].head(40), timed[~timed["is_clean_lap"]].head(40)])
        for row in candidates.itertuples():
            lap = tel[(tel["driver_number"] == str(row.driver_number)) & (tel["lap_number"] == row.lap_number)]
            if len(lap) > 100 and not lap["is_data_gap"].any() and lap[["x", "y", "speed_kmh"]].notna().all().all():
                length = float(lap["lap_distance_m"].max())
                return pd.DataFrame({
                    "X": lap["x"].to_numpy(dtype=float), "Y": lap["y"].to_numpy(dtype=float),
                    "Distance": lap["lap_distance_m"].to_numpy(dtype=float),
                    "RelativeDistance": lap["lap_distance_m"].to_numpy(dtype=float) / length if length else 0.0,
                    "Speed": lap["speed_kmh"].to_numpy(dtype=float),
                    "DRS": lap["drs"].astype("float64").fillna(0).to_numpy().astype(int),
                    "nGear": lap["gear"].astype("float64").fillna(0).to_numpy().astype(int),
                    "Throttle": lap["throttle_pct"].astype("float64").fillna(0).to_numpy(),
                    "Brake": lap["brake"].astype("float64").fillna(0).to_numpy().astype(bool),
                    "SessionTime": pd.to_timedelta(lap["session_time_sec"].to_numpy(dtype=float), unit="s"),
                })
        return None

    def circuit_rotation(self) -> float:
        """Degrees to turn the map so that it looks like the broadcast's. From MultiViewer, kept in the lake after the
        first time (src/lib/race_calendar.py). Without it the map is simply not turned."""
        try:
            from src.lib import race_calendar
            key = int(self._tables["sessions"]["circuit_key"].iloc[0])
            return race_calendar.rotation(self.year, key, self.root)
        except Exception as err:
            print(f"Circuit rotation not available ({type(err).__name__}: {err}); the map is not turned")
            return DEFAULT_ROTATION_DEG


def _gap_text(value: Any) -> Any:
    """OpenF1's gap (1.234, or "+1 LAP") as the timing stream writes it ("+1.234", "1L"); nothing stays nothing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().upper()
    if "LAP" in text:
        digits = "".join(ch for ch in text if ch.isdigit()) or "1"
        return f"{digits}L"
    try:
        return f"+{float(text):.3f}"
    except ValueError:
        return np.nan
