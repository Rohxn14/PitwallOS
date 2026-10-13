"""
Race overview (src/analytics/race_overview.py)

What the Analyze window shows today: facts the lake already holds and the Phase 0 tests pin
down. No model and no estimate, so nothing here can be "wrong on purpose":

- classification: the official result (Bronze results; the classification exists only there,
                  raw, until a Silver results table is added)
- stints:         one row per tyre stint (Silver laps: compound, laps, tyre age when fitted,
                  how it ended)
- strategies:     per driver, the compound sequence and number of stops; and which drivers ran
                  the same strategy (a retirement's strategy was never completed, so it is marked
                  and does not count towards the strategy's best result)
- data quality:   what the pipeline flagged for this race (clean, neutralised, wet, pit laps);
                  defined in src/pipeline/quality.py, shared with the backfill report

Everything that has to be estimated (pace, pit loss, fuel-corrected degradation, dirty air,
undercuts) is Phase 2, step by step, each with its own golden test: each has its own tab, and the
race report (src/analytics/report.py) puts them on one page. The old engine that guessed them
(race_engine.py, wrong in every output) was deleted in step 6.

The replay is not needed for any of this: everything is read from the lake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.lib.running_order import took_the_flag
from src.pipeline import lake
from src.pipeline.quality import data_quality  # one definition, shared with the backfill report

COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}

# How a stint ended: Silver's pit_kind on its last lap, or the end of the driver's race.
END_TYRE_CHANGE = "tyre change"
END_RETIRED_IN_PITS = "ended in the pits"
END_UNKNOWN_PASSAGE = "unknown pit passage"
END_RACE = "end of race"

DNF_MARK = "†"


@dataclass
class LakeStatus:
    """Which layers of the lake hold this race."""
    year: int
    round_number: int
    bronze: bool
    silver: bool
    gold: bool
    event_name: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    event_date: Optional[str] = None
    total_laps: Optional[int] = None
    sizes_mb: Dict[str, float] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """The overview needs Bronze (results, manifest) and Silver laps."""
        return self.bronze and self.silver


def _folder_mb(path) -> float:
    if not path.exists():
        return 0.0
    return round(sum(p.stat().st_size for p in path.glob("*") if p.is_file()) / 1024 / 1024, 2)


def lake_status(year: int, round_number: int, session: str = "R",
                root: str = lake.DEFAULT_LAKE_ROOT) -> LakeStatus:
    bronze_manifest = lake.session_dir(lake.BRONZE, year, round_number, session, root) / lake.MANIFEST_NAME
    silver_laps = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    gold = lake.gold_features_path(year, round_number, session, root)
    status = LakeStatus(year=year, round_number=round_number, bronze=bronze_manifest.exists(),
                        silver=silver_laps.exists(), gold=gold.exists())
    if status.bronze:
        manifest = lake.read_manifest(year, round_number, session, root)
        status.event_name = manifest.get("event_name")
        status.location = manifest.get("location")
        status.country = manifest.get("country")
        status.event_date = str(manifest.get("event_date", ""))[:10] or None
        status.total_laps = manifest.get("total_laps")
    status.sizes_mb = {layer: _folder_mb(lake.session_dir(layer, year, round_number, session, root))
                       for layer in lake.LAYERS}
    return status


# --- classification -----------------------------------------------------------

def _int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _race_time(seconds: float) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:06.3f}"


def classification(results: pd.DataFrame) -> pd.DataFrame:
    """
    The official result, one row per driver, in result order.

    `result` is what the timing screens print: the race time for the winner, the gap for a
    lead-lap finisher, laps down for a lapped car, the status otherwise. FastF1's `Time` is a
    gap to the winner only for lead-lap finishers, so it is used for nothing else.
    `finished` means took the chequered flag (src/lib/running_order.took_the_flag).
    """
    df = results.copy()
    df["_pos"] = df["Position"].map(_int)
    df = df.sort_values("_pos", na_position="last").reset_index(drop=True)
    winner_laps = _int(df["Laps"].iloc[0]) if len(df) else None

    rows = []
    for _, r in df.iterrows():
        status = _text(r.get("Status"))
        laps = _int(r.get("Laps"))
        time = r.get("Time")
        seconds = time.total_seconds() if pd.notna(time) and hasattr(time, "total_seconds") else None
        on_lead_lap = status.lower() == "finished"
        if r["_pos"] == 1 and on_lead_lap and seconds is not None:
            result = _race_time(seconds)
        elif on_lead_lap and seconds is not None and laps == winner_laps:
            result = f"+{seconds:.3f}s"
        elif winner_laps is not None and laps is not None and laps < winner_laps and (
                status.lower() == "lapped" or status.startswith("+")):
            down = winner_laps - laps
            result = f"+{down} lap" + ("s" if down > 1 else "")
        else:
            result = status
        rows.append({
            "position": r["_pos"],
            "classified": _text(r.get("ClassifiedPosition")),
            "driver_code": _text(r.get("Abbreviation")),
            "driver_number": _text(r.get("DriverNumber")),
            "driver_name": _text(r.get("FullName", None)),
            "team": _text(r.get("TeamName", None)),
            "team_color": _text(r.get("TeamColor", None)),
            "grid": _int(r.get("GridPosition")),
            "status": status,
            "finished": took_the_flag(status, finished_by_laps=False),
            "laps": laps,
            "result": result,
            "points": None if pd.isna(r.get("Points", None)) else float(r.get("Points")),
        })
    out = pd.DataFrame(rows)
    for col in ("position", "grid", "laps"):
        out[col] = out[col].astype("Int64")
    return out


# --- stints and strategies -------------------------------------------------------

def stints(laps: pd.DataFrame) -> pd.DataFrame:
    """
    One row per tyre stint per driver, in race order.

    A stint is a run of consecutive laps on one set of tyres. Silver's `stint` counts tyre
    changes only, so a pass-through (the safety car leading the field down the pit lane, a
    drive-through) does not start a new one. Where Silver could not tell (`stint` is null after
    an 'unknown' passage), a change of compound still starts a new stint and stint_number stays
    null: unknown is not guessed.
    """
    cols = ["driver_code", "stint_number", "compound", "first_lap", "last_lap", "laps",
            "tyre_age_at_fit", "fresh", "pass_throughs", "ended_by"]
    if laps.empty:
        return pd.DataFrame(columns=cols)

    df = laps.sort_values(["driver_code", "lap_number"]).reset_index(drop=True)
    known = df["stint"].astype("Int64").astype("string")
    key = known.where(df["stint"].notna(), "?" + df["compound"].astype("string").fillna("?"))
    new_run = (df["driver_code"] != df["driver_code"].shift()) | (key != key.shift())
    df["_run"] = new_run.cumsum()

    rows = []
    for _, run in df.groupby("_run", sort=True):
        first, last = run.iloc[0], run.iloc[-1]
        compounds = run["compound"].dropna()
        kind = last.get("pit_kind")
        if pd.isna(kind):
            ended = END_RACE
        elif kind == "tyre_change":
            ended = END_TYRE_CHANGE
        elif kind == "ended":
            ended = END_RETIRED_IN_PITS
        elif kind == "unknown":
            ended = END_UNKNOWN_PASSAGE
        else:
            ended = END_RACE        # a pass-through on the last lap of the data
        tyre_life = first.get("tyre_life")
        rows.append({
            "driver_code": str(first["driver_code"]),
            "stint_number": first["stint"],
            "compound": compounds.iloc[0] if len(compounds) else None,
            "first_lap": int(first["lap_number"]),
            "last_lap": int(last["lap_number"]),
            "laps": int(len(run)),
            "tyre_age_at_fit": None if pd.isna(tyre_life) else int(tyre_life) - 1,
            "fresh": None if pd.isna(first.get("fresh_tyre")) else bool(first.get("fresh_tyre")),
            "pass_throughs": int((run["pit_kind"] == "pass_through").sum()),
            "ended_by": ended,
        })
    out = pd.DataFrame(rows, columns=cols)
    out["stint_number"] = out["stint_number"].astype("Int64")
    out["tyre_age_at_fit"] = out["tyre_age_at_fit"].astype("Int64")
    return out


def strategy_label(compounds: List[Optional[str]]) -> str:
    """Compound sequence as letters, e.g. ['MEDIUM', 'HARD'] -> 'M-H' ('?' for an unknown set)."""
    return "-".join(COMPOUND_LETTER.get(str(c).upper(), "?") if c else "?" for c in compounds)


def strategies(stint_table: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    """
    One row per driver, in result order: strategy label, stops, stint lengths, result.

    stops = tyre changes (Silver's is_tyre_change), the same count the Phase 0 golden tests pin.
    """
    rows = []
    order = result["driver_code"].tolist() if len(result) else sorted(stint_table["driver_code"].unique())
    by_driver = {code: grp for code, grp in stint_table.groupby("driver_code")}
    res = result.set_index("driver_code") if len(result) else pd.DataFrame()
    for code in order:
        grp = by_driver.get(code)
        if grp is None:
            continue
        known = code in res.index
        rows.append({
            "position": res.loc[code, "position"] if known else pd.NA,
            "driver_code": code,
            "strategy": strategy_label(grp["compound"].tolist()),
            "stops": int((grp["ended_by"] == END_TYRE_CHANGE).sum()),
            "stint_laps": "-".join(str(n) for n in grp["laps"]),
            "pit_laps": ", ".join(str(n) for n in grp.loc[grp["ended_by"] == END_TYRE_CHANGE, "last_lap"]),
            "finished": bool(res.loc[code, "finished"]) if known else False,
            "result": res.loc[code, "result"] if known else "",
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["position"] = out["position"].astype("Int64")
    return out


def strategy_groups(strategy_table: pd.DataFrame) -> pd.DataFrame:
    """
    Which drivers ran each (compound sequence, stops) strategy.

    Retirements are listed with a DNF_MARK and do not count: their strategy was never
    completed, so `finishers` and `best_result` use finishers only. Most-finished first.
    """
    cols = ["strategy", "stops", "drivers", "finishers", "best_result"]
    if strategy_table.empty:
        return pd.DataFrame(columns=cols)
    groups = []
    for (label, stops), grp in strategy_table.groupby(["strategy", "stops"], sort=False):
        done = grp[grp["finished"]]
        positions = done["position"].dropna()
        groups.append({
            "strategy": label,
            "stops": int(stops),
            "drivers": ", ".join(c if f else c + DNF_MARK for c, f in zip(grp["driver_code"], grp["finished"])),
            "finishers": int(len(done)),
            "best_result": int(positions.min()) if len(positions) else None,
        })
    out = pd.DataFrame(groups, columns=cols)
    out["_best"] = out["best_result"].fillna(999)
    out = out.sort_values(["finishers", "_best"], ascending=[False, True]).drop(columns="_best")
    return out.reset_index(drop=True)


# --- everything for one race ----------------------------------------------------------

@dataclass
class RaceOverview:
    status: LakeStatus
    classification: pd.DataFrame
    stints: pd.DataFrame
    strategies: pd.DataFrame
    strategy_groups: pd.DataFrame
    quality: Dict[str, int]
    warnings: List[str]


def load_overview(year: int, round_number: int, session: str = "R",
                  root: str = lake.DEFAULT_LAKE_ROOT) -> RaceOverview:
    """Read one race from the lake and shape it for the Analyze window. Needs Bronze + Silver."""
    status = lake_status(year, round_number, session, root)
    if not status.ready:
        raise FileNotFoundError(f"{year} round {round_number} is not in the lake yet (Bronze + Silver needed).")
    results = lake.read_table(lake.table_path(lake.BRONZE, "results", year, round_number, session, root))
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", year, round_number, session, root))
    manifest = lake.read_manifest(year, round_number, session, root)

    result = classification(results)
    stint_table = stints(laps)
    strategy_table = strategies(stint_table, result)
    return RaceOverview(
        status=status,
        classification=result,
        stints=stint_table,
        strategies=strategy_table,
        strategy_groups=strategy_groups(strategy_table),
        quality=data_quality(laps),
        warnings=list(manifest.get("warnings", [])),
    )
