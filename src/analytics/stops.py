"""
Pit stops and pit loss (src/analytics/stops.py) — Phase 2 step 2

One row per tyre change (Silver's `is_tyre_change`, on the in-lap) and what it cost.

    loss = in-lap time + out-lap time - reference for those two laps

The reference depends on what the race was doing (Rohan's design note of 2026-10-12, after one
round of pushback):

  green, yellow   the driver's own pace: 2 x the median of their pace laps (Phase 2 step 1) in the
                  5 laps before the in-lap and the 5 laps after the out-lap. What they would have
                  lapped instead, at that point of the race (a whole-race median mixes heavy- and
                  light-fuel laps and was measurably biased by when the stop happened).
  SC, VSC, lap 1  the field: the median lap time, on the in-lap's and the out-lap's numbers, of the
                  other cars that did not pass through the pit lane on that lap. Under a safety car
                  both laps are slow for everybody, so only the difference to the cars that stayed
                  out is the cost of stopping (against the driver's own pace a safety-car stop
                  would "cost" ~71 s).
  red flag        no loss: tyres are changed during the stoppage, at no cost relative to anyone,
                  and the in-lap holds the stoppage itself (~2,470 s) or no time at all. The row
                  stays (every stop is kept, Rohan's note); its loss is unknown, with the reason.

The loss is also unknown (never guessed, rule 2) when the in-lap or out-lap has no time, when
there are too few laps to form the reference, or when the out-lap is itself an in-lap (two
passages back to back: a penalty, then a stop).

The circuit's pit loss is the median loss of its GREEN stops. It is per race, never one number
for every circuit: before building, per-race medians over 2023-2025 ran from 18 s (Spa, Miami)
to 29.5 s (Singapore), while stops inside one race agree to about 1.8 s (IQR).

Strategic stops exclude red-flag tyre changes (free, and made by nearly the whole field).

Run it over every race in the lake:

    python -m src.analytics.stops --season 2023 2024 2025
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analytics.laps import classify_laps
from src.pipeline import lake

LOCAL_WINDOW_LAPS = 5        # driver reference: pace laps up to 5 laps before the in-lap and after the out-lap
MIN_REFERENCE_LAPS = 3       # fewer pace laps than this around the stop: no reference
MIN_FIELD_CARS = 3           # field reference: at least this many cars that stayed out, on each lap

GREEN, YELLOW, SAFETY_CAR, VSC, LAP_1, RED_FLAG = "green", "yellow", "safety car", "VSC", "lap 1", "red flag"
CONDITIONS = (GREEN, YELLOW, SAFETY_CAR, VSC, LAP_1, RED_FLAG)
DRIVER_REFERENCE, FIELD_REFERENCE = "driver pace", "field"
REFERENCE_OF = {GREEN: DRIVER_REFERENCE, YELLOW: DRIVER_REFERENCE, SAFETY_CAR: FIELD_REFERENCE,
                VSC: FIELD_REFERENCE, LAP_1: FIELD_REFERENCE, RED_FLAG: None}

STOP_COLUMNS = [
    "driver_code", "team", "stop_number", "in_lap", "out_lap", "condition", "compound_before",
    "compound_after", "tyre_age_before", "in_lap_sec", "out_lap_sec", "reference_kind", "reference_sec",
    "reference_n", "loss_sec", "loss_note",
]


def stop_condition(in_lap: pd.Series, out_lap: pd.Series) -> str:
    """What the race was doing during the stop: the first that applies over the in-lap and the out-lap."""
    either = lambda flag: bool(in_lap[flag]) or bool(out_lap[flag])
    if either("is_red"):
        return RED_FLAG
    if int(in_lap["lap_number"]) == 1:
        return LAP_1
    if either("is_sc"):
        return SAFETY_CAR
    if either("is_vsc"):
        return VSC
    if bool(in_lap["is_green"]) and bool(out_lap["is_green"]):
        return GREEN
    return YELLOW


def _driver_reference(laps: pd.DataFrame, in_lap: int, out_lap: int):
    """2 x the median of the driver's pace laps around the stop, and how many laps it used."""
    near = laps[laps["is_pace_lap"]
                & (laps["lap_number"].between(in_lap - LOCAL_WINDOW_LAPS, in_lap - 1)
                   | laps["lap_number"].between(out_lap + 1, out_lap + LOCAL_WINDOW_LAPS))]
    if len(near) < MIN_REFERENCE_LAPS:
        return np.nan, len(near)
    return 2.0 * float(near["lap_time_sec"].median()), len(near)


def _field_reference(all_laps: pd.DataFrame, driver: str, in_lap: int, out_lap: int):
    """Sum over the in-lap and out-lap numbers of the median lap time of the cars that stayed out."""
    field = all_laps[(all_laps["driver_code"] != driver) & all_laps["lap_number"].isin([in_lap, out_lap])
                     & ~all_laps["is_pit_lap"].astype(bool) & all_laps["lap_time_sec"].notna()]
    counts = field.groupby("lap_number").size()
    cars = int(counts.min()) if len(counts) == 2 else 0
    if cars < MIN_FIELD_CARS:
        return np.nan, cars
    return float(field.groupby("lap_number")["lap_time_sec"].median().sum()), cars


def stops(classified: pd.DataFrame) -> pd.DataFrame:
    """
    One row per tyre change, in race order of each driver. `classified` = laps.classify_laps(silver).

    Columns: STOP_COLUMNS. reference_sec is the reference for the two laps together; reference_n is
    how many pace laps (driver pace) or cars (field) it rests on. loss_sec is null with loss_note
    saying why whenever it cannot be measured.
    """
    rows: List[Dict[str, Any]] = []
    for driver, laps in classified.groupby("driver_code", sort=True):
        laps = laps.sort_values("lap_number")
        by_lap = laps.set_index("lap_number")
        number = 0
        for _, in_lap in laps[laps["is_tyre_change"].astype(bool)].iterrows():
            number += 1
            lap = int(in_lap["lap_number"])
            out_lap = by_lap.loc[lap + 1] if lap + 1 in by_lap.index else None
            row = {"driver_code": str(driver), "team": in_lap.get("team"), "stop_number": number, "in_lap": lap,
                   "out_lap": lap + 1 if out_lap is not None else pd.NA,
                   "compound_before": in_lap["compound"], "tyre_age_before": in_lap["tyre_life"],
                   "compound_after": out_lap["compound"] if out_lap is not None else pd.NA,
                   "in_lap_sec": in_lap["lap_time_sec"],
                   "out_lap_sec": out_lap["lap_time_sec"] if out_lap is not None else np.nan,
                   "reference_kind": None, "reference_sec": np.nan, "reference_n": 0, "loss_sec": np.nan,
                   "loss_note": None}
            if out_lap is None:                 # Silver only marks a tyre change when a next lap exists
                row.update(condition=None, loss_note="no out-lap")
                rows.append(row)
                continue
            condition = stop_condition(in_lap, out_lap)
            kind = REFERENCE_OF[condition]
            row.update(condition=condition, reference_kind=kind)
            if condition == RED_FLAG:
                row["loss_note"] = "red flag: tyres changed during the stoppage"
            elif pd.isna(row["in_lap_sec"]) or pd.isna(row["out_lap_sec"]):
                row["loss_note"] = "no lap time on the in-lap or out-lap"
            elif bool(out_lap["is_in_lap"]):
                row["loss_note"] = "back-to-back pit passages"
            else:
                if kind == DRIVER_REFERENCE:
                    ref, n = _driver_reference(laps, lap, lap + 1)
                    short = f"fewer than {MIN_REFERENCE_LAPS} pace laps around the stop"
                else:
                    ref, n = _field_reference(classified, str(driver), lap, lap + 1)
                    short = f"fewer than {MIN_FIELD_CARS} cars stayed out on these laps"
                row.update(reference_sec=ref, reference_n=n)
                if np.isnan(ref):
                    row["loss_note"] = short
                else:
                    row["loss_sec"] = float(row["in_lap_sec"] + row["out_lap_sec"] - ref)
            rows.append(row)
    out = pd.DataFrame(rows, columns=STOP_COLUMNS)
    out["out_lap"] = out["out_lap"].astype("Int64")
    out["tyre_age_before"] = out["tyre_age_before"].astype("Int64")
    out["reference_n"] = out["reference_n"].astype("int64")
    return out.sort_values(["in_lap", "driver_code"]).reset_index(drop=True)


def pit_loss(stop_table: pd.DataFrame) -> pd.DataFrame:
    """
    The race's pit loss by condition: one row per condition that has stops, with stops (all), measured
    (with a loss), median_sec, q25_sec, q75_sec. The GREEN row's median is the circuit's pit loss.
    """
    rows = []
    for condition in CONDITIONS:
        group = stop_table[stop_table["condition"] == condition]
        if group.empty:
            continue
        loss = group["loss_sec"].dropna()
        rows.append({"condition": condition, "stops": int(len(group)), "measured": int(len(loss)),
                     "median_sec": float(loss.median()) if len(loss) else np.nan,
                     "q25_sec": float(loss.quantile(0.25)) if len(loss) else np.nan,
                     "q75_sec": float(loss.quantile(0.75)) if len(loss) else np.nan})
    return pd.DataFrame(rows, columns=["condition", "stops", "measured", "median_sec", "q25_sec", "q75_sec"])


def stop_counts(stop_table: pd.DataFrame) -> pd.DataFrame:
    """Per driver: tyre changes, strategic stops (red-flag tyre changes left out) and the in-laps."""
    rows = []
    for driver, group in stop_table.groupby("driver_code", sort=True):
        rows.append({"driver_code": driver, "tyre_changes": int(len(group)),
                     "strategic_stops": int((group["condition"] != RED_FLAG).sum()),
                     "in_laps": ", ".join(str(n) for n in group["in_lap"])})
    return pd.DataFrame(rows, columns=["driver_code", "tyre_changes", "strategic_stops", "in_laps"])


# --- one race, every race -----------------------------------------------------------------

@dataclass
class RaceStops:
    """Step 2 for one race."""
    stops: pd.DataFrame        # stops()
    pit_loss: pd.DataFrame     # pit_loss()
    counts: pd.DataFrame       # stop_counts()

    @property
    def green_loss(self) -> Optional[float]:
        """The circuit's pit loss: median loss of the green stops (None if there were none)."""
        green = self.pit_loss[self.pit_loss["condition"] == GREEN]
        return None if green.empty or pd.isna(green["median_sec"].iloc[0]) else float(green["median_sec"].iloc[0])


def race_stops(classified: pd.DataFrame) -> RaceStops:
    """Everything step 2 says about one race, from its classified laps (laps.classify_laps). Pure."""
    table = stops(classified)
    return RaceStops(stops=table, pit_loss=pit_loss(table), counts=stop_counts(table))


def load_race_stops(year: int, round_number: int, session: str = "R",
                    root: str = lake.DEFAULT_LAKE_ROOT) -> RaceStops:
    path = lake.table_path(lake.SILVER, "laps", year, round_number, session, root)
    if not path.exists():
        raise FileNotFoundError(f"{year} round {round_number}: no Silver laps in the lake ({path}).")
    return race_stops(classify_laps(lake.read_table(path)))


def season_summary(seasons: Sequence[int], root: str = lake.DEFAULT_LAKE_ROOT) -> pd.DataFrame:
    """One row per race: stops by condition and the circuit's pit loss with its spread."""
    wanted = {int(s) for s in seasons}
    rows = []
    for path in lake.list_sessions(lake.SILVER, "laps", root):
        if "session=R" not in path.parts:
            continue
        laps = lake.read_table(path)
        if laps.empty or int(laps["season"].iloc[0]) not in wanted:
            continue
        season, round_number = int(laps["season"].iloc[0]), int(laps["round"].iloc[0])
        result = race_stops(classify_laps(laps))
        loss = result.pit_loss.set_index("condition")
        get = lambda cond, col, default=np.nan: loss.loc[cond, col] if cond in loss.index else default
        try:
            event = lake.read_manifest(season, round_number, "R", root).get("event_name")
        except FileNotFoundError:
            event = None
        rows.append({
            "season": season, "round": round_number, "event_name": event,
            "tyre_changes": int(len(result.stops)), "unmeasured": int(result.stops["loss_sec"].isna().sum()),
            "green_stops": int(get(GREEN, "measured", 0)), "green_loss_sec": get(GREEN, "median_sec"),
            "green_q25_sec": get(GREEN, "q25_sec"), "green_q75_sec": get(GREEN, "q75_sec"),
            "sc_stops": int(get(SAFETY_CAR, "measured", 0)), "sc_loss_sec": get(SAFETY_CAR, "median_sec"),
            "vsc_stops": int(get(VSC, "measured", 0)), "vsc_loss_sec": get(VSC, "median_sec"),
            "red_flag_stops": int(get(RED_FLAG, "stops", 0)),
        })
    return pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True) if rows else pd.DataFrame()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 step 2 over every race in the lake: stops and pit loss")
    parser.add_argument("--season", type=int, nargs="+", required=True, help="one or more seasons, e.g. 2023 2024")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT, help="lake root folder (default: data)")
    args = parser.parse_args(argv)

    table = season_summary(args.season, args.lake)
    if table.empty:
        print("No Silver laps in the lake for these seasons.")
        return 1
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))
    green = table["green_loss_sec"].dropna()
    middle = lambda col: (f"{table[col].dropna().median():.1f} s" if table[col].notna().any() else "none")
    print(f"\n{len(table)} races, {table['tyre_changes'].sum():,} tyre changes ({table['unmeasured'].sum():,} without a "
          f"measurable loss). Circuit pit loss (median of green stops): median {green.median():.1f} s over races, "
          f"from {green.min():.1f} to {green.max():.1f} s. Safety-car stops: median {middle('sc_loss_sec')}, "
          f"VSC: {middle('vsc_loss_sec')} (medians of race medians).")
    print(f"Report: {lake.write_report(table, 'pit_loss', args.lake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
