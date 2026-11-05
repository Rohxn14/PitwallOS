"""Small hand-built races for the Phase 4 engine tests: a few cars, laps of known pace, chosen events."""

import numpy as np
import pandas as pd

from src.simulator.race import DEPLOY, END, FULL, GREEN, SINGLE
from src.simulator.rules import Rules

# Rules with round numbers, so every expected clock can be worked out by hand.
RULES = Rules(pit_loss_sec=20.0, in_lap_share=0.2, sc_pit_loss_sec=10.0, vsc_pit_loss_sec=12.0,
              queue_gap_sec=0.5, follow_gap_sec=0.7,
              sc_leader_ratio={DEPLOY: 1.3, FULL: 1.6, END: 1.6, SINGLE: 1.3},
              sc_catch_ratio={DEPLOY: 1.3, FULL: 1.2, END: 1.2, SINGLE: 1.3},
              vsc_ratio={DEPLOY: 1.3, FULL: 1.4, END: 1.1, SINGLE: 1.3},
              pass_max=0.8, pass_slope=4.0, pass_midpoint=1.0, pass_train_factor=1.0)    # no DRS trains unless asked


def race_inputs(cars, total_laps, pits=None, conditions=None, last_lap=None, laps_done=0):
    """
    The engine's inputs for a made-up race.

      cars        {driver: (clock when the start lap ended, pace)}: pace is one number (every lap) or a list
      total_laps  the race distance
      pits        {driver: {in-lap: "tyre_change" or "pass_through"}}
      conditions  {lap: (kind, phase)} for the leader's laps that are not green
      last_lap    {driver: its last lap} for a car that retires
      laps_done   laps completed when the start lap ended: one number, or {driver: laps} (a lapped car)
    Returns (start, planned, pace, conditions) for engine.simulate.
    """
    pits, last_lap = pits or {}, last_lap or {}
    done = laps_done if isinstance(laps_done, dict) else {d: laps_done for d in cars}
    rows = []
    for driver, (clock, pace) in cars.items():
        stops = pits.get(driver, {})
        for lap in range(done[driver] + 1, last_lap.get(driver, total_laps) + 1):
            p = pace if np.isscalar(pace) else pace[lap - done[driver] - 1]
            rows.append({"driver_code": driver, "lap_number": lap, "compound": "MEDIUM", "tyre_age": float(lap),
                         "stint_lap": lap, "fresh_tyre": 1.0, "is_wet_tyre": False,
                         "is_in_lap": lap in stops, "is_out_lap": (lap - 1) in stops,
                         "is_tyre_change": stops.get(lap) == "tyre_change", "pit_kind": stops.get(lap), "pace": float(p)})
    planned = pd.DataFrame(rows)
    pace = planned.pop("pace")
    start = pd.DataFrame({"laps_done": [done[d] for d in cars], "clock_sec": [c for c, _ in cars.values()]},
                         index=pd.Index(list(cars), name="driver_code"))
    table = pd.DataFrame({"kind": GREEN, "phase": ""}, index=pd.RangeIndex(1, total_laps + 1, name="lap_number"))
    for lap, (kind, phase) in (conditions or {}).items():
        table.loc[lap, ["kind", "phase"]] = [kind, phase]
    return start, planned, pace, table


def clocks(simulation, driver):
    """A car's crossing times, lap by lap."""
    laps = simulation.laps[simulation.laps["driver_code"] == driver].sort_values("lap_number")
    return laps["clock_sec"].tolist()


def order_after(simulation, lap):
    """The order at the line at the end of `lap` (cars that completed it)."""
    laps = simulation.laps[simulation.laps["lap_number"] == lap].sort_values("clock_sec")
    return laps["driver_code"].tolist()
