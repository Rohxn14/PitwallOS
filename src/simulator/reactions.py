"""
Rivals' answers (src/simulator/reactions.py) - Phase 4 step 2

When a "what if" changes one driver's plan, the rivals may answer it. The answer measured here is the
one Phase 2 step 5 is about: a car stops under green, dry, less than 3 s behind the car ahead (a threat:
an undercut), and the car ahead answers by stopping on the next lap (a cover).

Measured on 2023 + 2024 (411 threats where the car ahead had not just stopped): the car ahead answered
within 1-3 laps 26.8 % of the time (next lap 60, 2 laps 30, 3 laps 20; on the same lap 43, a decision
taken together, not an answer; 4+ laps later 194; never again 64). No group of threats is answered more
often than about 4 in 10, so "the rival answers when it is more likely than not" would almost never answer.
A cover is a CHANCE, and a "what if" is simulated both ways and weighted (decisions.py).

    P(cover) = 1 / (1 + exp(-(b0 + b . standardised inputs)))      (CoverModel, fitted on training races)
    inputs: the tyre age of the car ahead, its age minus the stopping car's, the gap, laps left (capped 30)

Out of fold (5 race folds): log loss 0.581 against 0.610 for the base rate, Brier 0.198 against 0.210,
and calibrated (threats given 0.4-0.5 were answered 47 % of the time). A modest model: whether a team
covers depends mostly on its own plan, which the data cannot see. The car ahead with fresher tyres than
the stopping car almost never answers (7 %); nobody answers with under 10 laps left.

One answer per "what if": to the changed driver's FIRST new stop. The car ahead that answers moves its next
planned tyre change to the next lap (the commonest answer), fitting what it really fitted; a car ahead with
no stop left does not answer (it ran to the end, as 64 did).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics import undercuts
from src.analytics.laps import classify_laps
from src.analytics.stops import stops
from src.pipeline import lake
from src.simulator import plans
from src.simulator.race import GREEN, Race

THREAT_GAP_SEC = undercuts.MAX_GAP_SEC           # 3 s: closer than this, stopping first is the way past
LAPS_LEFT_CAP = 30
INPUTS = ["ahead_age", "age_difference", "gap", "laps_left"]
PENALTY = 1.0                                    # L2 on the standardised slopes: a little shrinkage, 368 rows
MIN_THREATS = 20                                 # fewer threats to learn from: the measured rate below, for all
DEFAULT_COVER_RATE = 0.299                       # 2023 + 2024: answered within 1-3 laps, 110 of 368 answerable threats
# Every threat status of Phase 2 step 5 in which the car ahead had not stopped first or on the same lap.
ANSWERABLE = (undercuts.ATTEMPT, undercuts.A_NEVER, undercuts.A_LATE, undercuts.A_NOT_GREEN,
              undercuts.NEUTRALISED, undercuts.NO_GAP_AFTER)


def threats(race: Race, root: str = lake.DEFAULT_LAKE_ROOT, silver: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every threat of one race (see the module docstring): INPUTS and covered (answered within 1-3 laps)."""
    if silver is None:
        silver = lake.read_table(lake.table_path(lake.SILVER, "laps", *race.key, "R", root))
    classified = classify_laps(silver)
    events = undercuts.undercut_events(classified, stops(classified))
    by = classified.set_index(["driver_code", "lap_number"])["tyre_life"]
    rows = []
    for _, e in events[events["status"].isin(ANSWERABLE)].iterrows():
        n = int(e["in_lap"])
        ahead_age = by.get((e["car_ahead"], n - 1), np.nan)
        rows.append({"season": race.season, "round": race.round, "driver_code": e["driver_code"], "car_ahead": e["car_ahead"],
                     "in_lap": n, "ahead_age": float(ahead_age) if pd.notna(ahead_age) else np.nan,
                     "age_difference": (float(ahead_age) - float(e["tyre_age_before"]))
                     if pd.notna(ahead_age) and pd.notna(e["tyre_age_before"]) else np.nan,
                     "gap": float(e["gap_before_sec"]), "laps_left": float(min(race.total_laps - n, LAPS_LEFT_CAP)),
                     "covered": bool(pd.notna(e["lag"]) and 1 <= int(e["lag"]) <= 3)})
    return pd.DataFrame(rows, columns=["season", "round", "driver_code", "car_ahead", "in_lap"] + INPUTS + ["covered"])


@dataclass
class CoverModel:
    """P(the car ahead answers a threat on the next lap): a penalised logistic on standardised INPUTS."""
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray            # intercept first
    base_rate: float
    rows: int

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> "CoverModel":
        """Fitted on these threats; with fewer than MIN_THREATS, the measured rate for every threat."""
        x = rows.dropna(subset=INPUTS)
        if len(x) < MIN_THREATS:
            k = len(INPUTS)
            return cls(mean=np.zeros(k), scale=np.ones(k), rows=len(x), base_rate=DEFAULT_COVER_RATE,
                       coef=np.r_[np.log(DEFAULT_COVER_RATE / (1 - DEFAULT_COVER_RATE)), np.zeros(k)])
        y = x["covered"].astype(float).to_numpy()
        X = x[INPUTS].to_numpy(float)
        mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])

        def penalised_minus_log_likelihood(b):
            p = np.clip(1.0 / (1.0 + np.exp(-(Z @ b))), 1e-9, 1 - 1e-9)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + PENALTY * (b[1:] ** 2).sum()

        coef = minimize(penalised_minus_log_likelihood, np.zeros(Z.shape[1]), method="L-BFGS-B").x
        return cls(mean=mean, scale=scale, coef=coef, base_rate=float(y.mean()), rows=len(x))

    def chance(self, ahead_age: float, age_difference: float, gap: float, laps_left: float) -> float:
        x = np.array([ahead_age, age_difference, gap, min(laps_left, LAPS_LEFT_CAP)], dtype=float)
        if not np.isfinite(x).all():
            return self.base_rate
        z = self.coef[0] + ((x - self.mean) / self.scale) @ self.coef[1:]
        return float(1.0 / (1.0 + np.exp(-z)))

    def chances(self, rows: pd.DataFrame) -> np.ndarray:
        return np.array([self.chance(*r) for r in rows[INPUTS].to_numpy(float)])


# --- an answer to a "what if" ---------------------------------------------------------------------

@dataclass
class Answer:
    """How the car ahead may answer the changed driver's first new stop."""
    driver: str                 # the car ahead
    stop_lap: int               # the changed driver's new in-lap
    chance: float               # P(cover)
    plan: pd.DataFrame          # the plan in which it covers (its next stop moved to stop_lap + 1)
    gap: float                  # simulated gap at the line, the lap before the stop


def new_stops(real_plan: pd.DataFrame, plan: pd.DataFrame, driver: str) -> List[int]:
    """The changed driver's tyre changes that are not in its real plan, in lap order."""
    return sorted(set(plans.stops_of(plan, driver)) - set(plans.stops_of(real_plan, driver)))


def answer(race: Race, real_plan: pd.DataFrame, plan: pd.DataFrame, driver: str, laps: pd.DataFrame,
           conditions: pd.DataFrame, model: CoverModel) -> Optional[Answer]:
    """
    The car ahead's possible answer to the driver's first new stop, from a simulation of `plan` (its
    laps): the car directly ahead on the road when the lap before the stop ended, on the same lap, less
    than THREAT_GAP_SEC ahead, the stop green and on slicks, the car ahead with a tyre change still to
    come after it. None when there is no threat to answer.
    """
    added = new_stops(real_plan, plan, driver)
    if not added:
        return None
    n = added[0]
    before = laps[laps["lap_number"] == n - 1].set_index("driver_code")["clock_sec"]
    if driver not in before.index or conditions.at[n, "kind"] != GREEN:
        return None
    ahead = before[before < before[driver]]
    if ahead.empty:
        return None
    a = ahead.idxmax()
    gap = float(before[driver] - ahead.max())
    # Lap n-1 is planned, or already history when the stop is on the first planned lap.
    rows = pd.concat([race.laps[["driver_code", "lap_number", "tyre_age", "is_wet_tyre"]],
                      plan[["driver_code", "lap_number", "tyre_age", "is_wet_tyre"]]])
    rows = rows.drop_duplicates(["driver_code", "lap_number"], keep="last").set_index(["driver_code", "lap_number"])
    if gap >= THREAT_GAP_SEC or (driver, n - 1) not in rows.index or (a, n - 1) not in rows.index:
        return None
    if bool(rows.loc[(driver, n - 1), "is_wet_tyre"]) or bool(rows.loc[(a, n - 1), "is_wet_tyre"]):
        return None
    later = [lap for lap in plans.stops_of(plan, a) if lap >= n]
    if not later or later[0] <= n + 1:
        return None                                     # no stop left, or it stops now anyway
    ahead_age = float(rows.loc[(a, n - 1), "tyre_age"])
    p = model.chance(ahead_age, ahead_age - float(rows.loc[(driver, n - 1), "tyre_age"]), gap, race.total_laps - n)
    covering = plans.move_stop(plan, a, later[0], n + 1)
    return Answer(driver=a, stop_lap=n, chance=p, plan=covering, gap=gap)
