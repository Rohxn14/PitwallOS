"""Small made-up races for the Phase 3 tests: hand-picked lap times, and races built from the formula itself
(optionally with wander, noise, and an effect the formula does not know about)."""

import numpy as np
import pandas as pd

from src.analytics.laps import classify_laps
from src.models import evaluate
from src.models.evaluate import attach_answers
from src.models.linear import lap_terms
from src.pipeline.cleaner import clean_laps
from src.pipeline.feature_store import build_lap_features
from tests.helpers import make_bronze_laps, make_weather

# --- a race with hand-picked lap times, through the real pipeline --------------------------------

# Seconds per lap. Lap 1 is the standing start. BBB's lap 5 is behind the safety car. CCC runs on inters.
LAP_TIMES = {
    "AAA": [100.0, 90.0, 90.6, 91.2, 90.9, 91.5, 91.8, 92.0],
    "BBB": [101.0, 91.0, 91.0, 91.0, 120.0, 91.5, 91.5, 91.5],
    "CCC": [110.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0],
}
COMPOUNDS = {"AAA": "MEDIUM", "BBB": "MEDIUM", "CCC": "INTERMEDIATE"}
SAFETY_CAR = {("BBB", 5)}


def simulated_race(season=2024, round_number=1, event_name="Test Grand Prix"):
    """The race above as the harness sees it: Gold rows with the answers attached (evaluate.attach_answers)."""
    rows = []
    for driver, times in LAP_TIMES.items():
        start = 0.0
        for lap, lap_time in enumerate(times, start=1):
            rows.append({"driver": driver, "lap": lap, "start": start, "end": start + lap_time,
                         "lap_time": lap_time, "compound": COMPOUNDS[driver], "tyre_life": float(lap),
                         "status": "4" if (driver, lap) in SAFETY_CAR else "1"})
            start += lap_time
    silver, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), season, round_number, "R")
    gold = build_lap_features(silver, total_laps=len(LAP_TIMES["AAA"]))
    return attach_answers(gold, classify_laps(silver), event_name)


# --- races built from the lap-time formula itself (Gold-like rows, no pipeline) ------------------

# The physics (s, s per lap of tyre age, s per kg, s per lap by gap bin).
PHYSICS = {
    "offset SOFT": -0.40, "offset HARD": 0.15,
    "wear SOFT": 0.08, "wear MEDIUM": 0.05, "wear HARD": 0.03,
    "fuel": 0.03,
    "traffic 0-0.5 s": 0.70, "traffic 0.5-1 s": 0.40, "traffic 1-1.5 s": 0.25, "traffic 1.5-2 s": 0.15,
    "traffic 2-3 s": 0.10,
}
RACE_LAPS = 30
GAP_CYCLE = [0.2, 0.7, 1.2, 1.7, 2.5, 4.0, 6.0]      # the gap at the start of each lap cycles through every bin
PLANS = {                                           # driver: (base, first compound, stop lap, second compound)
    "AAA": (90.0, "SOFT", 12, "MEDIUM"),
    "BBB": (90.5, "MEDIUM", 15, "HARD"),
    "CCC": (91.0, "SOFT", 10, "HARD"),
}


def formula_race(season, round_number, base_shift=0.0, step_sd=0.0, noise_sd=0.0, seed=0, first_flying_lap_sec=0.0,
                 gap_cycle=GAP_CYCLE):
    """
    Gold-like rows of a race whose lap times follow PHYSICS, answers attached. With step_sd and noise_sd
    at 0 the times follow it exactly; otherwise each driver's base also wanders (a random step of
    step_sd every lap) and every lap gets its own noise (noise_sd): the state-space model's world.
    first_flying_lap_sec is added to the first lap after each out-lap: an effect the formula does not have.
    gap_cycle: the gaps at the start of the laps, in turn ([4.0]: clear air all race, so the gap held at
    lap N's is right any number of laps ahead).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d, (driver, (base, first, stop, second)) in enumerate(PLANS.items()):
        wander = np.cumsum(rng.normal(0.0, step_sd, RACE_LAPS)) if step_sd else np.zeros(RACE_LAPS)
        noise = rng.normal(0.0, noise_sd, RACE_LAPS) if noise_sd else np.zeros(RACE_LAPS)
        for lap in range(1, RACE_LAPS + 1):
            compound, age = (first, lap) if lap <= stop else (second, lap - stop)
            pit_lap = lap in (stop, stop + 1)
            slow = 20.0 if lap == 1 or pit_lap else 0.0     # standing start, pit lane: never clean, never scored
            effect = first_flying_lap_sec if lap == stop + 2 else 0.0
            rows.append({"season": season, "round": round_number, "driver_code": driver, "lap_number": lap,
                         "total_race_laps": RACE_LAPS, "compound": compound, "tyre_age": float(age),
                         "stint_lap": age, "fresh_tyre": 1.0,
                         "fuel_mass_est": 110.0 - (lap - 1) * 105.0 / RACE_LAPS,
                         "gap_on_track_start_sec": gap_cycle[(lap + d) % len(gap_cycle)],
                         "gap_on_track_end_sec": gap_cycle[(lap + 1 + d) % len(gap_cycle)],
                         "gap_ahead_sec": np.nan if d == 0 else 1.0, "gap_to_leader_sec": float(d),
                         "position": float(d + 1), "roll3_clean_std_sec": np.nan,
                         "is_clean_lap": lap > 1 and not pit_lap, "is_wet_tyre": False,
                         "is_in_lap": lap == stop, "is_out_lap": lap == stop + 1,
                         "is_yellow": False, "is_sc": False, "is_vsc": False, "is_red": False,
                         "event_name": "Formula Grand Prix",
                         "base": base + base_shift + wander[lap - 1] + noise[lap - 1] + slow + effect})
    laps = pd.DataFrame(rows)
    terms = lap_terms(laps["compound"], laps["tyre_age"], laps["fuel_mass_est"], laps["gap_on_track_start_sec"])
    laps["lap_time_sec"] = laps.pop("base") + terms[list(PHYSICS)] @ pd.Series(PHYSICS)
    by_driver = laps.groupby("driver_code")
    clean_time = laps["lap_time_sec"].where(laps["is_clean_lap"])
    laps["last_clean_lap_time_sec"] = clean_time.groupby(laps["driver_code"]).transform(lambda s: s.shift(1).ffill())
    laps[evaluate.TARGET] = by_driver["lap_time_sec"].shift(-1)
    laps["target_next_is_clean"] = by_driver["is_clean_lap"].shift(-1, fill_value=False).astype(bool)
    laps["scored"] = laps["target_next_is_clean"] & (laps["lap_number"] >= 2)
    laps["next_lap_status"] = np.where(laps["target_next_is_clean"], "pace", "pit lap")
    laps["next_lap_compound"] = by_driver["compound"].shift(-1)
    laps["conditions"] = "dry"
    laps["score_status"] = np.where(laps["scored"], evaluate.SCORED, "not scored")
    return laps
