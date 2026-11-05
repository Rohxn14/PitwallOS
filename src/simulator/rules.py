"""
The rules of racing (src/simulator/rules.py) - Phase 4 step 1

Everything the simulator does besides the pace model's lap times: what a pit stop costs, how a safety
car or a VSC slows the field and closes it up, and when a faster car gets past a slower one. Every
number is MEASURED on races (never typed in from memory), by `race_facts` on each race and `RuleBook`
over a set of races. A replay is scored on races the book was not measured on (cross-validation), so a
rule can never be tuned to the race it is judged on.

PIT STOPS (Phase 2 step 2's loss: in-lap + out-lap - 2 x the driver's pace around the stop)
    green        the circuit's pit loss: the median green stop of the same circuit in the book's races
                 (another season), else the median of all its races. Measured on 2023 + 2024 before
                 choosing: the same circuit's other season misses a race's loss by 1.5 s on average, the
                 median of all races by 2.1 s.
                 IN_LAP_SHARE of it is on the in-lap (median 18 %: the stop itself is after the line on
                 most circuits, so the out-lap holds most of the loss).
    SC / VSC     their own losses against staying out (a car in the pit lane is not slowed, the cars on
                 track are). Phase 2 step 6 found its lap-number reference unreliable there, so they are
                 CALIBRATED: replays of the 2023 + 2024 races with each driver's real pace (so only the
                 rules are wrong), a grid of SC 4-19 s x VSC 6-15 s, each value chosen on the other folds'
                 races: SC 4 s (4 of 5 folds, 7 the other), VSC 15 s (3 of 5; 12 and 9 the others), also
                 the best cell over all races (0.908 places off). The surface is flat (0.908 to 0.943);
                 Phase 2's 15 / 19 s gives 0.934. A stop behind a settled safety car costs little more
                 than staying in the queue; under a VSC the field is slowed less (x 1.37, not x 1.6), so a
                 stop costs more, and both less than a green one (22 s).
    red flag     nothing: tyres are changed while the race is stopped.
    pass-through a green drive-through costs the pit loss minus a typical stop (TYPICAL_STATIONARY_SEC);
                 under a safety car, VSC or red flag nothing (the field goes through, or nobody races).

SAFETY CAR (the leader's laps of each period, Phase 2 step 1)
    Every car's lap is its clear-air pace x a ratio (per car: lap duration / its pace before, measured
    on the cars that did not pit), by the part of the period. The lap it comes out on, every car alike
    (sc_leader_ratio = sc_catch_ratio: the median of all cars); that lap is raced until it comes out.
    Its later laps: the safety car runs where the race leader is, at the leader's pace x sc_leader_ratio
    (the leaders' median); every other car drives faster (sc_catch_ratio, the 10th percentile: cars
    catching up) until it reaches the car ahead, then follows it at queue_gap_sec (the median gap at the
    line on the last safety-car lap, 0.47 s). Nobody passes, except past a car in the pit lane.
VSC
    Every car's lap x vsc_ratio (by part). Its full laps: no passing, so the gaps hold; the lap it comes
    out on and the lap it ends on ("VSC ending", then green) are raced.
GREEN RACING
    A car that would reach the car ahead (closer than follow_gap_sec at the line) either passes it or
    is held follow_gap_sec behind (the median gap at the line of a car that started the lap within
    1.2 s and did not get past). The chance of passing on one lap rises with the pace advantage
    (seconds per lap, from the pace model): p = pass_max / (1 + exp(-slope x (advantage - midpoint))),
    fitted on every following situation of the book's races (14,920 on 2023 + 2024: 4 % at 0-0.25 s, 24 %
    at 0.5-0.75 s, 51 % at 1-1.5 s). The midpoint is the circuit's: a hard circuit needs a bigger
    advantage, not just more laps of trying (raw pass rates in the same band run from 1.6 % at Monaco to
    46 % in Spain). Each circuit's shift is pulled towards 0 by a normal prior (PASS_SHIFT_PRIOR_SEC,
    0.5 s: 0.25 / 0.5 / 1 / 2 s replay alike, 1.085 to 1.094 places; 0.5 or 1 chosen on the other folds).
    The laps of wet races on which both cars ran slicks count for passing (Monaco 2023 had rain; without
    it, Monaco 2024 replayed at the average circuit's pass rate). A car in the pit lane, or on its
    in-lap, out-lap or the lap it retires, never holds anyone up, and a car never fights a car a lap ahead
    or behind (blue flags). The deterministic replay passes once the chances added up over the laps stuck
    behind the same car say it would have (the median of the waiting time); the Monte Carlo draws them.
DRS TRAINS (step 3c)
    When the car ahead is itself within TRAIN_GAP_SEC (1 s, the DRS gap) of the car in front of it when the
    lap starts, it has DRS too, and the attacker's advantage is cancelled: the most it can hope for per lap
    is pass_max x pass_train_factor. Measured on the 8,611 following situations within 1 s of 2023 + 2024
    (46 % of them behind a car in a train): at 0.5-0.75 s a lap faster 28 % passed behind a car in clear air,
    9 % behind a car in a train; at 1-1.5 s 59 % against 26 %; against what the speed gap alone predicts 1.18x
    and 0.65x, the same in both seasons (2023 1.25 / 0.77, 2024 1.12 / 0.49). A factor on the maximum fitted
    best out of fold (log loss per situation 0.3022, against 0.3078 without trains, 0.3044 for a bigger
    advantage needed, 0.3027 for both): pass_max 0.83, train factor 0.51. Why it matters: the number of
    overtakes the simulator makes was right overall (39 per race after lap 5 against 36 real) but 40 % too
    many in the races with a safety car (52 against 37), where the restart turns the whole field into one
    train. Making every pass harder instead (a bigger advantage needed) would have cut the count everywhere
    (+0.5 s: 22 per race). The curve without trains stays in the book (plain_curve) to compare with.
    In the Monte Carlo (calibration.py, 2026-10-20): overtakes per race 38.4 -> 35.8 (real 36.2), in the races with a
    real safety car 52 -> 47 (real 37); the chances themselves: no clear difference on their own. The deterministic
    replay: boosted+prior 1.693 -> 1.698 places off from lap 5, 1.117 -> 1.106 from halfway; hindsight 1.028 -> 1.050.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics.laps import classify_laps
from src.analytics.stops import GREEN as GREEN_STOP, stops
from src.models import evaluate
from src.models.linear import LinearDecomposition, next_lap_terms
from src.pipeline import lake
from src.pipeline.feature_store import car_ahead_on_track
from src.simulator.race import DEPLOY, END, FULL, SINGLE, Race

TYPICAL_STATIONARY_SEC = 2.7      # Phase 2: F1's typical stop, median of race medians (2024 r19 on)
PASS_GAP_SEC = 1.0                # a "following situation" for the pass rates: within 1 s at the lap's start
TRAIN_GAP_SEC = 1.0               # the car ahead within this of the car in front of it: it has DRS too (a train)
PASS_SHIFT_PRIOR_SEC = 0.5        # how far a circuit's pass midpoint may move from all races' (prior sd, s): replay.py
MIN_RATIO_LAPS = 5                # a lap ratio needs this many laps, else the default stands

# Defaults: what 2023 + 2024 measured (replaced by the book's own measurements when it has enough laps).
DEFAULTS = {
    "in_lap_share": 0.18,
    "pit_loss_sec": 22.5,
    "sc_pit_loss_sec": 4.0,        # calibrated on the replays (see SC / VSC above): Phase 2 step 2's
    "vsc_pit_loss_sec": 15.0,      # 15.1 / 18.8 s (field reference, low confidence) replayed worse
    "queue_gap_sec": 0.47,
    "follow_gap_sec": 0.60,
    "sc_leader_ratio": {DEPLOY: 1.26, FULL: 1.56, END: 1.70, SINGLE: 1.26},
    "sc_catch_ratio": {DEPLOY: 1.26, FULL: 1.16, END: 1.36, SINGLE: 1.26},
    "vsc_ratio": {DEPLOY: 1.24, FULL: 1.30, END: 1.04, SINGLE: 1.24},
    "pass_max": 0.83, "pass_slope": 2.69, "pass_midpoint": 0.96, "pass_train_factor": 0.51,   # with DRS trains (3c)
}
CURVE = ("pass_max", "pass_slope", "pass_midpoint", "pass_train_factor")


@dataclass
class Rules:
    """The rules for one race."""
    pit_loss_sec: float = DEFAULTS["pit_loss_sec"]
    in_lap_share: float = DEFAULTS["in_lap_share"]
    sc_pit_loss_sec: float = DEFAULTS["sc_pit_loss_sec"]
    vsc_pit_loss_sec: float = DEFAULTS["vsc_pit_loss_sec"]
    queue_gap_sec: float = DEFAULTS["queue_gap_sec"]
    follow_gap_sec: float = DEFAULTS["follow_gap_sec"]
    sc_leader_ratio: Dict[str, float] = field(default_factory=lambda: dict(DEFAULTS["sc_leader_ratio"]))
    sc_catch_ratio: Dict[str, float] = field(default_factory=lambda: dict(DEFAULTS["sc_catch_ratio"]))
    vsc_ratio: Dict[str, float] = field(default_factory=lambda: dict(DEFAULTS["vsc_ratio"]))
    pass_max: float = DEFAULTS["pass_max"]
    pass_slope: float = DEFAULTS["pass_slope"]
    pass_midpoint: float = DEFAULTS["pass_midpoint"]
    pass_train_factor: float = DEFAULTS["pass_train_factor"]

    def pass_chance(self, advantage: float, train: bool = False) -> float:
        """The chance of getting past the car ahead on one lap, given the pace advantage (s per lap); train: the
        car ahead is itself within TRAIN_GAP_SEC of the car in front of it (it has DRS too)."""
        z = max(-700.0, min(700.0, -self.pass_slope * (advantage - self.pass_midpoint)))   # exp() stays finite
        return self.pass_max * (self.pass_train_factor if train else 1.0) / (1.0 + math.exp(z))

    def with_changes(self, **changes) -> "Rules":
        return replace(self, **changes)


# --- what one race shows ------------------------------------------------------------------------------

@dataclass
class RaceFacts:
    """What one race shows about the rules (rows to be pooled over races by RuleBook)."""
    key: Tuple[int, int]
    event_name: str
    mixed: bool
    green_stops: pd.DataFrame       # in-lap excess, out-lap excess, loss (s), per measured green stop
    neutralised: pd.DataFrame       # per car per neutralised lap: kind, phase, leader, pitting, ratio
    queue_gaps: pd.Series           # gaps at the line on the last lap of each safety car (road order)
    following: pd.DataFrame         # following situations: advantage, start gap, passed, gap at the end, train


def _green_stops(classified: pd.DataFrame) -> pd.DataFrame:
    table = stops(classified)
    green = table[(table["condition"] == GREEN_STOP) & table["loss_sec"].notna()]
    half = green["reference_sec"] / 2.0
    return pd.DataFrame({"in_excess": green["in_lap_sec"] - half, "out_excess": green["out_lap_sec"] - half,
                         "loss": green["loss_sec"]}).reset_index(drop=True)


def _neutralised(race: Race, classified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Per car per neutralised lap: duration / the car's pace before (median of its last 5 pace laps)."""
    c = classified.sort_values(["driver_code", "lap_number"]).copy()
    c["duration"] = c["lap_end_sec"] - c["lap_start_sec"]
    pace = c["lap_time_sec"].where(c["is_pace_lap"])
    c["pace_before"] = pace.groupby(c["driver_code"]).transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).median().ffill())
    rows, gaps = [], []
    for _, p in race.periods.iterrows():
        first, last = int(p["first_lap"]), int(p["last_lap"])
        for lap in range(first, last + 1):
            phase = SINGLE if first == last else DEPLOY if lap == first else END if lap == last else FULL
            on = c[(c["lap_number"] == lap) & c["duration"].notna() & c["pace_before"].notna()]
            rows.append(pd.DataFrame({"kind": p["kind"], "phase": phase,
                                      "leader": on["position"].eq(1).fillna(False).astype(bool).to_numpy(),
                                      "pitting": on["is_pit_lap"].astype(bool).to_numpy(),
                                      "ratio": (on["duration"] / on["pace_before"]).to_numpy()}))
        if p["kind"] == "SC":
            end = c[(c["lap_number"] == last) & ~c["is_pit_lap"].astype(bool) & c["lap_end_sec"].notna()]
            gaps.append(end["lap_end_sec"].sort_values().diff().dropna())
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["kind", "phase", "leader", "pitting", "ratio"])
    return table, (pd.concat(gaps, ignore_index=True) if gaps else pd.Series(dtype=float))


def _following(race: Race, rung2: LinearDecomposition) -> pd.DataFrame:
    """
    Every following situation on a green lap: the car behind started the lap within 2 s of the car ahead
    on the road (same lap), neither pitting on that lap or the one before. Advantage = the car ahead's
    clear-air time minus the car behind's, as rung 2 forecast them a lap earlier (positive: the car behind
    is faster). passed = the car behind crossed the line first at the end of the lap. train = the car ahead
    itself started the lap within TRAIN_GAP_SEC of the car in front of it, on the same lap (DRS too).
    """
    laps = race.laps.copy()
    inputs = evaluate.model_inputs(laps)
    coef = rung2.coefficients()
    traffic = [c for c in coef.index if c.startswith("traffic")]
    clear_next = rung2.predict(inputs, 1) - next_lap_terms(inputs, 1)[traffic] @ coef[traffic]
    laps["pace"] = clear_next.groupby(laps["driver_code"]).shift(1)          # lap N's forecast, made at N-1
    ahead = car_ahead_on_track(laps.assign(is_in_lap=laps["is_in_lap"].astype(bool)))
    previous = pd.concat([laps[["driver_code", "lap_number"]], ahead], axis=1).groupby("driver_code").shift(1)
    start_gap = previous["gap_sec"].where(previous["lap_number"] == laps["lap_number"] - 1)
    same_lap = previous["car_ahead_lap"] == laps["lap_number"] - 1
    laps["start_gap"] = start_gap.where(same_lap)                             # to the car ahead, same lap only
    pit = laps["is_pit_lap"].astype(bool)
    pit_before = pit.groupby(laps["driver_code"]).shift(1, fill_value=False).astype(bool)
    by = laps.assign(pit_before=pit_before).set_index(["driver_code", "lap_number"])
    candidate = (start_gap < 2.0) & same_lap & laps["is_green"].astype(bool) & ~pit & ~pit_before \
        & (laps["lap_number"] >= 3) & laps["lap_end_sec"].notna()
    rows = []
    for i in laps.index[candidate]:
        a_key = (previous.at[i, "car_ahead"], laps.at[i, "lap_number"])
        if a_key not in by.index:
            continue
        a = by.loc[a_key]
        if bool(a["is_pit_lap"]) or bool(a["pit_before"]) or not bool(a["is_green"]) or pd.isna(a["lap_end_sec"]):
            continue
        rows.append({"driver_code": laps.at[i, "driver_code"], "car_ahead": a_key[0], "lap_number": int(laps.at[i, "lap_number"]),
                     "advantage": a["pace"] - laps.at[i, "pace"], "start_gap": start_gap[i],
                     "passed": bool(laps.at[i, "lap_end_sec"] < a["lap_end_sec"]),
                     "end_gap": laps.at[i, "lap_end_sec"] - a["lap_end_sec"],
                     "slicks": not bool(laps.at[i, "is_wet_tyre"]) and not bool(a["is_wet_tyre"]),
                     "train": bool(a["start_gap"] < TRAIN_GAP_SEC)})            # NaN (no car ahead of it): False
    return pd.DataFrame(rows, columns=["driver_code", "car_ahead", "lap_number", "advantage", "start_gap", "passed",
                                       "end_gap", "slicks", "train"])


def race_facts(race: Race, rung2: LinearDecomposition, root: str = lake.DEFAULT_LAKE_ROOT,
               silver: Optional[pd.DataFrame] = None) -> RaceFacts:
    """Everything one race shows about the rules. rung2: fitted, for the pace advantage of following cars."""
    if silver is None:
        silver = lake.read_table(lake.table_path(lake.SILVER, "laps", *race.key, "R", root))
    classified = classify_laps(silver)
    neutralised, gaps = _neutralised(race, classified)
    return RaceFacts(key=race.key, event_name=race.event_name, mixed=race.mixed,
                     green_stops=_green_stops(classified), neutralised=neutralised, queue_gaps=gaps,
                     following=_following(race, rung2))


# --- the book: rules measured over a set of races -----------------------------------------------------

def fit_pass_curve(following: pd.DataFrame, shift_prior_sec: float,
                   trains: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    p = pass_max x (train factor if the car ahead is in a DRS train) / (1 + exp(-slope x (advantage -
    midpoint - shift of the circuit))), by maximum likelihood on the situations within PASS_GAP_SEC at the
    start of the lap (column `event` names the circuit, `train` the car ahead in a train). Each circuit's
    shift is pulled towards 0 by a normal prior of standard deviation shift_prior_sec (a penalty on the
    likelihood): a circuit with few situations stays near the average. The train factor runs from 0 to 2.
    trains=False: one curve for every situation, factor 1 (the curve before step 3c). With no train among the
    situations the factor cannot be measured: the default. Returns (curve, shifts).
    """
    x = following[(following["start_gap"] < PASS_GAP_SEC) & following["advantage"].notna()]
    if x["passed"].sum() < 20:
        return {k: DEFAULTS[k] if (trains or k != "pass_train_factor") else 1.0 for k in CURVE}, {}
    events = sorted(x["event"].unique())
    which = x["event"].map({e: i for i, e in enumerate(events)}).to_numpy()
    advantage, passed = x["advantage"].to_numpy(float), x["passed"].to_numpy(float)
    train = x["train"].fillna(False).astype(bool).to_numpy() if trains and "train" in x else np.zeros(len(x), dtype=bool)
    measured = bool(train.any())
    k = 4 if measured else 3

    def minus_log_posterior(theta):
        p_max, slope, midpoint, shift = 1.0 / (1.0 + np.exp(-theta[0])), np.exp(theta[1]), theta[2], theta[k:]
        factor = 2.0 / (1.0 + np.exp(-theta[3])) if measured else 1.0
        p = np.clip(np.where(train, p_max * factor, p_max) / (1.0 + np.exp(-slope * (advantage - midpoint - shift[which]))),
                    1e-6, 1 - 1e-6)
        return -(passed * np.log(p) + (1 - passed) * np.log(1 - p)).sum() + 0.5 * (shift ** 2).sum() / shift_prior_sec ** 2

    best = minimize(minus_log_posterior, np.r_[1.0, 1.0, 1.0, np.zeros(k - 3), np.zeros(len(events))], method="L-BFGS-B").x
    curve = {"pass_max": float(1.0 / (1.0 + np.exp(-best[0]))), "pass_slope": float(np.exp(best[1])),
             "pass_midpoint": float(best[2]),
             "pass_train_factor": float(2.0 / (1.0 + np.exp(-best[3]))) if measured else
             (DEFAULTS["pass_train_factor"] if trains else 1.0)}
    return curve, {e: float(s) for e, s in zip(events, best[k:])}


def _pooled(frames: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    """The frames stacked; an empty table with these columns when there are none."""
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _ratio(table: pd.DataFrame, kind: str, phase: str, leader: Optional[bool], quantile: float, default: float) -> float:
    # astype(bool): a column of Python objects (a table stacked with an empty one) would turn ~ into -1 / -2,
    # and every pit-lane lap would count (it did, until 2026-10-16: SC laps read ~5 % slow).
    rows = table[(table["kind"] == kind) & (table["phase"] == phase) & ~table["pitting"].astype(bool)]
    if leader is not None:
        rows = rows[rows["leader"].astype(bool) == leader]
    rows = rows["ratio"].astype(float).dropna()
    return float(rows.quantile(quantile)) if len(rows) >= MIN_RATIO_LAPS else default


@dataclass
class RuleBook:
    """Rules measured over a set of races (dry races only: rain changes everything, except for passing,
    where the laps of wet races on slicks count too), and per circuit."""
    base: Rules
    circuit_pit_loss: Dict[str, float]
    circuit_pass_shift: Dict[str, float]
    races: Tuple[Tuple[int, int], ...]
    following: pd.DataFrame                 # the pooled following situations (to refit the pass curve)
    shift_prior_sec: float = PASS_SHIFT_PRIOR_SEC
    wet_races: Tuple[Tuple[int, int], ...] = ()     # measured for passing only (laps on slicks)
    plain_curve: Dict[str, float] = field(default_factory=dict)      # the curve without trains (before 3c) ...
    plain_shift: Dict[str, float] = field(default_factory=dict)      # ... and its circuit shifts

    def with_pass_prior(self, shift_prior_sec: float) -> "RuleBook":
        """The same book with the pass curves refitted under another prior on the circuit shifts."""
        curve, shifts = fit_pass_curve(self.following, shift_prior_sec)
        plain, plain_shift = fit_pass_curve(self.following, shift_prior_sec, trains=False)
        return replace(self, base=self.base.with_changes(**curve), circuit_pass_shift=shifts,
                       shift_prior_sec=shift_prior_sec, plain_curve=plain, plain_shift=plain_shift)

    @classmethod
    def measure(cls, facts: Iterable[RaceFacts], shift_prior_sec: float = PASS_SHIFT_PRIOR_SEC) -> "RuleBook":
        facts = list(facts)
        dry = [f for f in facts if not f.mixed]
        stops_ = _pooled([f.green_stops.assign(event=f.event_name) for f in dry], ["in_excess", "out_excess", "loss", "event"])
        neutral = _pooled([f.neutralised for f in dry], ["kind", "phase", "leader", "pitting", "ratio"])
        gaps = pd.concat([f.queue_gaps for f in dry], ignore_index=True) if dry else pd.Series(dtype=float)
        # Passing: the laps of a wet race on which both cars were on slicks count too. A circuit is often
        # in the book only twice (2023, 2024); without them, Monaco 2024 replayed with no Monaco at all
        # (2023 had rain), at the average circuit's pass rate, and the cars on new tyres passed everyone.
        following = _pooled([f.following[f.following["slicks"].astype(bool)].assign(event=f.event_name) for f in facts],
                            ["advantage", "start_gap", "passed", "end_gap", "slicks", "train", "event"])

        base = Rules()
        if len(stops_):
            base.pit_loss_sec = float(stops_.groupby("event")["loss"].median().median())
            base.in_lap_share = float((stops_["in_excess"] / stops_["loss"]).median())
        if len(gaps) >= MIN_RATIO_LAPS:
            base.queue_gap_sec = float(gaps.median())
        end_gaps = following[(following["start_gap"] < 1.2) & ~following["passed"].astype(bool)]["end_gap"].dropna()
        if len(end_gaps) >= MIN_RATIO_LAPS:
            base.follow_gap_sec = float(end_gaps.median())
        for phase in (DEPLOY, SINGLE):          # the lap it comes out: every car slows alike (median of all)
            base.sc_leader_ratio[phase] = base.sc_catch_ratio[phase] = _ratio(
                neutral, "SC", phase, None, 0.5, base.sc_leader_ratio[phase])
        for phase in (FULL, END):               # the leader sets the queue's pace; the others catch up
            base.sc_leader_ratio[phase] = _ratio(neutral, "SC", phase, True, 0.5, base.sc_leader_ratio[phase])
            base.sc_catch_ratio[phase] = _ratio(neutral, "SC", phase, None, 0.1, base.sc_catch_ratio[phase])
        for phase in (DEPLOY, FULL, END, SINGLE):
            base.vsc_ratio[phase] = _ratio(neutral, "VSC", phase, None, 0.5, base.vsc_ratio[phase])
        curve, shifts = fit_pass_curve(following, shift_prior_sec)
        plain, plain_shift = fit_pass_curve(following, shift_prior_sec, trains=False)
        base = base.with_changes(**curve)
        circuit_loss = stops_.groupby("event")["loss"].median().to_dict() if len(stops_) else {}
        return cls(base=base, circuit_pit_loss=circuit_loss, circuit_pass_shift=shifts,
                   races=tuple(f.key for f in dry), following=following, shift_prior_sec=shift_prior_sec,
                   wet_races=tuple(f.key for f in facts if f.mixed), plain_curve=plain, plain_shift=plain_shift)

    def rules_for(self, race: Race, trains: bool = True, **changes) -> Rules:
        """The rules for one race: the book's, with its circuit's pit loss and pass midpoint when known.
        trains=False: the pass curve without DRS trains (as before step 3c), to compare with."""
        if race.key in self.races or race.key in self.wet_races:
            raise ValueError(f"{race.key} is in the rule book's own races: replay it with a book measured without it.")
        if trains or not self.plain_curve:
            curve, shift = {}, self.circuit_pass_shift.get(race.event_name, 0.0)
        else:
            curve, shift = dict(self.plain_curve), self.plain_shift.get(race.event_name, 0.0)
        rules = self.base.with_changes(**curve)
        rules = rules.with_changes(pit_loss_sec=self.circuit_pit_loss.get(race.event_name, self.base.pit_loss_sec),
                                   pass_midpoint=rules.pass_midpoint + shift)
        return rules.with_changes(**changes) if changes else rules

    def describe(self) -> str:
        b = self.base
        fmt = lambda d: ", ".join(f"{k} {v:.2f}" for k, v in d.items())
        return "\n".join([
            f"Rules measured on {len(self.races)} dry races:",
            f"  pit loss (green) median of circuit medians {b.pit_loss_sec:.1f} s, {100 * b.in_lap_share:.0f} % of it on the in-lap",
            f"  safety car: leader x ({fmt(b.sc_leader_ratio)}); cars catching up x ({fmt(b.sc_catch_ratio)}); "
            f"queue gap {b.queue_gap_sec:.2f} s",
            f"  VSC: x ({fmt(b.vsc_ratio)})",
            f"  following: held {b.follow_gap_sec:.2f} s behind; pass chance per lap = {b.pass_max:.2f} / (1 + exp(-"
            f"{b.pass_slope:.2f} x (advantage - {b.pass_midpoint:.2f}))), x {b.pass_train_factor:.2f} behind a car in a DRS train",
            f"  circuits with their own pit loss: {len(self.circuit_pit_loss)}, own pass midpoint: "
            f"{len(self.circuit_pass_shift)} (prior sd {self.shift_prior_sec:g} s)",
        ])
