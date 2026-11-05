"""
The race, lap by lap (src/simulator/engine.py) - Phase 4 step 1

Every car has a clock (the session time it last crossed the line) and a count of laps done. The race
moves in waves: in each wave every running car drives its next lap, in the order the cars are on the
road (the order of their clocks: the car that crossed the line longest ago is furthest round the lap,
so it crosses next). Lapped cars sit in their real places on the road, between the leaders.

For each car, in road order:

    1. its own lap   F = clock + clear-air pace (pace.py)
                         x the ratio of a safety-car or VSC lap (rules.py)
                         + its share of the pit loss on an in-lap or an out-lap
    2. traffic       GREEN or VSC: if F would put it closer than FOLLOW gap to the car ahead (or ahead
                     of it), it either passes (the rules' chance; not on a VSC's later laps) and is
                     checked against the next car ahead, or it is held:  crossing = the car ahead's
                     crossing + the gap. The lap a safety car or VSC comes out on is raced like this
                     (it comes out mid-lap: Qatar 2024 lap 40, Russell passed Gasly before it did).
                     SAFETY CAR (its later laps) or RED FLAG: a queue. The safety car runs where the race
                     leader is, at the leader's pace x the leader ratio, and stays there if the leader
                     pits. Every car on the leader's lap drives faster (catch ratio) until it reaches the
                     car ahead (or the safety car), then follows it at the QUEUE gap. A car that pits
                     drives the safety car's pace plus its pit loss (what the stop costs against staying
                     out) and holds nobody. Lapped cars are not in the queue (waved past, or told to
                     overtake) and hold nobody.
    3. clock = crossing, laps done + 1

The chance of getting past depends on the pace advantage and on whether the car ahead is itself in a DRS
train (within TRAIN_GAP_SEC of the car in front of it when the lap starts: step 3c, rules.py).

Nobody fights a car on another lap under green (blue flags); a car on its in-lap or out-lap (in the
pit lane), or on the lap it retires (Qatar 2024: Perez's 150 s lap 39 under the safety car, passed by
the queue before he stopped), never holds anyone up and is never held; nothing else is passed under a
VSC, a safety car or a red flag. What the race is doing on a wave (green, SC, VSC, red flag) is the leader's lap's
(race.lap_conditions, in the leader's laps like Race Control's).

The chequered flag falls when the first car completes the race distance; every car finishes at its
first crossing from then on, so lapped cars finish a lap or more down, as in reality. They are ranked
by laps completed, then by the time they crossed. A car that retired in the real race stops after its
real last lap (its plan ends there) and is not ranked.

Passing is deterministic by default: a car stuck behind the same car passes once its chances, added
over the laps (as -ln(1 - p) per lap, so the sum crosses ln 2 exactly when the chance of having passed
reaches one half), say it should have: the median of the waiting time. A pass is not held up again
until the car is ahead at the line (it may need another lap: 0.5 s a lap faster from 0.7 s behind is
still 0.2 s behind at the line). Given a random generator, every lap's chance is drawn instead (the
Monte Carlo of step 3).

The teams (step 3b): simulate_plans takes the plans as dictionaries, and calls on_neutralisation(Neutralisation)
on the first lap of each safety car, VSC or red flag, before the leader starts that lap, so that the teams
(teams.py) can change the plans of laps not yet driven: a stop on that lap is a stop under it. The lap status
comes as {lap: (kind, phase)} (race.status_of), so a drawn race needs no table.

Speed (step 3, which runs the race hundreds of times): the plan is turned into plain dictionaries once,
and the finishing order is worked out from each car's crossings in plain Python; record_laps=False skips
the lap-by-lap table, which only a single replay needs. ~0.007 s a race (it was ~0.2 s); the results are
the same to the last digit (the exact-replay check and every engine test).

The same luck for two plans (step 4, the game): a generator hands out its numbers in the order the passes come
up, so changing one car's plan shifts every later draw and two options end up in different futures. PassDraws
fixes each pass's draw in advance, by (car, car ahead, lap): wherever two races meet the same situation, they
draw the same number (common random numbers), and the difference between two options is the options', not luck's.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics.neutralisations import RED_FLAG, SAFETY_CAR, VIRTUAL_SAFETY_CAR
from src.simulator.race import DEPLOY, END, FULL, GREEN, SINGLE
from src.simulator.rules import TRAIN_GAP_SEC, TYPICAL_STATIONARY_SEC, Rules

LN2 = float(np.log(2.0))
PASS_THROUGH = "pass_through"
LAP_COLUMNS = ["driver_code", "lap_number", "clock_sec", "lap_sec", "pace_sec", "held", "kind"]
FINISH_COLUMNS = ["driver_code", "laps", "clock_sec", "rank"]


@dataclass
class Simulation:
    """What one simulated race gives."""
    laps: pd.DataFrame          # per car per lap: driver_code, lap_number, clock_sec, lap_sec, pace_sec, held, kind
                                # (empty when simulate(record_laps=False))
    finish: pd.DataFrame        # per car that took the flag: driver_code, laps, clock_sec, rank (1 = winner)
    passes: int                 # passes on track (not counting cars in the pit lane, or lapped cars)
    no_pace: int                # laps with no pace (the field's median lap was used)


@dataclass
class Neutralisation:
    """
    What the teams see when a safety car, VSC or red flag comes out (step 3b): handed to the
    on_neutralisation hook before the leader starts the lap it comes out on. The hook may edit the plans of
    laps not yet driven (the dictionaries are the engine's own).
    """
    lap: int                    # the leader's lap it comes out on
    kind: str
    length: int                 # laps the race stays neutralised from here (the leader's laps)
    order: List[str]            # the running cars in road order: the car that crosses the line next first
    leader: str
    done: Dict[str, int]        # laps completed by every car
    plans: Dict[str, Dict[int, tuple]]                      # {driver: {lap: (pace, in-lap, out-lap, passage)}}
    out_lap_passage: Dict[Tuple[str, int], Optional[str]]   # {(driver, out-lap): what the passage before it was}


class PassDraws:
    """
    The pass draws of one run, fixed in advance: one uniform number per (car, car ahead, lap of the car). A car
    stuck behind another on its lap n gets past if that number is below the pass chance. Two races run with the
    same PassDraws meet the same luck in the same situation, whatever else differs (step 4's options).
    """

    def __init__(self, drivers, total_laps: int, rng: np.random.Generator) -> None:
        self.index = {d: i for i, d in enumerate(drivers)}
        self.table = rng.random((len(self.index), len(self.index), total_laps + 2))

    def __call__(self, car: str, ahead: str, lap: int) -> float:
        return float(self.table[self.index[car], self.index[ahead], lap])


def stop_loss(rules: Rules, kind: str, passage: Optional[str]) -> float:
    """The whole pit loss of one passage through the pit lane under this kind of lap."""
    if kind == RED_FLAG:
        return 0.0
    if passage == PASS_THROUGH:
        return max(0.0, rules.pit_loss_sec - TYPICAL_STATIONARY_SEC) if kind == GREEN else 0.0
    if kind == SAFETY_CAR:
        return rules.sc_pit_loss_sec
    if kind == VIRTUAL_SAFETY_CAR:
        return rules.vsc_pit_loss_sec
    return rules.pit_loss_sec


def _passage_of_out_lap(planned: pd.DataFrame) -> Dict[Tuple[str, int], Optional[str]]:
    """The passage kind (on the in-lap) for each (driver, out-lap)."""
    in_laps = planned[planned["is_in_lap"]]
    return {(d, int(lap) + 1): kind for d, lap, kind in zip(in_laps["driver_code"], in_laps["lap_number"],
                                                              in_laps["pit_kind"])}


def _plans(planned: pd.DataFrame, pace: np.ndarray) -> Dict[str, Dict[int, tuple]]:
    """{driver: {lap: (pace, in-lap, out-lap, passage kind)}}: the plan as plain dictionaries."""
    plans: Dict[str, Dict[int, tuple]] = {}
    for d, lap, t, is_in, is_out, kind in zip(planned["driver_code"], planned["lap_number"], pace,
                                              planned["is_in_lap"], planned["is_out_lap"], planned["pit_kind"]):
        plans.setdefault(d, {})[int(lap)] = (float(t), bool(is_in), bool(is_out), kind)
    return plans


def simulate(start: pd.DataFrame, planned: pd.DataFrame, pace: pd.Series, conditions: pd.DataFrame,
             rules: Rules, total_laps: int, rng: Optional[np.random.Generator] = None,
             stop_extra: Optional[Dict[Tuple[str, int], float]] = None, record_laps: bool = True,
             on_neutralisation: Optional[Callable[[Neutralisation], None]] = None,
             draws: Optional[PassDraws] = None) -> Simulation:
    """
    One race from the start state to the flag.

      start        race.start_state(): index driver_code, columns laps_done and clock_sec
      planned      race.plan(): one row per car per lap still to drive (tyres, pit passages)
      pace         clear-air lap time of every planned row (same index as planned), s
      conditions   race.lap_conditions(): kind and phase of each of the leader's laps
      rules        rules.Rules for this race
      total_laps   the race distance
      rng          None: deterministic passing; a numpy Generator: passes drawn (Monte Carlo)
      stop_extra   {(driver, in-lap): seconds} added to one stop's loss, split like it (a slow or quick
                   stop against the circuit's typical one); none by default
      record_laps  False: no lap-by-lap table (the Monte Carlo only needs the result)
      on_neutralisation  called with a Neutralisation when a safety car, VSC or red flag comes out (step 3b)
      draws        the passes drawn from these fixed numbers instead (PassDraws: the same luck for two plans)
    """
    keep = planned["driver_code"].isin(start.index).to_numpy()
    planned = planned[keep]
    lap_status = dict(zip(conditions.index, zip(conditions["kind"], conditions["phase"])))
    return simulate_plans(start, _plans(planned, np.asarray(pace, dtype=float)[keep]), _passage_of_out_lap(planned),
                          lap_status, rules, total_laps, rng, stop_extra, record_laps, on_neutralisation, draws)


def laps_neutralised(lap_status: Dict[int, Tuple[str, str]], lap: int) -> int:
    """How many laps, from the lap a neutralisation comes out on, it lasts: until the race is green or another
    one comes out (Qatar 2024: the safety car of laps 35-39 and the one deployed on lap 40 are two)."""
    n = 1
    while True:
        kind, phase = lap_status.get(lap + n, (GREEN, ""))
        if kind == GREEN or phase in (DEPLOY, SINGLE):
            return n
        n += 1


def simulate_plans(start: pd.DataFrame, plans: Dict[str, Dict[int, tuple]],
                   out_lap_passage: Dict[Tuple[str, int], Optional[str]], lap_status: Dict[int, Tuple[str, str]],
                   rules: Rules, total_laps: int, rng: Optional[np.random.Generator] = None,
                   stop_extra: Optional[Dict[Tuple[str, int], float]] = None, record_laps: bool = True,
                   on_neutralisation: Optional[Callable[[Neutralisation], None]] = None,
                   draws: Optional[PassDraws] = None) -> Simulation:
    """
    simulate() on plans already turned into dictionaries (_plans), with the status of each neutralised
    lap as {lap: (kind, phase)} (race.status_of; a green lap is absent). The Monte Carlo of step 3b builds
    these itself, once per run, so that the teams can edit a plan while the race is run.
    """
    # A car whose plan ends before the race distance retires after its last planned lap.
    retires_after = {d: max(p) for d, p in plans.items() if p and max(p) < total_laps}
    clock: Dict[str, float] = start["clock_sec"].astype(float).to_dict()
    done: Dict[str, int] = start["laps_done"].astype(int).to_dict()
    extra = stop_extra or {}
    follow_gap, queue_gap, in_share = rules.follow_gap_sec, rules.queue_gap_sec, rules.in_lap_share
    stuck: Dict[Tuple[str, str], Tuple[int, float]] = {}       # (car, car ahead) -> (wave last stuck, hazard)
    finished: Dict[str, Tuple[int, float]] = {}                 # car -> (laps done, crossing) at the flag
    crossed: Dict[str, List[Tuple[int, float]]] = {d: [] for d in clock}   # every car's (lap, crossing)
    flag: Optional[float] = None
    records: List[tuple] = []
    passes = no_pace = 0
    wave = 0

    while True:
        running = [d for d in clock if d not in finished and d in plans and done[d] + 1 in plans[d]]
        if not running:
            break
        wave += 1
        order = sorted(running, key=lambda d: clock[d])                          # on the road
        # Within TRAIN_GAP_SEC of the car ahead on the road when the lap starts (same lap): it has DRS too, a train.
        train = {x: done[x] == done[a] and clock[x] - clock[a] < TRAIN_GAP_SEC for a, x in zip(order, order[1:])}
        leader = min(running, key=lambda d: (-done[d], clock[d]))
        kind, phase = lap_status.get(done[leader] + 1, (GREEN, ""))
        if on_neutralisation is not None and kind != GREEN and phase in (DEPLOY, SINGLE):
            # It comes out now: the teams decide before the leader starts this lap (a stop on it is a stop under it).
            on_neutralisation(Neutralisation(lap=done[leader] + 1, kind=kind,
                                             length=laps_neutralised(lap_status, done[leader] + 1), order=order,
                                             leader=leader, done=done, plans=plans, out_lap_passage=out_lap_passage))
        # The queue forms behind the safety car after the lap it comes out on; that lap (and a VSC's first)
        # is raced until the neutralisation starts mid-lap, so it keeps the green rules (with its ratio).
        # A VSC ends mid-lap too ("VSC ending", then green): its last lap is raced. A safety car's last lap is
        # not: it comes in, and nobody may pass before the control line.
        settled = phase in ((FULL, END) if kind == SAFETY_CAR else (FULL,))
        queue = kind == RED_FLAG or (kind == SAFETY_CAR and settled)
        racing = kind == GREEN or not settled
        rows = {d: plans[d][done[d] + 1] for d in order}
        field_pace = None
        if any(not math.isfinite(rows[d][0]) for d in order):        # a car with no pace: the field's median
            known = [rows[d][0] for d in order if math.isfinite(rows[d][0])]
            if not known:
                raise ValueError(f"Wave {wave}: no car has a pace. Start the race where every car's pace is known "
                                 "(race.first_start_lap).")
            field_pace = float(np.median(known))

        lap_time, cost, pitting = {}, {}, {}
        for d in order:
            (t, is_in, is_out, passage), lap = rows[d], done[d] + 1
            if not math.isfinite(t):
                t, no_pace = field_pace, no_pace + 1
            if kind == VIRTUAL_SAFETY_CAR:
                t *= rules.vsc_ratio.get(phase or FULL, rules.vsc_ratio[FULL])
            elif kind == SAFETY_CAR and not settled:
                t *= rules.sc_leader_ratio.get(phase, rules.sc_leader_ratio[FULL])     # every car alike
            lap_time[d] = t
            c = 0.0
            if is_in:
                c += in_share * (stop_loss(rules, kind, passage) + extra.get((d, lap), 0.0))
            if is_out:
                c += (1.0 - in_share) * (stop_loss(rules, kind, out_lap_passage.get((d, lap)))
                                         + extra.get((d, lap - 1), 0.0))
            cost[d] = c
            # In the pit lane, or on the lap it retires (slowing, pulling off): it holds nobody, nobody holds it.
            pitting[d] = is_in or is_out or retires_after.get(d) == lap

        crossings: List[float] = []          # sorted crossing times of the cars already placed this wave
        placed: List[str] = []               # ... and those cars, in the same order
        held: Dict[str, bool] = {}
        new_clock: Dict[str, float] = {}
        queued_ahead: List[float] = []       # crossings of the queue's cars not in the pit lane, so far
        # The safety car itself: where the race leader is, at the safety car's pace. It stays there when
        # the leader pits, and the next car closes up to it.
        sc_ratio = rules.sc_leader_ratio.get(phase or FULL, rules.sc_leader_ratio[FULL])
        catch_ratio = rules.sc_catch_ratio.get(phase or FULL, rules.sc_catch_ratio[FULL])
        safety_car = clock[leader] + lap_time[leader] * sc_ratio
        behind_leader = False
        for d in order:
            behind_leader = behind_leader or d == leader
            if queue:
                # The queue: cars on the leader's lap behind the safety car. A lapped car is waved past or
                # told to overtake (Qatar 2024 lap 38: "LAPPED CARS MAY NOW OVERTAKE"): it holds nobody.
                in_queue = behind_leader and done[d] == done[leader]
                if in_queue and pitting[d]:
                    # In the pit lane, not in the queue: the safety car's pace plus the pit loss (what the
                    # stop costs against staying out); it holds nobody.
                    f = clock[d] + lap_time[d] * sc_ratio
                    crossing = f
                elif in_queue:
                    # Faster than the safety car until it reaches the car ahead (or the safety car).
                    f = clock[d] + lap_time[d] * catch_ratio
                    crossing = max(f, queued_ahead[-1] + queue_gap if queued_ahead else safety_car)
                    queued_ahead.append(crossing)
                else:
                    f = clock[d] + lap_time[d] * catch_ratio
                    crossing = f
                held[d] = crossing > f
                crossing += cost[d]
            else:
                f = clock[d] + lap_time[d] + cost[d]
                j, blocker = len(crossings) - 1, None
                while j >= 0 and crossings[j] > f - follow_gap:                      # d would reach this car
                    x = placed[j]
                    if pitting[d] or pitting[x] or done[x] != done[d]:               # pit lane, or another lap
                        j -= 1
                        continue
                    if not racing:
                        blocker = j                                                  # no passing under a VSC
                        break
                    # Stuck behind the same car since the last wave: its chances add up. Once it has got by,
                    # it stays by (the hazard stays above ln 2) until it is ahead at the line.
                    last, hazard = stuck.get((d, x), (None, 0.0))
                    hazard = hazard if last == wave - 1 else 0.0
                    by_already = hazard >= LN2
                    p = rules.pass_chance(lap_time[x] - lap_time[d], train.get(x, False))
                    if by_already:
                        passed = True
                    elif draws is not None:                                          # the same luck for every plan
                        passed = draws(d, x, lap) < p
                        hazard = math.inf if passed else 0.0
                    elif rng is not None:
                        passed = bool(rng.random() < p)
                        hazard = math.inf if passed else 0.0
                    else:
                        hazard -= math.log1p(-min(p, 0.999999))
                        passed = hazard >= LN2
                    stuck[(d, x)] = (wave, hazard)
                    if passed:
                        passes += 0 if by_already else 1
                        j -= 1
                        continue
                    blocker = j
                    break
                crossing = f if blocker is None else max(f, crossings[blocker] + follow_gap)
                held[d] = crossing > f
            k = bisect.bisect_right(crossings, crossing)
            crossings.insert(k, crossing)
            placed.insert(k, d)
            new_clock[d] = crossing
            crossed[d].append((done[d] + 1, crossing))
            if record_laps:
                records.append((d, done[d] + 1, crossing, crossing - clock[d], rows[d][0], held[d], kind))

        for d in placed:
            clock[d] = new_clock[d]
            done[d] += 1
        if flag is None and any(done[d] >= total_laps for d in placed):
            flag = min(clock[d] for d in placed if done[d] >= total_laps)
        if flag is not None:                    # a car that has crossed the line after the flag has finished
            finished.update({d: (done[d], clock[d]) for d in placed if clock[d] >= flag})

    laps = pd.DataFrame(records, columns=LAP_COLUMNS)
    return Simulation(laps=laps, finish=_finish(crossed, total_laps), passes=passes, no_pace=no_pace)


def _finish(crossed: Dict[str, List[Tuple[int, float]]], total_laps: int) -> pd.DataFrame:
    """finish_order() from every car's (lap, crossing), in plain Python."""
    at_distance = [t for laps in crossed.values() for lap, t in laps if lap >= total_laps]
    if not at_distance:
        return pd.DataFrame(columns=FINISH_COLUMNS)
    flag = min(at_distance)
    first = []
    for d, laps in crossed.items():
        after = [(t, lap) for lap, t in laps if t >= flag]
        if after:
            t, lap = min(after)
            first.append((-lap, t, d))
    first.sort()
    return pd.DataFrame([(d, -minus_laps, t, i + 1) for i, (minus_laps, t, d) in enumerate(first)],
                        columns=FINISH_COLUMNS)


def finish_order(laps: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """
    The result: the flag falls at the first crossing of the race distance, and every car finishes at its
    first crossing from then on. That crossing can come from an earlier wave: a car lapped during the race
    completes its lap N-1 after the leader's lap N (Monaco 2024 from half distance: Tsunoda's lap 77
    ended 39 s after Leclerc's lap 78). Ranked by laps completed, then by time.
    """
    crossed: Dict[str, List[Tuple[int, float]]] = {}
    for d, lap, t in zip(laps["driver_code"], laps["lap_number"], laps["clock_sec"]):
        crossed.setdefault(d, []).append((int(lap), float(t)))
    return _finish(crossed, total_laps)
