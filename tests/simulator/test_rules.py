"""
Phase 4 step 1: the rules of racing (src/simulator/rules.py).

Qatar 2024's facts against the timing and Phase 2: its green pit loss is Phase 2's 28.3 s; the queue at the
line when the second safety car came in (end of lap 39: Verstappen, Norris 0.100 s behind, Leclerc 0.604,
Piastri 0.205); Verstappen's laps behind the safety car. The pass curve recovers a made-up curve and a
circuit that needs a bigger advantage. A rule book never replays a race it was measured on; a book from a
wet race alone falls back to the measured defaults.

Step 3c, DRS trains: at the end of Qatar's lap 6 Zhou, Bottas (0.877 s behind: a train) and Albon (0.861 s behind
Bottas) crossed the line in that order, so Albon's lap 7 is behind a car in a train; Piastri's lap 8 is not (Russell was
1.343 s behind Norris). The curve recovers a made-up train factor of 0.5, and the book keeps the curve without trains.
"""

import types
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate
from src.models.linear import LinearDecomposition
from src.simulator import race as R
from src.simulator.rules import DEFAULTS, RuleBook, Rules, fit_pass_curve, race_facts

QATAR, BRITISH = (2024, 23), (2024, 12)


@pytest.fixture(scope="module")
def facts(golden_gold_lake):
    rung2 = LinearDecomposition().fit(evaluate.load_race(*QATAR, root=golden_gold_lake))
    return {key: race_facts(R.load(*key, root=golden_gold_lake), rung2, golden_gold_lake) for key in (QATAR, BRITISH)}


def test_qatar_facts_match_the_timing_and_phase_2(facts):
    qatar = facts[QATAR]
    assert qatar.green_stops["loss"].median() == pytest.approx(28.35, abs=0.01)          # Phase 2 step 2: 28.3 s
    gaps = qatar.queue_gaps.round(3).tolist()
    assert any(gaps[i:i + 3] == [0.100, 0.604, 0.205] for i in range(len(gaps) - 2))     # end of lap 39
    leader = qatar.neutralised[qatar.neutralised["leader"]]
    # Lap 36, behind the safety car (led through the pit lane): about twice his green lap.
    assert leader[(leader["phase"] == "full")]["ratio"].iloc[0] == pytest.approx(2.005, abs=0.001)
    assert len(qatar.following) > 300 and qatar.following["slicks"].all()
    assert facts[BRITISH].mixed


def test_the_pass_curve_recovers_a_made_up_one():
    rng = np.random.default_rng(1)
    rows = []
    for event, shift in (("Easy", -0.3), ("Average", 0.0), ("Hard", 1.0)):
        advantage = rng.uniform(-1.0, 3.0, 6000)
        p = 0.7 / (1.0 + np.exp(-3.0 * (advantage - 0.9 - shift)))
        rows.append(pd.DataFrame({"advantage": advantage, "start_gap": 0.5, "passed": rng.random(6000) < p,
                                  "end_gap": 0.7, "slicks": True, "event": event}))
    curve, shifts = fit_pass_curve(pd.concat(rows, ignore_index=True), shift_prior_sec=2.0)
    assert curve["pass_max"] == pytest.approx(0.7, abs=0.05) and curve["pass_slope"] == pytest.approx(3.0, abs=0.5)
    # The midpoint and the shifts are known up to a shared constant: compare the circuits with each other.
    assert shifts["Hard"] - shifts["Average"] == pytest.approx(1.0, abs=0.15)
    assert shifts["Easy"] - shifts["Average"] == pytest.approx(-0.3, abs=0.15)


def test_pass_chance():
    rules = Rules(pass_max=0.8, pass_slope=4.0, pass_midpoint=1.0)
    assert rules.pass_chance(1.0) == pytest.approx(0.4)
    assert rules.pass_chance(-5.0) < 1e-9 and rules.pass_chance(10.0) == pytest.approx(0.8)


def test_a_book_never_replays_its_own_races_and_gives_the_circuits_numbers(facts, golden_gold_lake):
    book = RuleBook.measure(facts.values())
    assert book.races == (QATAR,) and book.wet_races == (BRITISH,)
    qatar = R.load(*QATAR, root=golden_gold_lake)
    with pytest.raises(ValueError):
        book.rules_for(qatar)
    with pytest.raises(ValueError):
        book.rules_for(R.load(*BRITISH, root=golden_gold_lake))
    assert book.circuit_pit_loss["Qatar Grand Prix"] == pytest.approx(28.35, abs=0.01)


def test_pit_lane_laps_never_set_the_safety_cars_pace(facts):
    # A dry race with no neutralisation adds an empty table. Stacked with it, the `pitting` column became
    # Python objects, ~ gave -1 / -2 instead of False / True, and every pit-lane lap counted (until
    # 2026-10-16: the safety car's laps read ~5 % slow). Pooling it in must change nothing.
    qatar = facts[QATAR]
    quiet = replace(qatar, key=(2024, 99), event_name="Quiet Grand Prix",
                    neutralised=qatar.neutralised.iloc[0:0].astype(object), queue_gaps=pd.Series(dtype=float))
    alone, pooled = RuleBook.measure([qatar]), RuleBook.measure([qatar, quiet])
    for ratios in ("sc_leader_ratio", "sc_catch_ratio", "vsc_ratio"):
        assert getattr(pooled.base, ratios) == getattr(alone.base, ratios)
    # By hand: the cars catching up on the full laps, 10th percentile of the laps outside the pit lane only.
    n = qatar.neutralised
    by_hand = n[(n["kind"] == "SC") & (n["phase"] == "full") & ~n["pitting"]]["ratio"].quantile(0.1)
    assert pooled.base.sc_catch_ratio["full"] == pytest.approx(by_hand)
    assert n[(n["kind"] == "SC") & (n["phase"] == "full") & n["pitting"]]["ratio"].min() > by_hand    # all slower


def test_a_book_from_a_wet_race_alone_keeps_the_defaults(facts):
    book = RuleBook.measure([facts[BRITISH]])
    assert book.races == () and book.base.pit_loss_sec == DEFAULTS["pit_loss_sec"]
    assert book.base.sc_pit_loss_sec == DEFAULTS["sc_pit_loss_sec"]
    assert book.base.pass_midpoint == DEFAULTS["pass_midpoint"]


# --- step 3c: DRS trains ------------------------------------------------------------------------------------

def test_drs_trains_in_qatar(facts):
    f = facts[QATAR].following.set_index(["driver_code", "lap_number"])
    # End of lap 6: ZHO 4018.465, BOT 4019.342 (0.877 s behind: in a train), ALB 4020.203 (0.861 s behind BOT).
    alb = f.loc[("ALB", 7)]
    assert alb["car_ahead"] == "BOT" and alb["start_gap"] == pytest.approx(0.861, abs=0.001) and bool(alb["train"])
    # End of lap 7: VER 4092.844, NOR 4094.045, RUS 4095.388, PIA 4096.392: RUS was 1.343 s behind NOR, no train.
    pia = f.loc[("PIA", 8)]
    assert pia["car_ahead"] == "RUS" and pia["start_gap"] == pytest.approx(1.004, abs=0.001) and not bool(pia["train"])


def test_the_pass_curve_finds_a_made_up_train_factor():
    rng = np.random.default_rng(2)
    advantage = rng.uniform(-1.0, 3.0, 20000)
    train = rng.random(20000) < 0.5
    p = np.where(train, 0.5, 1.0) * 0.8 / (1.0 + np.exp(-3.0 * (advantage - 0.9)))
    rows = pd.DataFrame({"advantage": advantage, "start_gap": 0.5, "passed": rng.random(20000) < p, "end_gap": 0.7,
                         "slicks": True, "train": train, "event": "Average"})
    curve, _ = fit_pass_curve(rows, shift_prior_sec=0.5)
    assert curve["pass_train_factor"] == pytest.approx(0.5, abs=0.06) and curve["pass_max"] == pytest.approx(0.8, abs=0.06)
    plain, _ = fit_pass_curve(rows, shift_prior_sec=0.5, trains=False)
    assert plain["pass_train_factor"] == 1.0 and plain["pass_max"] < curve["pass_max"]     # one curve for both: a lower top


def test_the_pass_chance_behind_a_train_and_the_book_without_one(facts):
    rules = Rules(pass_max=0.8, pass_slope=4.0, pass_midpoint=1.0, pass_train_factor=0.5)
    assert rules.pass_chance(1.0) == pytest.approx(0.4) and rules.pass_chance(1.0, train=True) == pytest.approx(0.2)
    book = RuleBook.measure(facts.values())
    another = types.SimpleNamespace(key=(2024, 99), event_name="Qatar Grand Prix")       # a race the book never saw
    assert book.rules_for(another).pass_train_factor == book.base.pass_train_factor
    plain = book.rules_for(another, trains=False)
    assert plain.pass_train_factor == 1.0 and plain.pass_max == pytest.approx(book.plain_curve["pass_max"])
    assert "DRS train" in book.describe()
