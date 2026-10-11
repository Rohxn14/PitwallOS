"""
Running order (src/lib/running_order.py): who is P1..PN at each replay frame.

Unit tests use tiny hand-built races. The golden test uses the real 2026 Azerbaijan GP
(tests/fixtures/replay/2026_r15_baku: laps, results and the official timing-tower positions),
the race where the old replay showed the winner 10th once the cars had crossed the line.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lib.running_order import (
    DISQUALIFIED, FINISHED, RETIRED, RUNNING, RaceOutcome, official_gap_between,
    official_positions, outcome_from_dict, race_outcomes, running_order, took_the_flag,
)

BAKU = Path(__file__).parents[1] / "fixtures" / "replay" / "2026_r15_baku"


def outcome(code, ended_at=None, took_flag=True, classified=None, grid=None, laps=None,
            disqualified=False, gap=None):
    return RaceOutcome(code=code, ended_at=ended_at, took_flag=took_flag, disqualified=disqualified,
                       laps_completed=laps, classified_position=classified, grid_position=grid,
                       official_gap_s=gap)


def board(positions, codes, frame):
    """Driver codes in board order at one frame."""
    return [code for _, code in sorted(zip(positions[:, frame], codes))]


# --- took_the_flag ------------------------------------------------------------

@pytest.mark.parametrize("status, by_laps, expected", [
    ("Finished", False, True),
    ("Lapped", False, True),         # F1 API style
    ("+1 Lap", False, True),         # Ergast style
    ("+3 Laps", False, True),
    ("Retired", True, False),        # the status wins when it is explicit
    ("Accident", True, False),
    ("Engine", True, False),
    ("Did not start", True, False),
    ("Disqualified", True, True),    # a disqualified car may have crossed the line: laps decide
    ("Disqualified", False, False),
    ("", True, True),                # no status: laps decide
    ("Not classified", False, False),
])
def test_took_the_flag(status, by_laps, expected):
    assert took_the_flag(status, by_laps) is expected


# --- running_order: synthetic races ------------------------------------------

def test_finished_cars_are_not_sorted_by_frozen_telemetry():
    """The Baku bug: after the line every car's telemetry is frozen near the same spot."""
    codes = ["AAA", "BBB", "CCC"]
    times = np.array([90.0, 110.0])
    # Frozen progress after the line: noise puts CCC "ahead" of AAA.
    progress = np.array([[2.99, 2.990], [2.99, 2.991], [2.99, 2.995]])
    outcomes = {
        "AAA": outcome("AAA", ended_at=100, classified=1, laps=3),
        "BBB": outcome("BBB", ended_at=101, classified=2, laps=3),
        "CCC": outcome("CCC", ended_at=102, classified=3, laps=3),
    }
    official = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    positions, states = running_order(times, codes, outcomes, progress, official)
    assert board(positions, codes, 1) == ["AAA", "BBB", "CCC"]
    assert states[:, 1].tolist() == [FINISHED] * 3

    # Without the official stream, finished cars are ordered by laps done, then the result.
    positions, _ = running_order(times, codes, outcomes, progress, None)
    assert board(positions, codes, 1) == ["AAA", "BBB", "CCC"]


def test_board_becomes_the_official_classification_when_every_race_is_over():
    """A post-race penalty: BBB crosses 2nd but is classified 3rd."""
    codes = ["AAA", "BBB", "CCC"]
    times = np.array([100.5, 101.5, 103.0])        # AAA+BBB done | all done
    progress = np.zeros((3, 3))
    outcomes = {
        "AAA": outcome("AAA", ended_at=100, classified=1, laps=3),
        "BBB": outcome("BBB", ended_at=101, classified=3, laps=3),
        "CCC": outcome("CCC", ended_at=102, classified=2, laps=3),
    }
    crossing = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    positions, _ = running_order(times, codes, outcomes, progress, crossing)
    assert board(positions, codes, 1) == ["AAA", "BBB", "CCC"]   # as it happened on the road
    assert board(positions, codes, 2) == ["AAA", "CCC", "BBB"]   # the official result


def test_retired_car_drops_to_the_bottom_at_the_end_of_its_last_lap():
    codes = ["AAA", "BBB", "CCC"]
    times = np.array([49.0, 51.0])
    progress = np.zeros((3, 2))
    outcomes = {
        "AAA": outcome("AAA", ended_at=50, took_flag=False, classified=3, laps=1),   # stops at 50 s
        "BBB": outcome("BBB", ended_at=300, classified=1, laps=5),
        "CCC": outcome("CCC", ended_at=301, classified=2, laps=5),
    }
    # The timing tower still shows AAA leading at 51 s: it reacts late; the stop is what counts.
    official = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    positions, states = running_order(times, codes, outcomes, progress, official)
    assert board(positions, codes, 0) == ["AAA", "BBB", "CCC"]
    assert board(positions, codes, 1) == ["BBB", "CCC", "AAA"]
    assert states[0].tolist() == [RUNNING, RETIRED]


def test_retirees_are_ordered_by_the_classification():
    codes = ["AAA", "BBB", "CCC", "DDD"]
    times = np.array([200.0])
    progress = np.zeros((4, 1))
    outcomes = {
        "AAA": outcome("AAA", ended_at=10, took_flag=False, classified=4, laps=0),
        "BBB": outcome("BBB", ended_at=150, took_flag=False, classified=3, laps=8),
        "CCC": outcome("CCC", ended_at=500, classified=1, laps=20),
        "DDD": outcome("DDD", ended_at=501, classified=2, laps=20),
    }
    official = np.array([[4], [3], [1], [2]], dtype=float)
    positions, _ = running_order(times, codes, outcomes, progress, official)
    assert board(positions, codes, 0) == ["CCC", "DDD", "BBB", "AAA"]


def test_disqualified_finisher_keeps_its_place_until_the_result_then_drops():
    codes = ["AAA", "BBB", "CCC"]
    times = np.array([100.5, 103.0])
    progress = np.zeros((3, 2))
    outcomes = {
        "AAA": outcome("AAA", ended_at=100, classified=3, laps=3, disqualified=True),  # crossed 1st
        "BBB": outcome("BBB", ended_at=101, classified=1, laps=3),
        "CCC": outcome("CCC", ended_at=102, classified=2, laps=3),
    }
    official = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    positions, states = running_order(times, codes, outcomes, progress, official)
    assert board(positions, codes, 0) == ["AAA", "BBB", "CCC"]
    assert board(positions, codes, 1) == ["BBB", "CCC", "AAA"]
    assert states[0].tolist() == [DISQUALIFIED, DISQUALIFIED]


def test_without_official_positions_a_lapped_finisher_is_behind_a_running_lead_lap_car():
    codes = ["LEAD", "RUN", "LAPPED"]
    times = np.array([105.0])
    progress = np.array([[2.999], [2.5], [1.999]])      # RUN is half way round its last lap
    outcomes = {
        "LEAD": outcome("LEAD", ended_at=100, classified=1, laps=3),
        "RUN": outcome("RUN", ended_at=130, classified=2, laps=3),
        "LAPPED": outcome("LAPPED", ended_at=104, classified=3, laps=2),
    }
    positions, states = running_order(times, codes, outcomes, progress, None)
    assert board(positions, codes, 0) == ["LEAD", "RUN", "LAPPED"]
    assert states[:, 0].tolist() == [FINISHED, RUNNING, FINISHED]


def test_a_hole_in_the_official_stream_falls_back_to_progress_for_that_frame():
    codes = ["AAA", "BBB"]
    times = np.array([10.0, 20.0])
    progress = np.array([[0.2, 0.4], [0.3, 0.5]])          # BBB ahead on the road
    outcomes = {c: outcome(c, ended_at=1000, classified=i + 1, laps=10) for i, c in enumerate(codes)}
    official = np.array([[1, 1], [2, np.nan]])             # frame 1: BBB has no update yet
    positions, _ = running_order(times, codes, outcomes, progress, official)
    assert board(positions, codes, 0) == ["AAA", "BBB"]    # official
    assert board(positions, codes, 1) == ["BBB", "AAA"]    # progress


def test_grid_order_breaks_the_tie_at_the_start():
    codes = ["AAA", "BBB", "CCC"]
    outcomes = {
        "AAA": outcome("AAA", ended_at=1000, classified=1, grid=3, laps=10),
        "BBB": outcome("BBB", ended_at=1000, classified=2, grid=1, laps=10),
        "CCC": outcome("CCC", ended_at=1000, classified=3, grid=2, laps=10),
    }
    positions, _ = running_order(np.array([0.0]), codes, outcomes, np.zeros((3, 1)), None)
    assert board(positions, codes, 0) == ["BBB", "CCC", "AAA"]


def test_driver_without_an_outcome_is_running_throughout():
    codes = ["AAA", "BBB"]
    progress = np.array([[5.0], [4.0]])
    positions, states = running_order(np.array([1e6]), codes, {"AAA": outcome("AAA", ended_at=10, laps=5)},
                                      progress, None)
    assert states[:, 0].tolist() == [FINISHED, RUNNING]
    assert board(positions, codes, 0) == ["AAA", "BBB"]


# --- official_positions ------------------------------------------------------

def test_official_positions_are_held_between_updates():
    stream = pd.DataFrame({
        "Time": pd.to_timedelta([10, 20, 20, 30], unit="s"),
        "Driver": ["44", "44", "44", "1"],
        "Position": [2, 3, 1, 5],        # two updates at 20 s: the later row wins
    })
    out = official_positions(stream, {"44": "HAM", "1": "VER"}, ["HAM", "VER"], np.array([5, 10, 15, 20, 40]))
    assert np.isnan(out[0, 0]) and out[0, 1:].tolist() == [2, 2, 1, 1]
    assert np.isnan(out[1, :4]).all() and out[1, 4] == 5


def test_official_positions_without_a_stream_are_unknown():
    out = official_positions(None, {}, ["HAM"], np.array([1.0, 2.0]))
    assert np.isnan(out).all()


# --- race_outcomes -----------------------------------------------------------

def _laps(rows):
    return pd.DataFrame({
        "Driver": [r[0] for r in rows],
        "LapNumber": [float(r[1]) for r in rows],
        "LapTime": pd.to_timedelta([r[2] for r in rows], unit="s"),
        "Time": pd.to_timedelta([r[3] for r in rows], unit="s"),
    })


def test_race_outcomes_from_laps_and_results():
    laps = _laps([
        ("AAA", 1, 90, 100), ("AAA", 2, 90, 190),
        ("BBB", 1, 91, 101), ("BBB", 2, 91, 192),
        ("CCC", 1, 95, 105), ("CCC", 2, None, 150),           # stopped on lap 2
        ("DDD", 1, 120, 130),                                   # a lap down, took the flag
    ])
    results = pd.DataFrame({
        "Abbreviation": ["AAA", "BBB", "DDD", "CCC"],
        "Status": ["Finished", "Finished", "Lapped", "Retired"],
        "Position": [1.0, 2.0, 3.0, 4.0],
        "GridPosition": [2.0, 1.0, 0.0, 3.0],                  # 0 = pit-lane start
        "Laps": [2.0, 2.0, 1.0, 1.0],
        "Time": pd.to_timedelta([3600.0, 2.0, 9.9, None], unit="s"),
    })
    out = race_outcomes(laps, results, t0=10.0)
    assert out["AAA"] == RaceOutcome("AAA", 180.0, True, False, 2, 1, 2, 0.0)
    assert out["BBB"].official_gap_s == 2.0 and out["BBB"].ended_at == 182.0
    assert out["CCC"].took_flag is False and out["CCC"].ended_at == 140.0 and out["CCC"].laps_completed == 1
    assert out["DDD"].took_flag is True and out["DDD"].laps_completed == 1
    assert out["DDD"].official_gap_s is None          # lapped: results Time is not a gap to the winner
    assert out["DDD"].grid_position is None           # pit-lane start


def test_race_outcomes_without_results_use_the_laps():
    laps = _laps([("AAA", 1, 90, 100), ("AAA", 2, 90, 190), ("BBB", 1, 91, 101), ("BBB", 2, None, 150)])
    out = race_outcomes(laps, None, t0=0.0)
    assert out["AAA"].took_flag is True
    assert out["BBB"].took_flag is False
    assert out["AAA"].classified_position is None


def test_outcomes_survive_the_round_trip_through_the_cache():
    o = outcome("AAA", ended_at=1.5, classified=1, grid=2, laps=3, gap=0.0)
    assert outcome_from_dict(o.to_dict()) == o


def test_official_gap_between_two_finishers():
    assert official_gap_between(outcome("A", gap=0.0), outcome("B", gap=0.196)) == pytest.approx(0.196)
    assert official_gap_between(outcome("A", gap=0.0), outcome("B", gap=None)) is None


# --- golden: 2026 Azerbaijan GP -------------------------------------------------

BAKU_CLASSIFICATION = ["RUS", "VER", "HAD", "LEC", "ANT", "HAM", "LIN", "OCO", "BEA", "SAI", "HUL",
                       "LAW", "PIA", "PER", "BOR", "BOT", "COL", "GAS", "NOR", "ALB", "ALO", "STR"]
BAKU_RETIRED = {"BOT", "COL", "GAS", "NOR", "ALB", "ALO", "STR"}


@pytest.fixture(scope="module")
def baku():
    laps = pd.read_parquet(BAKU / "laps.parquet")
    results = pd.read_parquet(BAKU / "results.parquet")
    stream = pd.read_parquet(BAKU / "timing_positions.parquet")
    t0 = laps.loc[laps["LapNumber"] == 1, "LapStartTime"].min().total_seconds()   # the start
    number_to_code = dict(zip(results["DriverNumber"].astype(str), results["Abbreviation"]))
    codes = sorted(results["Abbreviation"])
    outcomes = race_outcomes(laps, results, t0)
    return dict(laps=laps, stream=stream, t0=t0, codes=codes, outcomes=outcomes, n2c=number_to_code)


def baku_board(baku, replay_times):
    times = np.asarray(replay_times, dtype=float)
    official = official_positions(baku["stream"], baku["n2c"], baku["codes"], times + baku["t0"])
    progress = np.zeros((len(baku["codes"]), len(times)))     # the official stream covers the race
    positions, states = running_order(times, baku["codes"], baku["outcomes"], progress, official)
    return positions, states


def test_baku_outcomes(baku):
    outcomes = baku["outcomes"]
    assert {c for c, o in outcomes.items() if not o.took_flag} == BAKU_RETIRED
    assert outcomes["VER"].official_gap_s == pytest.approx(0.196)
    assert outcomes["BOR"].official_gap_s == pytest.approx(44.230)       # includes the 10 s penalty


def test_baku_starts_in_grid_order(baku):
    positions, _ = baku_board(baku, [0.0])
    grid = sorted(baku["outcomes"].values(), key=lambda o: (o.grid_position or 99))
    assert board(positions, baku["codes"], 0) == [o.code for o in grid]


def test_baku_finish_as_it_happened_then_the_official_result(baku):
    per_crosses = baku["outcomes"]["PER"].ended_at        # the last car to take the flag
    positions, states = baku_board(baku, [per_crosses - 1.0, per_crosses + 1.0, per_crosses + 600.0])

    on_the_road = board(positions, baku["codes"], 0)
    assert on_the_road[:15] == ["RUS", "VER", "HAD", "LEC", "ANT", "HAM", "LIN", "OCO", "BEA", "SAI",
                                "HUL", "LAW", "BOR", "PIA", "PER"]           # BOR crossed 13th
    assert set(on_the_road[15:]) == BAKU_RETIRED

    assert board(positions, baku["codes"], 1) == BAKU_CLASSIFICATION        # penalty applied: BOR 15th
    assert board(positions, baku["codes"], 2) == BAKU_CLASSIFICATION
    finished = {c for c, s in zip(baku["codes"], states[:, 1]) if s == FINISHED}
    assert finished == set(BAKU_CLASSIFICATION) - BAKU_RETIRED


def test_baku_stroll_is_out_from_the_end_of_his_last_lap(baku):
    stroll = baku["outcomes"]["STR"]
    positions, states = baku_board(baku, [stroll.ended_at - 1.0, stroll.ended_at + 1.0])
    i = baku["codes"].index("STR")
    assert states[i].tolist() == [RUNNING, RETIRED]
    assert positions[i, 1] == len(baku["codes"])
