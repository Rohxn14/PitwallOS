"""
Undercuts, measured (src/analytics/undercuts.py), Phase 2 step 5.

Two-car races built through the real cleaner, where every number can be worked out by hand: AAA leads,
BBB follows; both lap in 90 s on old tyres and 89 s on new ones; an in-lap is 2 s slower (its last
7 s in the pit lane) and an out-lap costs the rest of the pit lane. Each ledger reason gets its own
small race. The golden case is the 2024 Hungarian GP: Norris, 2.051 s behind Piastri, stops on lap 45;
Piastri stops on lap 47; Norris ends lap 48 3.377 s ahead (Silver's line crossings), and the timing
positions show Norris ahead until the lead goes back to Piastri on lap 68 (McLaren's team order).
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.laps import classify_laps, race_pace
from src.analytics.undercuts import (
    A_FIRST, A_LATE, A_NEVER, A_NOT_GREEN, ATTEMPT, LEADING, NEUTRALISED, NO_GAP_AFTER, NOT_GREEN, RED_FLAG_CHANGE,
    SAME_LAP, TOO_FAR, WET, attempt_laps, main, race_undercuts, undercut_events, wilson,
)
from src.pipeline.cleaner import clean_laps
from src.pipeline.feature_store import build_lap_features
from tests.conftest import HUNGARY
from tests.helpers import make_bronze_laps, make_weather

QATAR, BRITISH = (2024, 23), (2024, 12)
PACE_OLD, PACE_NEW = 90.0, 89.0
IN_EXTRA, IN_PIT = 2.0, 7.0            # an in-lap is 2 s slower; its last 7 s are in the pit lane
RACE_LAPS = 18


def car(code, start_gap=0.0, stop=None, pit_lane=20.0, after="HARD", last_lap=RACE_LAPS, status=None):
    """One car's Bronze rows. pit_lane = pit entry to pit exit (s); status = {lap: track status code}."""
    rows, t = [], 0.0
    for lap in range(1, last_lap + 1):
        if lap == 1:
            lap_time = 95.0 + start_gap
        elif stop is not None and lap == stop:
            lap_time = PACE_OLD + IN_EXTRA
        elif stop is not None and lap == stop + 1:
            lap_time = PACE_NEW + pit_lane - IN_EXTRA
        else:
            lap_time = PACE_NEW if stop is not None and lap > stop else PACE_OLD
        new = stop is not None and lap > stop
        row = {"driver": code, "lap": lap, "start": t, "end": t + lap_time, "lap_time": lap_time,
               "compound": after if new else "MEDIUM", "tyre_life": float(lap - stop if new else lap),
               "status": (status or {}).get(lap, "1")}
        if stop is not None and lap == stop:
            row["pit_in"] = row["end"] - IN_PIT
        if stop is not None and lap == stop + 1:
            row["pit_out"] = t + pit_lane - IN_PIT
        rows.append(row)
        t += lap_time
    return rows


def race(*cars):
    """Classified laps; each lap's position is its order at the line among the cars that completed it."""
    rows = [r for c in cars for r in c]
    by_lap = {}
    for r in rows:
        by_lap.setdefault(r["lap"], []).append(r)
    for lap_rows in by_lap.values():
        for p, r in enumerate(sorted(lap_rows, key=lambda r: r["end"]), start=1):
            r["position"] = float(p)
    laps, _ = clean_laps(make_bronze_laps(rows), make_weather([(0.0, 30.0, False)]), 2024, 1, "R")
    return classify_laps(laps)


def bbb_stop(*cars):
    events = undercut_events(race(*cars))
    return events[events["driver_code"] == "BBB"].iloc[0]


def test_an_undercut_that_works_and_its_gain_split():
    # BBB 1.5 s behind stops on lap 10, AAA on lap 12 with a stop 0.5 s slower in the pit lane.
    # End of lap 9: AAA 815.0, BBB 816.5. End of lap 13 (AAA's out-lap): AAA 1194.5, BBB 1193.5.
    e = bbb_stop(car("AAA", stop=12, pit_lane=20.5), car("BBB", start_gap=1.5, stop=10))
    assert e["status"] == ATTEMPT and e["car_ahead"] == "AAA" and e["lag"] == 2
    assert e["gap_before_sec"] == pytest.approx(1.5) and e["gap_after_sec"] == pytest.approx(-1.0)
    assert bool(e["worked"]) and e["gain_sec"] == pytest.approx(2.5)
    assert e["stop_gain_sec"] == pytest.approx(0.5)                 # 20.5 s in the pit lane against 20.0
    assert e["between_gain_sec"] == 0.0                             # AAA stopped 2 laps later: no lap between
    assert e["track_gain_sec"] == pytest.approx(2.0)


def test_laps_between_are_what_the_new_tyres_gained():
    # AAA waits until lap 13: on lap 12 BBB laps in 89 s on new tyres, AAA in 90 s on old ones.
    e = bbb_stop(car("AAA", stop=13, pit_lane=20.5), car("BBB", start_gap=1.5, stop=10))
    assert e["status"] == ATTEMPT and e["lag"] == 3
    assert e["between_gain_sec"] == pytest.approx(1.0)
    assert e["gain_sec"] == pytest.approx(3.5) and e["gap_after_sec"] == pytest.approx(-2.0)
    assert e["stop_gain_sec"] + e["track_gain_sec"] + e["between_gain_sec"] == pytest.approx(e["gain_sec"])


def test_an_undercut_covered_on_the_next_lap_fails():
    e = bbb_stop(car("AAA", stop=11, pit_lane=20.5), car("BBB", start_gap=2.5, stop=10))
    assert e["status"] == ATTEMPT and e["lag"] == 1
    assert e["gain_sec"] == pytest.approx(1.5) and e["gap_after_sec"] == pytest.approx(1.0)
    assert not bool(e["worked"])                                    # gained 1.5 s, needed 2.5


@pytest.mark.parametrize("aaa, bbb, status", [
    (dict(stop=10), dict(start_gap=1.5, stop=10), SAME_LAP),
    (dict(), dict(start_gap=1.5, stop=10), A_NEVER),
    (dict(stop=15), dict(start_gap=1.5, stop=10), A_LATE),
    (dict(stop=12), dict(start_gap=3.5, stop=10), TOO_FAR),
    # AAA starts 21.5 s ahead and stops on lap 8; it rejoins 2.0 s ahead, so BBB's stop answers it
    (dict(stop=8), dict(start_gap=21.5, stop=10), A_FIRST),
    (dict(stop=12, status={10: "4"}), dict(start_gap=1.5, stop=10, status={10: "4"}), NOT_GREEN),
    (dict(stop=12, status={10: "5", 11: "5"}), dict(start_gap=1.5, stop=10, status={10: "5", 11: "5"}),
     RED_FLAG_CHANGE),
    (dict(stop=12, status={12: "4"}), dict(start_gap=1.5, stop=10, status={12: "4"}), A_NOT_GREEN),
    (dict(stop=13, status={12: "6"}), dict(start_gap=1.5, stop=10, status={12: "6"}), NEUTRALISED),
    (dict(stop=12), dict(start_gap=1.5, stop=10, after="INTERMEDIATE"), WET),
    (dict(stop=12), dict(start_gap=1.5, stop=10, last_lap=12), NO_GAP_AFTER),
])
def test_every_stop_that_is_not_an_attempt_says_why(aaa, bbb, status):
    e = bbb_stop(car("AAA", **aaa), car("BBB", **bbb))
    assert e["status"] == status
    assert pd.isna(e["gain_sec"])                                   # a gain only for attempts


def test_the_leaders_stop_is_never_an_attempt_and_every_stop_is_counted():
    r = race_undercuts(race(car("AAA", stop=12, pit_lane=20.5), car("BBB", start_gap=1.5, stop=10)))
    assert r.events.set_index("driver_code").loc["AAA", "status"] == LEADING
    ledger = r.ledger.set_index("reason")["count"]
    assert ledger["tyre changes"] == ledger.drop("tyre changes").sum() == 2
    assert r.summary["attempts"] == 1 and r.summary["worked"] == 1


def test_attempt_laps_reads_one_attempt_lap_by_lap():
    classified = race(car("AAA", stop=12, pit_lane=20.5), car("BBB", start_gap=1.5, stop=10))
    e = undercut_events(classified).set_index("driver_code").loc["BBB"].to_dict()
    e["driver_code"] = "BBB"
    laps = attempt_laps(classified, e)
    assert laps["lap"].tolist() == [9, 10, 11, 12, 13]
    assert laps["gap_sec"].tolist() == pytest.approx([1.5, 3.5, 20.5, 17.5, -1.0])
    assert laps["b_kind"].tolist() == ["old tyres", "in-lap", "out-lap", "new tyres", "new tyres"]
    assert laps["a_kind"].tolist() == ["old tyres", "old tyres", "old tyres", "in-lap", "out-lap"]
    assert laps["b_gained_sec"].sum() == pytest.approx(e["gain_sec"])


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)   # the textbook value
    lo, hi = wilson(67, 148)
    assert lo < 67 / 148 < hi and 0.37 < lo < 0.38 and 0.53 < hi < 0.54


# --- golden ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hungary(hungary_silver_laps):
    classified = race_pace(hungary_silver_laps).laps
    return classified, race_undercuts(classified)


def test_hungary_norris_undercuts_piastri_on_lap_45(hungary):
    _, r = hungary
    e = r.events.set_index(["driver_code", "in_lap"]).loc[("NOR", 45)]
    assert e["status"] == ATTEMPT and e["car_ahead"] == "PIA" and bool(e["teammates"])
    assert e["car_ahead_in_lap"] == 47
    # Silver's line crossings: end of lap 44 NOR 7021.103, PIA 7019.052; end of lap 48 NOR 7371.626, PIA 7375.003
    assert e["gap_before_sec"] == pytest.approx(7021.103 - 7019.052)
    assert e["gap_after_sec"] == pytest.approx(7371.626 - 7375.003) and bool(e["worked"])
    # Pit lane: NOR 7105.665 -> 7126.944, PIA 7271.866 -> 7293.722
    assert e["stop_gain_sec"] == pytest.approx((7293.722 - 7271.866) - (7126.944 - 7105.665))
    assert e["gain_sec"] == pytest.approx(5.428)
    assert e["stop_gain_sec"] + e["track_gain_sec"] + e["between_gain_sec"] == pytest.approx(e["gain_sec"])
    first = r.events.set_index(["driver_code", "in_lap"]).loc[("NOR", 17)]
    assert first["status"] == TOO_FAR and first["gap_before_sec"] == pytest.approx(3.502)   # PIA stopped on 18


def test_hungary_timing_positions_agree(hungary):
    classified, _ = hungary
    pos = classified.set_index(["driver_code", "lap_number"])["position"]
    assert all(pos[("NOR", k)] < pos[("PIA", k)] for k in range(48, 68))   # ahead after the stops
    assert all(pos[("PIA", k)] < pos[("NOR", k)] for k in range(68, 71))   # the lead handed back on lap 68


def test_hungary_gap_before_is_golds_gap_ahead(hungary, hungary_silver_laps):
    _, r = hungary
    gold = build_lap_features(hungary_silver_laps, 70).set_index(["driver_code", "lap_number"])
    known = r.events.dropna(subset=["gap_before_sec"])
    expected = [gold.loc[(d, int(n) - 1), "gap_ahead_sec"] for d, n in zip(known["driver_code"], known["in_lap"])]
    np.testing.assert_allclose(known["gap_before_sec"].to_numpy(dtype=float), np.array(expected, dtype=float))


@pytest.fixture(scope="module")
def golden(golden_silver_laps):
    return {race_: race_undercuts(race_pace(laps).laps) for race_, laps in golden_silver_laps.items()}


def test_qatar_stops_under_the_safety_car_are_not_undercuts(golden):
    r = golden[QATAR]
    assert r.summary["attempts"] == 0
    assert r.events.set_index(["driver_code", "in_lap"]).loc[("VER", 35), "status"] == NOT_GREEN
    ledger = r.ledger.set_index("reason")["count"]
    assert ledger["tyre changes"] == ledger.drop("tyre changes").sum() == 26


def test_british_weather_stops_are_not_undercuts(golden):
    events = golden[BRITISH].events.set_index(["driver_code", "in_lap"])
    assert events.loc[("HAM", 27), "status"] == WET                    # mediums -> inters
    assert events.loc[("HAM", 38), "status"] == WET                    # inters -> softs
    assert golden[BRITISH].summary["attempts"] == 0


def test_command_line_writes_an_undercut_report(golden_lake, capsys):
    import shutil
    from pathlib import Path

    from src.pipeline import lake
    from src.pipeline.cleaner import build_silver
    from tests.conftest import HUNGARY_FIXTURE_LAKE

    shutil.copytree(Path(HUNGARY_FIXTURE_LAKE) / "bronze", Path(golden_lake) / "bronze", dirs_exist_ok=True)
    for race_ in (BRITISH, QATAR, HUNGARY):
        build_silver(*race_, "R", lake_root=golden_lake, telemetry=False)
    assert main(["--season", "2024", "--lake", golden_lake]) == 0
    report = lake.read_report("undercuts", golden_lake)
    assert len(report) == 26 + 45 + 40                                 # every tyre change of the three races
    attempts = report[report["status"] == ATTEMPT]
    assert attempts[["driver_code", "car_ahead", "in_lap"]].values.tolist() == [["NOR", "PIA", 45]]
    assert "1 of 1" in capsys.readouterr().out
