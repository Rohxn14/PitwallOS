"""The undercut chart (off-screen, matplotlib only) and the Undercuts tab in the Analyze window (Qt offscreen)."""

import os

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analytics.laps import race_pace
from src.analytics.undercuts import race_undercuts
from src.gui.undercuts_charts import draw_attempts
from tests.conftest import HUNGARY

QATAR = (2024, 23)


@pytest.fixture(scope="module")
def hungary(hungary_silver_laps):
    return race_undercuts(race_pace(hungary_silver_laps).laps)


def test_attempt_chart_has_a_row_per_attempt_with_before_and_after(hungary):
    fig = Figure(figsize=(12, 2.4))
    points = draw_attempts(fig, hungary)
    FigureCanvasAgg(fig).draw()
    ax = points[0].ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["NOR on PIA\nlaps 45 / 47"]
    assert points[0].x.tolist() == pytest.approx([2.051, -3.377])        # before, after
    assert "worked · gained +5.43 s" in [t.get_text() for t in ax.texts]
    assert "1 of 1 worked" in fig._suptitle.get_text()


def test_attempt_chart_says_so_when_a_race_has_none(golden_silver_laps):
    fig = Figure(figsize=(12, 2.4))
    assert draw_attempts(fig, race_undercuts(race_pace(golden_silver_laps[QATAR]).laps)) == []
    FigureCanvasAgg(fig).draw()
    assert "No undercut attempt" in fig.axes[0].texts[0].get_text()


def test_analyze_window_has_an_undercuts_tab(hungary_lake):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*HUNGARY, "R", lake_root=hungary_lake, telemetry=False)
    win = RaceAnalysisWindow(*HUNGARY, root=hungary_lake)
    try:
        tab = win.undercuts_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Undercuts"
        assert "1 undercut attempt: 1 worked" in tab.summary_lbl.text()
        assert tab.attempt_table.rowCount() == 1 and tab.attempt_table.item(0, 10).text() == "worked"
        assert tab.attempt_table.item(0, 2).text() == "yes"                    # NOR and PIA: McLaren
        assert [tab.attempt_table.item(0, c).text() for c in (4, 5)] == ["2.051 behind", "3.377 ahead"]
        # The selected attempt names the drivers, never "B" ...
        assert tab.lap_title.text() == "NOR on PIA"
        assert tab.outcome_lbl.text().startswith("Worked: NOR went from 2.051 behind PIA to 3.377 ahead, "
                                                 "gaining 5.428 s.")
        headers = [tab.lap_table.horizontalHeaderItem(c).text() for c in range(tab.lap_table.columnCount())]
        assert headers == ["Lap", "NOR", "NOR's lap", "PIA", "PIA's lap", "NOR at the line", "NOR gained"]
        ledger = [tab.ledger_table.item(i, 0).text() for i in range(tab.ledger_table.rowCount())]
        assert not any(reason.startswith("B ") or "B's" in reason for reason in ledger)
        assert tab.lap_table.rowCount() == 5                               # laps 44-48, the attempt selected
        assert [tab.lap_table.item(i, 5).text() for i in (0, 4)] == ["2.051 behind", "3.377 ahead"]
        # ... and shows both stops against the race's typical one (pit lane from the timing)
        stops = [[tab.stop_table.item(i, c).text() for c in range(3)] for i in range(tab.stop_table.rowCount())]
        assert stops[:2] == [["NOR", "45", "21.279"], ["PIA", "47", "21.856"]]
        assert stops[2][0].startswith("Race typical (")
        assert "none for this race" in tab.stops_note.text()               # Hungary 2024: before F1 published them
        assert tab.ledger_table.rowCount() == len(tab.result.ledger)
    finally:
        win.close()
        win.deleteLater()


def test_outcome_words_for_an_attempt_that_failed():
    from src.gui.undercuts_tab import outcome_words

    # Bahrain 2025, laps 26 / 27: VER stood 6.2 s in its box, ANT 2.8 s (F1's own times)
    attempt = {"driver_code": "VER", "car_ahead": "ANT", "gap_before_sec": 2.184, "gap_after_sec": 1.909,
               "gain_sec": 0.275, "stop_gain_sec": -3.825, "track_gain_sec": 4.100, "between_gain_sec": 0.0,
               "worked": False, "stationary_sec": 6.2, "car_ahead_stationary_sec": 2.8}
    assert outcome_words(attempt) == ("Did not work: VER went from 2.184 behind ANT to 1.909 behind, gaining 0.275 s. "
                                      "Where it came from: stops -3.825 s (ANT's pit-lane time minus VER's; "
                                      "stationary VER 6.2 s, ANT 2.8 s), track around the stops +4.100 s.")
