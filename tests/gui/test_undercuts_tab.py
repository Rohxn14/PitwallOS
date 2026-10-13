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
        assert tab.lap_table.rowCount() == 5                               # laps 44-48, the attempt selected
        assert tab.lap_table.item(4, 5).text() == "-3.377"
        assert tab.ledger_table.rowCount() == len(tab.result.ledger)
    finally:
        win.close()
        win.deleteLater()
