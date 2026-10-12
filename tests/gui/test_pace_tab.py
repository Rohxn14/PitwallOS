"""
The Pace tab inside the Analyze window, built off-screen (Qt's "offscreen" platform, no display):
the window opens a race from a lake, fills the tab, follows the driver selector, shows a tooltip
on hover, and the Laps tab shows the classification as a table.
"""

import os
import types

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_analyze_window_shows_step_1_for_qatar(qapp, golden_lake):
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(2024, 23, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(2024, 23, root=golden_lake)
    try:
        tab = win.pace_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Pace"
        assert tab.driver_box.itemText(0) == "VER"                       # result order: the winner first
        assert "SC 1-4, SC 35-39, VSC 40, SC 40-42" in tab.summary_lbl.text()
        ledger = [tab.ledger_table.item(i, 0).text() for i in range(tab.ledger_table.rowCount())]
        assert ledger[0] == "all laps" and ledger[-1] == "pace laps" and len(ledger) == 13
        assert tab.lap_canvas.points and tab.compound_canvas.points and tab.driver_canvas.points

        # hovering a lap shows what it was
        canvas = tab.lap_canvas
        canvas.draw()
        hp = canvas.points[0]
        px, py = hp.ax.transData.transform((hp.x[10], hp.y[10]))
        canvas._on_move(types.SimpleNamespace(x=px + 2, y=py - 2))
        tip = canvas._tips[hp.ax]
        assert tip.get_visible() and tip.get_text() == hp.text[10]
        canvas._on_move(types.SimpleNamespace(x=None, y=None))           # pointer left the plot
        assert not tip.get_visible()

        # another driver redraws the lap chart for them
        second = tab.driver_box.itemText(1)
        tab.driver_box.setCurrentIndex(1)
        assert canvas.figure._suptitle.get_text() == f"Lap times · {second}"

        # the Laps tab's table twin, filtered by status
        win.laps_source.setCurrentIndex(2)
        win.laps_filter.setText("safety car")
        headers = [win.laps_table.horizontalHeaderItem(c).text() for c in range(win.laps_table.columnCount())]
        assert "pace_status" in headers and "reference_sec" in headers
        assert win.laps_table.rowCount() == int((win.pace.laps["pace_status"] == "safety car").sum())
    finally:
        win.close()
        win.deleteLater()


def test_pace_tab_shows_an_error_instead_of_numbers(qapp):
    from src.gui.pace_tab import PaceTab

    tab = PaceTab()
    tab.show_error("ValueError: something broke")
    assert "something broke" in tab.summary_lbl.text()
    assert tab.ledger_table.rowCount() == 0 and tab.driver_box.count() == 0
    assert not tab.lap_canvas.points
    tab.deleteLater()
