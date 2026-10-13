"""The Degradation charts (off-screen, matplotlib only) and the Degradation tab in the Analyze window (Qt offscreen)."""

import os

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analytics.degradation import race_degradation
from src.analytics.laps import race_pace
from src.gui.degradation_charts import draw_fuel_correction, draw_stint_wear

QATAR = (2024, 23)


@pytest.fixture(scope="module")
def qatar(golden_silver_laps):
    return race_degradation(race_pace(golden_silver_laps[QATAR]).laps)


def test_fuel_correction_chart_has_a_row_per_compound_with_raw_and_corrected(qatar):
    fig = Figure(figsize=(12, 2.6))
    points = draw_fuel_correction(fig, qatar)
    FigureCanvasAgg(fig).draw()
    ax = points[0].ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Soft", "Medium", "Hard"]
    assert len(points[0].x) == 2 * 3                                     # raw and corrected for each compound
    assert "raw slope" in points[0].text[0] and "fuel-corrected" in points[0].text[1]


def test_stint_wear_chart_has_a_panel_per_compound_and_a_point_per_lap_used(qatar):
    fig = Figure(figsize=(12, 3.6))
    points = draw_stint_wear(fig, qatar)
    FigureCanvasAgg(fig).draw()
    assert [hp.ax.get_title(loc="left").split(" · ")[0] for hp in points] == ["Soft", "Medium", "Hard"]
    assert sum(len(hp.x) for hp in points) == int((qatar.laps["deg_status"] == "used").sum())


def test_analyze_window_has_a_degradation_tab(golden_lake):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(*QATAR, root=golden_lake)
    try:
        tab = win.degradation_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Degradation"
        assert "Medium" in tab.summary_lbl.text() and "fuel check" in tab.summary_lbl.text()
        assert tab.stint_table.rowCount() == len(tab.result.stints) > 0
        assert tab.fuel_canvas.points and tab.wear_canvas.points
    finally:
        win.close()
        win.deleteLater()
