"""The Stops chart (off-screen, matplotlib only) and the Stops tab in the Analyze window (Qt offscreen)."""

import os

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analytics.laps import race_pace
from src.analytics.stops import race_stops
from src.gui.stops_charts import draw_stop_losses

BRITISH, QATAR = (2024, 12), (2024, 23)


def test_stop_chart_has_one_row_per_condition_and_a_point_per_measured_stop(golden_silver_laps):
    result = race_stops(race_pace(golden_silver_laps[QATAR]).laps)
    fig = Figure(figsize=(12, 3.4))
    points = draw_stop_losses(fig, result)
    FigureCanvasAgg(fig).draw()
    ax = points[0].ax
    rows = {t.get_text().split("\n")[0]: y for t, y in zip(ax.get_yticklabels(), ax.get_yticks())}
    assert sorted(rows, key=rows.get, reverse=True) == ["Green", "Safety car", "Lap 1"]   # green on top
    assert len(points[0].x) == int(result.stops["loss_sec"].notna().sum())
    assert "circuit loss" in fig._suptitle.get_text()
    assert "= loss" in points[0].text[0]


def test_analyze_window_has_a_stops_tab(golden_lake):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*BRITISH, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(*BRITISH, root=golden_lake)
    try:
        tab = win.stops_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Stops"
        assert tab.table.rowCount() == 45
        assert tab.summary_lbl.text().startswith("Circuit pit loss (green stops):")
        assert tab.canvas.points
    finally:
        win.close()
        win.deleteLater()
