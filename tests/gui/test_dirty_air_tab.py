"""The dirty-air chart (off-screen, matplotlib only) and the Dirty air tab in the Analyze window (Qt offscreen)."""

import os

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analytics.dirty_air import BIN_LABELS, CLEAR_LABEL, USED, race_dirty_air
from src.analytics.laps import race_pace
from src.gui.dirty_air_charts import draw_following_cost

BRITISH, QATAR = (2024, 12), (2024, 23)


@pytest.fixture(scope="module")
def races(golden_silver_laps):
    return {r: race_dirty_air(race_pace(laps).laps) for r, laps in golden_silver_laps.items()}


def test_cost_chart_has_a_dot_per_lap_used_and_a_bar_per_gap(races):
    qatar = races[QATAR]
    fig = Figure(figsize=(12, 4.2))
    points = draw_following_cost(fig, qatar)
    FigureCanvasAgg(fig).draw()
    dots, bars = points
    assert len(dots.x) == int((qatar.laps["air_status"] == USED).sum())
    assert len(bars.x) == len(BIN_LABELS)                                # Qatar has 10+ laps in every gap
    assert "behind RUS" in dots.text[[i for i, t in enumerate(dots.text) if t.startswith("PIA · lap 8 ")][0]]
    labels = [t.get_text() for t in dots.ax.get_xticklabels()]
    assert labels[0].startswith("0-0.5 s") and labels[-1].startswith(f"{CLEAR_LABEL} (reference)")
    assert "beta" in fig._suptitle.get_text()


def test_cost_chart_says_why_when_a_race_has_too_few_laps(races):
    fig = Figure(figsize=(12, 4.2))
    points = draw_following_cost(fig, races[BRITISH])                     # 10 laps, none 3 s+ behind: no fit
    FigureCanvasAgg(fig).draw()
    assert points == []
    assert "Too few dry laps" in fig.axes[0].texts[0].get_text()


def test_analyze_window_has_a_dirty_air_tab(golden_lake):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(*QATAR, root=golden_lake)
    try:
        tab = win.dirty_air_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Dirty air"
        assert "costs +0.40" in tab.summary_lbl.text() and "with following in the fit" in tab.summary_lbl.text()
        assert tab.bin_table.rowCount() == len(BIN_LABELS) + 1
        assert tab.driver_table.rowCount() == len(tab.result.drivers) > 0
        assert tab.cost_canvas.points
    finally:
        win.close()
        win.deleteLater()
