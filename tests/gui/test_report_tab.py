"""The Race report tab in the Analyze window (Qt offscreen), on Qatar 2024: its numbers and its one explained check."""

import os

import pytest

from src.gui.report_tab import format_value

QATAR = (2024, 23)


def test_format_value_follows_the_unit():
    assert format_value("fuel check, s per kg", 0.031062) == "0.0311"
    assert format_value("tyre wear medium, s per lap", 0.0491) == "+0.049"
    assert format_value("green pit loss, s", 22.166) == "22.2"
    assert format_value("undercuts that worked, %", 45.27) == "45.3"
    assert format_value("tyre changes", 26) == "26"
    assert format_value("neutralised periods", "SC 1-4") == "SC 1-4"
    assert format_value("tyre changes", float("nan")) == "—"


def test_analyze_window_has_a_race_report_tab(golden_lake):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.analytics.report import EXPLAINED_OUTSIDE, headline_numbers, sanity_checks
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(*QATAR, root=golden_lake)
    try:
        tab = win.report_tab
        assert win.tabs.indexOf(tab) == 1 and win.tabs.tabText(1) == "Race report"     # right after Result
        assert win.report is tab.report and win.report.pace is win.pace                 # built once, shared
        assert tab.number_table.rowCount() == len(headline_numbers(win.report))
        assert tab.check_table.rowCount() == len(sanity_checks(win.report))
        verdicts = [tab.check_table.item(i, 4).text() for i in range(tab.check_table.rowCount())]
        assert verdicts.count(EXPLAINED_OUTSIDE) == 1                                   # Qatar's hard tyres
        assert "numbers checked" in tab.summary_lbl.text()
    finally:
        win.close()
        win.deleteLater()
