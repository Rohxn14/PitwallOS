"""
Race report tab of the Analyze window (src/gui/report_tab.py) — Phase 2 step 6

The race on one page (src/analytics/report.py): every number Phase 2 measures, with its n, its
uncertainty and a note on what was left out, then every sanity check with its range and verdict.
The step tabs show how each number was measured.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QScrollArea, QVBoxLayout, QWidget

from src.analytics.report import (
    EXPLAINED_OUTSIDE, LOW, NOT_CHECKED, NOT_MEASURED, OK, UNEXPLAINED, RaceReport, headline_numbers, sanity_checks,
)
from src.gui.analysis_theme import ACCENT, BAD, DIM, GOOD, TEXT
from src.gui.degradation_tab import _table
from src.gui.pace_tab import table_cell

STEPS = {1: "1 · Pace laps", 2: "2 · Stops", 3: "3 · Tyre wear", 4: "4 · Dirty air", 5: "5 · Undercuts"}
NUMBER_COLUMNS = ["Step", "Number", "Value", "Uncertainty", "n", "Note"]
CHECK_COLUMNS = ["Check", "Item", "Value", "Range", "Verdict", "Explanation"]
VERDICT_COLOUR = {OK: GOOD, EXPLAINED_OUTSIDE: TEXT, LOW: TEXT, UNEXPLAINED: BAD, NOT_CHECKED: DIM, NOT_MEASURED: DIM}

NOTE = (
    "Every number is checked against a range written down before the 2023-2025 races were measured: what is "
    "physically plausible, not what is typical. Outside its range, a number is explained (with its evidence) or "
    "a problem to find; across the 70 races of 2023-2025 none is unexplained. Safety-car and VSC stop losses are "
    "low confidence: step 2 compares laps by their number, and a safety car that starts or ends mid-lap makes "
    "those laps not comparable. In races where someone ran inters or wets, tyre wear, the fuel check and the "
    "cost of following are not checked."
)


def format_value(quantity: str, value) -> str:
    """A number as the report shows it: its precision follows its unit."""
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "—"
    if "s per kg" in quantity:
        return f"{value:.4f}"
    if "s per lap" in quantity:
        return f"{value:+.3f}"
    if "%" in quantity or ", s" in quantity:                     # percentages and seconds
        return f"{value:.1f}"
    return f"{value:g}"


class ReportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report: Optional[RaceReport] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        outer.addWidget(self.summary_lbl)

        page = QWidget(objectName="scrollpage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.number_table = _table(NUMBER_COLUMNS)
        self.check_table = _table(CHECK_COLUMNS)
        for title, table in (("What Phase 2 measures in this race", self.number_table),
                             ("Sanity checks", self.check_table)):
            label = QLabel(title)
            label.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
            lay.addWidget(label)
            table.horizontalHeader().setSectionResizeMode(table.columnCount() - 1, QHeaderView.Stretch)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)       # sized to show every row
            lay.addWidget(table)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

    def set_report(self, report: RaceReport):
        self.report = report
        numbers, checks = headline_numbers(report), sanity_checks(report)
        counts = checks["status"].value_counts()
        parts = [f"{len(checks)} numbers checked: {int(counts.get(OK, 0))} in range"]
        for status in (EXPLAINED_OUTSIDE, LOW, UNEXPLAINED, NOT_CHECKED, NOT_MEASURED):
            if counts.get(status, 0):
                parts.append(f"{int(counts[status])} {status}")
        self.summary_lbl.setStyleSheet(f"color: {BAD if counts.get(UNEXPLAINED, 0) else DIM};")
        self.summary_lbl.setText("  ·  ".join(parts))
        self._fill_numbers(numbers)
        self._fill_checks(checks)

    def show_error(self, message: str):
        self.report = None
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not build the race report: {message}")
        for table in (self.number_table, self.check_table):
            table.setRowCount(0)

    def _fill_numbers(self, numbers: pd.DataFrame):
        self.number_table.setRowCount(len(numbers))
        previous = None
        for i, r in enumerate(numbers.to_dict("records")):
            step = STEPS.get(r["step"], str(r["step"])) if r["step"] != previous else ""
            previous = r["step"]
            n = "" if r["n"] is None or pd.isna(r["n"]) else f"{int(r['n']):,}"
            cells = [step, r["quantity"], format_value(r["quantity"], r["value"]), r["uncertainty"], n, r["note"]]
            for c, text in enumerate(cells):
                align = Qt.AlignRight if c in (2, 4) else Qt.AlignLeft
                self.number_table.setItem(i, c, table_cell(text, align, ACCENT if c == 0 else TEXT, bold=(c in (0, 2))))
        self._fit(self.number_table)

    def _fill_checks(self, checks: pd.DataFrame):
        self.check_table.setRowCount(len(checks))
        for i, r in enumerate(checks.to_dict("records")):
            colour = VERDICT_COLOUR.get(r["status"], TEXT)
            cells = [r["check"], r["item"], format_value(r["check"], r["value"]), f"{r['low']:g} to {r['high']:g}",
                     r["status"], r["explanation"]]
            for c, text in enumerate(cells):
                align = Qt.AlignRight if c in (2, 3) else Qt.AlignLeft
                self.check_table.setItem(i, c, table_cell(text, align, colour if c == 4 else TEXT, bold=(c == 4)))
        self._fit(self.check_table)

    @staticmethod
    def _fit(table):
        rows = sum(table.rowHeight(i) for i in range(table.rowCount()))
        table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + 2 * table.frameWidth())
