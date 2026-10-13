"""
Undercuts tab of the Analyze window (src/gui/undercuts_tab.py) — Phase 2 step 5

The race's undercut attempts: a summary line, the chart of each attempt's gap before and after, the
attempts in a table (the chart's table twin, with the gain split), the selected attempt read in full
(its outcome in words, both cars' stops against the race's typical stop, and lap by lap), and the
ledger saying what happened at every other tyre change.

Everything a reader sees names the drivers: the analytics call the driver who stops first B and the
car ahead A, the tab never does.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QScrollArea, QVBoxLayout, QWidget

from src.analytics.undercuts import (
    A_FIRST, A_LATE, A_NEVER, MAX_GAP_SEC, MAX_RESPONSE_LAPS, SAME_LAP, RaceUndercuts, attempt_laps,
)
from src.gui.analysis_theme import ACCENT, BAD, DIM, TEXT
from src.gui.degradation_tab import _signed, _table
from src.gui.pace_charts import lap_clock
from src.gui.pace_tab import ChartCanvas, table_cell
from src.gui.undercuts_charts import draw_attempts, gap_words, is_yes

ATTEMPT_COLUMNS = ["Driver", "Car ahead", "Team-mates", "Stop laps", "Gap before", "Gap after", "Gain", "Stop", "Track",
                   "Between", "Outcome"]
ATTEMPT_TIPS = {
    "Driver": "the driver who stopped first: the one trying the undercut",
    "Car ahead": "the car one position ahead when the driver stopped",
    "Stop laps": "the driver's in-lap / the car ahead's in-lap",
    "Gap before": "the driver's gap to the car ahead at the line, end of the lap before the driver's in-lap (s)",
    "Gap after": "the same at the end of the car ahead's out-lap, once both are on new tyres (s)",
    "Gain": "how much the driver gained: gap before - gap after = stop + track + between (s)",
    "Stop": "the car ahead's pit-lane time minus the driver's, entry to exit (s): positive = the driver's stop was "
            "quicker",
    "Track": "the track around the stops: both cars' in-laps and out-laps on track (s)",
    "Between": "full racing laps between the driver's out-lap and the car ahead's in-lap, new tyres against old (s)",
}
STOP_COLUMNS = ["Driver", "In-lap", "Pit lane, s", "vs typical", "Stationary, s", "vs typical"]
STOP_TIPS = {
    "Pit lane, s": "pit entry to pit exit, from the timing",
    "Stationary, s": "the car standing still in its box: F1's own figure, published since the 2024 United States GP",
    "vs typical": "minus the race's typical stop (the median of its green stops): positive = slower than usual",
}
SELECTED = "#26394D"          # a muted blue for anything selected, quieter than the default highlight
TABLE_STYLE = f"QTableWidget {{ selection-background-color: {SELECTED}; selection-color: {TEXT}; }}"

NOTE = (
    f"An undercut attempt: a driver stops under green on dry tyres less than {MAX_GAP_SEC:g} s behind the car directly "
    f"ahead, which stops 1 to {MAX_RESPONSE_LAPS} laps later, with no safety car, VSC or red flag in between. It "
    "worked if the driver is ahead at the line when the car ahead completes its out-lap. Gain = gap before - gap "
    "after, in three parts that add up to it: the stop (pit-lane time, entry to exit), the laps between (full racing "
    "laps on new tyres against old ones; only when the car ahead waits 3 laps), and the track around the stops (the "
    "rest: both cars' in-laps and out-laps on track). Stationary times are F1's own, published since the 2024 United "
    "States GP. 2023-2025: 67 of 148 attempts worked (45 %); 76 % from under 1 s, 23 % from 2-3 s; 38 % when the car "
    "ahead stopped the next lap, 70 % when it waited 3 laps. Only the car directly ahead is considered."
)


def outcome_words(attempt: Dict[str, Any]) -> str:
    """One attempt's outcome in a sentence, with the drivers' codes and the gain split."""
    b, a = attempt["driver_code"], attempt["car_ahead"]
    gain = float(attempt["gain_sec"])
    verdict = "Worked" if is_yes(attempt["worked"]) else "Did not work"
    still_b, still_a = attempt.get("stationary_sec"), attempt.get("car_ahead_stationary_sec")
    stationary = ("" if pd.isna(still_b) or pd.isna(still_a)
                  else f"; stationary {b} {still_b:.1f} s, {a} {still_a:.1f} s")
    parts = [f"stops {gain_part(attempt['stop_gain_sec'])} ({a}'s pit-lane time minus {b}'s{stationary})",
             f"track around the stops {gain_part(attempt['track_gain_sec'])}"]
    if not pd.isna(attempt["between_gain_sec"]) and attempt["between_gain_sec"] != 0:
        parts.append(f"laps between {gain_part(attempt['between_gain_sec'])}")
    return (f"{verdict}: {b} went from {gap_words(attempt['gap_before_sec'])} {a} to "
            f"{gap_words(attempt['gap_after_sec'])}, {'gaining' if gain >= 0 else 'losing'} {abs(gain):.3f} s. "
            f"Where it came from: {', '.join(parts)}.")


def gain_part(value) -> str:
    return "—" if pd.isna(value) else f"{value:+.3f} s"


def _versus(value, typical, digits) -> str:
    return "—" if pd.isna(value) or pd.isna(typical) else f"{value - typical:+.{digits}f}"


def _value(value, digits) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


class UndercutsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[RaceUndercuts] = None
        self.classified: Optional[pd.DataFrame] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        outer.addWidget(self.summary_lbl)

        page = QWidget(objectName="scrollpage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.attempt_canvas = ChartCanvas(220)
        lay.addWidget(self.attempt_canvas)
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        self.attempt_table = _table(ATTEMPT_COLUMNS)
        self.attempt_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.attempt_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.attempt_table.itemSelectionChanged.connect(self._show_selected)
        self.stop_table = _table(STOP_COLUMNS)
        self.lap_table = _table(["Lap", "Driver", "Driver's lap", "Car ahead", "Its lap", "Gap at the line",
                                 "Driver gained"])
        self.ledger_table = _table(["What happened at the stop", "Stops"])
        for table, tips in ((self.attempt_table, ATTEMPT_TIPS), (self.stop_table, STOP_TIPS)):
            for c in range(table.columnCount()):
                name = table.horizontalHeaderItem(c).text()
                if name in tips:
                    table.horizontalHeaderItem(c).setToolTip(tips[name])
        for table in (self.attempt_table, self.stop_table, self.lap_table):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for name in ("Gap before", "Gap after", "Outcome"):           # words: as wide as they need
            self.attempt_table.horizontalHeader().setSectionResizeMode(ATTEMPT_COLUMNS.index(name),
                                                                       QHeaderView.ResizeToContents)
        self.stop_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.lap_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.ledger_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        self.lap_title = QLabel("The selected attempt")
        self.outcome_lbl = QLabel("")
        self.outcome_lbl.setWordWrap(True)
        self.outcome_lbl.setStyleSheet(f"color: {TEXT};")
        self.stops_note = QLabel("")
        self.stops_note.setWordWrap(True)
        self.stops_note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        attempts_lbl = QLabel("Attempts (select one to read it in full)")
        ledger_lbl = QLabel("Every tyre change: the first reason it is not an attempt")
        for label in (attempts_lbl, self.lap_title, ledger_lbl):
            label.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        for widget in (attempts_lbl, self.attempt_table, self.lap_title, self.outcome_lbl, self.stop_table,
                       self.stops_note, self.lap_table):
            left.addWidget(widget)
        right.addWidget(ledger_lbl)
        right.addWidget(self.ledger_table)
        for table in (self.attempt_table, self.stop_table, self.lap_table, self.ledger_table):
            table.setStyleSheet(TABLE_STYLE)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)          # sized to show every row
        left.addStretch()
        right.addStretch()
        row.addLayout(left, 3)
        row.addLayout(right, 2)
        lay.addLayout(row)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

    def set_race(self, result: RaceUndercuts, classified: pd.DataFrame):
        self.result, self.classified = result, classified
        s = result.summary
        ledger = result.ledger.set_index("reason")["count"]
        if s["attempts"]:
            parts = [f"{s['attempts']} undercut attempt{'s' if s['attempts'] != 1 else ''}: {s['worked']} worked",
                     f"median gain {s['gain_median']:+.2f} s"]
        else:
            parts = ["No undercut attempt in this race"]
        close = {SAME_LAP: "covered on the same lap", A_FIRST: "the car ahead had stopped first",
                 A_LATE: "the car ahead stopped 4+ laps later", A_NEVER: "the car ahead never stopped again"}
        others = [f"{label} {int(ledger[reason])}" for reason, label in close.items() if ledger[reason]]
        if others:
            parts.append("other close green stops: " + ", ".join(others))
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        self.summary_lbl.setText("  ·  ".join(parts))
        self.attempt_canvas.set_height(150 + 52 * max(len(result.attempts), 1))
        self.attempt_canvas.show_points(draw_attempts(self.attempt_canvas.figure, result))
        self._fill_tables(result)

    def show_error(self, message: str):
        self.result, self.classified = None, None
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not measure the undercuts: {message}")
        for table in (self.attempt_table, self.stop_table, self.lap_table, self.ledger_table):
            table.setRowCount(0)
        self.outcome_lbl.setText("")
        self.stops_note.setText("")
        self.attempt_canvas.figure.clear()
        self.attempt_canvas.show_points([])

    def _fill_tables(self, result: RaceUndercuts):
        attempts = result.attempts
        self.attempt_table.setRowCount(len(attempts))
        for i, a in enumerate(attempts.to_dict("records")):
            cells = [a["driver_code"], a["car_ahead"], "yes" if is_yes(a["teammates"]) else "",
                     f"{int(a['in_lap'])} / {int(a['car_ahead_in_lap'])}", gap_words(a["gap_before_sec"]),
                     gap_words(a["gap_after_sec"]), _signed(a["gain_sec"]), _signed(a["stop_gain_sec"]),
                     _signed(a["track_gain_sec"]), _signed(a["between_gain_sec"]),
                     "worked" if is_yes(a["worked"]) else "did not work"]
            outcome = len(cells) - 1
            for c, text in enumerate(cells):
                align = Qt.AlignLeft if c in (0, 1, 2, outcome) else Qt.AlignRight
                self.attempt_table.setItem(i, c, table_cell(text, align, TEXT, bold=(c in (6, outcome))))
        ledger = result.ledger
        self.ledger_table.setRowCount(len(ledger))
        for i, r in enumerate(ledger.to_dict("records")):
            ends = r["reason"] in ("tyre changes", "undercut attempts")
            colour = ACCENT if r["reason"] == "undercut attempts" else (DIM if r["count"] == 0 and not ends else TEXT)
            self.ledger_table.setItem(i, 0, table_cell(r["reason"], Qt.AlignLeft, colour, bold=ends))
            self.ledger_table.setItem(i, 1, table_cell(f"{r['count']:,}", Qt.AlignRight, colour, bold=ends))
        self._clear_selected()
        if len(attempts):
            self.attempt_table.selectRow(0)                 # fills the selected-attempt tables
        self._fit_heights()

    def _clear_selected(self):
        self.lap_title.setText("The selected attempt")
        self.outcome_lbl.setText("")
        self.stops_note.setText("")
        self.stop_table.setRowCount(0)
        self.lap_table.setRowCount(0)

    def _show_selected(self):
        rows = self.attempt_table.selectionModel().selectedRows() if self.result is not None else []
        if not rows or self.classified is None:
            self._clear_selected()
            self._fit_heights()
            return
        attempt = self.result.attempts.iloc[rows[0].row()].to_dict()
        b, a = attempt["driver_code"], attempt["car_ahead"]
        self.lap_title.setText(f"{b} on {a}")
        self.outcome_lbl.setText(outcome_words(attempt))
        self._fill_stops(attempt)
        self._fill_laps(attempt)
        self._fit_heights()

    def _fill_stops(self, attempt: Dict[str, Any]):
        """Both cars' stops, and the race's typical stop to read them against."""
        typical = self.result.typical or {}
        lane_t, still_t = typical.get("pit_lane_sec", float("nan")), typical.get("stationary_sec", float("nan"))
        rows = [(attempt["driver_code"], str(int(attempt["in_lap"])), attempt["pit_lane_sec"],
                 attempt["stationary_sec"], False),
                (attempt["car_ahead"], str(int(attempt["car_ahead_in_lap"])), attempt["car_ahead_pit_lane_sec"],
                 attempt["car_ahead_stationary_sec"], False),
                ("Race typical (median green stop)", "", lane_t, still_t, True)]
        self.stop_table.setRowCount(len(rows))
        for i, (who, lap, lane, still, is_typical) in enumerate(rows):
            cells = [who, lap, _value(lane, 3), "" if is_typical else _versus(lane, lane_t, 2),
                     _value(still, 1), "" if is_typical else _versus(still, still_t, 1)]
            for c, text in enumerate(cells):
                self.stop_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c < 2 else Qt.AlignRight,
                                                         DIM if is_typical else TEXT))
        if pd.isna(still_t) and pd.isna(attempt["stationary_sec"]):
            self.stops_note.setText("Stationary times: F1 publishes them since the 2024 United States GP; "
                                    "there are none for this race.")
        else:
            self.stops_note.setText("")

    def _fill_laps(self, attempt: Dict[str, Any]):
        b, a = attempt["driver_code"], attempt["car_ahead"]
        laps = attempt_laps(self.classified, attempt)
        self.lap_table.setHorizontalHeaderLabels(["Lap", b, f"{b}'s lap", a, f"{a}'s lap", f"{b} at the line",
                                                  f"{b} gained"])
        self.lap_table.horizontalHeaderItem(5).setToolTip(f"where {b} is at the end of the lap: behind or ahead of {a}")
        self.lap_table.setRowCount(len(laps))
        for i, r in enumerate(laps.to_dict("records")):
            cells = [str(int(r["lap"])), "—" if pd.isna(r["b_lap_sec"]) else lap_clock(r["b_lap_sec"]), r["b_kind"],
                     "—" if pd.isna(r["a_lap_sec"]) else lap_clock(r["a_lap_sec"]), r["a_kind"],
                     gap_words(r["gap_sec"]), _signed(r["b_gained_sec"])]
            for c, text in enumerate(cells):
                self.lap_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c in (2, 4) else Qt.AlignRight, TEXT))

    def _fit_heights(self):
        for table in (self.attempt_table, self.stop_table, self.lap_table, self.ledger_table):
            rows = sum(table.rowHeight(i) for i in range(table.rowCount()))
            table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + 2 * table.frameWidth())
