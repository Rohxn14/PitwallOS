"""
Undercuts tab of the Analyze window (src/gui/undercuts_tab.py) — Phase 2 step 5

The race's undercut attempts: a summary line, the chart of each attempt's gap before and after, the
attempts in a table (the chart's table twin, with the gain split), the selected attempt lap by lap,
and the ledger saying what happened at every other tyre change.
"""

from __future__ import annotations

from typing import Optional

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
from src.gui.undercuts_charts import draw_attempts, is_yes

ATTEMPT_COLUMNS = ["B", "Car ahead", "Team-mates", "Stop laps", "Gap before", "Gap after", "Gain", "Stop", "Track",
                   "Between", "Outcome"]
ATTEMPT_TIPS = {
    "Stop laps": "B's in-lap / the car ahead's in-lap",
    "Gap before": "B's crossing minus the car ahead's, end of the lap before B's in-lap (s)",
    "Gap after": "the same at the end of the car ahead's out-lap; negative: B is ahead (s)",
    "Gain": "gap before - gap after = stop + track + between (s)",
    "Stop": "the car ahead's pit-lane time minus B's, entry to exit (s)",
    "Track": "the track around the stops: both cars' in-laps and out-laps on track (s)",
    "Between": "full racing laps between B's out-lap and the car ahead's in-lap, B on new tyres (s)",
}
SELECTED = "#26394D"          # a muted blue for the selected attempt, quieter than the default highlight
LAP_COLUMNS = ["Lap", "B", "B's lap", "Car ahead", "Its lap", "Gap at the line", "B gained"]

NOTE = (
    f"An undercut attempt: B stops under green on dry tyres less than {MAX_GAP_SEC:g} s behind the car directly "
    f"ahead, which stops 1 to {MAX_RESPONSE_LAPS} laps later, with no safety car, VSC or red flag in between. It "
    "worked if B is ahead at the line when the car ahead completes its out-lap. Gain = gap before - gap after, in "
    "three parts that add up to it: the stop (pit-lane time, entry to exit), the laps between (full racing laps, B on "
    "new tyres against old ones; only when the car ahead waits 3 laps), and the track around the stops (the rest: "
    "both cars' in-laps and out-laps on track). 2023-2025: 67 of 148 attempts worked (45 %); 76 % from under 1 s, "
    "23 % from 2-3 s; 38 % when the car ahead stopped the next lap, 70 % when it waited 3 laps. Only the car directly "
    "ahead is considered."
)


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
        self.attempt_table.setStyleSheet(f"QTableWidget {{ selection-background-color: {SELECTED}; "
                                         f"selection-color: {TEXT}; }}")
        for c, name in enumerate(ATTEMPT_COLUMNS):
            if name in ATTEMPT_TIPS:
                self.attempt_table.horizontalHeaderItem(c).setToolTip(ATTEMPT_TIPS[name])
        self.lap_table = _table(LAP_COLUMNS)
        self.ledger_table = _table(["What happened at the stop", "Stops"])
        for table in (self.attempt_table, self.lap_table):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ledger_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.lap_title = QLabel("The selected attempt, lap by lap")
        for box, label, table in ((left, QLabel("Attempts (select one to read it lap by lap)"), self.attempt_table),
                                  (left, self.lap_title, self.lap_table),
                                  (right, QLabel("Every tyre change: the first reason it is not an attempt"),
                                   self.ledger_table)):
            label.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
            box.addWidget(label)
            box.addWidget(table)
        for table in (self.attempt_table, self.lap_table, self.ledger_table):
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
        for table in (self.attempt_table, self.lap_table, self.ledger_table):
            table.setRowCount(0)
        self.attempt_canvas.figure.clear()
        self.attempt_canvas.show_points([])

    def _fill_tables(self, result: RaceUndercuts):
        attempts = result.attempts
        self.attempt_table.setRowCount(len(attempts))
        for i, a in enumerate(attempts.to_dict("records")):
            cells = [a["driver_code"], a["car_ahead"], "yes" if is_yes(a["teammates"]) else "",
                     f"{int(a['in_lap'])} / {int(a['car_ahead_in_lap'])}", f"{a['gap_before_sec']:+.3f}",
                     f"{a['gap_after_sec']:+.3f}", _signed(a["gain_sec"]), _signed(a["stop_gain_sec"]),
                     _signed(a["track_gain_sec"]), _signed(a["between_gain_sec"]),
                     "worked" if a["worked"] else "did not work"]
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
        self.lap_table.setRowCount(0)
        if len(attempts):
            self.attempt_table.selectRow(0)                 # fills the lap table
        self._fit_heights()

    def _show_selected(self):
        rows = self.attempt_table.selectionModel().selectedRows() if self.result is not None else []
        if not rows or self.classified is None:
            self.lap_table.setRowCount(0)
            self._fit_heights()
            return
        attempt = self.result.attempts.iloc[rows[0].row()].to_dict()
        laps = attempt_laps(self.classified, attempt)
        self.lap_title.setText(f"{attempt['driver_code']} on {attempt['car_ahead']}, lap by lap "
                               "(gap: B's crossing minus the car ahead's)")
        self.lap_table.setRowCount(len(laps))
        for i, r in enumerate(laps.to_dict("records")):
            cells = [str(int(r["lap"])), "—" if pd.isna(r["b_lap_sec"]) else lap_clock(r["b_lap_sec"]), r["b_kind"],
                     "—" if pd.isna(r["a_lap_sec"]) else lap_clock(r["a_lap_sec"]), r["a_kind"],
                     f"{r['gap_sec']:+.3f}", _signed(r["b_gained_sec"])]
            for c, text in enumerate(cells):
                self.lap_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c in (2, 4) else Qt.AlignRight, TEXT))
        self._fit_heights()

    def _fit_heights(self):
        for table in (self.attempt_table, self.lap_table, self.ledger_table):
            rows = sum(table.rowHeight(i) for i in range(table.rowCount()))
            table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + 2 * table.frameWidth())
