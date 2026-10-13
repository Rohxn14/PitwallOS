"""
Stops tab of the Analyze window (src/gui/stops_tab.py) — Phase 2 step 2

The race's pit loss and typical stop (summary line), every measured stop on one chart
(src/gui/stops_charts.py, with hover), and every tyre change in a table, including those whose
loss cannot be measured and why: the chart's table twin. The table also says how each stop itself
went: its time in the pit lane and, where F1 published it, the time the car stood still.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QVBoxLayout, QWidget

from src.analytics.stops import GREEN, RED_FLAG, SAFETY_CAR, VSC, RaceStops
from src.gui.analysis_theme import BAD, DIM, TEXT
from src.gui.pace_charts import compound_name
from src.gui.pace_tab import ChartCanvas, table_cell
from src.gui.stops_charts import draw_stop_losses

COLUMNS = ["Driver", "Stop", "In-lap", "Condition", "Tyres", "In-lap s", "Out-lap s", "Pit lane s", "Stationary s",
           "Reference s", "Loss s", "Why no loss"]
TIPS = {"Pit lane s": "pit entry to pit exit, from the timing",
        "Stationary s": "the car standing still in its box: F1's own figure, published since the 2024 United States GP"}
LOSS_COLUMN = COLUMNS.index("Loss s")


def _num(value, digits=3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


class StopsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[RaceStops] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        lay.addWidget(self.summary_lbl)
        self.canvas = ChartCanvas(330)
        lay.addWidget(self.canvas)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        for c, name in enumerate(COLUMNS):
            if name in TIPS:
                self.table.horizontalHeaderItem(c).setToolTip(TIPS[name])
        lay.addWidget(self.table, 1)

    def set_race(self, result: RaceStops):
        self.result = result
        loss = result.pit_loss.set_index("condition")
        parts = []
        for condition, label in ((GREEN, "Circuit pit loss (green stops)"), (SAFETY_CAR, "under a safety car"),
                                 (VSC, "under a VSC")):
            if condition in loss.index and loss.loc[condition, "measured"]:
                r = loss.loc[condition]
                parts.append(f"{label}: {r['median_sec']:.1f} s (middle half {r['q25_sec']:.1f}-{r['q75_sec']:.1f}, "
                             f"n = {int(r['measured'])})")
        if RED_FLAG in loss.index:
            parts.append(f"red-flag tyre changes: {int(loss.loc[RED_FLAG, 'stops'])} (free, no loss)")
        parts.append(f"{len(result.stops)} tyre changes")
        typical = result.typical
        if typical["pit_lane_n"]:
            stop = (f", {typical['stationary_sec']:.1f} s stationary" if typical["stationary_n"]
                    else "; stationary times not published for this race (F1 publishes them since late 2024)")
            parts.append(f"typical green stop: {typical['pit_lane_sec']:.2f} s in the pit lane{stop}")
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        self.summary_lbl.setText("  ·  ".join(parts))
        self.canvas.show_points(draw_stop_losses(self.canvas.figure, result))
        self._fill_table(result.stops)

    def show_error(self, message: str):
        self.result = None
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not compute the stops: {message}")
        self.table.setRowCount(0)
        self.canvas.figure.clear()
        self.canvas.show_points([])

    def _fill_table(self, stops: pd.DataFrame):
        self.table.setRowCount(len(stops))
        for i, r in enumerate(stops.to_dict("records")):
            measured = not pd.isna(r["loss_sec"])
            colour = TEXT if measured else DIM
            ref = "" if pd.isna(r["reference_sec"]) else f"{r['reference_sec']:.3f} ({r['reference_kind']})"
            cells = [r["driver_code"], str(r["stop_number"]), str(r["in_lap"]), r["condition"] or "—",
                     f"{compound_name(r['compound_before'])} → {compound_name(r['compound_after'])}",
                     _num(r["in_lap_sec"]), _num(r["out_lap_sec"]), _num(r["pit_lane_sec"]),
                     _num(r["stationary_sec"], 1), ref or "—", _num(r["loss_sec"], 2), r["loss_note"] or ""]
            for c, text in enumerate(cells):
                align = Qt.AlignLeft if c in (0, 3, 4, len(cells) - 1) else Qt.AlignRight
                self.table.setItem(i, c, table_cell(text, align, colour, bold=(c == LOSS_COLUMN and measured)))
