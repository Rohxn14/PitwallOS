"""
Dirty air tab of the Analyze window (src/gui/dirty_air_tab.py) — Phase 2 step 4

The race's cost of following: a summary line (beta, its laps, and what it does to step 3's wear
slopes), the chart of the cost by gap, and three tables: the cost per gap (the chart's table twin),
how much each driver spent following and behind whom, and the ledger of laps left out.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QScrollArea, QVBoxLayout, QWidget

from src.analytics.dirty_air import CLEAR_AIR_SEC, CLEAR_LABEL, FOLLOW_THRESHOLD_SEC, MIN_LAPS, RaceDirtyAir
from src.gui.analysis_theme import ACCENT, BAD, DIM, TEXT
from src.gui.degradation_tab import _signed, _table
from src.gui.dirty_air_charts import draw_following_cost
from src.gui.pace_charts import compound_name
from src.gui.pace_tab import ChartCanvas, table_cell

BIN_COLUMNS = ["Gap at the lap's start", "Laps", "Cost s/lap", "± 95 %"]
DRIVER_COLUMNS = ["Driver", "Laps used", "Following", "Share", "Followed most"]

NOTE = (
    f"Following = starting the lap less than {FOLLOW_THRESHOLD_SEC:g} s behind the car ahead on the road (Gold's "
    "dirty-air line). Beta compares those laps with every other lap of the same stint, after fuel and tyre wear; the "
    f"costs per gap compare with laps {CLEAR_AIR_SEC:g} s or more behind, so they are larger: cars 1.5-3 s behind "
    "still lose time. Lap times cannot separate the three things that happen behind another car: losing downforce "
    "in its wake, being held up by it, and DRS (within 1 s) giving time back. The number is all three together; "
    "at Monaco 2025 (+1.37 s/lap, the highest of 2023-2025) it is mostly tactics: teams backed the field up in a "
    "race with a two-stop rule. Laps are step 3's (pace laps on slicks, dry track), minus laps that start within "
    f"{CLEAR_AIR_SEC:g} s of a car on another lap."
)


class DirtyAirTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[RaceDirtyAir] = None
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
        self.cost_canvas = ChartCanvas(400)
        lay.addWidget(self.cost_canvas)
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        self.bin_table = _table(BIN_COLUMNS)
        self.driver_table = _table(DRIVER_COLUMNS)
        for table in (self.bin_table, self.driver_table):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.driver_table.setMinimumHeight(300)
        self.ledger_table = _table(["Reason", "Laps"])
        self.ledger_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for box, title, table in ((left, "Cost per gap (the chart's numbers)", self.bin_table),
                                  (left, "Time spent following, per driver", self.driver_table),
                                  (right, "Laps left out", self.ledger_table)):
            label = QLabel(title)
            label.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
            box.addWidget(label)
            box.addWidget(table)
        for table in (self.bin_table, self.ledger_table):
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)          # sized to show every row
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

    def set_race(self, result: RaceDirtyAir):
        self.result = result
        f = result.following
        parts = []
        if result.mixed:
            parts.append("MIXED CONDITIONS: someone ran inters or wets, so few laps qualify; read with care "
                         "(left out of season-wide figures)")
        if np.isnan(f["beta"]):
            parts.append(f"too few laps for a cost of following ({f['following_laps']} following, "
                         f"{f['laps'] - f['following_laps']} not; {MIN_LAPS} of each needed)")
        else:
            parts.append(f"Following within {FOLLOW_THRESHOLD_SEC:g} s at the lap's start costs "
                         f"{f['beta']:+.3f} ± {1.96 * f['se']:.3f} s/lap (95 %)")
            parts.append(f"{f['following_laps']} of {f['laps']} laps following, {f['stints_both']} stints with both")
        wear = [f"{compound_name(w['compound']).lower()} {w['slope']:+.3f} → {w['slope_with_following']:+.3f}"
                for w in result.wear.dropna(subset=["slope_with_following"]).to_dict("records")]
        if wear:
            parts.append("step 3's wear with following in the fit: " + ", ".join(wear) + " s/lap")
        self.summary_lbl.setStyleSheet(f"color: {BAD if result.mixed else DIM};")
        self.summary_lbl.setText("  ·  ".join(parts))
        self.cost_canvas.show_points(draw_following_cost(self.cost_canvas.figure, result))
        self._fill_tables(result)

    def show_error(self, message: str):
        self.result = None
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not compute the cost of following: {message}")
        for table in (self.bin_table, self.driver_table, self.ledger_table):
            table.setRowCount(0)
        self.cost_canvas.figure.clear()
        self.cost_canvas.show_points([])

    def _fill_tables(self, result: RaceDirtyAir):
        bins = result.bins
        self.bin_table.setRowCount(len(bins))
        for i, b in enumerate(bins.to_dict("records")):
            reference = b["bin"] == CLEAR_LABEL
            cells = [b["bin"] + (" (reference)" if reference else ""), f"{b['laps']:,}",
                     "0 by definition" if reference else _signed(b["cost"]),
                     "—" if pd.isna(b["se"]) else f"{1.96 * b['se']:.3f}"]
            for c, text in enumerate(cells):
                self.bin_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c == 0 else Qt.AlignRight,
                                                        DIM if reference else TEXT, bold=(c == 2 and not reference)))
        drivers = result.drivers
        self.driver_table.setRowCount(len(drivers))
        for i, d in enumerate(drivers.to_dict("records")):
            most = "—" if d["most_followed"] is None else f"{d['most_followed']} ({d['most_followed_laps']} laps)"
            cells = [d["driver_code"], str(d["laps"]), str(d["following_laps"]), f"{100 * d['share']:.0f} %", most]
            for c, text in enumerate(cells):
                self.driver_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c in (0, 4) else Qt.AlignRight, TEXT))
        ledger = result.ledger
        self.ledger_table.setRowCount(len(ledger))
        for i, r in enumerate(ledger.to_dict("records")):
            ends = r["reason"] in ("pace laps", "used for the cost of following")
            colour = ACCENT if r["reason"] == "used for the cost of following" else (
                DIM if r["laps"] == 0 and not ends else TEXT)
            self.ledger_table.setItem(i, 0, table_cell(r["reason"], Qt.AlignLeft, colour, bold=ends))
            self.ledger_table.setItem(i, 1, table_cell(f"{r['laps']:,}", Qt.AlignRight, colour, bold=ends))
        for table in (self.bin_table, self.ledger_table):
            rows = sum(table.rowHeight(i) for i in range(table.rowCount()))
            table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + 2 * table.frameWidth())
