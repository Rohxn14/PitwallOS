"""
Degradation tab of the Analyze window (src/gui/degradation_tab.py) — Phase 2 step 3

The race's fuel-corrected tyre wear: a summary line (fuel assumption and the race's own fuel
check), what the fuel correction does per compound, every stint's wear, and the stints in a table
(the charts' table twin) next to the ledger of laps left out.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QScrollArea, QTableWidget, QVBoxLayout, QWidget

from src.analytics.degradation import FUEL_EFFECT_SEC_PER_KG, MIN_STINT_LAPS, RaceDegradation
from src.gui.analysis_theme import ACCENT, BAD, DIM, TEXT
from src.gui.degradation_charts import draw_fuel_correction, draw_stint_wear
from src.gui.pace_charts import compound_name
from src.gui.pace_tab import ChartCanvas, table_cell

STINT_COLUMNS = ["Driver", "Compound", "Laps in race", "Laps used", "Tyre age", "Raw s/lap", "Corrected s/lap", "± 95 %"]


def _signed(value, digits=3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:+.{digits}f}"


def _table(columns) -> QTableWidget:
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(22)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    return t


class DegradationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[RaceDegradation] = None
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
        self.fuel_canvas = ChartCanvas(250)
        lay.addWidget(self.fuel_canvas)
        self.wear_canvas = ChartCanvas(360)
        lay.addWidget(self.wear_canvas)
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        for box, title in ((left, "Stints (fuel-corrected slope per stint)"), (right, "Laps left out")):
            label = QLabel(title)
            label.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
            box.addWidget(label)
        self.stint_table = _table(STINT_COLUMNS)
        self.stint_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stint_table.setMinimumHeight(300)
        left.addWidget(self.stint_table)
        self.ledger_table = _table(["Reason", "Laps"])
        self.ledger_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ledger_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # sized to show every row
        right.addWidget(self.ledger_table)
        right.addStretch()
        row.addLayout(left, 3)
        row.addLayout(right, 2)
        lay.addLayout(row)
        note = QLabel(
            f"Fuel correction: lap time minus {FUEL_EFFECT_SEC_PER_KG} s per kg of fuel on board (110 kg at the start, "
            "burned evenly to 5 kg), i.e. the lap as if the tank were empty. Slopes use step 1's pace laps on slick "
            f"tyres, on a dry track, in stints of {MIN_STINT_LAPS} or more such laps. The race's own fuel check fits "
            "the fuel effect from the data (possible because tyre age resets at a stop and fuel does not); it also "
            "contains track evolution. A negative slope after correction means something made laps faster faster "
            "than the tyres wore (in 2023-2025: Qatar both years and Las Vegas 2023, likely a track gaining rubber).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

    def set_race(self, result: RaceDegradation):
        self.result = result
        parts = []
        if result.mixed:
            parts.append("MIXED CONDITIONS: someone ran inters or wets, so the track kept changing; read these slopes "
                         "with care (left out of season-wide figures)")
        for r in result.compounds.dropna(subset=["slope"]).to_dict("records"):
            parts.append(f"{compound_name(r['compound'])} {r['slope']:+.3f} ± {1.96 * r['se']:.3f} s/lap")
        fuel = result.fuel
        burn = result.compounds["fuel_per_lap"].dropna()
        assumed = f"fuel assumed {FUEL_EFFECT_SEC_PER_KG} s/kg" + (f" (= {burn.iloc[0]:.3f} s/lap here)" if len(burn) else "")
        parts.append(assumed)
        if not np.isnan(fuel["fuel_sec_per_kg"]):
            parts.append(f"this race's fuel check {fuel['fuel_sec_per_kg']:.4f} ± {1.96 * fuel['fuel_se']:.4f} s/kg")
        self.summary_lbl.setStyleSheet(f"color: {BAD if result.mixed else DIM};")
        self.summary_lbl.setText("  ·  ".join(parts))
        self.fuel_canvas.set_height(110 + 46 * max(len(result.compounds), 1))
        self.fuel_canvas.show_points(draw_fuel_correction(self.fuel_canvas.figure, result))
        self.wear_canvas.show_points(draw_stint_wear(self.wear_canvas.figure, result))
        self._fill_tables(result)

    def show_error(self, message: str):
        self.result = None
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not compute the degradation: {message}")
        for table in (self.stint_table, self.ledger_table):
            table.setRowCount(0)
        for canvas in (self.fuel_canvas, self.wear_canvas):
            canvas.figure.clear()
            canvas.show_points([])

    def _fill_tables(self, result: RaceDegradation):
        stints = result.stints
        self.stint_table.setRowCount(len(stints))
        for i, s in enumerate(stints.to_dict("records")):
            cells = [s["driver_code"], compound_name(s["compound"]), f"{s['first_lap']}-{s['last_lap']}", str(s["laps"]),
                     f"{s['age_first']}-{s['age_last']}", _signed(s["raw_slope"]), _signed(s["slope"]),
                     "—" if pd.isna(s["se"]) else f"{1.96 * s['se']:.3f}"]
            for c, text in enumerate(cells):
                self.stint_table.setItem(i, c, table_cell(text, Qt.AlignLeft if c < 2 else Qt.AlignRight, TEXT,
                                                          bold=(c == 6)))
        ledger = result.ledger
        self.ledger_table.setRowCount(len(ledger))
        for i, r in enumerate(ledger.to_dict("records")):
            ends = r["reason"] in ("pace laps", "used for slopes")
            colour = ACCENT if r["reason"] == "used for slopes" else (DIM if r["laps"] == 0 and not ends else TEXT)
            self.ledger_table.setItem(i, 0, table_cell(r["reason"], Qt.AlignLeft, colour, bold=ends))
            self.ledger_table.setItem(i, 1, table_cell(f"{r['laps']:,}", Qt.AlignRight, colour, bold=ends))
        rows = sum(self.ledger_table.rowHeight(i) for i in range(self.ledger_table.rowCount()))
        self.ledger_table.setFixedHeight(self.ledger_table.horizontalHeader().sizeHint().height() + rows
                                         + 2 * self.ledger_table.frameWidth())
