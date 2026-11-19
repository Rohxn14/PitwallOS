"""
Compare tab of the Analyze window (src/gui/compare_tab.py) - Phase 5 step 5.3

Two drivers of one race, corner by corner: where one gains time on the other and how. Pick both drivers (there is no
default pair: Rohan's choice, so the tab never suggests a story); the tab then shows a summary line, the chart (the gap
building up over a lap, and each corner's gap, src/gui/compare_charts.py) and the table of every corner with the "how".

The numbers come from src/telemetry/compare.py, on the laps both drivers ran clean, lap N against lap N. The race's
corner tables are read once (compare.load_race); each new pair is computed from them in a fraction of a second.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from src.gui.analysis_theme import ACCENT, BAD, DIM, TEXT
from src.gui.compare_charts import corner_tick, draw_comparison
from src.gui.degradation_tab import _table
from src.gui.pace_tab import ChartCanvas, table_cell
from src.telemetry.compare import CONSISTENT, SLOW_KMH, Comparison, NoPairedLaps, RaceCorners, compare_race

PICK = "pick a driver"
TYRE_GAP_WARN_LAPS = 5        # tyres this many laps apart in age (median): said in the summary, it biases the gap
TRAFFIC_WARN_SHARE = 0.25     # one driver following on this much more of the paired laps than the other: said too

NOTE = (
    "Paired laps: only laps both drivers ran clean (green, not lap 1, no pit stop), lap N against lap N, so both "
    "carry the same fuel on the same track. Tyre age is NOT paired: the summary says how far apart it was. "
    "Traffic is not paired either: laps spent following a car (under 1.5 s behind at the lap's start, step 2's "
    "definition) are kept and counted, because leaving them out did not make teammate gaps agree any better from one "
    "race to the next (2023-2024). "
    "Gap = the first driver minus the second: negative = the first is faster. Each corner's segment runs from halfway "
    "after the corner before to halfway before the next, so the segments of a lap add up to the lap. The chart's bars "
    "and the table's first gap are medians (a typical lap); the mean gaps add up exactly to the mean lap gap (one bad "
    "lap moves a mean, not a median). \"How\" names a difference only if the corner's winner has it on at least "
    f"{CONSISTENT:.0%} of the paired laps: braking point (where the brakes go on, before the slowest point), speed at "
    "the corner, exit speed (100 m after the slowest point), throttle back to 50 %. Telemetry: OpenF1, about 3.7 "
    "positions a second resampled to 10; a corner's time is good to about 20 ms on one lap, braking points to about "
    "one sample (8 m at 300 km/h)."
)


def _seconds(value: float) -> str:
    return f"{value:+.3f}"


class CompareTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.race: Optional[RaceCorners] = None
        self.comparison: Optional[Comparison] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Compare"))
        self.box_a, self.box_b = QComboBox(), QComboBox()
        for box in (self.box_a, self.box_b):
            box.setMinimumWidth(120)
            box.currentIndexChanged.connect(self._on_pick)
        bar.addWidget(self.box_a)
        bar.addWidget(QLabel("against"))
        bar.addWidget(self.box_b)
        self.swap_btn = QPushButton("Swap")
        self.swap_btn.setToolTip("Swap the two drivers: every sign flips")
        self.swap_btn.clicked.connect(self._swap)
        bar.addWidget(self.swap_btn)
        bar.addStretch()
        outer.addLayout(bar)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        outer.addWidget(self.summary_lbl)

        page = QWidget(objectName="scrollpage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.canvas = ChartCanvas(640)
        lay.addWidget(self.canvas)
        title = QLabel("Corner by corner")
        title.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        lay.addWidget(title)
        self.table = _table(self._columns("A"))
        self.table.horizontalHeader().setSectionResizeMode(len(self._columns("A")) - 1, QHeaderView.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)          # sized to show every row
        lay.addWidget(self.table)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)
        self._clear("")

    @staticmethod
    def _columns(a: str):
        return ["Corner", "km/h", "Gap (median)", "Gap (mean)", f"{a} faster on", "Faster", "How"]

    # --- data ---------------------------------------------------------------------------

    def set_race(self, race: RaceCorners, order: Sequence[str] = ()):
        """One race's corner tables. `order`: driver codes in result order (listed first)."""
        self.race = race
        present = set(race.drivers)
        drivers = [d for d in order if d in present] + sorted(present - set(order))
        for box in (self.box_a, self.box_b):
            box.blockSignals(True)
            box.clear()
            box.addItems([PICK] + drivers)
            box.setEnabled(True)
            box.blockSignals(False)
        self.swap_btn.setEnabled(True)
        self._clear("Pick two drivers to compare them corner by corner, on the laps both ran clean.")

    def show_error(self, message: str):
        """The race cannot be compared (no telemetry, no track model ...): say why and show nothing else."""
        self.race = None
        for box in (self.box_a, self.box_b):
            box.blockSignals(True)
            box.clear()
            box.setEnabled(False)
            box.blockSignals(False)
        self.swap_btn.setEnabled(False)
        self._clear(f"Cannot compare drivers in this race: {message}", BAD)

    def pick(self, a: str, b: str):
        """Select two drivers (as a user would)."""
        self.box_a.blockSignals(True)
        self.box_a.setCurrentText(a)
        self.box_a.blockSignals(False)
        self.box_b.setCurrentText(b)
        self._on_pick()

    def _swap(self):
        self.pick(self.box_b.currentText(), self.box_a.currentText())

    def _clear(self, message: str, colour: str = DIM):
        self.comparison = None
        self.summary_lbl.setStyleSheet(f"color: {colour};")
        self.summary_lbl.setText(message)
        self.table.setRowCount(0)
        self.canvas.figure.clear()
        self.canvas.show_points([])

    def _on_pick(self, _index: int = 0):
        if self.race is None:
            return
        a, b = self.box_a.currentText(), self.box_b.currentText()
        if PICK in (a, b) or not a or not b:
            self._clear("Pick two drivers to compare them corner by corner, on the laps both ran clean.")
            return
        if a == b:
            self._clear("Pick two different drivers.")
            return
        try:
            c = compare_race(self.race, a, b)
        except NoPairedLaps:
            self._clear(f"{a} and {b} have no lap that both ran clean (green, not lap 1, no pit stop, whole telemetry): "
                        "nothing to compare like for like.")
            return
        self.comparison = c
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        self.summary_lbl.setText(summary(c))
        self.canvas.show_points(draw_comparison(self.canvas.figure, c))
        self._fill_table(c)

    def _fill_table(self, c: Comparison):
        corners = c.corners.sort_values("corner_order")
        self.table.setHorizontalHeaderLabels(self._columns(c.a))
        self.table.setRowCount(len(corners))
        for i, r in enumerate(corners.to_dict("records")):
            winner = r["winner"] or "level"
            cells = [corner_tick(r["corner"]), f"{r['corner_speed_kmh']:.0f}", _seconds(r["gap_median_sec"]),
                     _seconds(r["gap_mean_sec"]), f"{100 * r['a_faster_share']:.0f} %", winner, r["how"]]
            slow = r["corner_speed_kmh"] < SLOW_KMH
            for col, text in enumerate(cells):
                align = Qt.AlignLeft if col in (0, 5, 6) else Qt.AlignRight
                colour = DIM if (col == 6 and r["how"] in ("no single difference", "level")) else TEXT
                self.table.setItem(i, col, table_cell(text, align, colour, bold=(col == 2),
                                                      tip=f"{'slow' if slow else 'fast'} corner ({r['corner_speed_kmh']:.0f} km/h)"))
        rows = sum(self.table.rowHeight(i) for i in range(self.table.rowCount()))
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height() + rows + 2 * self.table.frameWidth())


def summary(c: Comparison) -> str:
    """The comparison in one line: laps, the lap gap, corners won, tyre ages, and any caution."""
    gap_median, gap_mean = c.lap_gap_median_sec, c.lap_gap_mean_sec
    if abs(gap_median) < 0.0005:
        who = f"level on a typical lap (mean {_seconds(gap_mean)} s, {c.a} minus {c.b})"
    else:
        faster = c.a if gap_median < 0 else c.b
        who = f"{faster} faster by {abs(gap_median):.3f} s a lap (median; mean {abs(gap_mean):.3f} s)"
    n = len(c.laps)
    parts = [f"{c.a} against {c.b}: {n} paired lap{'s' if n != 1 else ''}", who,
             f"{c.a} faster in {c.corners_won_by_a} of {len(c.corners)} corners"]
    if abs(gap_mean) >= 0.05:                  # a share of a tiny gap means nothing
        parts.append(f"corners under {SLOW_KMH:.0f} km/h: {_seconds(c.slow_gap_sec())} s of the {_seconds(gap_mean)} s "
                     "mean gap")
    tyre = c.tyre_age_gap
    if np.isfinite(tyre):
        if abs(tyre) < 1:
            parts.append("tyres the same age (median)")
        else:
            older = c.a if tyre > 0 else c.b
            laps = round(abs(tyre))
            parts.append(f"{older}'s tyres {laps} lap{'s' if laps != 1 else ''} older (median)"
                         + (": pairing does not equalise that, read with care" if abs(tyre) >= TYRE_GAP_WARN_LAPS else ""))
    if c.following:
        fa, fb = c.following.get(c.a, 0), c.following.get(c.b, 0)
        traffic = f"following a car at the lap's start: {c.a} {fa}, {c.b} {fb} of the {n} laps"
        if abs(fa - fb) >= TRAFFIC_WARN_SHARE * n:
            traffic += ": the gap includes traffic, read with care"
        parts.append(traffic)
    parts += c.notes
    return "  ·  ".join(parts)
