"""
Pace tab of the Analyze window (src/gui/pace_tab.py) — Phase 2 step 1

What step 1 says about one race, on screen. The driver selector (one row, above everything it
scopes) picks the lap chart's driver and highlights that driver in the driver chart; the compound
chart and the ledger cover the whole race.

The charts are drawn by src/gui/pace_charts.py (matplotlib, no Qt). This module places them,
adds a tooltip for the point under the pointer, and fills the exclusion ledger. The table twin
of the lap chart is the Laps tab's "Pace classification" view (PACE_TABLE_COLUMNS).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
# PySide6 before matplotlib's Qt backend: the backend binds to the Qt library already imported, so it
# can never pick another one (PyQt6, if installed) and mix two Qt libraries in one window.
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402  (after PySide6, see above)
from matplotlib.figure import Figure

from src.analytics.laps import (
    MIN_REFERENCE_LAPS, OUTLIER_THRESHOLD_SEC, REASON_GENERATED, REASON_INACCURATE, REASON_LAP_1,
    REASON_NO_LAP_TIME, REASON_OTHER_STATUS, REASON_OUTLIER, REASON_PIT_LAP, REASON_RED_FLAG, REASON_SAFETY_CAR,
    REASON_VSC, REASON_YELLOW, REFERENCE_WINDOW_LAPS, RacePace,
)
from src.analytics.neutralisations import describe_periods
from src.gui import pace_charts as charts
from src.gui.analysis_theme import ACCENT, BAD, DIM, TEXT

# The Laps tab's "Pace classification" view: every lap with its status and the numbers behind it
PACE_TABLE_COLUMNS = ["driver_code", "lap_number", "lap_time_sec", "compound", "tyre_life", "tyre_run",
                      "track_status", "pace_status", "reference_sec", "over_reference_sec", "outlier_checked",
                      "is_deleted"]

REASON_NOTES = {
    "all laps": "Every lap in Silver for this race. Nothing is deleted: each lap gets exactly one status.",
    REASON_LAP_1: "The standing start: every driver's first lap.",
    REASON_RED_FLAG: "Track status 5 during the lap: the session was stopped.",
    REASON_SAFETY_CAR: "Track status 4 during the lap. A stop made under the safety car counts here.",
    REASON_VSC: "Track status 6 or 7 (virtual safety car) during the lap.",
    REASON_YELLOW: "Track status 2 during the lap: a yellow flag somewhere on the track.",
    REASON_OTHER_STATUS: "Not green for the whole lap for another reason, e.g. no status recorded.",
    REASON_PIT_LAP: "In-lap or out-lap with nothing else wrong: the time includes the pit lane.",
    REASON_NO_LAP_TIME: "Green, not a pit lap, yet the timing has no lap time.",
    REASON_INACCURATE: "FastF1 marks the lap time as not accurate.",
    REASON_GENERATED: "A row FastF1 added itself; its times are made up.",
    REASON_OUTLIER: f"More than {OUTLIER_THRESHOLD_SEC:g} s slower than the median of the {REFERENCE_WINDOW_LAPS} "
                    "clean laps around it on the same set of tyres.",
    "pace laps": "What steps 3 and 4 measure from.",
}

NOTE = (
    "How a pace lap is chosen (Phase 2 step 1, design note of 2026-10-12). Every lap gets one status, the first "
    "reason in the ledger's order that applies, so the ledger adds up to every lap. A pace lap is clean (timed, "
    "accurate, green for the whole lap, not lap 1, not a pit lap) and not an outlier: more than "
    f"{OUTLIER_THRESHOLD_SEC:g} s slower than the median of the {REFERENCE_WINDOW_LAPS} clean laps around it on the "
    "same set of tyres. That reference moves with tyre wear and fuel, so only a sudden jump counts (a mistake, the "
    "first lap of a rain shower). A set of tyres with fewer than "
    f"{MIN_REFERENCE_LAPS} clean laps cannot be checked: its laps stay, counted as unchecked. Wet-tyre and deleted "
    "laps stay pace laps; pace is grouped by compound instead. Known limit: the first green lap after a safety car is "
    "slow (a median 0.8 s over 2023-2025), but so are the laps right after it, so the rule rarely flags it. "
    "Neutralised periods come from the official track status, counted in the leader's laps as Race Control does."
)


class ChartCanvas(FigureCanvasQTAgg):
    """A matplotlib figure in the window, with a tooltip for the point under the pointer."""

    def __init__(self, height_px: int, parent=None):
        super().__init__(Figure(facecolor=charts.SURFACE, layout="constrained"))
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.set_height(height_px)
        self.points: List[charts.HoverPoints] = []
        self._tips: Dict[object, object] = {}
        self._shown = None                  # (chart index, point index) under the pointer
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("figure_leave_event", lambda _event: self._hide())

    def set_height(self, height_px: int):
        """Exactly this tall (the figure's own default size would win otherwise); the width follows the window."""
        self.setMinimumHeight(height_px)
        self.setMaximumHeight(height_px)

    def show_points(self, points: List[charts.HoverPoints]):
        """Call after a chart was drawn into self.figure: remember its points and repaint."""
        self.points, self._tips, self._shown = points, {}, None
        self.draw_idle()

    def _tip(self, ax):
        tip = self._tips.get(ax)
        if tip is None:
            tip = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", fontsize=8,
                              color=charts.INK, zorder=20, annotation_clip=False,
                              bbox={"boxstyle": "round,pad=0.5", "fc": charts.TIP_FACE, "ec": charts.TIP_EDGE})
            tip.set_in_layout(False)        # a tooltip must never make the layout move
            tip.set_visible(False)
            self._tips[ax] = tip
        return tip

    def _hide(self):
        if self._shown is None:
            return
        for tip in self._tips.values():
            tip.set_visible(False)
        self._shown = None
        self.draw_idle()

    def _on_move(self, event):
        if event.x is None or event.y is None or not self.points:
            self._hide()
            return
        radius = charts.HOVER_RADIUS_PX * float(getattr(self, "device_pixel_ratio", 1) or 1)
        best = None                         # (squared distance, chart index, point index)
        for k, hp in enumerate(self.points):
            if len(hp.x) == 0:
                continue
            px = hp.ax.transData.transform(np.column_stack([hp.x, hp.y]))
            i = charts.nearest(px, event.x, event.y, radius)
            if i is not None:
                d2 = float(((px[i] - np.array([event.x, event.y])) ** 2).sum())
                if best is None or d2 < best[0]:
                    best = (d2, k, i)
        if best is None:
            self._hide()
            return
        _, k, i = best
        if self._shown == (k, i):
            return
        hp = self.points[k]
        for ax, tip in self._tips.items():
            if ax is not hp.ax:
                tip.set_visible(False)
        tip = self._tip(hp.ax)
        fx, fy = hp.ax.transAxes.inverted().transform(hp.ax.transData.transform((hp.x[i], hp.y[i])))
        right, top = fx > 0.55, fy > 0.55   # open the box towards the middle of the plot
        tip.xy = (hp.x[i], hp.y[i])
        tip.xyann = (-12 if right else 12, -12 if top else 12)
        tip.set_horizontalalignment("right" if right else "left")
        tip.set_verticalalignment("top" if top else "bottom")
        tip.set_text(hp.text[i])
        tip.set_visible(True)
        self._shown = (k, i)
        self.draw_idle()


def table_cell(text: str, align=Qt.AlignCenter, colour: Optional[str] = None, bold=False, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align | Qt.AlignVCenter)
    if colour:
        item.setForeground(QColor(colour))
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    if tip:
        item.setToolTip(tip)
    return item


class PaceTab(QWidget):
    """Step 1 for one race: lap chart for one driver, raw compound pace, the ledger, driver pace by compound."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pace: Optional[RacePace] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Driver"))
        self.driver_box = QComboBox()
        self.driver_box.setMinimumWidth(110)
        self.driver_box.currentIndexChanged.connect(self._on_driver)
        bar.addWidget(self.driver_box)
        bar.addSpacing(14)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        bar.addWidget(self.summary_lbl, 1)
        outer.addLayout(bar)

        page = QWidget(objectName="scrollpage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.lap_canvas = ChartCanvas(440)
        lay.addWidget(self.lap_canvas)

        row = QHBoxLayout()
        self.compound_canvas = ChartCanvas(400)
        row.addWidget(self.compound_canvas, 3)
        ledger_box = QVBoxLayout()
        title = QLabel("Where the laps went")
        title.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        ledger_box.addWidget(title)
        self.ledger_table = QTableWidget(0, 4)
        self.ledger_table.setHorizontalHeaderLabels(["Reason", "Laps", "Share", "Left"])
        self.ledger_table.verticalHeader().setVisible(False)
        self.ledger_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ledger_table.setSelectionMode(QTableWidget.NoSelection)
        self.ledger_table.setAlternatingRowColors(True)
        self.ledger_table.verticalHeader().setDefaultSectionSize(24)
        self.ledger_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # sized to show every row
        header = self.ledger_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        ledger_box.addWidget(self.ledger_table)
        ledger_box.addStretch()
        row.addLayout(ledger_box, 2)
        lay.addLayout(row)

        self.driver_canvas = ChartCanvas(420)
        lay.addWidget(self.driver_canvas)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

    # --- data ---------------------------------------------------------------------------

    def set_race(self, pace: RacePace, order: Sequence[str] = ()):
        """Show one race. `order`: driver codes in result order (the selector lists them first)."""
        self.pace = pace
        present = set(pace.laps["driver_code"].astype(str))
        drivers = [c for c in order if c in present] + sorted(present - set(order))
        self.driver_box.blockSignals(True)
        self.driver_box.clear()
        self.driver_box.addItems(drivers)
        self.driver_box.blockSignals(False)

        s = pace.summary
        self.summary_lbl.setStyleSheet(f"color: {DIM};")
        self.summary_lbl.setText(
            f"Pace laps {s['pace_laps']:,} of {s['laps']:,} ({s['pace_pct']} %)  ·  outliers {s['outliers']}  ·  "
            f"unchecked {s['unchecked']}  ·  neutralised: {describe_periods(pace.periods)}")
        self.compound_canvas.show_points(charts.draw_compound_pace(self.compound_canvas.figure, pace))
        self._fill_ledger(pace)
        self.driver_canvas.set_height(150 + 19 * max(charts.driver_chart_rows(pace), 4))
        self._on_driver()

    def show_error(self, message: str):
        """The race could not be read: say so, and show no number rather than an old one."""
        self.pace = None
        self.driver_box.clear()
        self.summary_lbl.setStyleSheet(f"color: {BAD};")
        self.summary_lbl.setText(f"Could not compute the pace laps: {message}")
        self.ledger_table.setRowCount(0)
        for canvas in (self.lap_canvas, self.compound_canvas, self.driver_canvas):
            canvas.figure.clear()
            canvas.show_points([])

    def _on_driver(self, _index: int = 0):
        if self.pace is None or not self.driver_box.count():
            return
        driver = self.driver_box.currentText()
        self.lap_canvas.show_points(charts.draw_lap_times(self.lap_canvas.figure, self.pace, driver))
        self.driver_canvas.show_points(charts.draw_driver_pace(self.driver_canvas.figure, self.pace, highlight=driver))

    def _fill_ledger(self, pace: RacePace):
        ledger = pace.ledger
        self.ledger_table.setRowCount(len(ledger))
        for i, r in enumerate(ledger.to_dict("records")):
            reason = r["reason"]
            ends = reason in ("all laps", "pace laps")
            colour = ACCENT if reason == "pace laps" else (DIM if r["laps"] == 0 and not ends else TEXT)
            tip = REASON_NOTES.get(reason, "")
            self.ledger_table.setItem(i, 0, table_cell(reason, Qt.AlignLeft, colour, bold=ends, tip=tip))
            self.ledger_table.setItem(i, 1, table_cell(f"{r['laps']:,}", Qt.AlignRight, colour, bold=ends, tip=tip))
            share = "" if reason == "all laps" else f"{100 * r['share']:.1f} %"
            self.ledger_table.setItem(i, 2, table_cell(share, Qt.AlignRight, colour, tip=tip))
            self.ledger_table.setItem(i, 3, table_cell(f"{r['remaining']:,}", Qt.AlignRight, DIM, tip=tip))
        rows = sum(self.ledger_table.rowHeight(i) for i in range(self.ledger_table.rowCount()))
        frame = 2 * self.ledger_table.frameWidth()
        self.ledger_table.setFixedHeight(self.ledger_table.horizontalHeader().sizeHint().height() + rows + frame)
