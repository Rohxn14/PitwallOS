"""
Race Analysis window (src/gui/race_analysis_window.py)

The "Analyze" half of the session picker: one race, read from the data lake. It needs no replay
and no telemetry stream, which is why it is not in the insights menu (those windows are fed live
by a running replay).

If the race is not in the lake yet, the window builds it first (Bronze -> Silver -> Gold, laps
only). FastF1 reads from the same download cache as the replay, so a race that was replayed
before is built without downloading anything.

The tabs show only what the lake holds as facts (src/analytics/race_overview.py). The Phase 2
analytics appear as placeholders until each step is built and golden-tested; the old engine's
numbers (src/analytics/race_engine.py, known wrong) are deliberately not shown.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QRect, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.analytics.race_overview import DNF_MARK, RaceOverview, lake_status, load_overview
from src.pipeline import lake

BG, PANEL, BORDER = "#121212", "#1A1A1A", "#2A2A2A"
TEXT, DIM, ACCENT, GOOD, BAD = "#E0E0E0", "#8A8A8A", "#00E5FF", "#00E676", "#FF5252"

COMPOUND_COLOURS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#E8E8E8",
    "INTERMEDIATE": "#39B54A", "WET": "#0067FF",
}
COMPOUND_TEXT = {"SOFT": "#FFFFFF", "MEDIUM": "#000000", "HARD": "#000000",
                 "INTERMEDIATE": "#FFFFFF", "WET": "#FFFFFF"}

STYLE = f"""
    QMainWindow, QWidget#page, QWidget#scrollpage {{ background: {BG}; color: {TEXT}; }}
    QWidget#scrollpage {{ background: {PANEL}; }}
    QLabel {{ color: {TEXT}; }}
    QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL}; }}
    QTabBar::tab {{ background: #202020; color: #AAAAAA; padding: 8px 18px; border: 1px solid {BORDER}; }}
    QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
    QTableWidget {{ background: #1E1E1E; alternate-background-color: #232323; gridline-color: {BORDER};
                   color: #FFFFFF; border: none; }}
    QHeaderView::section {{ background: #262626; color: {ACCENT}; font-weight: bold; border: 1px solid {BORDER};
                           padding: 4px; }}
    QPushButton {{ background: {ACCENT}; color: #000000; font-weight: bold; border-radius: 4px; padding: 7px 14px; }}
    QPushButton:hover {{ background: #18FFFF; }}
    QPushButton:disabled {{ background: #3A3A3A; color: #777777; }}
    QLineEdit, QComboBox {{ background: #1E1E1E; border: 1px solid #444444; color: #FFFFFF; padding: 5px;
                           border-radius: 4px; }}
    QScrollArea {{ border: none; background: {PANEL}; }}
    QProgressBar {{ background: #1E1E1E; border: 1px solid {BORDER}; border-radius: 4px; height: 8px; }}
    QProgressBar::chunk {{ background: {ACCENT}; }}
"""

# Phase 2 of the roadmap (CLAUDE.md): what each step will add to this window, and its golden test.
PHASE2_STEPS = [
    ("0", "Silver stint and pit-passage repair", "Stops = tyre changes; pass-throughs are not stops.",
     "British 45 tyre changes; Qatar 26 + 34 pass-throughs", True),
    ("1", "Lap selection and neutralisations", "pace_laps() with an exclusion ledger; SC / VSC periods from "
     "the official flags only.", "Qatar SC laps 1-4 and 34-42, VSC lap 40; British none", False),
    ("2", "Stops and pit loss", "One row per tyre change. Loss = in-lap + out-lap - 2 x reference pace; "
     "circuit loss = median of green stops; SC stops measured separately.",
     "Per-driver stop counts; your predicted loss, written first", False),
    ("3", "Fuel-corrected degradation", "Per-stint slope +- SE, per-compound slope pooled across stints, "
     "joint fuel-coefficient cross-check.", "Your expected medium slope, written first", False),
    ("4", "Dirty air", "Fixed-effects coefficient beta +- SE, then gap bins.", "beta >= 0, caveats written down", False),
    ("5", "Undercuts, measured", "Real attempts: gap before and after both stops, outcome, gain split into "
     "in-lap / out-lap / laps between.", "A hand-verified case", False),
    ("6", "Race report", "RaceReport in this window with n, uncertainty and every exclusion; delete "
     "race_engine.py.", "Sanity ranges across the Phase 1 backfill", False),
]


def _cell_text(value) -> str:
    """Missing values show as an em dash, never as an invented number."""
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)


def _item(text, align=Qt.AlignCenter, colour: Optional[str] = None, bold=False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align | Qt.AlignVCenter)
    if colour:
        item.setForeground(QColor(colour))
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    return item


def _table(columns) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return table


def _team_colour(hex_code: str) -> str:
    hex_code = (hex_code or "").strip().lstrip("#")
    return f"#{hex_code}" if len(hex_code) == 6 else TEXT


def _scroll_page(inner: QWidget) -> QScrollArea:
    """A scroll area whose page is painted dark (Qt's default page is light grey)."""
    inner.setObjectName("scrollpage")
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    return scroll


class PipelineBuildThread(QThread):
    """Bronze -> Silver (laps) -> Gold for one race, off the UI thread."""
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, year: int, round_number: int, root: str):
        super().__init__()
        self.year, self.round_number, self.root = year, round_number, root

    def run(self):
        try:
            from src.pipeline.runner import run_pipeline
            self.done.emit(run_pipeline(self.year, self.round_number, "R", lake_root=self.root, telemetry=False))
        except Exception as e:   # shown in the window: a silent failure is how the old preview broke
            self.failed.emit(f"{type(e).__name__}: {e}")


class StintChart(QWidget):
    """Tyre strategy as driven: one row per driver in result order, one bar per stint."""

    ROW_H, AXIS_H, POS_W, CODE_W, RESULT_W = 24, 26, 30, 46, 92

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []          # [(position, code, team_colour, [stint dicts], result)]
        self.total_laps = 1
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_overview(self, ov: RaceOverview):
        colours = dict(zip(ov.classification["driver_code"], ov.classification["team_color"]))
        by_driver = {code: grp.to_dict("records") for code, grp in ov.stints.groupby("driver_code")}
        self.rows = []
        for _, s in ov.strategies.iterrows():
            pos = "" if pd.isna(s["position"]) else str(int(s["position"]))
            self.rows.append((pos, s["driver_code"], _team_colour(colours.get(s["driver_code"], "")),
                              by_driver.get(s["driver_code"], []), str(s["result"])))
        laps = ov.status.total_laps or int(ov.stints["last_lap"].max() if len(ov.stints) else 1)
        self.total_laps = max(int(laps), 1)
        self.setMinimumHeight(len(self.rows) * self.ROW_H + self.AXIS_H + 8)
        self.update()

    def _x(self, lap: float, left: int, width: int) -> int:
        return left + int(round(lap / self.total_laps * width))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        left = self.POS_W + self.CODE_W + 8
        width = max(self.width() - left - self.RESULT_W - 8, 50)

        # lap grid
        step = 5 if self.total_laps <= 40 else 10
        p.setFont(QFont("Arial", 8))
        bottom = len(self.rows) * self.ROW_H
        for lap in range(0, self.total_laps + 1, step):
            x = self._x(lap, left, width)
            p.setPen(QPen(QColor("#2E2E2E"), 1))
            p.drawLine(x, 0, x, bottom)
            p.setPen(QColor(DIM))
            p.drawText(QRect(x - 20, bottom + 4, 40, 16), Qt.AlignCenter, str(lap))
        p.drawText(QRect(left, bottom + 4, width, 16), Qt.AlignRight, "lap")

        for i, (pos, code, team_colour, stint_list, result) in enumerate(self.rows):
            y = i * self.ROW_H
            if i % 2:
                p.fillRect(0, y, self.width(), self.ROW_H, QColor("#1E1E1E"))
            p.setPen(QColor(DIM))
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.drawText(QRect(0, y, self.POS_W - 4, self.ROW_H), Qt.AlignRight | Qt.AlignVCenter, pos)
            p.setPen(QColor(team_colour))
            p.drawText(QRect(self.POS_W + 6, y, self.CODE_W, self.ROW_H), Qt.AlignLeft | Qt.AlignVCenter, code)

            for s in stint_list:
                compound = str(s.get("compound") or "").upper()
                x1 = self._x(s["first_lap"] - 1, left, width)
                x2 = self._x(s["last_lap"], left, width)
                bar = QRect(x1 + 1, y + 5, max(x2 - x1 - 2, 2), self.ROW_H - 10)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(COMPOUND_COLOURS.get(compound, "#777777")))
                p.drawRoundedRect(bar, 3, 3)
                if s.get("fresh") is False:      # a used set: hatched
                    p.setBrush(QBrush(QColor(0, 0, 0, 90), Qt.BDiagPattern))
                    p.drawRoundedRect(bar, 3, 3)
                if bar.width() > 26:
                    p.setPen(QColor(COMPOUND_TEXT.get(compound, "#FFFFFF")))
                    p.setFont(QFont("Arial", 8, QFont.Bold))
                    p.drawText(bar, Qt.AlignCenter, f"{compound[:1] or '?'} {s['laps']}")

            p.setPen(QColor(DIM))
            p.setFont(QFont("Arial", 8))
            p.drawText(QRect(left + width + 8, y, self.RESULT_W, self.ROW_H), Qt.AlignLeft | Qt.AlignVCenter, result)
        p.end()


class RaceAnalysisWindow(QMainWindow):
    """One race from the lake: result, strategies, laps, data quality, and Phase 2 placeholders."""

    def __init__(self, year: int, round_number: int, event_name: str = "", root: str = lake.DEFAULT_LAKE_ROOT,
                 parent=None):
        super().__init__(parent)
        self.year, self.round_number, self.root = year, round_number, root
        self.event_name = event_name or f"Round {round_number}"
        self.overview: Optional[RaceOverview] = None
        self.worker: Optional[PipelineBuildThread] = None
        self._laps_df: Optional[pd.DataFrame] = None

        self.setWindowTitle(f"PitwallOS — Race Analysis — {self.event_name} {year}")
        self.resize(1240, 820)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._open()

    # --- layout ---------------------------------------------------------------

    def _build_ui(self):
        page = QWidget(objectName="page")
        self.setCentralWidget(page)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        top = QHBoxLayout()
        titles = QVBoxLayout()
        kicker = QLabel("PITWALLOS · RACE ANALYSIS")
        kicker.setStyleSheet(f"color: {ACCENT}; font-weight: bold; letter-spacing: 2px;")
        kicker.setFont(QFont("Arial", 9))
        self.title_lbl = QLabel(f"{self.event_name} {self.year}")
        self.title_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        self.subtitle_lbl = QLabel(f"Round {self.round_number}")
        self.subtitle_lbl.setStyleSheet(f"color: {DIM};")
        for w in (kicker, self.title_lbl, self.subtitle_lbl):
            titles.addWidget(w)
        top.addLayout(titles)
        top.addStretch()

        chips = QVBoxLayout()
        chips.setSpacing(6)
        chip_row = QHBoxLayout()
        self.layer_chips = {}
        for layer in lake.LAYERS:
            chip = QLabel(layer.capitalize())
            chip.setStyleSheet(self._chip_style(False))
            self.layer_chips[layer] = chip
            chip_row.addWidget(chip)
        chips.addLayout(chip_row)
        self.rebuild_btn = QPushButton("Rebuild data")
        self.rebuild_btn.setToolTip("Run Bronze -> Silver -> Gold again for this race (laps only)")
        self.rebuild_btn.clicked.connect(self._start_build)
        chips.addWidget(self.rebuild_btn, alignment=Qt.AlignRight)
        top.addLayout(chips)
        outer.addLayout(top)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_waiting_page())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_result_tab(), "Result")
        self.tabs.addTab(self._build_strategy_tab(), "Strategy")
        self.tabs.addTab(self._build_laps_tab(), "Laps")
        self.tabs.addTab(self._build_quality_tab(), "Data quality")
        self.tabs.addTab(self._build_phase2_tab(), "Phase 2 analytics")
        self.stack.addWidget(self.tabs)
        outer.addWidget(self.stack, 1)

        foot = QLabel("Read from the data lake (data/). No replay needed. FastF1 downloads are shared with the replay "
                      "(.fastf1-cache/).")
        foot.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        outer.addWidget(foot)

    @staticmethod
    def _chip_style(ok: bool) -> str:
        colour = GOOD if ok else "#555555"
        return (f"QLabel {{ color: {colour}; border: 1px solid {colour}; border-radius: 10px; "
                f"padding: 3px 10px; font-weight: bold; }}")

    def _build_waiting_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch()
        self.wait_title = QLabel("This race is not in the data lake yet")
        self.wait_title.setFont(QFont("Arial", 15, QFont.Bold))
        self.wait_title.setAlignment(Qt.AlignCenter)
        self.wait_text = QLabel("")
        self.wait_text.setWordWrap(True)
        self.wait_text.setAlignment(Qt.AlignCenter)
        self.wait_text.setStyleSheet(f"color: {DIM};")
        self.wait_progress = QProgressBar()
        self.wait_progress.setRange(0, 0)
        self.wait_progress.setMaximumWidth(420)
        self.retry_btn = QPushButton("Try again")
        self.retry_btn.clicked.connect(self._start_build)
        self.retry_btn.setVisible(False)
        lay.addWidget(self.wait_title)
        lay.addWidget(self.wait_text)
        lay.addWidget(self.wait_progress, alignment=Qt.AlignCenter)
        lay.addWidget(self.retry_btn, alignment=Qt.AlignCenter)
        lay.addStretch()
        return w

    def _build_result_tab(self) -> QWidget:
        self.result_table = _table(["Pos", "Driver", "Team", "Grid", "+/-", "Laps", "Result", "Points"])
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        return self.result_table

    def _build_strategy_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        legend = QHBoxLayout()
        for name, colour in COMPOUND_COLOURS.items():
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {colour}; font-size: 14px;")
            legend.addWidget(dot)
            legend.addWidget(QLabel(name.capitalize()))
        hatch = QLabel("hatched = used set at fitting   ·   number = laps in the stint")
        hatch.setStyleSheet(f"color: {DIM};")
        legend.addSpacing(14)
        legend.addWidget(hatch)
        legend.addStretch()
        lay.addLayout(legend)

        self.stint_chart = StintChart()
        lay.addWidget(_scroll_page(self.stint_chart), 3)

        lower = QHBoxLayout()
        groups_box = QVBoxLayout()
        groups_title = QLabel(f"Strategies run in this race   ({DNF_MARK} did not finish: listed, not counted)")
        groups_title.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        self.groups_table = _table(["Strategy", "Stops", "Drivers", "Finished", "Best result"])
        self.groups_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        groups_box.addWidget(groups_title)
        groups_box.addWidget(self.groups_table)
        lower.addLayout(groups_box, 1)

        drivers_box = QVBoxLayout()
        drivers_title = QLabel("Per driver")
        drivers_title.setStyleSheet(f"color: {ACCENT}; font-weight: bold;")
        self.strategy_table = _table(["Pos", "Driver", "Strategy", "Stops", "Pit laps", "Stint laps"])
        drivers_box.addWidget(drivers_title)
        drivers_box.addWidget(self.strategy_table)
        lower.addLayout(drivers_box, 1)
        lay.addLayout(lower, 2)
        return w

    def _build_laps_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout()
        self.laps_source = QComboBox()
        self.laps_source.addItems(["Silver laps (cleaned, flagged)", "Gold lap features (model input)"])
        self.laps_source.currentIndexChanged.connect(self._load_laps_table)
        self.laps_filter = QLineEdit()
        self.laps_filter.setPlaceholderText("Filter by driver or compound, e.g. VER, HARD")
        self.laps_filter.textChanged.connect(self._fill_laps_table)
        bar.addWidget(self.laps_source)
        bar.addWidget(self.laps_filter, 1)
        lay.addLayout(bar)
        self.laps_status = QLabel("")
        self.laps_status.setStyleSheet(f"color: {DIM};")
        lay.addWidget(self.laps_status)
        self.laps_table = QTableWidget()
        self.laps_table.setAlternatingRowColors(True)
        self.laps_table.verticalHeader().setVisible(False)
        self.laps_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.laps_table, 1)
        return w

    def _build_quality_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        note = QLabel("What the pipeline flagged for this race. Nothing is deleted: every lap stays in Silver with "
                      "its flags, and each consumer decides what to exclude.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM};")
        lay.addWidget(note)
        self.quality_grid = QGridLayout()
        self.quality_grid.setSpacing(10)
        lay.addLayout(self.quality_grid)
        self.warnings_lbl = QLabel("")
        self.warnings_lbl.setWordWrap(True)
        self.warnings_lbl.setStyleSheet(f"color: {DIM};")
        lay.addWidget(self.warnings_lbl)
        lay.addStretch()
        return w

    def _build_phase2_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        intro = QLabel("These numbers have to be estimated, so each one waits for its Phase 2 step: your design note "
                       "and expected number first, then the code, then a golden test on a real race. Until then "
                       "this window shows nothing rather than a number that might be wrong.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {DIM};")
        lay.addWidget(intro)
        for step, title, shows, golden, done in PHASE2_STEPS:
            card = QFrame()
            card.setStyleSheet(f"QFrame {{ background: #202020; border: 1px solid {BORDER}; border-radius: 6px; }}"
                               f"QLabel {{ border: none; background: transparent; }}")
            c = QGridLayout(card)
            c.setContentsMargins(12, 10, 12, 10)
            head = QLabel(f"Step {step} · {title}")
            head.setFont(QFont("Arial", 11, QFont.Bold))
            chip = QLabel("DONE" if done else "NOT BUILT")
            chip.setStyleSheet(self._chip_style(done))
            body = QLabel(shows)
            body.setWordWrap(True)
            test = QLabel(f"Golden test: {golden}")
            test.setWordWrap(True)
            test.setStyleSheet(f"color: {DIM};")
            c.addWidget(head, 0, 0)
            c.addWidget(chip, 0, 1, alignment=Qt.AlignRight)
            c.addWidget(body, 1, 0, 1, 2)
            c.addWidget(test, 2, 0, 1, 2)
            lay.addWidget(card)
        lay.addStretch()
        return _scroll_page(inner)

    # --- data -------------------------------------------------------------------

    def _refresh_chips(self):
        status = lake_status(self.year, self.round_number, root=self.root)
        for layer, chip in self.layer_chips.items():
            ok = getattr(status, layer)
            size = status.sizes_mb.get(layer, 0.0)
            chip.setText(f"{layer.capitalize()} {'✓' if ok else '—'}" + (f"  {size:.1f} MB" if ok else ""))
            chip.setStyleSheet(self._chip_style(ok))
        return status

    def _open(self):
        status = self._refresh_chips()
        if status.ready:
            self._load()
        else:
            self._start_build()

    def _start_build(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.stack.setCurrentIndex(0)
        self.wait_title.setText(f"Building {self.event_name} {self.year} into the data lake")
        self.wait_text.setText("Bronze (raw FastF1 tables) → Silver (cleaned, flagged laps) → Gold (lap features).\n"
                               "Reads FastF1's shared download cache first; a race never downloaded before takes a "
                               "few minutes.")
        self.wait_text.setStyleSheet(f"color: {DIM};")
        self.wait_progress.setVisible(True)
        self.retry_btn.setVisible(False)
        self.rebuild_btn.setEnabled(False)
        self.worker = PipelineBuildThread(self.year, self.round_number, self.root)
        self.worker.done.connect(self._on_built)
        self.worker.failed.connect(self._on_build_failed)
        self.worker.start()

    def _on_built(self, _summary: dict):
        self.rebuild_btn.setEnabled(True)
        self._refresh_chips()
        self._load()

    def _on_build_failed(self, message: str):
        self.rebuild_btn.setEnabled(True)
        self.stack.setCurrentIndex(0)
        self.wait_progress.setVisible(False)
        self.wait_title.setText("Could not build this race")
        self.wait_text.setText(message)
        self.wait_text.setStyleSheet(f"color: {BAD};")
        self.retry_btn.setVisible(True)

    def _load(self):
        try:
            self.overview = load_overview(self.year, self.round_number, root=self.root)
        except Exception as e:
            self._on_build_failed(f"{type(e).__name__}: {e}")
            return
        ov = self.overview
        if ov.status.event_name:
            self.event_name = ov.status.event_name
            self.title_lbl.setText(f"{ov.status.event_name} {self.year}")
            self.setWindowTitle(f"PitwallOS — Race Analysis — {ov.status.event_name} {self.year}")
        details = [f"Round {self.round_number}", ov.status.location, ov.status.event_date,
                   f"{ov.status.total_laps} laps" if ov.status.total_laps else None]
        self.subtitle_lbl.setText("  ·  ".join(d for d in details if d))
        self._fill_result(ov)
        self._fill_strategy(ov)
        self._fill_quality(ov)
        self._load_laps_table()
        self.stack.setCurrentIndex(1)

    def _fill_result(self, ov: RaceOverview):
        c = ov.classification
        self.result_table.setRowCount(len(c))
        for i, r in c.iterrows():
            pos = "" if pd.isna(r["position"]) else str(int(r["position"]))
            grid = "PL" if pd.isna(r["grid"]) and str(r["status"]).lower() != "did not start" else _cell_text(r["grid"])
            delta, delta_colour = "", DIM
            if not pd.isna(r["position"]) and not pd.isna(r["grid"]) and str(r["classified"]).isdigit():
                gained = int(r["grid"]) - int(r["position"])
                delta = f"+{gained}" if gained > 0 else str(gained) if gained < 0 else "="
                delta_colour = GOOD if gained > 0 else BAD if gained < 0 else DIM
            self.result_table.setItem(i, 0, _item(pos, bold=True))
            self.result_table.setItem(i, 1, _item(f"{r['driver_code']}  {r['driver_name']}", Qt.AlignLeft,
                                                  _team_colour(r["team_color"]), bold=True))
            self.result_table.setItem(i, 2, _item(r["team"], Qt.AlignLeft))
            self.result_table.setItem(i, 3, _item(grid))
            self.result_table.setItem(i, 4, _item(delta, colour=delta_colour))
            self.result_table.setItem(i, 5, _item(_cell_text(r["laps"])))
            self.result_table.setItem(i, 6, _item(r["result"], colour=None if r["finished"] else BAD))
            self.result_table.setItem(i, 7, _item("" if r["points"] in (None, 0.0) else f"{r['points']:g}"))

    def _fill_strategy(self, ov: RaceOverview):
        self.stint_chart.set_overview(ov)
        g = ov.strategy_groups
        self.groups_table.setRowCount(len(g))
        for i, r in g.iterrows():
            self.groups_table.setItem(i, 0, _item(r["strategy"], bold=True))
            self.groups_table.setItem(i, 1, _item(str(r["stops"])))
            self.groups_table.setItem(i, 2, _item(r["drivers"], Qt.AlignLeft))
            self.groups_table.setItem(i, 3, _item(str(r["finishers"])))
            best = r["best_result"]
            self.groups_table.setItem(i, 4, _item("—" if best is None or pd.isna(best) else f"P{int(best)}"))
        s = ov.strategies
        self.strategy_table.setRowCount(len(s))
        for i, r in s.iterrows():
            code = r["driver_code"] if r["finished"] else r["driver_code"] + DNF_MARK
            self.strategy_table.setItem(i, 0, _item(_cell_text(r["position"])))
            self.strategy_table.setItem(i, 1, _item(code, bold=True, colour=None if r["finished"] else DIM))
            self.strategy_table.setItem(i, 2, _item(r["strategy"]))
            self.strategy_table.setItem(i, 3, _item(str(r["stops"])))
            self.strategy_table.setItem(i, 4, _item(r["pit_laps"] or "—"))
            self.strategy_table.setItem(i, 5, _item(r["stint_laps"]))

    def _fill_quality(self, ov: RaceOverview):
        while self.quality_grid.count():
            item = self.quality_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        q = ov.quality
        share = lambda n: f"{n / q['laps']:.0%} of laps" if q["laps"] else ""
        tiles = [
            ("Laps", f"{q['laps']:,}", f"{q['drivers']} drivers"),
            ("Clean laps", f"{q['clean_laps']:,}", share(q["clean_laps"]) + " · timed, accurate, green, no pit, not lap 1"),
            ("Not green", f"{q['not_green']:,}", f"SC {q['safety_car_laps']} · VSC {q['vsc_laps']} · red {q['red_flag_laps']}"),
            ("Wet-tyre laps", f"{q['wet_tyre_laps']:,}", "inters or wets (own flag, not excluded from clean)"),
            ("Tyre changes", f"{q['tyre_changes']}", "the stops"),
            ("Pass-throughs", f"{q['pass_throughs']}", "pit lane without new tyres (SC, penalties)"),
            ("Ended in the pits", f"{q['ended_in_pits']}", f"unknown passages: {q['unknown_passages']}"),
            ("Pit laps", f"{q['pit_laps']:,}", "in-laps + out-laps"),
            ("Not accurate", f"{q['inaccurate']:,}", "FastF1 IsAccurate = False"),
            ("Deleted", f"{q['deleted_laps']}", "track limits (own flag)"),
            ("No lap time", f"{q['no_lap_time']}", "kept, never invented"),
            ("Lake size", f"{sum(ov.status.sizes_mb.values()):.1f} MB",
             " · ".join(f"{k} {v:.1f}" for k, v in ov.status.sizes_mb.items())),
        ]
        for n, (label, value, note) in enumerate(tiles):
            tile = QFrame()
            tile.setStyleSheet(f"QFrame {{ background: #202020; border: 1px solid {BORDER}; border-radius: 6px; }}"
                               f"QLabel {{ border: none; background: transparent; }}")
            t = QVBoxLayout(tile)
            t.setContentsMargins(12, 8, 12, 8)
            cap = QLabel(label.upper())
            cap.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
            val = QLabel(value)
            val.setFont(QFont("Arial", 18, QFont.Bold))
            sub = QLabel(note)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color: {DIM}; font-size: 10px;")
            for x in (cap, val, sub):
                t.addWidget(x)
            self.quality_grid.addWidget(tile, n // 4, n % 4)
        warnings = ov.warnings
        self.warnings_lbl.setText("Bronze warnings: " + ("; ".join(warnings) if warnings else "none"))

    def _load_laps_table(self):
        gold = self.laps_source.currentIndex() == 1
        path = (lake.gold_features_path(self.year, self.round_number, root=self.root) if gold
                else lake.table_path(lake.SILVER, "laps", self.year, self.round_number, "R", self.root))
        try:
            self._laps_df = lake.read_table(path)
        except Exception as e:
            self._laps_df = None
            self.laps_table.setRowCount(0)
            self.laps_status.setText(f"Could not load {path}: {e}")
            return
        self._fill_laps_table()

    def _fill_laps_table(self):
        df = self._laps_df
        if df is None:
            return
        text = self.laps_filter.text().strip().upper()
        if text:
            mask = df["driver_code"].astype(str).str.upper().str.contains(text, regex=False)
            if "compound" in df.columns:
                mask |= df["compound"].astype(str).str.upper().str.contains(text, regex=False)
            df = df[mask]
        shown = df.head(500)
        self.laps_table.setColumnCount(len(shown.columns))
        self.laps_table.setHorizontalHeaderLabels([str(c) for c in shown.columns])
        self.laps_table.setRowCount(len(shown))
        for r, row in enumerate(shown.itertuples(index=False)):
            for c, value in enumerate(row):
                self.laps_table.setItem(r, c, _item(_cell_text(value)))
        self.laps_table.resizeColumnsToContents()
        more = f" (showing the first {len(shown)})" if len(df) > len(shown) else ""
        self.laps_status.setText(f"{len(df):,} rows{more}")


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    args = sys.argv[1:]
    year = int(args[0]) if len(args) > 0 else 2024
    rnd = int(args[1]) if len(args) > 1 else 12
    win = RaceAnalysisWindow(year, rnd)
    win.show()
    sys.exit(app.exec())
