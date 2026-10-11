"""
PySide6 Race Analytics Dashboard (src/gui/analytics_window.py)

Interactive dashboard for viewing race analytics results:
- Clean pace comparison table
- Stint degradation breakdown with slope values
- Pit stop analysis table
- Undercut/overcut window evaluations
- Safety Car impact summary
"""

import os
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QSpinBox, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

from src.analytics.race_engine import RaceAnalyticsEngine, RaceAnalyticsReport
from src.pipeline import lake


class AnalyticsWorkerThread(QThread):
    """Background thread to run analytics without freezing the GUI."""
    finished_signal = Signal(object)
    error_signal = Signal(str)
    
    def __init__(self, features_path: str):
        super().__init__()
        self.features_path = features_path
        
    def run(self):
        try:
            engine = RaceAnalyticsEngine()
            report = engine.analyze(self.features_path)
            self.finished_signal.emit(report)
        except Exception as e:
            self.error_signal.emit(str(e))


class RaceAnalyticsWindow(QMainWindow):
    """Interactive Race Analytics Dashboard."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PitwallOS — Race Analytics & Strategy Dashboard")
        self.resize(1200, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #E0E0E0; }
            QGroupBox { font-weight: bold; border: 1px solid #333; border-radius: 6px; margin-top: 10px; padding-top: 10px; color: #00E5FF; }
            QLabel { color: #E0E0E0; }
            QPushButton { background-color: #00E5FF; color: #000; font-weight: bold; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background-color: #18FFFF; }
            QTabWidget::pane { border: 1px solid #333; background: #1A1A1A; }
            QTabBar::tab { background: #262626; color: #AAA; padding: 8px 16px; border: 1px solid #333; }
            QTabBar::tab:selected { background: #1A1A1A; color: #00E5FF; border-bottom: 2px solid #00E5FF; }
            QTableWidget { background-color: #1E1E1E; gridline-color: #333; color: #FFF; border: 1px solid #333; }
            QHeaderView::section { background-color: #262626; color: #00E5FF; font-weight: bold; border: 1px solid #333; padding: 4px; }
        """)
        
        self.report = None
        self._init_ui()
        self._try_load_default()
        
    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("RACE ANALYTICS & STRATEGY ENGINE")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #00E5FF; margin-bottom: 5px;")
        layout.addWidget(title)
        
        # Controls
        ctrl_box = QGroupBox("Analysis Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)
        
        ctrl_layout.addWidget(QLabel("Year:"))
        self.year_spin = QSpinBox()
        self.year_spin.setRange(2018, 2026)
        self.year_spin.setValue(2024)
        ctrl_layout.addWidget(self.year_spin)
        
        ctrl_layout.addWidget(QLabel("Round:"))
        self.round_spin = QSpinBox()
        self.round_spin.setRange(1, 24)
        self.round_spin.setValue(12)
        ctrl_layout.addWidget(self.round_spin)
        
        self.analyze_btn = QPushButton("Analyze Race")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.clicked.connect(self._run_analysis)
        ctrl_layout.addWidget(self.analyze_btn)
        
        self.status_lbl = QLabel("")
        ctrl_layout.addWidget(self.status_lbl)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl_box)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Tabbed Results
        self.tabs = QTabWidget()
        
        self.pace_table = QTableWidget()
        self.stint_table = QTableWidget()
        self.pit_table = QTableWidget()
        self.undercut_table = QTableWidget()
        self.sc_table = QTableWidget()
        
        for tbl in [self.pace_table, self.stint_table, self.pit_table, self.undercut_table, self.sc_table]:
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.setAlternatingRowColors(True)
        
        self.tabs.addTab(self.pace_table, "Clean Pace & Dirty Air")
        self.tabs.addTab(self.stint_table, "Stint Degradation")
        self.tabs.addTab(self.pit_table, "Pit Stop Analysis")
        self.tabs.addTab(self.undercut_table, "Undercut / Overcut")
        self.tabs.addTab(self.sc_table, "Safety Car Impact")
        
        layout.addWidget(self.tabs)
        
    def _try_load_default(self):
        default_path = str(lake.gold_features_path(2024, 12))
        if os.path.exists(default_path):
            self._start_analysis(default_path)
            
    def _run_analysis(self):
        year = self.year_spin.value()
        rnd = self.round_spin.value()
        path = str(lake.gold_features_path(year, rnd))
        
        if not os.path.exists(path):
            QMessageBox.warning(self, "Feature Store Missing",
                                f"No feature store found at:\n{path}\n\nRun the Data Pipeline first.")
            return
        self._start_analysis(path)
        
    def _start_analysis(self, path: str):
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self.progress.setVisible(True)
        
        self.worker = AnalyticsWorkerThread(path)
        self.worker.finished_signal.connect(self._on_success)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()
        
    def _on_success(self, report: RaceAnalyticsReport):
        self.report = report
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Race")
        self.progress.setVisible(False)
        self.status_lbl.setText(f"{report.event_name} {report.year} — Analysis Complete")
        self.status_lbl.setStyleSheet("color: #00E676; font-weight: bold;")
        
        self._populate_pace_table(report)
        self._populate_stint_table(report)
        self._populate_pit_table(report)
        self._populate_undercut_table(report)
        self._populate_sc_table(report)
        
    def _on_error(self, msg: str):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Race")
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", msg)
        
    # --- Table Population Methods ---
        
    def _populate_pace_table(self, report):
        data = sorted(report.driver_clean_pace.items(), key=lambda x: x[1])
        cols = ["Pos", "Driver", "Clean Pace (s)", "Dirty Air Delta (s)", "True vs Leader"]
        self.pace_table.setRowCount(len(data))
        self.pace_table.setColumnCount(len(cols))
        self.pace_table.setHorizontalHeaderLabels(cols)
        
        leader_pace = data[0][1] if data else 0.0
        
        for i, (drv, pace) in enumerate(data):
            delta = report.driver_dirty_air_delta.get(drv, 0.0)
            gap = pace - leader_pace
            
            self._set_cell(self.pace_table, i, 0, f"P{i+1}")
            self._set_cell(self.pace_table, i, 1, drv)
            self._set_cell(self.pace_table, i, 2, f"{pace:.3f}")
            self._set_cell(self.pace_table, i, 3, f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}")
            self._set_cell(self.pace_table, i, 4, f"+{gap:.3f}" if gap > 0 else "LEADER")
            
    def _populate_stint_table(self, report):
        stints = report.stint_summaries
        cols = ["Driver", "Stint", "Compound", "Laps", "Length", "Avg Pace (s)", "Deg (s/lap)", "Best", "Worst"]
        self.stint_table.setRowCount(len(stints))
        self.stint_table.setColumnCount(len(cols))
        self.stint_table.setHorizontalHeaderLabels(cols)
        
        for i, s in enumerate(stints):
            self._set_cell(self.stint_table, i, 0, s.driver_code)
            self._set_cell(self.stint_table, i, 1, str(s.stint_number))
            self._set_cell(self.stint_table, i, 2, s.compound)
            self._set_cell(self.stint_table, i, 3, f"{s.start_lap}-{s.end_lap}")
            self._set_cell(self.stint_table, i, 4, str(s.stint_length))
            self._set_cell(self.stint_table, i, 5, f"{s.avg_pace_sec:.3f}")
            self._set_cell(self.stint_table, i, 6, f"{s.degradation_slope:+.4f}")
            self._set_cell(self.stint_table, i, 7, f"{s.best_lap_sec:.3f}")
            self._set_cell(self.stint_table, i, 8, f"{s.worst_lap_sec:.3f}")
            
    def _populate_pit_table(self, report):
        pits = report.pit_stop_analyses
        cols = ["Driver", "Pit Lap", "Compound Change", "Pit Loss (s)", "Position Change"]
        self.pit_table.setRowCount(len(pits))
        self.pit_table.setColumnCount(len(cols))
        self.pit_table.setHorizontalHeaderLabels(cols)
        
        for i, p in enumerate(pits):
            pos_change = ""
            if p.position_before is not None and p.position_after is not None:
                diff = p.position_after - p.position_before
                pos_change = f"P{p.position_before} -> P{p.position_after} ({'+' if diff > 0 else ''}{diff})"
            
            self._set_cell(self.pit_table, i, 0, p.driver_code)
            self._set_cell(self.pit_table, i, 1, str(p.pit_lap))
            self._set_cell(self.pit_table, i, 2, f"{p.compound_before} -> {p.compound_after}")
            self._set_cell(self.pit_table, i, 3, f"{p.estimated_pit_loss_sec:.1f}")
            self._set_cell(self.pit_table, i, 4, pos_change)
            
    def _populate_undercut_table(self, report):
        windows = report.undercut_windows
        cols = ["Attacker", "Defender", "Lap", "Gap (s)", "Pit Loss (s)", "Tyre Adv (s/lap)", "Adv Laps", "Margin (s)", "Verdict"]
        self.undercut_table.setRowCount(len(windows))
        self.undercut_table.setColumnCount(len(cols))
        self.undercut_table.setHorizontalHeaderLabels(cols)
        
        for i, u in enumerate(windows):
            self._set_cell(self.undercut_table, i, 0, u.attacking_driver)
            self._set_cell(self.undercut_table, i, 1, u.defending_driver)
            self._set_cell(self.undercut_table, i, 2, str(u.evaluation_lap))
            self._set_cell(self.undercut_table, i, 3, f"{u.gap_sec:.2f}")
            self._set_cell(self.undercut_table, i, 4, f"{u.pit_loss_sec:.1f}")
            self._set_cell(self.undercut_table, i, 5, f"{u.fresh_tyre_advantage_per_lap:.3f}")
            self._set_cell(self.undercut_table, i, 6, str(u.laps_of_advantage))
            self._set_cell(self.undercut_table, i, 7, f"{u.undercut_margin_sec:+.2f}")
            
            verdict_item = QTableWidgetItem(u.verdict)
            verdict_item.setTextAlignment(Qt.AlignCenter)
            if u.verdict == "UNDERCUT_WIN":
                verdict_item.setForeground(QColor(0, 230, 118))
            elif u.verdict == "UNDERCUT_FAIL":
                verdict_item.setForeground(QColor(255, 82, 82))
            else:
                verdict_item.setForeground(QColor(255, 215, 0))
            self.undercut_table.setItem(i, 8, verdict_item)
            
    def _populate_sc_table(self, report):
        impacts = report.safety_car_impacts
        cols = ["Type", "Laps", "Normal Pit Loss (s)", "SC Pit Loss (s)", "Time Saved (s)", "Drivers Who Pitted"]
        self.sc_table.setRowCount(len(impacts))
        self.sc_table.setColumnCount(len(cols))
        self.sc_table.setHorizontalHeaderLabels(cols)
        
        for i, sc in enumerate(impacts):
            self._set_cell(self.sc_table, i, 0, sc.sc_type)
            self._set_cell(self.sc_table, i, 1, f"{sc.start_lap}-{sc.end_lap}")
            self._set_cell(self.sc_table, i, 2, f"{sc.normal_pit_loss_sec:.1f}")
            self._set_cell(self.sc_table, i, 3, f"{sc.sc_pit_loss_sec:.1f}")
            self._set_cell(self.sc_table, i, 4, f"{sc.time_saved_sec:.1f}")
            self._set_cell(self.sc_table, i, 5, ", ".join(sc.drivers_who_pitted) if sc.drivers_who_pitted else "None")
            
    def _set_cell(self, table, row, col, text):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, col, item)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = RaceAnalyticsWindow()
    win.show()
    sys.exit(app.exec())
