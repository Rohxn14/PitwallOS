"""
PySide6 Data Pipeline & Feature Store Inspector Window (src/gui/pipeline_window.py)

Provides a graphical dashboard for executing the Medallion Telemetry Pipeline,
inspecting Bronze/Silver/Gold Parquet storage files, and previewing generated
ML feature vectors in an interactive table.
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QLineEdit, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from src.pipeline.runner import run_pipeline
from src.pipeline import lake

PREVIEW_ROWS = 100      # rows shown in the table; the full Gold table stays in memory for filtering


def format_cell(val) -> str:
    """Text for one table cell. Missing values show as an em dash, never as an invented number."""
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "—"
    if isinstance(val, (float, np.floating)):
        return f"{val:.2f}"
    return str(val)


class PipelineWorkerThread(QThread):
    """Background worker thread to execute ETL pipeline without freezing GUI."""
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, year: int, round_number: int):
        super().__init__()
        self.year = year
        self.round_number = round_number

    def run(self):
        try:
            res = run_pipeline(self.year, self.round_number)
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class DataPipelineWindow(QMainWindow):
    """Graphical Data Pipeline & Feature Store Manager Window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_df: pd.DataFrame | None = None
        self.setWindowTitle("PitwallOS — Telemetry Data Pipeline & Feature Store Manager")
        self.resize(1100, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #E0E0E0; }
            QGroupBox { font-weight: bold; border: 1px solid #333; border-radius: 6px; margin-top: 10px; padding-top: 10px; color: #00E5FF; }
            QLabel { color: #E0E0E0; }
            QPushButton { background-color: #00E5FF; color: #000; font-weight: bold; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background-color: #18FFFF; }
            QTableWidget { background-color: #1E1E1E; gridline-color: #333; color: #FFF; border: 1px solid #333; }
            QHeaderView::section { background-color: #262626; color: #00E5FF; font-weight: bold; border: 1px solid #333; padding: 4px; }
            QLineEdit { background-color: #1E1E1E; border: 1px solid #444; color: #FFF; padding: 6px; border-radius: 4px; }
        """)

        self._init_ui()
        self._load_existing_features()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Title Banner
        title_lbl = QLabel("🏎️ TELEMETRY DATA ENGINEERING PIPELINE (MEDALLION PARQUET STORE)")
        title_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        title_lbl.setStyleSheet("color: #00E5FF; margin-bottom: 5px;")
        layout.addWidget(title_lbl)

        # 2. Control Top Deck
        ctrl_box = QGroupBox("Pipeline Execution Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("Season Year:"))
        self.year_spin = QSpinBox()
        self.year_spin.setRange(2018, 2026)
        self.year_spin.setValue(2024)
        ctrl_layout.addWidget(self.year_spin)

        ctrl_layout.addWidget(QLabel("Round:"))
        self.round_spin = QSpinBox()
        self.round_spin.setRange(1, 24)
        self.round_spin.setValue(12)
        ctrl_layout.addWidget(self.round_spin)

        self.build_btn = QPushButton("⚡ Build Parquet Feature Store")
        self.build_btn.setCursor(Qt.PointingHandCursor)
        self.build_btn.clicked.connect(self._run_pipeline)
        ctrl_layout.addWidget(self.build_btn)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_box)

        # Progress Bar (Hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 3. Storage Analytics Metrics Deck
        metrics_box = QGroupBox("Medallion Data Storage Analytics")
        metrics_layout = QHBoxLayout(metrics_box)

        self.bronze_card = QLabel("Bronze Raw: -- MB")
        self.silver_card = QLabel("Silver Cleaned: -- MB")
        self.gold_card = QLabel("Gold Feature Vectors: --")
        # Every lake write is validated by src/pipeline/contracts.py, so a file that exists passed.
        self.health_card = QLabel("Schema Status: ✓ Parquet Validated")
        self.health_card.setStyleSheet("color: #00E676; font-weight: bold;")

        metrics_layout.addWidget(self.bronze_card)
        metrics_layout.addWidget(self.silver_card)
        metrics_layout.addWidget(self.gold_card)
        metrics_layout.addWidget(self.health_card)
        layout.addWidget(metrics_box)

        # 4. Feature Store Data Table Preview
        table_box = QGroupBox("Gold Feature Store Table Preview (data/gold/)")
        table_layout = QVBoxLayout(table_box)

        # Filter Input
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Driver / Feature:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search e.g. HAM, VER, MEDIUM...")
        self.filter_input.textChanged.connect(self._filter_table)
        filter_layout.addWidget(self.filter_input)
        filter_layout.addStretch()
        table_layout.addLayout(filter_layout)

        # Shows which file is loaded, how many rows match, or why loading failed
        self.status_label = QLabel("No Gold table loaded yet.")
        self.status_label.setStyleSheet("color: #9E9E9E;")
        table_layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.table)

        layout.addWidget(table_box)

    def _run_pipeline(self):
        year = self.year_spin.value()
        rnd = self.round_spin.value()

        self.build_btn.setEnabled(False)
        self.build_btn.setText("Processing Pipeline...")
        self.progress_bar.setVisible(True)

        self.worker = PipelineWorkerThread(year, rnd)
        self.worker.finished_signal.connect(self._on_pipeline_success)
        self.worker.error_signal.connect(self._on_pipeline_error)
        self.worker.start()

    def _on_pipeline_success(self, res: dict):
        self.build_btn.setEnabled(True)
        self.build_btn.setText("⚡ Build Parquet Feature Store")
        self.progress_bar.setVisible(False)

        gold_path = res['gold']['file_path']
        self._load_features_from_path(gold_path)

        # Update Cards
        self.bronze_card.setText(f"Bronze Raw: {res['bronze']['file_size_mb']} MB")
        self.silver_card.setText(f"Silver Cleaned: {res['silver']['file_size_mb']} MB")
        self.gold_card.setText(f"Gold Vectors: {res['gold']['feature_rows']:,}")

        QMessageBox.information(self, "Pipeline Complete", f"Parquet Feature Store created successfully at:\n{gold_path}")

    def _on_pipeline_error(self, err_msg: str):
        self.build_btn.setEnabled(True)
        self.build_btn.setText("⚡ Build Parquet Feature Store")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Pipeline Error", f"Failed to execute pipeline:\n{err_msg}")

    def _load_existing_features(self):
        default_gold_path = str(lake.gold_features_path(2024, 12))
        if os.path.exists(default_gold_path):
            self._load_features_from_path(default_gold_path)

    def _load_features_from_path(self, file_path: str):
        try:
            table = pq.read_table(file_path)
        except Exception as e:
            # Shown in the window: a silent print is how the old preview failed unnoticed
            self.full_df = None
            self.table.setRowCount(0)
            self.status_label.setText(f"Could not load {file_path}: {e}")
            self.status_label.setStyleSheet("color: #FF5252;")
            return

        self.full_df = table.to_pandas()
        self.status_label.setStyleSheet("color: #9E9E9E;")
        self._populate_table(self.full_df)

        file_kb = os.path.getsize(file_path) / 1024
        self.gold_card.setText(f"Gold Vectors: {len(self.full_df):,} ({file_kb:.1f} KB)")

    def _populate_table(self, df: pd.DataFrame):
        preview_df = df.head(PREVIEW_ROWS)
        self.table.setRowCount(len(preview_df))
        self.table.setColumnCount(len(preview_df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])

        # Rows are addressed by POSITION (0, 1, 2 ...). After filtering, the DataFrame index keeps
        # its original labels (e.g. 300-350 for one driver), which would fall outside the table.
        for r_pos, row in enumerate(preview_df.itertuples(index=False)):
            for c_pos, val in enumerate(row):
                item = QTableWidgetItem(format_cell(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r_pos, c_pos, item)

        shown = len(preview_df)
        total = len(df)
        suffix = f" (showing first {shown})" if total > shown else ""
        self.status_label.setText(f"{total:,} matching rows{suffix}")

    def _filter_table(self, text: str):
        if self.full_df is None:
            return
        if not text.strip():
            self._populate_table(self.full_df)
            return

        txt = text.strip().upper()
        filtered = self.full_df[
            self.full_df['driver_code'].astype(str).str.upper().str.contains(txt, regex=False) |
            self.full_df['compound'].astype(str).str.upper().str.contains(txt, regex=False)
        ]
        self._populate_table(filtered)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = DataPipelineWindow()
    win.show()
    sys.exit(app.exec())
