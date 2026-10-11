import math
import sys

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox
)
from PySide6.QtGui import QFont
from src.gui.pit_wall_window import PitWallWindow
from src.insights.telemetry_trace import TIME_WINDOW_SEC, DriverTrace

# Colours matching the qualifying viewer
_BG        = "#282828"   # panel background
_SPEED_COL = "#F0F0F0"   # anti-flash white
_GEAR_COL  = "#B0B0B0"   # light gray
_THROT_COL = "#2ECC71"   # green
_BRAKE_COL = "#E74C3C"   # red

# Order of the values stored per sample in a DriverTrace
_COLUMNS = ("speed", "gear", "throttle", "brake")


class DriverTelemetryWindow(PitWallWindow):
    """
    Pit wall insight that shows live telemetry for a selected driver as
    three stacked line charts: Speed (top 50%), Gear (middle 25%),
    and Throttle / Brake (bottom 25%).

    Rewinds and seeks are handled by DriverTrace (src/insights/telemetry_trace.py): each chart is
    kept strictly increasing in its x value, so a rewind cuts the line back to the new position
    instead of drawing over itself, and a forward seek leaves a gap instead of a straight line.
    """

    def __init__(self):
        self._known_drivers = []
        self._traces: dict[str, DriverTrace] = {}
        # Circuit length from the session (metres), refreshed from every message
        self._circuit_length_m: float | None = None
        # Furthest lap distance seen; FastF1's integrated distance can exceed the official length
        self._max_lap_dist_m = 0.0
        # Last seek counter seen from the replay (timeline_epoch); None until the first message
        self._last_epoch = None
        self._x_mode = "time"   # "time" | "lap"
        super().__init__()
        self.setWindowTitle("F1 Race Replay - Driver Live Telemetry")

    # ── UI setup ─────────────────────────────────────────────────────────

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(6)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # Top control row: driver selector + x-axis mode
        control_row = QHBoxLayout()

        driver_label = QLabel("Driver:")
        driver_label.setFont(QFont("Arial", 11))
        self.driver_combo = QComboBox()
        self.driver_combo.setMinimumWidth(100)
        self.driver_combo.setPlaceholderText("Waiting for data…")
        self.driver_combo.setFont(QFont("Arial", 11))
        self.driver_combo.currentTextChanged.connect(self._on_driver_selected)

        xmode_label = QLabel("X Axis:")
        xmode_label.setFont(QFont("Arial", 11))
        self.xmode_combo = QComboBox()
        self.xmode_combo.setFont(QFont("Arial", 11))
        self.xmode_combo.addItems(["Last 30 seconds", "Current Lap"])
        self.xmode_combo.currentIndexChanged.connect(self._on_xmode_changed)

        control_row.addWidget(driver_label)
        control_row.addWidget(self.driver_combo)
        control_row.addSpacing(20)
        control_row.addWidget(xmode_label)
        control_row.addWidget(self.xmode_combo)
        control_row.addStretch()
        root_layout.addLayout(control_row)

        # Matplotlib figure – three stacked panels, same proportions as qualifying viewer
        self._fig = plt.figure(figsize=(10, 6), facecolor=_BG)
        gs = gridspec.GridSpec(
            3, 1,
            figure=self._fig,
            height_ratios=[2, 1, 1],
            hspace=0.08,
        )

        # Speed panel
        self._ax_speed = self._fig.add_subplot(gs[0])
        self._line_speed, = self._ax_speed.plot([], [], color=_SPEED_COL, linewidth=1.5)
        self._ax_speed.set_facecolor(_BG)
        self._ax_speed.set_ylabel("Speed (km/h)", color=_SPEED_COL, fontsize=10)
        self._ax_speed.set_ylim(0, 380)
        self._ax_speed.tick_params(colors=_SPEED_COL, labelbottom=False)
        for spine in self._ax_speed.spines.values():
            spine.set_edgecolor("#555555")

        # Gear panel
        self._ax_gear = self._fig.add_subplot(gs[1])
        self._line_gear, = self._ax_gear.plot([], [], color=_GEAR_COL, linewidth=1.5, drawstyle="steps-post")
        self._ax_gear.set_facecolor(_BG)
        self._ax_gear.set_ylabel("Gear", color=_GEAR_COL, fontsize=10)
        self._ax_gear.set_ylim(0, 9)
        self._ax_gear.set_yticks(range(1, 9))
        self._ax_gear.tick_params(colors=_GEAR_COL, labelbottom=False)
        for spine in self._ax_gear.spines.values():
            spine.set_edgecolor("#555555")

        # Throttle / Brake panel
        self._ax_ctrl = self._fig.add_subplot(gs[2])
        self._line_throt, = self._ax_ctrl.plot([], [], color=_THROT_COL, linewidth=1.5)
        self._line_brake, = self._ax_ctrl.plot([], [], color=_BRAKE_COL, linewidth=1.5)
        self._ax_ctrl.set_facecolor(_BG)
        self._ax_ctrl.set_ylabel("Throttle / Brake (%)", color=_SPEED_COL, fontsize=10)
        self._ax_ctrl.set_ylim(-5, 105)
        self._ax_ctrl.tick_params(colors=_SPEED_COL)
        for spine in self._ax_ctrl.spines.values():
            spine.set_edgecolor("#555555")

        # x-axis label lives on the bottom panel only
        self._ax_ctrl.set_xlabel("Time (s)", color=_SPEED_COL, fontsize=9)

        self._canvas = FigureCanvas(self._fig)
        root_layout.addWidget(self._canvas)

        self._apply_xmode_labels()

    # ── X-axis mode helpers ───────────────────────────────────────────────

    def _on_xmode_changed(self, index: int):
        self._x_mode = "time" if index == 0 else "lap"
        self._apply_xmode_labels()
        self._redraw(self.driver_combo.currentText())

    def _apply_xmode_labels(self):
        if self._x_mode == "time":
            self._ax_ctrl.set_xlabel("Time (s)", color=_SPEED_COL, fontsize=9)
        else:
            self._ax_ctrl.set_xlabel("Distance (m)", color=_SPEED_COL, fontsize=9)
        self._ax_ctrl.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # ── Buffer management ─────────────────────────────────────────────────

    def _trace_for(self, code: str) -> DriverTrace:
        if code not in self._traces:
            self._traces[code] = DriverTrace(len(_COLUMNS))
        return self._traces[code]

    def _reset_state(self):
        """Forget everything from the previous connection, so two sessions never mix."""
        self._traces.clear()
        self._circuit_length_m = None
        self._max_lap_dist_m = 0.0
        self._last_epoch = None

    @staticmethod
    def _number(value) -> float | None:
        """A finite float, or None. A missing reading stays missing rather than becoming 0."""
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None

    def _add_sample(self, code: str, driver: dict, session_t: float, playback_speed: float,
                    seeked: bool | None):
        speed    = self._number(driver.get("speed"))
        gear     = self._number(driver.get("gear"))
        throttle = self._number(driver.get("throttle"))
        brake    = self._number(driver.get("brake"))
        if brake is not None:
            brake *= 100   # the stream sends brake as 0-1; the chart shows 0-100
        nan = float("nan")
        row = tuple(nan if v is None else v for v in (speed, gear, throttle, brake))

        # `dist` is already the distance within the current lap (f1_data.py resets it every lap)
        dist = self._number(driver.get("dist"))
        lap = driver.get("lap")
        lap = int(lap) if self._number(lap) is not None else None
        if dist is not None:
            self._max_lap_dist_m = max(self._max_lap_dist_m, dist)

        self._trace_for(code).add(session_t, lap, dist, row, playback_speed, seeked)

    # ── Driver selector ───────────────────────────────────────────────────

    def _on_driver_selected(self, driver_code: str):
        self._redraw(driver_code)

    def _refresh_driver_list(self, drivers: dict):
        incoming = sorted(drivers.keys())
        if incoming == self._known_drivers:
            return
        current = self.driver_combo.currentText()
        self.driver_combo.blockSignals(True)
        self.driver_combo.clear()
        self.driver_combo.addItems(incoming)
        if current in incoming:
            self.driver_combo.setCurrentText(current)
        elif incoming:
            self.driver_combo.setCurrentIndex(0)
        self.driver_combo.blockSignals(False)
        self._known_drivers = incoming

    # ── Chart redraw ──────────────────────────────────────────────────────

    def _redraw(self, driver_code: str):
        trace = self._traces.get(driver_code) if driver_code else None
        if trace is None:
            self._clear_lines()
            return

        if self._x_mode == "time":
            self._redraw_time(trace)
        else:
            self._redraw_lap(trace)

        self._canvas.draw_idle()

    def _redraw_time(self, trace: DriverTrace):
        xs, cols = trace.time_series()
        self._set_lines(xs, cols)
        for ax in (self._ax_speed, self._ax_gear, self._ax_ctrl):
            ax.set_xlim(-TIME_WINDOW_SEC, 0)

    def _redraw_lap(self, trace: DriverTrace):
        xs, cols = trace.lap_series()
        self._set_lines(xs, cols)
        # Whole lap on the axis: the official length, stretched if the measured distance runs past it
        lap_length = max(self._circuit_length_m or 0.0, self._max_lap_dist_m)
        if lap_length <= 0:
            lap_length = 1.0
        for ax in (self._ax_speed, self._ax_gear, self._ax_ctrl):
            ax.set_xlim(0, lap_length)

    def _set_lines(self, xs, cols):
        speeds, gears, throttles, brakes = cols
        self._line_speed.set_data(xs, speeds)
        self._line_gear.set_data(xs, gears)
        self._line_throt.set_data(xs, throttles)
        self._line_brake.set_data(xs, brakes)

    def _clear_lines(self):
        for line in (self._line_speed, self._line_gear, self._line_throt, self._line_brake):
            line.set_data([], [])
        self._canvas.draw_idle()

    # ── PitWallWindow overrides ───────────────────────────────────────────

    def on_telemetry_data(self, data):
        frame = data.get("frame")
        if not frame:
            return
        drivers = frame.get("drivers", {})
        session_t = self._number(frame.get("t"))
        if not drivers or session_t is None:
            return

        # Refreshed every message: a different race may have been loaded
        circuit_length = self._number(data.get("circuit_length_m"))
        if circuit_length:
            self._circuit_length_m = circuit_length

        playback_speed = self._number(data.get("playback_speed")) or 1.0

        # The replay counts seeks (timeline_epoch). A change means the playhead jumped, so lines may
        # need a break; no change means normal playback, however coarse the steps. An older replay
        # without the counter falls back to guessing from time gaps (seeked=None).
        epoch = data.get("timeline_epoch")
        if epoch is None:
            seeked = None
        else:
            seeked = self._last_epoch is not None and epoch != self._last_epoch
            self._last_epoch = epoch

        self._refresh_driver_list(drivers)

        for code, driver in drivers.items():
            self._add_sample(code, driver, session_t, playback_speed, seeked)

        selected = self.driver_combo.currentText()
        if selected:
            self._redraw(selected)

    def on_connection_status_changed(self, status):
        if status != "Connected":
            self._reset_state()
            self._clear_lines()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Driver Live Telemetry")
    window = DriverTelemetryWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
