"""
Rewind-safe buffers for live telemetry charts (src/insights/telemetry_trace.py).

The replay streams one snapshot per update, and the user can rewind or seek at any time. The
stream carries no "seek" event, so a chart that simply appends what arrives draws lines backwards
after a rewind.

The rule used here: a chart is a function (speed as a function of distance, or of time), so it can
never hold two values at the same x. Every trace keeps its samples strictly increasing in a key,
and adding a sample at key k first discards everything stored at key >= k. A rewind therefore
removes the "future" we left, a paused replay re-sending the same frame replaces it, and no
explicit rewind detection is needed.

Pure Python: no Qt, no matplotlib, so it is unit-tested directly.
"""

import bisect
import math

# A pessimistic replay update rate. Arcade aims for 60 per second, but a heavy frame can drop it to
# single digits (measured: 8.5/s when every message carried the 600 KB lap table), and assuming 60
# made normal playback look like a string of seeks.
MIN_STREAM_UPDATES_PER_SEC = 5
# Below this much race time between two samples, a jump is never treated as a seek.
MIN_SEEK_GAP_SEC = 0.5
# How many normal update steps a gap may span before it counts as a seek (absorbs frame-rate dips).
SEEK_GAP_STEPS = 3
# Seconds of history shown in "Last 30 seconds" mode.
TIME_WINDOW_SEC = 30.0

NAN = float("nan")


def seek_gap_seconds(playback_speed: float) -> float:
    """
    Largest race-time gap between consecutive samples that still counts as normal playback.

    Only a fallback: the replay now sends an explicit seek signal (timeline_epoch), and this is
    used only right after a seek, or with a replay that does not send the signal. At playback
    speed s, one update advances about s / updates-per-second of race time, so the threshold
    scales with speed.
    """
    speed = playback_speed if playback_speed and playback_speed > 0 else 1.0
    return max(MIN_SEEK_GAP_SEC, SEEK_GAP_STEPS * speed / MIN_STREAM_UPDATES_PER_SEC)


class MonotonicTrace:
    """
    Samples kept strictly increasing in `key`.

    Each sample also records its session time `t` and whether it follows a seek (a jump in time
    larger than the allowed gap), so the renderer can leave a gap instead of drawing a straight
    line across track that was never shown.
    """

    def __init__(self):
        self._keys: list[float] = []
        self._times: list[float] = []
        self._rows: list[tuple] = []
        self._breaks: list[bool] = []

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> list[float]:
        return list(self._keys)

    @property
    def last_key(self) -> float | None:
        return self._keys[-1] if self._keys else None

    def clear(self) -> None:
        self._keys.clear()
        self._times.clear()
        self._rows.clear()
        self._breaks.clear()

    def append(self, key: float, t: float, row: tuple, max_gap: float) -> None:
        """Add a sample, first discarding every stored sample at key >= `key`."""
        while self._keys and self._keys[-1] >= key:
            self._keys.pop()
            self._times.pop()
            self._rows.pop()
            self._breaks.pop()
        follows_seek = bool(self._times) and (t - self._times[-1]) > max_gap
        self._keys.append(key)
        self._times.append(t)
        self._rows.append(tuple(row))
        self._breaks.append(follows_seek)

    def trim_before(self, key_min: float) -> None:
        """Drop samples with key < key_min (used for the rolling time window)."""
        cut = bisect.bisect_left(self._keys, key_min)
        if cut:
            del self._keys[:cut]
            del self._times[:cut]
            del self._rows[:cut]
            del self._breaks[:cut]

    def series(self, n_columns: int) -> tuple[list[float], list[list[float]]]:
        """
        (xs, columns) ready for plotting, with a NaN row inserted before every sample that
        follows a seek. matplotlib does not connect points across a NaN.
        """
        xs: list[float] = []
        cols: list[list[float]] = [[] for _ in range(n_columns)]
        for i, (key, row, brk) in enumerate(zip(self._keys, self._rows, self._breaks)):
            if brk and i > 0:
                xs.append(NAN)
                for col in cols:
                    col.append(NAN)
            xs.append(key)
            for col, value in zip(cols, row):
                col.append(value)
        return xs, cols


class DriverTrace:
    """
    The two traces the telemetry window shows for one driver.

    - lap trace: keyed by distance within the lap; reset whenever the lap number changes, so a
      rewind into another lap starts a fresh trace at the car's true distance.
    - time trace: keyed by session time; only the last TIME_WINDOW_SEC seconds are kept.
    """

    def __init__(self, n_columns: int):
        self.n_columns = n_columns
        self.lap: int | None = None
        self.lap_trace = MonotonicTrace()
        self.time_trace = MonotonicTrace()

    def clear(self) -> None:
        self.lap = None
        self.lap_trace.clear()
        self.time_trace.clear()

    def add(self, t: float, lap: int | None, dist: float | None, row: tuple,
            playback_speed: float, seeked: bool | None = None) -> None:
        """
        seeked: True if the replay reports a seek since the previous message, False if it reports
        none (then the line is never broken), None if the replay sends no seek signal.

        After a seek the line is broken only if the new sample does not continue from what is
        kept: a rewind into a part already drawn joins up seamlessly.
        """
        max_gap = math.inf if seeked is False else seek_gap_seconds(playback_speed)

        self.time_trace.append(t, t, row, max_gap)
        self.time_trace.trim_before(t - TIME_WINDOW_SEC)

        if lap is None or dist is None or not math.isfinite(dist):
            return
        if lap != self.lap:
            self.lap_trace.clear()
            self.lap = lap
        self.lap_trace.append(dist, t, row, max_gap)

    def lap_series(self) -> tuple[list[float], list[list[float]]]:
        """x = distance into the current lap (m)."""
        return self.lap_trace.series(self.n_columns)

    def time_series(self) -> tuple[list[float], list[list[float]]]:
        """x = seconds relative to the newest sample (0 = now, -30 = thirty seconds ago)."""
        xs, cols = self.time_trace.series(self.n_columns)
        if not xs:
            return xs, cols
        t_now = self.time_trace.last_key
        return [x - t_now for x in xs], cols
