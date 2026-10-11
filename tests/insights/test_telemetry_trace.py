"""
Tests for the rewind-safe telemetry buffers.

Each test feeds a sequence of (time, lap, distance) samples shaped like the replay stream and
checks what the chart would draw. No Qt and no matplotlib are needed.
"""

import math

import pytest

from src.insights.telemetry_trace import (
    TIME_WINDOW_SEC,
    DriverTrace,
    MonotonicTrace,
    seek_gap_seconds,
)

ROW = (250.0, 7, 100.0, 0.0)     # speed, gear, throttle, brake


def play(trace, t0, t1, lap, dist0, speed_mps=80.0, dt=0.04, playback_speed=1.0, seeked=None):
    """
    Feed samples every `dt` seconds from t0 to t1, driving at a constant speed. `seeked` is
    the replay's seek signal for the FIRST sample (None = an old replay that sends no signal).
    """
    n = int(round((t1 - t0) / dt))
    for i in range(n + 1):
        t = t0 + i * dt
        flag = seeked if (i == 0 or seeked is None) else False
        trace.add(t, lap, dist0 + (t - t0) * speed_mps, ROW, playback_speed, flag)


def segments(xs):
    """Split plotted x values at the NaN breaks."""
    out, cur = [], []
    for x in xs:
        if math.isnan(x):
            out.append(cur)
            cur = []
        else:
            cur.append(x)
    out.append(cur)
    return out


def strictly_increasing(values):
    return all(b > a for a, b in zip(values, values[1:]))


# --- MonotonicTrace ---------------------------------------------------------------------------

def test_append_keeps_keys_strictly_increasing():
    tr = MonotonicTrace()
    for k in (1.0, 2.0, 3.0, 2.5):
        tr.append(k, k, ROW, max_gap=10)
    assert tr.keys == [1.0, 2.0, 2.5]


def test_repeated_key_replaces_instead_of_growing():
    # A paused replay (or the last frame of the race) re-sends the same frame ~60 times a second
    tr = MonotonicTrace()
    for _ in range(100):
        tr.append(5.0, 5.0, ROW, max_gap=10)
    assert len(tr) == 1


def test_trim_before_drops_old_samples():
    tr = MonotonicTrace()
    for k in range(10):
        tr.append(float(k), float(k), ROW, max_gap=10)
    tr.trim_before(6.5)
    assert tr.keys == [7.0, 8.0, 9.0]


def test_series_inserts_nan_only_after_a_seek():
    tr = MonotonicTrace()
    tr.append(0.0, 0.0, ROW, max_gap=1.0)
    tr.append(1.0, 0.5, ROW, max_gap=1.0)
    tr.append(2.0, 9.0, ROW, max_gap=1.0)    # 8.5 s jump in time: a seek
    xs, cols = tr.series(len(ROW))
    assert segments(xs) == [[0.0, 1.0], [2.0]]
    assert all(len(c) == len(xs) for c in cols)


# --- seek threshold -----------------------------------------------------------------------------

def test_seek_gap_scales_with_playback_speed():
    assert seek_gap_seconds(0.1) == 0.5                  # floor at slow speeds
    assert seek_gap_seconds(1.0) == pytest.approx(3 * 1 / 5)
    assert seek_gap_seconds(256.0) == pytest.approx(3 * 256 / 5)
    # one normal update at any speed is never mistaken for a seek
    for speed in (0.1, 1, 4, 16, 64, 256):
        assert speed / 60 < seek_gap_seconds(speed)


def test_missing_or_zero_speed_falls_back_to_normal_speed():
    assert seek_gap_seconds(0) == seek_gap_seconds(1.0)
    assert seek_gap_seconds(None) == seek_gap_seconds(1.0)


# --- lap mode -----------------------------------------------------------------------------------

def test_normal_play_draws_one_increasing_line():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 30.0, lap=5, dist0=0.0)
    xs, _ = d.lap_series()
    assert len(segments(xs)) == 1
    assert strictly_increasing(xs)


def test_rewind_inside_a_lap_keeps_the_part_before_the_rewind_point():
    # The bug in the screenshot: play to ~3800 m, rewind to ~900 m, keep playing
    d = DriverTrace(len(ROW))
    play(d, 0.0, 48.0, lap=5, dist0=0.0)              # reaches 3840 m
    play(d, 11.25, 20.0, lap=5, dist0=900.0)           # rewound to 900 m
    xs, _ = d.lap_series()
    assert strictly_increasing(xs)                     # no backward strokes
    assert len(segments(xs)) == 1                      # and no spurious gap
    assert xs[0] == 0.0                                # the start of the lap is still drawn
    assert max(xs) == pytest.approx(1600.0, abs=1.0)   # the stale tail (to 3840 m) is gone


def test_rewind_into_an_earlier_lap_starts_at_the_true_distance():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 20.0, lap=6, dist0=0.0)
    play(d, -50.0, -45.0, lap=5, dist0=2500.0)         # seek back into lap 5, mid-lap
    xs, _ = d.lap_series()
    assert d.lap == 5
    assert xs[0] == pytest.approx(2500.0)              # not shifted to x = 0
    assert strictly_increasing(xs)


def test_forward_seek_inside_a_lap_leaves_a_gap():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 10.0, lap=5, dist0=0.0)               # up to 800 m
    play(d, 30.0, 35.0, lap=5, dist0=2400.0)           # jumped 20 s ahead
    xs, _ = d.lap_series()
    segs = segments(xs)
    assert len(segs) == 2
    assert segs[0][-1] == pytest.approx(800.0)
    assert segs[1][0] == pytest.approx(2400.0)


def test_fast_playback_is_not_mistaken_for_seeks():
    # At 256x one update advances 4.3 s of race time
    d = DriverTrace(len(ROW))
    play(d, 0.0, 80.0, lap=5, dist0=0.0, dt=256 / 60, playback_speed=256)
    xs, _ = d.lap_series()
    assert len(segments(xs)) == 1


def test_new_lap_resets_the_lap_trace():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 60.0, lap=5, dist0=0.0)
    play(d, 60.04, 62.0, lap=6, dist0=0.0)
    xs, _ = d.lap_series()
    assert d.lap == 6
    assert max(xs) < 200.0


def test_missing_distance_skips_the_lap_trace_but_not_the_time_trace():
    d = DriverTrace(len(ROW))
    d.add(1.0, 5, None, ROW, 1.0)
    assert len(d.lap_trace) == 0
    assert len(d.time_trace) == 1


# --- time mode ----------------------------------------------------------------------------------

def test_time_trace_is_relative_to_now_and_capped_at_the_window():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 100.0, lap=5, dist0=0.0, speed_mps=1.0)
    xs, _ = d.time_series()
    assert xs[-1] == 0.0
    assert min(xs) >= -TIME_WINDOW_SEC


def test_time_trace_never_exceeds_the_window_after_a_rewind():
    # Old behaviour: after a rewind the oldest sample was "in the future", trimming stopped and the
    # buffer grew without bound
    d = DriverTrace(len(ROW))
    play(d, 0.0, 100.0, lap=5, dist0=0.0, speed_mps=1.0)
    play(d, 40.0, 50.0, lap=5, dist0=40.0, speed_mps=1.0)    # rewound 60 s
    keys = d.time_trace.keys
    assert keys[-1] == pytest.approx(50.0)
    assert keys[0] >= 50.0 - TIME_WINDOW_SEC
    assert strictly_increasing(keys)


def test_time_trace_forward_seek_leaves_a_gap():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 10.0, lap=5, dist0=0.0)
    play(d, 20.0, 25.0, lap=5, dist0=1600.0)
    xs, _ = d.time_series()
    assert len(segments(xs)) == 2


def test_clear_forgets_everything():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 5.0, lap=5, dist0=0.0)
    d.clear()
    assert d.lap is None
    assert len(d.lap_trace) == 0 and len(d.time_trace) == 0


# --- with the replay's seek signal (timeline_epoch) ---------------------------------------------

def test_slow_replay_without_seeks_never_breaks_the_line():
    # The bug in the second screenshot: the replay ran at ~8.5 updates/s at 4x, so each message
    # advanced ~0.5 s and frame-rate dips looked like seeks. With the signal, no seek = no break.
    d = DriverTrace(len(ROW))
    play(d, 0.0, 60.0, lap=5, dist0=0.0, dt=1.2, playback_speed=4.0, seeked=False)
    xs, _ = d.lap_series()
    assert len(segments(xs)) == 1


def test_signalled_forward_seek_breaks_the_line():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 10.0, lap=5, dist0=0.0, seeked=False)
    play(d, 30.0, 35.0, lap=5, dist0=2400.0, seeked=True)
    xs, _ = d.lap_series()
    assert len(segments(xs)) == 2


def test_signalled_rewind_into_a_drawn_part_joins_up():
    d = DriverTrace(len(ROW))
    play(d, 0.0, 48.0, lap=5, dist0=0.0, seeked=False)
    play(d, 11.25, 20.0, lap=5, dist0=900.0, seeked=True)
    xs, _ = d.lap_series()
    assert len(segments(xs)) == 1 and strictly_increasing(xs)
