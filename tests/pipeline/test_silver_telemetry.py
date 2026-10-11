"""cleaner.resample_driver: irregular ~4 Hz streams -> true 10 Hz."""

import numpy as np
import pandas as pd

from src.pipeline.cleaner import resample_driver


def car_stream(times, speed=180.0, gear=None, throttle=None):
    n = len(times)
    return pd.DataFrame({
        "SessionTime": pd.to_timedelta(times, unit="s"),
        "Speed": np.full(n, speed) if np.isscalar(speed) else speed,
        "RPM": np.full(n, 10000.0),
        "Throttle": np.full(n, 50.0) if throttle is None else throttle,
        "nGear": np.full(n, 6) if gear is None else gear,
        "Brake": np.zeros(n, dtype=bool),
        "DRS": np.zeros(n, dtype=int),
    })


def pos_stream(times):
    n = len(times)
    return pd.DataFrame({"SessionTime": pd.to_timedelta(times, unit="s"),
                         "X": np.linspace(10, 1000, n), "Y": np.full(n, 5.0), "Z": np.full(n, 1.0)})


# Irregular source clock, like the real feed.
SRC = np.round(np.arange(99.87, 121.0, 0.27), 3)
LAP_STARTS = np.array([100.0, 110.0])
LAP_NUMBERS = np.array([1, 2])


def resample(car, pos=None, **kw):
    return resample_driver(car, pos_stream(SRC) if pos is None else pos, LAP_STARTS, LAP_NUMBERS, 120.0, **kw)


def test_grid_is_exactly_10hz_on_round_ticks():
    out = resample(car_stream(SRC))
    t = out["session_time_sec"].to_numpy()
    assert t[0] == 100.0 and t[-1] == 120.0
    assert np.allclose(np.diff(t), 0.1)
    assert np.allclose(t * 10, np.round(t * 10))     # every timestamp is a whole tenth


def test_every_row_gets_the_right_lap_number():
    out = resample(car_stream(SRC))
    assert set(out.loc[out["session_time_sec"] < 110.0, "lap_number"]) == {1}
    assert set(out.loc[out["session_time_sec"] >= 110.0, "lap_number"]) == {2}
    assert out.loc[out["session_time_sec"] == 110.0, "lap_elapsed_sec"].iloc[0] == 0.0


def test_lap_distance_integrates_speed_and_restarts_each_lap():
    out = resample(car_stream(SRC, speed=180.0))     # 180 km/h = 50 m/s
    lap1 = out[out["lap_number"] == 1]["lap_distance_m"]
    assert abs(lap1.iloc[-1] - 500.0) < 1.0          # 10 s at 50 m/s
    assert out[out["lap_number"] == 2]["lap_distance_m"].iloc[0] < 6.0


def test_continuous_signal_is_linearly_interpolated():
    out = resample(car_stream(SRC, speed=SRC * 2.0))  # speed is a straight line in time
    assert np.allclose(out["speed_kmh"], out["session_time_sec"] * 2.0, atol=1e-3)


def test_gear_holds_last_value_and_is_never_interpolated():
    gear = np.where(SRC < 105.0, 3, 5)                # jumps 3 -> 5, there is no gear 4
    out = resample(car_stream(SRC, gear=gear))
    assert set(out["gear"].dropna().astype(int)) == {3, 5}


def test_throttle_104_means_no_reading_and_is_ignored():
    throttle = np.full(len(SRC), 50.0)
    throttle[10:13] = 104.0
    out = resample(car_stream(SRC, throttle=throttle))
    assert out["throttle_pct"].max() <= 50.0 + 1e-4


def test_long_hole_is_left_null_and_flagged_not_interpolated():
    keep = (SRC < 103.0) | (SRC > 108.0)              # 5 second hole in the car feed
    out = resample(car_stream(SRC[keep]))
    inside = out[(out["session_time_sec"] > 103.5) & (out["session_time_sec"] < 107.5)]
    assert inside["speed_kmh"].isna().all() and inside["gear"].isna().all()
    assert inside["is_data_gap"].all()
    outside = out[out["session_time_sec"] > 109.0]
    assert outside["speed_kmh"].notna().all() and not outside["is_data_gap"].any()


def test_position_zero_zero_is_treated_as_no_fix():
    pos = pos_stream(SRC)
    pos.loc[5:7, ["X", "Y"]] = 0.0
    out = resample(car_stream(SRC), pos)
    assert out["x"].min() > 0.0
