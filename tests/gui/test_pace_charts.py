"""
Pace charts (src/gui/pace_charts.py), drawn off-screen with matplotlib only (no Qt).

They check that each chart draws what the analytics say, on the two reference races: the four
neutralised periods of Qatar shaded, outliers labelled, laps beyond the scale kept on its edge,
one box per compound, one panel per compound. And the small pieces the tab relies on: tooltip
text, nearest-point search, lap-time format.
"""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analytics.laps import race_pace
from src.gui.pace_charts import (
    draw_compound_pace, draw_driver_pace, draw_lap_times, driver_chart_rows, lap_clock, lap_tooltip, nearest,
)

BRITISH, QATAR = (2024, 12), (2024, 23)


@pytest.fixture(scope="module")
def paces(golden_silver_laps):
    return {race: race_pace(laps) for race, laps in golden_silver_laps.items()}


def drawn(fig):
    """Render for real, off-screen (a bare Figure's canvas draws nothing): layout, legend, annotations."""
    FigureCanvasAgg(fig).draw()
    return fig


def test_lap_chart_shades_each_neutralised_period_and_keeps_slow_laps_on_the_edge(paces):
    fig = Figure(figsize=(12, 4.6))
    points = draw_lap_times(fig, paces[QATAR], "VER")
    drawn(fig)
    ax = points[0].ax
    assert len(ax.patches) == 4                                          # SC 1-4, SC 35-39, VSC 40, SC 40-42
    assert [t.get_text() for t in ax.texts] == ["SC", "SC", "VSC", "SC"]   # no outlier labels: none in Qatar
    laps = paces[QATAR].laps
    timed = laps[(laps["driver_code"] == "VER") & laps["lap_time_sec"].notna()]
    assert len(points[0].x) == len(timed) == 53                          # 57 laps, 4 without a time
    top = ax.get_ylim()[1]
    assert (points[0].y < top).all()                                     # safety-car laps sit inside the edge
    assert np.isclose(points[0].y[points[0].x == 2], top - 0.025 * (top - ax.get_ylim()[0])).all()
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[:2] == ["Medium", "Hard"] and "neutralised" in legend


def test_lap_chart_labels_every_outlier(paces):
    fig = Figure(figsize=(12, 4.6))
    points = draw_lap_times(fig, paces[BRITISH], "HAM")
    drawn(fig)
    labels = [t.get_text() for t in points[0].ax.texts if t.get_text().startswith("+")]
    laps = paces[BRITISH].laps
    assert len(labels) == int(laps[laps["driver_code"] == "HAM"]["is_outlier"].sum()) == 3
    assert "+8.6 s" in labels                                            # lap 19, the first lap of the shower
    assert "outlier (> 3 s over)" in [t.get_text() for t in fig.legends[0].get_texts()]


def test_lap_chart_for_a_driver_without_laps_says_so(paces):
    fig = Figure()
    assert draw_lap_times(fig, paces[QATAR], "XXX") == []
    assert "No timed laps" in fig.axes[0].texts[0].get_text()


def test_compound_chart_has_one_box_per_compound_with_its_count_and_timing(paces):
    fig = Figure(figsize=(6.6, 4))
    points = draw_compound_pace(fig, paces[BRITISH])
    drawn(fig)
    ax = points[0].ax
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert [t.split("\n")[0] for t in ticks] == ["Soft", "Medium", "Hard", "Intermediate"]
    assert "around lap 41" in ticks[2]                                   # the hards ran late: the raw confound
    assert len(points[0].x) == 4 and "raw: not fuel-corrected" in points[0].text[0]


def test_driver_chart_has_one_panel_per_compound_and_highlights_the_driver(paces):
    fig = Figure(figsize=(12, 5))
    points = draw_driver_pace(fig, paces[QATAR], highlight="VER")
    drawn(fig)
    assert [hp.ax.get_title(loc="left").split(" · ")[0] for hp in points] == ["Soft", "Medium", "Hard"]
    medium = points[1].ax
    bold = [t.get_text() for t in medium.get_yticklabels() if t.get_fontweight() == "bold"]
    assert bold == ["VER"]
    assert driver_chart_rows(paces[QATAR]) == 16                         # the medium panel is the tallest


def test_lap_tooltip_says_why_a_lap_is_or_is_not_a_pace_lap():
    base = {"lap_number": 19, "driver_code": "HAM", "lap_time_sec": 103.711, "compound": "MEDIUM", "tyre_life": 19,
            "reference_sec": 95.07, "over_reference_sec": 8.641, "outlier_checked": True, "is_deleted": False}
    assert lap_tooltip({**base, "pace_status": "outlier"}).splitlines() == [
        "Lap 19 · HAM", "103.711 s  (1:43.711)", "Medium · tyre age 19",
        "outlier · +8.64 s vs reference 95.07 s (limit +3 s)"]
    assert lap_tooltip({**base, "pace_status": "safety car"}).endswith("not a pace lap: safety car")
    unchecked = lap_tooltip({**base, "pace_status": "pace", "outlier_checked": False})
    assert "not checked" in unchecked
    assert "deleted" in lap_tooltip({**base, "pace_status": "pace", "is_deleted": True})
    assert "No compound" in lap_tooltip({**base, "compound": None, "pace_status": "pace"})


def test_nearest_point_within_the_radius_only():
    points = np.array([[10.0, 10.0], [40.0, 10.0]])
    assert nearest(points, 12.0, 11.0, radius=12) == 0
    assert nearest(points, 37.0, 14.0, radius=12) == 1
    assert nearest(points, 25.0, 40.0, radius=12) is None
    assert nearest(np.empty((0, 2)), 0.0, 0.0, radius=12) is None


def test_lap_clock():
    assert lap_clock(84.468) == "1:24.468"
    assert lap_clock(59.9999) == "1:00.000"
    assert lap_clock(125.0) == "2:05.000"
