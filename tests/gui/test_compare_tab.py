"""The Compare chart (off-screen, matplotlib only) and the Compare tab in the Analyze window (Qt offscreen).

On a copy of the OpenF1 test lake (Qatar 2024: VER and NOR, laps 1-7, lap 6 the one both ran clean and whole), with its
track model and corner tables built and Lusail's official corners in the copy's reference folder (no network)."""

import json
import os
import shutil
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.gui.compare_charts import draw_comparison
from src.pipeline import lake
from src.telemetry import build
from src.telemetry import compare as cp
from tests.conftest import QATAR

LUSAIL = Path(__file__).parent.parent / "fixtures" / "multiviewer" / "circuit_150_2024.json"


@pytest.fixture(scope="module")
def qatar_root(openf1_qatar_lake, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("compare_tab") / "lake")
    shutil.copytree(openf1_qatar_lake, root)
    lake.write_json(json.loads(LUSAIL.read_text(encoding="utf-8")), lake.reference_path("circuit_150_2024", root))
    assert build.build_race_telemetry(*QATAR, root=root)["corner_rows"] > 0
    return root


@pytest.fixture(scope="module")
def qatar(qatar_root):
    return cp.load_race(*QATAR, root=qatar_root)


def qt():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_chart_has_a_point_after_each_corner_and_a_bar_per_corner(qatar):
    c = cp.compare_race(qatar, "VER", "NOR")
    fig = Figure(figsize=(13, 6.6))
    build_up, bars = draw_comparison(fig, c)
    FigureCanvasAgg(fig).draw()
    assert len(build_up.x) == len(bars.x) == 16
    assert build_up.y[-1] == pytest.approx(86.502 - 86.683)                   # the last point is the lap
    assert bars.text[5].startswith("T6 ·") and "how:" in bars.text[5]
    assert "VER against NOR" in fig._suptitle.get_text()
    legend = [t.get_text() for t in bars.ax.get_legend().get_texts()]
    assert legend == ["VER faster", "NOR faster"]                              # identity is never colour alone


def test_the_bars_are_each_corners_median_not_its_mean():
    """The made-up race of test_compare.py with a lock-up: corner 2 is level on a typical lap (median 0) although one
    lap lost 0.6 s there (mean +0.2 s). The bar shows the typical lap; the tooltip gives both."""
    from tests.telemetry.test_compare import made_up_race

    features, laps = made_up_race()
    lockup = (features["driver_code"] == "AAA") & (features["corner_order"] == 2) & (features["lap_number"] == 3)
    features.loc[lockup, "segment_time_sec"] += 0.6
    _, bars = draw_comparison(Figure(figsize=(10, 6)), cp.compare(features, laps, "AAA", "BBB"))
    assert list(bars.y) == pytest.approx([-0.1, 0.0, 0.05])
    assert "mean +0.200 s" in bars.text[1]


def test_the_tab_shows_nothing_until_two_drivers_are_picked_and_swapping_flips_every_sign(qatar):
    qt()
    from src.gui.compare_tab import PICK, CompareTab

    tab = CompareTab()
    tab.set_race(qatar, order=["VER", "NOR"])
    assert [tab.box_a.itemText(i) for i in range(tab.box_a.count())] == [PICK, "VER", "NOR"]
    assert "Pick two drivers" in tab.summary_lbl.text() and tab.comparison is None and tab.table.rowCount() == 0
    tab.box_b.setCurrentText("NOR")                                              # one driver is not a pair
    assert "Pick two drivers" in tab.summary_lbl.text() and tab.comparison is None

    tab.pick("VER", "NOR")
    text = tab.summary_lbl.text()
    assert text.startswith("VER against NOR: 1 paired lap") and "VER faster by 0.181 s a lap" in text
    assert "only 1 paired laps: read with care" in text and "tyres the same age" in text
    assert "following a car at the lap's start: VER" in text
    assert tab.table.rowCount() == 16 and tab.table.horizontalHeaderItem(4).text() == "VER faster on"
    assert tab.canvas.points

    tab._swap()
    assert tab.summary_lbl.text().startswith("NOR against VER") and tab.comparison.lap_gap_mean_sec > 0
    assert tab.table.horizontalHeaderItem(4).text() == "NOR faster on"

    tab.pick("VER", "VER")
    assert tab.summary_lbl.text() == "Pick two different drivers." and tab.table.rowCount() == 0


def test_the_window_has_a_compare_tab_with_the_races_drivers(qatar_root):
    qt()
    from src.gui.race_analysis_window import RaceAnalysisWindow

    win = RaceAnalysisWindow(*QATAR, root=qatar_root)
    try:
        tab = win.compare_tab
        assert win.tabs.tabText(win.tabs.indexOf(tab)) == "Compare"
        assert tab.box_a.isEnabled() and {"VER", "NOR"} <= {tab.box_a.itemText(i) for i in range(tab.box_a.count())}
    finally:
        win.close()
        win.deleteLater()


def test_a_race_built_laps_only_says_why_it_cannot_compare_and_how_to_build_it(golden_lake):
    qt()
    from src.gui.race_analysis_window import RaceAnalysisWindow
    from src.pipeline.cleaner import build_silver

    build_silver(*QATAR, "R", lake_root=golden_lake, telemetry=False)
    win = RaceAnalysisWindow(*QATAR, root=golden_lake)
    try:
        tab = win.compare_tab
        assert not tab.box_a.isEnabled() and "without telemetry (laps only)" in tab.summary_lbl.text()
        assert "python -m src.pipeline.runner --year 2024 --round 23" in tab.summary_lbl.text()
    finally:
        win.close()
        win.deleteLater()
