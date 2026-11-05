"""
The strategist game's window (src/gui/game_window.py), Qt offscreen, on the game's three-race lake (conftest: game_lake;
the work done on the test's thread, wait=True).

Qatar 2024, named: the car page lists the order at the end of lap 6 with the cars that did not finish greyed out (they
cannot be taken); taking Verstappen leads to his first big moment, with one row and one button per call and the
running order with his row marked; making the first call at every moment plays the race to the flag, where the result
names the race and the real team's place. A blind race hides every name until the flag. A race the game cannot play
(a wet one: British 2024) says so instead of crashing.
"""

import os

import pytest

QATAR, BRITISH = (2024, 23), (2024, 12)


@pytest.fixture(scope="module")
def qt():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def text_of(table, column):
    return [table.item(r, column).text() for r in range(table.rowCount())]


def test_a_whole_game_on_qatar(qt, game_lake):
    from src.gui.game_window import GameWindow
    _, root = game_lake
    window = GameWindow(QATAR, root=root, futures=4, seed=3, wait=True)
    try:
        assert window.stack.currentWidget() is window.car_page
        cars = text_of(window.car_table, 1)
        assert len(cars) == 18 and "VER" in cars and "RUS" in cars                 # named: driver codes
        from PySide6.QtCore import Qt
        grey = [window.car_rows[r] for r in range(len(cars)) if window.car_table.item(r, 0).flags() == Qt.NoItemFlags]
        assert set(grey) == {"PER", "HUL", "STR"}                                  # retired later: cannot be taken
        window.car_table.selectRow(window.car_rows.index("VER"))
        assert window.take_btn.isEnabled()
        window.take_btn.click()
        assert window.stack.currentWidget() is window.race_page
        assert window.option_table.rowCount() == len(window.turn.options) == len(window.choice_buttons) >= 2
        assert "YOU" in text_of(window.order_table, 1)
        assert window.moment_title.text().startswith("Lap ")
        calls = 0
        while window.stack.currentWidget() is window.race_page and calls < 40:
            window.choice_buttons[0].click()
            calls += 1
        assert window.stack.currentWidget() is window.result_page
        assert window.result_title.text().startswith("You finished")
        assert "2024 Qatar Grand Prix" in window.result_text.text() and "VER" in window.result_text.text()
        assert len(window.game.decisions) == calls
    finally:
        window.close()
        window.deleteLater()


def test_a_blind_race_hides_the_names_until_the_flag(qt, game_lake):
    from src.gui.game_window import GameWindow
    _, root = game_lake
    window = GameWindow(None, root=root, futures=4, seed=11, wait=True)
    try:
        cars = text_of(window.car_table, 1)
        assert cars and all(c.startswith("car ") for c in cars)                   # letters, no driver codes
        assert window.race.key in {QATAR, (2024, 13)}                              # a dry race of the lake
        window.random_btn.click()
        assert window.stack.currentWidget() is window.race_page
        shown = window.situation.text() + " ".join(text_of(window.order_table, 1)) + window.moment_why.text()
        assert window.game.driver not in shown and window.race.event_name not in shown
    finally:
        window.close()
        window.deleteLater()


def test_a_race_the_game_cannot_play_says_so(qt, game_lake):
    from src.gui.game_window import GameWindow
    _, root = game_lake
    window = GameWindow(BRITISH, root=root, futures=4, wait=True)
    try:
        assert window.stack.currentWidget() is window.wait_page
        assert "dry races" in window.wait_text.text() and "rain" in window.wait_text.text()
        assert not window.wait_close.isHidden()                                    # a Close button, no crash
    finally:
        window.close()
        window.deleteLater()
