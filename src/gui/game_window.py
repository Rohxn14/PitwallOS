"""
The strategist game window (src/gui/game_window.py) - Phase 4 step 4

The game of src/simulator/game.py in a window, opened from the session picker (src/gui/race_selection.py): "Play" on a
2023-2024 race plays that race, named; "Play a blind race" picks a random dry one and hides the circuit, the year and
the names (the other cars are letters in grid order, you are YOU) until the flag. After the models load, three pages:

    your car    the order at the end of the start lap (lap 5 or so); click a car that finished the real race (your
                result is compared with the real team's) and "Take this car", or "A random car"
    the race    why the game stopped, where you are, the running order, and your calls: each with your chance to win,
                to finish on the podium, to score points and your average place, and a button to make it
    the flag    your place, the real team's strategy in the same race with the same luck, what really happened, the
                race revealed, your stops and theirs

The slow work runs off the screen's thread (GameWorker, a QThread): loading the models (fitting them the first time,
about 10 minutes) and, after every call, running the race on to the next big moment and counting each call's futures
(10-20 s) while a bar moves, so the window never freezes. The window decides nothing: every number comes from game.py.

    python -m src.gui.game_window                 a blind race
    python -m src.gui.game_window 2024 13         that race, named
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QProgressBar, QPushButton, QStackedWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.gui.analysis_theme import ACCENT, BAD, COMPOUND_COLOURS, DIM, GOOD, TEXT
from src.gui.race_analysis_window import STYLE, _item, _table
from src.pipeline import lake
from src.simulator import game as game_module, race as race_module

PLAYABLE_SEASONS = (2023, 2024)     # the development races the models learned from (2025 stays locked)
COMPOUND_OF = {"softs": "SOFT", "mediums": "MEDIUM", "hards": "HARD"}
_LEFT_RUNNING = set()               # windows closed while their worker runs: kept alive until it ends


class NotPlayable(Exception):
    """A race the game cannot play (another season, or rain)."""


class GameWorker(QThread):
    """One piece of the game's slow work, off the screen's thread; `progress` says what it is doing."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, work: Callable[[Callable[[str], None]], object]):
        super().__init__()
        self.work = work

    def run(self):
        try:
            self.done.emit(self.work(self.progress.emit))
        except NotPlayable as e:
            self.failed.emit(str(e))
        except Exception as e:      # shown in the window: never a silent failure
            self.failed.emit(f"{type(e).__name__}: {e}")


def _tyre_item(text: str) -> QTableWidgetItem:
    """A tyres cell in its compound's colour ("softs, 12 laps" in soft red)."""
    compound = COMPOUND_OF.get(text.split(",")[0].strip())
    return _item(text, colour=COMPOUND_COLOURS.get(compound, DIM))


def _heading(text: str, size: int = 12) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setPointSize(size)
    font.setBold(True)
    label.setFont(font)
    return label


class GameWindow(QMainWindow):
    """The strategist game: pick a car, make the calls at the big moments, and see how you did against the real team."""

    def __init__(self, race_key: Optional[Tuple[int, int]] = None, root: str = lake.DEFAULT_LAKE_ROOT,
                 futures: int = game_module.FUTURES, seed: Optional[int] = None, wait: bool = False, parent=None):
        """race_key None: a blind race. wait=True does the slow work on this thread, so a test needs no event loop."""
        super().__init__(parent)
        self.race_key, self.root, self.futures, self.wait = race_key, root, futures, wait
        self.blind = race_key is None
        self.seed = seed if seed is not None else random.randrange(1_000_000)
        self.models: Optional[game_module.Models] = None
        self.race: Optional[race_module.Race] = None
        self.game: Optional[game_module.Game] = None
        self.turn: Optional[game_module.Turn] = None
        self.after = 0
        self.worker: Optional[GameWorker] = None
        self.closed = False
        self.car_rows: List[str] = []
        self.choice_buttons: List[QPushButton] = []
        self.calls: List[str] = []
        self.other_windows: List[GameWindow] = []

        self.setWindowTitle("PitwallOS — The strategist game"
                            + ("" if self.blind else f" — {race_key[0]} round {race_key[1]}"))
        self.resize(1240, 840)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._load()

    # --- layout -------------------------------------------------------------------------------------------

    def _build_ui(self):
        page = QWidget(objectName="page")
        self.setCentralWidget(page)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(_heading("🎮  The strategist game", 16))
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)                     # it moves without a number: how long is unknown
        self.busy_bar.setTextVisible(False)
        self.busy_text = QLabel("")
        self.busy_text.setStyleSheet(f"color: {DIM};")
        outer.addWidget(self.busy_text)
        outer.addWidget(self.busy_bar)
        for build in (self._build_wait_page, self._build_car_page, self._build_race_page, self._build_result_page):
            self.stack.addWidget(build())

    def _build_wait_page(self) -> QWidget:
        self.wait_page = QWidget(objectName="page")
        layout = QVBoxLayout(self.wait_page)
        layout.addStretch()
        self.wait_title = _heading("", 14)
        self.wait_title.setAlignment(Qt.AlignCenter)
        self.wait_text = QLabel("")
        self.wait_text.setAlignment(Qt.AlignCenter)
        self.wait_text.setWordWrap(True)
        self.wait_close = QPushButton("Close")
        self.wait_close.clicked.connect(self.close)
        self.wait_close.setVisible(False)
        layout.addWidget(self.wait_title)
        layout.addWidget(self.wait_text)
        layout.addWidget(self.wait_close, alignment=Qt.AlignCenter)
        layout.addStretch()
        return self.wait_page

    def _build_car_page(self) -> QWidget:
        self.car_page = QWidget(objectName="page")
        layout = QVBoxLayout(self.car_page)
        layout.addWidget(_heading("Pick your car"))
        self.car_intro = QLabel("")
        self.car_intro.setWordWrap(True)
        layout.addWidget(self.car_intro)
        self.car_table = _table(["Place", "Car", "Tyres", ""])
        self.car_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.car_table.itemSelectionChanged.connect(self._car_selected)
        self.car_table.cellDoubleClicked.connect(lambda row, _column: self._take_car())
        layout.addWidget(self.car_table, 1)
        buttons = QHBoxLayout()
        self.take_btn = QPushButton("Take this car")
        self.take_btn.setEnabled(False)
        self.take_btn.clicked.connect(self._take_car)
        self.random_btn = QPushButton("A random car")
        self.random_btn.clicked.connect(self._take_random_car)
        buttons.addStretch()
        buttons.addWidget(self.random_btn)
        buttons.addWidget(self.take_btn)
        layout.addLayout(buttons)
        return self.car_page

    def _build_race_page(self) -> QWidget:
        self.race_page = QWidget(objectName="page")
        layout = QVBoxLayout(self.race_page)
        self.moment_title = _heading("", 15)
        self.moment_why = QLabel("")
        self.moment_why.setWordWrap(True)
        self.moment_why.setStyleSheet(f"color: {ACCENT}; font-size: 13px;")
        self.situation = QLabel("")
        self.situation.setWordWrap(True)
        layout.addWidget(self.moment_title)
        layout.addWidget(self.moment_why)
        layout.addWidget(self.situation)
        columns = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(_heading("Your calls"))
        hint = QLabel("Each call is played out over the same futures of the rest of the race: the safety cars, the "
                      "other teams' answers and the retirements drawn, nothing of the real race's future in them.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        left.addWidget(hint)
        self.option_table = _table(["Call", "Win", "Podium", "Points", "Avg place", ""])
        header = self.option_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        left.addWidget(self.option_table, 1)
        right.addWidget(_heading("Running order"))
        self.order_table = _table(["P", "Car", "Gap ahead", "To the leader", "Tyres"])
        right.addWidget(self.order_table, 1)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)
        layout.addLayout(columns, 1)
        self.calls_label = QLabel("")
        self.calls_label.setWordWrap(True)
        self.calls_label.setStyleSheet(f"color: {DIM};")
        layout.addWidget(self.calls_label)
        return self.race_page

    def _build_result_page(self) -> QWidget:
        self.result_page = QWidget(objectName="page")
        layout = QVBoxLayout(self.result_page)
        layout.addWidget(_heading("Chequered flag", 15))
        self.result_title = _heading("", 22)
        self.result_title.setStyleSheet(f"color: {ACCENT};")
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        self.result_text.setTextFormat(Qt.RichText)
        layout.addWidget(self.result_title)
        layout.addWidget(self.result_text)
        layout.addStretch()
        buttons = QHBoxLayout()
        self.again_btn = QPushButton("Play a blind race")
        self.again_btn.clicked.connect(self._play_again)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addStretch()
        buttons.addWidget(self.again_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        return self.result_page

    # --- the slow work -------------------------------------------------------------------------------------

    def _run(self, work: Callable[[Callable[[str], None]], object], on_done: Callable[[object], None], message: str):
        """Do `work` off the screen's thread (on it with wait=True), then on_done with its result."""
        self.busy_bar.setVisible(True)
        self._say(message)
        if self.wait:
            try:
                result = work(self._say)
            except NotPlayable as e:
                self._failed(str(e))
                return
            except Exception as e:
                self._failed(f"{type(e).__name__}: {e}")
                return
            on_done(result)
            return
        self.worker = GameWorker(work)
        self.worker.progress.connect(self._say)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _say(self, text: str):
        self.busy_text.setText(text)

    def _idle(self):
        self.busy_bar.setVisible(False)
        self.busy_text.setText("")

    def _failed(self, message: str):
        if self.closed:
            return
        self._idle()
        self.wait_title.setText("The game stopped")
        self.wait_text.setText(message)
        self.wait_text.setStyleSheet(f"color: {BAD};")
        self.wait_close.setVisible(True)
        self.stack.setCurrentWidget(self.wait_page)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.closed = True
            _LEFT_RUNNING.add(self)                       # Qt must not delete a window whose thread still runs
            self.worker.finished.connect(lambda: _LEFT_RUNNING.discard(self))
        super().closeEvent(event)

    # --- the models and the race ---------------------------------------------------------------------------

    def _load(self):
        saved = lake.model_path(game_module.MODELS_NAME, self.root).exists()
        self.wait_title.setText("Loading the game" if saved else "Fitting the game's models (once: about 10 minutes)")
        self.wait_text.setText("The models learned from the 2023 and 2024 races; each race is played with the ones "
                               "that never saw it.")
        self.stack.setCurrentWidget(self.wait_page)
        self._run(self._load_work, self._on_loaded, "Loading ...")

    def _load_work(self, say: Callable[[str], None]):
        models = game_module.load_models(self.root, log=say)
        key = self.race_key or random.Random(self.seed).choice(sorted(models.dry))
        if tuple(key) not in models.dry:
            raise NotPlayable(f"The game plays the dry races of {PLAYABLE_SEASONS[0]} and {PLAYABLE_SEASONS[-1]}, "
                              f"the races its models learned from (each played with the models that never saw it). "
                              f"{key[0]} round {key[1]} is not one of them: another season, or rain (the pace model "
                              f"cannot see rain coming).")
        return models, race_module.load(*key, root=self.root)

    def _on_loaded(self, loaded):
        if self.closed:
            return
        self._idle()
        self.models, self.race = loaded
        start_lap = race_module.first_start_lap(self.race)
        state = race_module.start_state(self.race, start_lap)
        labels = game_module.labels_of(self.race, "", self.blind)
        self.car_rows = game_module.order_of(state)
        what = ("A dry race of 2023-2024, played blind: the circuit, the year and the names stay hidden until the "
                "flag." if self.blind else f"The {self.race.season} {self.race.event_name}.")
        self.car_intro.setText(f"{what} {self.race.total_laps} laps. The order at the end of lap {start_lap}, where "
                               "you take over (the simulator needs a few real laps to know each driver's pace). "
                               "Take a car that finished the real race: your result is compared with its team's.")
        laps = self.race.laps
        self.car_table.setRowCount(len(self.car_rows))
        for r, d in enumerate(self.car_rows):
            done = int(state.at[d, "laps_done"])
            row = laps[(laps["driver_code"] == d) & (laps["lap_number"] == done)]
            compound = row["compound"].iloc[0] if len(row) else None
            tyres = (f"{game_module.PLURAL.get(compound, str(compound).lower())}, {float(row['tyre_age'].iloc[0]):.0f} "
                     "laps") if isinstance(compound, str) else "?"
            can = d in self.race.finishers
            cells = [_item(f"P{r + 1}"), _item(labels[d], bold=True), _tyre_item(tyres),
                     _item("" if can else "did not finish: cannot be chosen", colour=DIM)]
            for c, cell in enumerate(cells):
                if not can:
                    cell.setFlags(Qt.NoItemFlags)
                self.car_table.setItem(r, c, cell)
        self.stack.setCurrentWidget(self.car_page)

    def _car_selected(self):
        rows = self.car_table.selectionModel().selectedRows()
        self.take_btn.setEnabled(bool(rows) and self.car_rows[rows[0].row()] in self.race.finishers)

    def _take_car(self):
        rows = self.car_table.selectionModel().selectedRows()
        if rows and self.car_rows[rows[0].row()] in self.race.finishers:
            self._start(self.car_rows[rows[0].row()])

    def _take_random_car(self):
        self._start(random.Random(self.seed).choice([d for d in self.car_rows if d in self.race.finishers]))

    # --- the race ------------------------------------------------------------------------------------------

    def _start(self, driver: str):
        self.take_btn.setEnabled(False)
        self.random_btn.setEnabled(False)
        self.moment_title.setText("Lights out")
        self.moment_why.setText("")
        self.situation.setText("")
        self.option_table.setRowCount(0)
        self.order_table.setRowCount(0)
        self.stack.setCurrentWidget(self.race_page)
        self._run(lambda say: self._begin(driver, say), self._on_step, "The race starts ...")

    def _begin(self, driver: str, say: Callable[[str], None]):
        self.game = game_module.new_game(self.race, driver, self.models, seed=self.seed, futures=self.futures,
                                         blind=self.blind)
        self.after = self.game.start_lap
        plan = ", ".join(f"box lap {s} for {game_module.PLURAL[c]}" for s, c in game_module.stops_in(
            self.game.plan, driver) if s > self.game.start_lap) or "no stop"
        self.calls = [f"From lap {self.game.start_lap} the plan was the fastest finish by the tyre model: {plan}."]
        return self._next_step(say)

    def _next_step(self, say: Callable[[str], None]):
        """Your race on to the next big moment, its calls counted; or, at the flag, the result."""
        say("The race runs on to the next big moment ...")
        turn = game_module.next_turn(self.game, self.after, counting=lambda moment, options: say(
            f"Lap {moment.lap}: counting {self.futures} futures for each of your {len(options)} calls ..."))
        if turn is not None:
            return turn
        say("The chequered flag: running the real team's strategy in the same race ...")
        return game_module.finish(self.game, game_module.played(self.game))

    def _on_step(self, step):
        if self.closed:
            return
        self._idle()
        if isinstance(step, game_module.Turn):
            self._show_turn(step)
        else:
            self._show_result(step)

    def _show_turn(self, turn: game_module.Turn):
        self.turn = turn
        game, moment = self.game, turn.moment
        self.moment_title.setText(f"Lap {moment.lap} of {game.total_laps}")
        why = "; ".join(moment.reasons)
        self.moment_why.setText(why[0].upper() + why[1:] + ".")
        self.situation.setText(game_module.where_you_are(game, moment.state))
        order = game_module.running_order(game, moment.state)
        self.order_table.setRowCount(len(order))
        for r, row in enumerate(order.to_dict("records")):
            mine = bool(row["yours"])
            colour = ACCENT if mine else TEXT
            cells = [_item(f"{row['place']}", colour=colour, bold=mine), _item(row["car"], colour=colour, bold=mine),
                     _item(row["gap ahead"], colour=colour), _item(row["to the leader"], colour=colour),
                     _tyre_item(row["tyres"])]
            for c, cell in enumerate(cells):
                self.order_table.setItem(r, c, cell)
        self.option_table.setRowCount(len(turn.options))
        self.choice_buttons = []
        for r, (option, odds) in enumerate(zip(turn.options, turn.table.itertuples(index=False))):
            self.option_table.setItem(r, 0, _item(option.label, align=Qt.AlignLeft, bold=True))
            for c, value in enumerate((odds.win, odds.podium, odds.points), start=1):
                self.option_table.setItem(r, c, _item(f"{value:.0%}", colour=GOOD if value >= 0.5 else TEXT))
            self.option_table.setItem(r, 4, _item(f"{odds.place:.1f}"))
            button = QPushButton("Choose")
            button.clicked.connect(lambda _=False, i=r: self._choose(i))
            self.option_table.setCellWidget(r, 5, button)
            self.choice_buttons.append(button)
        self.option_table.resizeRowsToContents()
        self.calls_label.setText("Your calls so far: " + " · ".join(self.calls[1:]) if len(self.calls) > 1 else self.calls[0])

    def _choose(self, pick: int):
        for button in self.choice_buttons:
            button.setEnabled(False)
        option = self.turn.options[pick]
        self.after = game_module.take(self.game, self.turn, pick)
        self.calls.append(f"lap {self.turn.moment.lap}: {option.label}")
        self.calls_label.setText("Your calls so far: " + " · ".join(self.calls[1:]))
        self._run(self._next_step, self._on_step, "The race runs on ...")

    def _show_result(self, result: game_module.Result):
        game, race = self.game, self.game.race
        place = lambda p: f"P{p}" if p is not None else "did not finish"
        team = race.laps.loc[race.laps["driver_code"] == game.driver, "team_code"].iloc[0]
        stops = lambda found: ", ".join(f"lap {s} {game_module.PLURAL.get(c, c)}" for s, c in found
                                        if s > game.start_lap) or "none"
        self.result_title.setText(f"You finished {place(result.place)}")
        lines = [f"The real team's strategy, in the same race with the same luck: <b>{place(result.real_team_place)}</b>",
                 f"What really happened: <b>{place(result.real_place)}</b>",
                 "",
                 f"It was the <b>{race.season} {race.event_name}</b>; you were <b>{game.driver}</b> ({team}).",
                 f"Your stops after lap {game.start_lap}: {stops(result.your_stops)}.",
                 f"The real team's: {stops(result.real_stops)}.",
                 "",
                 "Your calls: " + ("; ".join(self.calls[1:]) or "none")]
        self.result_text.setText("<br>".join(lines))
        self.stack.setCurrentWidget(self.result_page)

    def _play_again(self):
        window = GameWindow(None, root=self.root, futures=self.futures, wait=self.wait)
        window.show()
        self.other_windows = [w for w in self.other_windows if w.isVisible()] + [window]


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    args = sys.argv[1:]
    window = GameWindow((int(args[0]), int(args[1])) if len(args) >= 2 else None)
    window.show()
    sys.exit(app.exec())
