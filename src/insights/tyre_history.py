"""
Rewind-safe tyre history for the tyre strategy window (src/insights/tyre_history.py).

The window used to append a new stint whenever the compound changed, and never removed one. After
a rewind it kept stints from the "future", a same-compound stop (e.g. fresh intermediates) was
invisible, and saved state leaked into the next race.

Here the window only records what tyre each driver had on each lap. A sample on lap L first
forgets everything recorded after lap L (so a rewind removes the future), and stints are derived
from the per-lap record with the same rule the data pipeline uses (src/pipeline/cleaner.py):
a new stint starts when the compound changes or the tyre age does not simply continue.

Pure Python, no Qt, so it is unit-tested directly.
"""

import math


def _number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class TyreHistory:
    """What tyre one driver had on each lap they were seen on."""

    def __init__(self):
        # lap -> (compound, tyre_life or None)
        self._by_lap: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self._by_lap)

    def clear(self) -> None:
        self._by_lap.clear()

    def observe(self, lap: int, compound, tyre_life=None) -> None:
        """Record the tyre on `lap`, first forgetting every later lap (a rewind)."""
        for later in [l for l in self._by_lap if l > lap]:
            del self._by_lap[later]
        self._by_lap[lap] = (compound, _number(tyre_life))

    def stints(self, current_lap: int | None = None) -> list[dict]:
        """
        [{"tyre", "start_lap", "end_lap"}], oldest first. A finished stint ends where the next
        one starts; the running stint has end_lap None (drawn up to the current lap).
        """
        result: list[dict] = []
        prev_lap = prev_compound = prev_life = None
        for lap in sorted(self._by_lap):
            compound, life = self._by_lap[lap]
            new_stint = (
                not result
                or compound != prev_compound
                or self._tyres_changed(prev_lap, prev_life, lap, life)
            )
            if new_stint:
                if result:
                    result[-1]["end_lap"] = lap
                result.append({"tyre": compound, "start_lap": lap, "end_lap": None})
            prev_lap, prev_compound, prev_life = lap, compound, life
        return result

    @staticmethod
    def _tyres_changed(prev_lap, prev_life, lap, life) -> bool:
        """Same compound: the set was changed if its age did not grow by the laps that passed."""
        if prev_life is None or life is None:
            return False      # cannot tell; do not invent a stop
        return round(life - prev_life) != (lap - prev_lap)
