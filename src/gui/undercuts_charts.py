"""
Undercut chart (src/gui/undercuts_charts.py) — Phase 2 step 5

`draw_attempts`: one row per undercut attempt of the race. Hollow dot: the gap of the driver who
stopped first to the car ahead, at the end of the lap before its in-lap; filled dot: the gap at the
end of the car ahead's out-lap, once both are on new tyres; an arrow between them. Left of the zero
line, the driver is ahead: the undercut worked. One series, so one colour; the outcome is said in
words at the end of each row, never by colour alone. Plain matplotlib like pace_charts.py, so the
tests render it off-screen.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.analytics.undercuts import RaceUndercuts
from src.gui.pace_charts import (
    AXIS, GRID, INK, INK_2, MUTED, SURFACE, HoverPoints, compound_name, empty_figure, figure_title, start_figure,
    style_axes,
)
from src.gui.stops_charts import DOT


def is_yes(value) -> bool:
    """True only for a known True (team-mates can be unknown)."""
    return not pd.isna(value) and bool(value)


def gap_words(sec) -> str:
    """A gap at the line (this car's crossing minus the other's) in words: '2.051 behind', '3.377 ahead'."""
    if sec is None or pd.isna(sec):
        return "—"
    if sec == 0:
        return "level"
    return f"{abs(sec):.3f} {'behind' if sec > 0 else 'ahead'}"


def stop_words(pit_lane, stationary) -> str:
    """'21.283 s in the pit lane, 2.4 s stationary' (stationary left out where F1 published none)."""
    lane = "pit-lane time unknown" if pd.isna(pit_lane) else f"{pit_lane:.3f} s in the pit lane"
    return lane if pd.isna(stationary) else f"{lane}, {stationary:.1f} s stationary"


def attempt_tooltip(row: dict) -> str:
    b, a = row["driver_code"], row["car_ahead"]
    lines = [f"{b} on {a}" + (" (team-mates)" if is_yes(row["teammates"]) else ""),
             f"{b} stopped on lap {int(row['in_lap'])} ({compound_name(row['compound_before'])} → "
             f"{compound_name(row['compound_after'])}), {a} on lap {int(row['car_ahead_in_lap'])}",
             f"{b} was {gap_words(row['gap_before_sec'])}, then {gap_words(row['gap_after_sec'])}: "
             f"gained {row['gain_sec']:+.3f} s",
             f"  stop (pit lane) {row['stop_gain_sec']:+.3f} · track around the stops {row['track_gain_sec']:+.3f} · "
             f"laps between {row['between_gain_sec']:+.3f}",
             f"  {b}: {stop_words(row.get('pit_lane_sec'), row.get('stationary_sec'))}",
             f"  {a}: {stop_words(row.get('car_ahead_pit_lane_sec'), row.get('car_ahead_stationary_sec'))}"]
    return "\n".join(lines)


def draw_attempts(fig: Figure, result: RaceUndercuts) -> List[HoverPoints]:
    start_figure(fig)
    table = result.attempts.sort_values(["in_lap", "driver_code"]).reset_index(drop=True)
    if table.empty:
        empty_figure(fig, "No undercut attempt in this race (the ledger says what happened at every stop)")
        return []
    ax = fig.add_subplot()
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0.0, color=AXIS, linewidth=1.2, zorder=1)
    rows = table.to_dict("records")
    ys = np.arange(len(rows))[::-1].astype(float)                # first attempt on top
    xs, yy, texts = [], [], []
    for y, r in zip(ys, rows):
        before, after = float(r["gap_before_sec"]), float(r["gap_after_sec"])
        ax.annotate("", xy=(after, y), xytext=(before, y), zorder=2,
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.2, "shrinkA": 5, "shrinkB": 6})
        ax.scatter([before], [y], s=46, facecolors=SURFACE, edgecolors=DOT, linewidths=1.6, zorder=3)
        ax.scatter([after], [y], s=56, color=DOT, edgecolors=SURFACE, linewidths=1.0, zorder=4)
        verdict = "worked" if r["worked"] else "did not work"
        ax.annotate(f"{verdict} · gained {r['gain_sec']:+.2f} s", (max(before, after), y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
        xs += [before, after]
        yy += [y, y]
        tip = attempt_tooltip(r)
        texts += [f"before {r['driver_code']}'s stop: {gap_words(before)}\n" + tip,
                  f"after {r['car_ahead']}'s stop: {gap_words(after)}\n" + tip]
    ax.set_yticks(ys, [f"{r['driver_code']} on {r['car_ahead']}\nlaps {int(r['in_lap'])} / {int(r['car_ahead_in_lap'])}"
                       for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    span = max(3.5, float(np.nanmax(np.abs(xs))) + 0.5)
    ax.set_xlim(-span, span * 1.45)                               # room on the right for the verdicts
    ax.set_xlabel("gap to the car ahead at the line (s) · positive: behind it · negative: ahead of it",
                  color=MUTED, fontsize=8.5)
    s = result.summary
    figure_title(fig, f"Undercut attempts · {s['worked']} of {s['attempts']} worked")
    ax.set_title("Hollow: the gap at the end of the lap before the driver's in-lap. Filled: at the end of the car "
                 "ahead's out-lap, once both are on new tyres.", loc="left", color=MUTED, fontsize=8, pad=8)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                      markeredgecolor=DOT, markeredgewidth=1.6, label="before the driver's stop"),
               Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=DOT,
                      markeredgecolor=SURFACE, label="after both stops")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8, labelcolor=INK_2)
    return [HoverPoints(ax, np.array(xs, dtype=float), np.array(yy, dtype=float), texts)]
