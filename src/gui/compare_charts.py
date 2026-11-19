"""
Compare chart (src/gui/compare_charts.py) - Phase 5 step 5.3

`draw_comparison`: two drivers on one race, in one figure of two rows that share the x axis, the corners in the order
they are driven, so each corner's bar sits under the point where it moved the gap:

  top     the gap building up over a lap: after each corner, the running gap of each paired lap (A minus B), its
          median as a line and the middle half of the laps as a band. The area between the line and zero is tinted in
          the colour of whoever is ahead.
  bottom  each corner's median gap as a bar, in the colour of the driver who wins the corner.

Below zero, A is ahead (top) or faster in that corner (bottom). The build-up line is the median of running totals, not
the sum of the bars: medians do not add up (src/telemetry/compare.py says why both are shown).

Colours: A and B are the reference palette's first two slots (analysis_theme.DRIVER_A, DRIVER_B), validated on the
window's panel. A legend names both drivers, the tooltips repeat every number, and the tab's table holds them all: colour
is never the only cue. Plain matplotlib like the other charts, so the tests render it off-screen.
"""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.gui.analysis_theme import DRIVER_A, DRIVER_B
from src.gui.pace_charts import (
    AXIS, INK, INK_2, MUTED, SURFACE, WASH, HoverPoints, empty_figure, figure_title, start_figure, style_axes,
)
from src.telemetry.compare import Comparison

TINT = 0.16                   # the area between the build-up and zero, in the colour of whoever is ahead
BAND = 0.07                   # the middle half of the paired laps


def corner_tick(label: str) -> str:
    return f"T{label}"


def _ahead(gap: float, a: str, b: str) -> str:
    if abs(gap) < 0.0005:
        return "level"
    return f"{a if gap < 0 else b} ahead by {abs(gap):.3f} s"


def build_up_tooltip(row: dict, corner: str, c: Comparison) -> str:
    return "\n".join([
        f"After {corner_tick(corner)}: {_ahead(row['median_sec'], c.a, c.b)} (median of {len(c.laps)} laps)",
        f"middle half of the laps: {row['q25_sec']:+.3f} to {row['q75_sec']:+.3f} s ({c.a} minus {c.b})",
    ])


def corner_tooltip(row: dict, c: Comparison) -> str:
    gap = row["gap_median_sec"]
    head = f"{corner_tick(row['corner'])} · {row['corner_speed_kmh']:.0f} km/h at the corner"
    if row["winner"]:
        verdict = f"{row['winner']} faster by {abs(gap):.3f} s (median); mean {row['gap_mean_sec']:+.3f} s"
    else:
        verdict = f"level (median); mean {row['gap_mean_sec']:+.3f} s"
    return "\n".join([head, verdict, f"{c.a} faster on {100 * row['a_faster_share']:.0f} % of {len(c.laps)} paired laps",
                      f"how: {row['how']}"])


def draw_comparison(fig: Figure, c: Comparison) -> List[HoverPoints]:
    start_figure(fig)
    corners = c.corners.sort_values("corner_order").reset_index(drop=True)
    n = len(corners)
    if n == 0:
        empty_figure(fig, "No corner to compare")
        return []
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (top, bottom):
        style_axes(ax)
        ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)

    # --- top: the gap building up over the lap ---
    build = c.build_up.sort_values("corner_order").reset_index(drop=True)
    x = np.concatenate([[0.5], np.arange(1, n + 1) + 0.5])            # the line, then after each corner
    y = np.concatenate([[0.0], build["median_sec"].to_numpy(dtype=float)])
    lo = np.concatenate([[0.0], build["q25_sec"].to_numpy(dtype=float)])
    hi = np.concatenate([[0.0], build["q75_sec"].to_numpy(dtype=float)])
    top.fill_between(x, lo, hi, color=WASH, alpha=BAND, linewidth=0, zorder=1)
    top.fill_between(x, y, 0.0, where=y <= 0, interpolate=True, color=DRIVER_A, alpha=TINT, linewidth=0, zorder=2)
    top.fill_between(x, y, 0.0, where=y >= 0, interpolate=True, color=DRIVER_B, alpha=TINT, linewidth=0, zorder=2)
    top.plot(x, y, color=INK, linewidth=2.0, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    top.scatter(x[1:], y[1:], s=36, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=4)
    top.set_ylabel(f"{c.a} minus {c.b}, running (s)", color=MUTED, fontsize=8.5)
    top.text(0.004, 0.03, f"{c.a} ahead", transform=top.transAxes, color=MUTED, fontsize=8, va="bottom")
    top.text(0.004, 0.97, f"{c.b} ahead", transform=top.transAxes, color=MUTED, fontsize=8, va="top")
    reach = max(np.nanmax(np.abs(np.concatenate([lo, hi, y]))), 0.02) * 1.15
    top.set_ylim(-reach, reach)
    final = y[-1]
    top.annotate(f"{final:+.3f} s", (x[-1], final), xytext=(6, 0), textcoords="offset points", ha="left",
                 va="center", fontsize=8, color=INK_2)
    top.legend(handles=[Line2D([], [], color=INK, linewidth=2, label="median of the paired laps"),
                        Patch(color=WASH, alpha=0.25, label="middle half of the laps"),
                        Patch(color=DRIVER_A, alpha=0.55, label=f"{c.a} ahead"),
                        Patch(color=DRIVER_B, alpha=0.55, label=f"{c.b} ahead")],
               loc="best", fontsize=7.5, frameon=False, labelcolor=INK_2, ncol=2)
    top.set_title(f"The gap after each corner on the {len(c.laps)} laps both drove clean, lap N against lap N. "
                  f"Below zero: {c.a} ahead. The last point is the lap.", loc="left", color=MUTED, fontsize=8, pad=8)
    top_points = HoverPoints(top, x[1:], y[1:], [build_up_tooltip(r, corners.loc[k, "corner"], c)
                                                 for k, r in enumerate(build.to_dict("records"))])

    # --- bottom: each corner ---
    gaps = corners["gap_median_sec"].to_numpy(dtype=float)
    bars_x = np.arange(1, n + 1)
    colours = [DRIVER_A if g < 0 else DRIVER_B if g > 0 else MUTED for g in gaps]
    bottom.bar(bars_x, gaps, width=0.55, color=colours, linewidth=0, zorder=2)
    for side in (gaps < 0, gaps > 0):                                    # label the biggest gain of each driver
        if side.any():
            k = int(np.argmax(np.where(side, np.abs(gaps), -1)))
            bottom.annotate(f"{gaps[k]:+.3f}", (bars_x[k], gaps[k]), xytext=(0, -10 if gaps[k] < 0 else 4),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    reach = max(np.nanmax(np.abs(gaps)), 0.01) * 1.35
    bottom.set_ylim(-reach, reach)
    bottom.set_ylabel(f"{c.a} minus {c.b} (s)", color=MUTED, fontsize=8.5)
    bottom.legend(handles=[Patch(color=DRIVER_A, label=f"{c.a} faster"), Patch(color=DRIVER_B, label=f"{c.b} faster")],
                  loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_2, ncol=2)
    bottom.set_xticks(bars_x, [corner_tick(t) for t in corners["corner"]])
    bottom.tick_params(axis="x", length=0, labelcolor=INK_2)
    bottom.set_xlim(0.3, n + 0.9)
    bottom.set_xlabel("corners, in the order they are driven", color=MUTED, fontsize=8.5)
    bottom_points = HoverPoints(bottom, bars_x.astype(float), gaps, [corner_tooltip(r, c) for r in corners.to_dict("records")])

    won = int((gaps < 0).sum())
    figure_title(fig, f"{c.a} against {c.b}: where the gap is made  ·  {c.a} faster in {won} of {n} corners")
    return [top_points, bottom_points]
