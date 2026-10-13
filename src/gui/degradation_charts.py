"""
Degradation charts (src/gui/degradation_charts.py) — Phase 2 step 3

  draw_fuel_correction  per compound, the slope before (raw, hollow) and after the fuel correction
                        (filled, with its 95 % interval): a before -> after "dumbbell". It shows what
                        the correction does: raw slopes near zero, corrected ones above it.
  draw_stint_wear       one panel per compound, every stint as a thin line of fuel-corrected lap
                        time against tyre age, both relative to the stint's own average (the stint
                        fixed effects made visible), and the pooled slope as one bold line.

Plain matplotlib like pace_charts.py, so the tests render them off-screen.
"""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.analytics.degradation import USED, RaceDegradation
from src.gui.pace_charts import (
    AXIS, GRID, INK, INK_2, MUTED, SURFACE, HoverPoints, compound_colour, compound_marker, compound_name,
    empty_figure, figure_title, start_figure, style_axes,
)

Z95 = 1.96


def draw_fuel_correction(fig: Figure, result: RaceDegradation) -> List[HoverPoints]:
    start_figure(fig)
    table = result.compounds.dropna(subset=["slope"])
    if table.empty:
        empty_figure(fig, "No compound has two or more usable stints in this race")
        return []
    ax = fig.add_subplot()
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    records = table.to_dict("records")
    ys = np.arange(len(records))[::-1].astype(float)          # soft on top
    xs, yy, texts = [], [], []
    for y, r in zip(ys, records):
        colour, marker = compound_colour(r["compound"]), compound_marker(r["compound"])
        ax.plot([r["raw_slope"], r["slope"]], [y, y], color=MUTED, linewidth=1.0, zorder=2)
        ax.plot([r["slope"] - Z95 * r["se"], r["slope"] + Z95 * r["se"]], [y, y], color=colour, linewidth=2.5,
                alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter([r["raw_slope"]], [y], s=46, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.4, zorder=3)
        ax.scatter([r["slope"]], [y], s=60, marker=marker, color=colour, edgecolors=SURFACE, linewidths=1.0, zorder=4)
        ax.annotate(f"{r['slope']:+.3f} ± {Z95 * r['se']:.3f}", (r["slope"], y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
        xs += [r["raw_slope"], r["slope"]]
        yy += [y, y]
        common = f"{compound_name(r['compound'])}: {r['stints']} stints, {r['laps']} laps"
        texts += [f"{common}\nraw slope {r['raw_slope']:+.3f} s per lap of tyre age\n(fuel burn hides "
                  f"{r['fuel_per_lap']:.3f} s per lap of it)",
                  f"{common}\nfuel-corrected slope {r['slope']:+.3f} s/lap\n95 % interval "
                  f"{r['slope'] - Z95 * r['se']:+.3f} to {r['slope'] + Z95 * r['se']:+.3f} (SE clustered by stint)"]
    ax.set_yticks(ys, [compound_name(r["compound"]) for r in records])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.set_ylim(-0.7, len(records) - 0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("seconds lost per lap of tyre age", color=MUTED, fontsize=8.5)
    figure_title(fig, "What the fuel correction does")
    ax.set_title("Hollow: raw slope (fuel burning off hides the wear). Filled: fuel-corrected, with its 95 % interval.",
                 loc="left", color=MUTED, fontsize=8, pad=14)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6.5, markerfacecolor=SURFACE,
                      markeredgecolor=MUTED, markeredgewidth=1.4, label="raw"),
               Line2D([], [], linestyle="none", marker="o", markersize=6.5, markerfacecolor=INK_2,
                      markeredgecolor=SURFACE, label="fuel-corrected")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8, labelcolor=INK_2)
    return [HoverPoints(ax, np.array(xs, dtype=float), np.array(yy, dtype=float), texts)]


def draw_stint_wear(fig: Figure, result: RaceDegradation) -> List[HoverPoints]:
    start_figure(fig)
    used = result.laps[result.laps["deg_status"] == USED]
    table = result.compounds.dropna(subset=["slope"])
    if table.empty or used.empty:
        empty_figure(fig, "No stint here can show tyre wear (see the ledger)")
        return []
    axes = fig.subplots(1, len(table), squeeze=False, sharey=True)[0]
    points = []
    stint_info = result.stints.set_index(["driver_code", "tyre_run", "compound"])
    for ax, r in zip(axes, table.to_dict("records")):
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        colour = compound_colour(r["compound"])
        laps = used[used["compound"] == r["compound"]]
        xs, ys, texts = [], [], []
        for _, g in laps.groupby("stint", sort=False):
            g = g.sort_values("lap_number")
            x = (g["age"] - g["age"].mean()).to_numpy(dtype=float)
            y = (g["corrected_sec"] - g["corrected_sec"].mean()).to_numpy(dtype=float)
            ax.plot(x, y, color=colour, linewidth=1.0, alpha=0.35, zorder=2)
            key = (g["driver_code"].iloc[0], int(g["tyre_run"].iloc[0]), r["compound"])
            s = stint_info.loc[key] if key in stint_info.index else None
            label = (f"{key[0]} on {compound_name(r['compound']).lower()}, laps {int(g['lap_number'].min())}-"
                     f"{int(g['lap_number'].max())} ({len(g)} laps)")
            if s is not None and not np.isnan(s["slope"]):
                label += f"\nthis stint: {s['slope']:+.3f} ± {Z95 * s['se']:.3f} s/lap (fuel-corrected)"
            for xi, yi, lap in zip(x, y, g["lap_number"]):
                xs.append(xi)
                ys.append(yi)
                texts.append(f"{label}\nlap {int(lap)}: {yi:+.2f} s vs the stint's average")
        span = np.array([min(xs), max(xs)])
        ax.plot(span, r["slope"] * span, color=INK, linewidth=2.0, zorder=3)
        ax.set_title(f"{compound_name(r['compound'])} · {r['stints']} stints · {r['slope']:+.3f} s/lap", loc="left",
                     color=INK, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        points.append(HoverPoints(ax, np.array(xs, dtype=float), np.array(ys, dtype=float), texts))
    axes[0].set_ylabel("fuel-corrected lap time vs the stint's average (s)", color=MUTED, fontsize=8.5)
    lo, hi = np.nanpercentile(np.concatenate([hp.y for hp in points]), [1, 99])
    axes[0].set_ylim(lo - 0.5, hi + 0.5)
    figure_title(fig, "Tyre wear inside each stint · fuel-corrected")
    fig.supxlabel("tyre age, laps from the middle of the stint · thin lines: one stint each · bold line: the pooled "
                  "slope (only the change within each stint counts)", color=MUTED, fontsize=8)
    return points
