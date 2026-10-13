"""
Dirty-air chart (src/gui/dirty_air_charts.py) — Phase 2 step 4

`draw_following_cost`: one column per gap to the car ahead at the lap's start, closest on the left,
the reference (3 s or more behind) on the right. Each lap used is a dot: its time against what the
same car does in clear air on that lap (its stint's level, tyre wear, fuel), so the dots are the
evidence. Each column's cost is a bold bar with its 95 % interval; it is exactly the mean of the
column's dots, and the reference column averages to 0 by definition. The shaded columns are the
laps counted as following for beta (under 1.5 s).

One series, so one colour (the stops chart's dot colour) and no legend: the title names it, and the
tab's table repeats every number. Plain matplotlib like pace_charts.py, so tests render it off-screen.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.analytics.dirty_air import (
    BIN_LABELS, CLEAR_LABEL, FOLLOW_THRESHOLD_SEC, GAP_EDGES, MIN_LAPS, USED, RaceDirtyAir,
)
from src.gui.pace_charts import (
    AXIS, INK, INK_2, MUTED, SURFACE, WASH, HoverPoints, compound_name, empty_figure, figure_title, lap_clock,
    start_figure, style_axes,
)
from src.gui.stops_charts import DOT

Z95 = 1.96
HALF = 0.34                   # half the width of a column's bar, in columns


def lap_tooltip(row: dict) -> str:
    return "\n".join([
        f"{row['driver_code']} · lap {int(row['lap_number'])} · {compound_name(row['compound'])}, "
        f"{int(row['age'])} laps old",
        f"starts {row['gap_start_sec']:.3f} s behind {row['car_ahead']}",
        f"{lap_clock(row['lap_time_sec'])}: {row['lap_cost_sec']:+.2f} s against the car's clear-air pace",
    ])


def bin_tooltip(row: dict) -> str:
    lo, hi = row["cost"] - Z95 * row["se"], row["cost"] + Z95 * row["se"]
    return (f"{row['bin']} behind at the lap's start: {row['cost']:+.3f} s/lap\n"
            f"95 % interval {lo:+.3f} to {hi:+.3f} (SE clustered by stint)\n"
            f"{row['laps']} laps, against {int(row['reference_laps'])} laps {CLEAR_LABEL} behind")


def draw_following_cost(fig: Figure, result: RaceDirtyAir) -> List[HoverPoints]:
    start_figure(fig)
    used = result.laps[(result.laps["air_status"] == USED) & result.laps["lap_cost_sec"].notna()]
    if used.empty:
        empty_figure(fig, "Too few dry laps here to measure the cost of following (see the ledger)")
        return []
    ax = fig.add_subplot()
    style_axes(ax)
    columns = list(BIN_LABELS) + [CLEAR_LABEL]
    following = sum(hi <= FOLLOW_THRESHOLD_SEC for hi in GAP_EDGES[1:])          # the columns under 1.5 s
    ax.axvspan(-0.5, following - 0.5, color=WASH, alpha=0.05, linewidth=0, zorder=0)
    ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)

    xs, ys, texts = [], [], []
    for k, label in enumerate(columns):
        group = used[used["gap_bin"] == label].sort_values(["driver_code", "lap_number"])
        cost = group["lap_cost_sec"].to_numpy(dtype=float)
        jitter = ((np.arange(len(cost)) * 0.618) % 1.0 - 0.5) * 0.56          # spread dots without randomness
        ax.scatter(k + jitter, cost, s=14, color=DOT, alpha=0.55, edgecolors=SURFACE, linewidths=0.5, zorder=2)
        xs.extend(k + jitter)
        ys.extend(cost)
        texts.extend(lap_tooltip(r) for r in group.to_dict("records"))

    bins = result.bins.assign(reference_laps=int(result.bins["laps"].iloc[-1]))
    bx, by, btexts = [], [], []
    for k, row in enumerate(bins.to_dict("records")):
        if row["bin"] == CLEAR_LABEL:
            continue
        if pd.isna(row["cost"]):
            ax.annotate(f"fewer than {MIN_LAPS} laps:\nno estimate", (k, 0.0), xytext=(0, -14),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5, color=MUTED)
            continue
        lo, hi = row["cost"] - Z95 * row["se"], row["cost"] + Z95 * row["se"]
        ax.plot([k, k], [lo, hi], color=INK, linewidth=1.5, zorder=3)
        ax.plot([k - HALF, k + HALF], [row["cost"], row["cost"]], color=INK, linewidth=2.5,
                solid_capstyle="round", zorder=4)
        ax.annotate(f"{row['cost']:+.2f}", (k + HALF, row["cost"]), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK)
        bx.append(float(k))
        by.append(float(row["cost"]))
        btexts.append(bin_tooltip(row))

    counts = bins.set_index("bin")["laps"]
    ticks = [f"{c}\n{int(counts.get(c, 0))} laps" for c in columns]
    ticks[-1] = f"{CLEAR_LABEL} (reference)\n{int(counts.get(CLEAR_LABEL, 0))} laps"
    ax.set_xticks(range(len(columns)), ticks)
    ax.tick_params(axis="x", length=0, labelcolor=INK_2)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    values = np.concatenate([np.array(ys, dtype=float), np.array(by, dtype=float)])
    low, high = np.nanpercentile(values, [1, 99])
    ax.set_ylim(min(low, 0.0) - 0.4, max(high, 0.0) + 0.4)
    ax.set_ylabel("lap time against the car's clear-air pace (s)", color=MUTED, fontsize=8.5)
    ax.set_xlabel("gap to the car ahead on the road when the lap starts", color=MUTED, fontsize=8.5)

    f = result.following
    title = "The cost of following"
    if not np.isnan(f["beta"]):
        title += f" · beta {f['beta']:+.3f} ± {Z95 * f['se']:.3f} s/lap"
    figure_title(fig, title)
    ax.set_title("Dots: each lap against what the same car does in clear air on that lap (its stint's level, tyre "
                 "wear, fuel). Bars: the cost for each gap, with its 95 % interval;\nthe mean of the dots above it. "
                 f"Shaded: the laps counted as following for beta (under {FOLLOW_THRESHOLD_SEC:g} s). Laps that start "
                 "next to a car on another lap (backmarkers, being lapped) are left out.",
                 loc="left", color=MUTED, fontsize=8, pad=8)
    points = [HoverPoints(ax, np.array(xs, dtype=float), np.array(ys, dtype=float), texts)]
    if bx:
        points.append(HoverPoints(ax, np.array(bx), np.array(by), btexts))
    return points
