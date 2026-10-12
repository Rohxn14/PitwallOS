"""
Stops chart (src/gui/stops_charts.py) — Phase 2 step 2

`draw_stop_losses`: every measured stop of the race as a dot on one row per condition (green,
yellow, safety car, VSC, lap 1), its loss on the x axis, the row's median as a line with the middle
half shaded. One series, so one colour: the rows say which condition, and the tooltip which stop.
Plain matplotlib like pace_charts.py, so the tests render it off-screen.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.analytics.stops import CONDITIONS, RED_FLAG, RaceStops
from src.gui.pace_charts import (
    INK, INK_2, MUTED, SURFACE, WASH, HoverPoints, compound_name, empty_figure, figure_title, start_figure,
    style_axes,
)

DOT = "#3987E5"        # the dataviz reference palette's first categorical slot, dark-mode step


def stop_tooltip(row: dict) -> str:
    lines = [f"{row['driver_code']} · stop {int(row['stop_number'])} · in-lap {int(row['in_lap'])}",
             f"{compound_name(row['compound_before'])} → {compound_name(row['compound_after'])}",
             f"in-lap {row['in_lap_sec']:.3f} s + out-lap {row['out_lap_sec']:.3f} s",
             f"- reference {row['reference_sec']:.3f} s ({row['reference_kind']}, n = {int(row['reference_n'])})",
             f"= loss {row['loss_sec']:.2f} s"]
    return "\n".join(lines)


def draw_stop_losses(fig: Figure, result: RaceStops) -> List[HoverPoints]:
    start_figure(fig)
    table = result.stops
    measured = table.dropna(subset=["loss_sec"])
    if measured.empty:
        empty_figure(fig, "No stop in this race has a measurable loss")
        return []
    ax = fig.add_subplot()
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#2C2C2A", linewidth=0.8)

    shown = [c for c in CONDITIONS if c != RED_FLAG and (measured["condition"] == c).any()]
    xs, ys, texts, labels = [], [], [], []
    for row_i, condition in enumerate(shown):
        y0 = len(shown) - 1 - row_i                     # first condition on top
        group = measured[measured["condition"] == condition].reset_index(drop=True)
        loss = group["loss_sec"].to_numpy(dtype=float)
        q25, med, q75 = np.quantile(loss, [0.25, 0.5, 0.75])
        ax.fill_betweenx([y0 - 0.32, y0 + 0.32], q25, q75, color=WASH, alpha=0.07, linewidth=0, zorder=0)
        ax.plot([med, med], [y0 - 0.36, y0 + 0.36], color=INK, linewidth=2.0, zorder=3)
        ax.annotate(f"{med:.1f} s", (med, y0 + 0.36), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=INK)
        jitter = ((np.arange(len(loss)) * 0.618) % 1.0 - 0.5) * 0.44   # spread dots without randomness
        ax.scatter(loss, y0 + jitter, s=26, color=DOT, edgecolors=SURFACE, linewidths=0.8, zorder=2)
        xs.extend(loss)
        ys.extend(y0 + jitter)
        texts.extend(stop_tooltip(r) for r in group.to_dict("records"))
        n_all = int((table["condition"] == condition).sum())
        labels.append(f"{condition.capitalize()}\n{len(loss)} of {n_all} stops")
    ax.set_yticks(list(range(len(shown) - 1, -1, -1)), labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.set_ylim(-0.7, len(shown) - 0.3 + 0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_xlabel("time lost to the stop (s)", color=MUTED, fontsize=8.5)

    green = result.green_loss
    figure_title(fig, "Pit loss per stop" + (f" · circuit loss {green:.1f} s" if green is not None else ""))
    red = int((table["condition"] == RED_FLAG).sum())
    unknown = int(table["loss_sec"].isna().sum())
    ax.set_title("loss = in-lap + out-lap - reference. Green and yellow stops: the driver's own pace around the stop. "
                 "Safety car, VSC, lap 1: the cars that stayed out on those laps.\nLine = median, band = middle half. "
                 f"{unknown} stop" + ("" if unknown == 1 else "s") + " without a measurable loss are in the table"
                 + (f" (including {red} red-flag tyre change" + ("" if red == 1 else "s") + ")" if red else "") + ".",
                 loc="left", color=MUTED, fontsize=8, pad=8)
    return [HoverPoints(ax, np.array(xs, dtype=float), np.array(ys, dtype=float), texts)]
