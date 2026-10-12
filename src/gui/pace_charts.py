"""
Pace charts (src/gui/pace_charts.py) — Phase 2 step 1

The Pace tab's charts, drawn with matplotlib into a Figure the caller owns. No Qt in here, so the
tests render them off-screen; the tab (src/gui/pace_tab.py) puts them on canvases.

  draw_lap_times      one driver's race lap by lap: pace laps by compound, outliers, the laps that
                      are not pace laps, the reference the outlier rule compared against, and the
                      neutralised periods
  draw_compound_pace  raw pace of each compound, over every driver's pace laps
  draw_driver_pace    each driver's median pace lap on each compound, fastest first

Each draw function returns the points it drew (HoverPoints) so the tab can show a tooltip for the
point under the pointer. The tooltip adds detail; every number in it is also in the Laps tab's
"Pace classification" table.

Design (the dataviz method): emphasis, not a rainbow. Pace laps carry their compound's colour;
every lap that is not a pace lap is one recessive gray. The compound colours are Pirelli's
(soft red, medium yellow, hard white, inter green, wet blue) because that is how every F1 viewer
reads them. Checked with the palette validator on this dark surface: every pair separates for
normal and colour-blind vision (worst: soft against inter, deuteranopia ΔE 8.0, at the target),
but medium and hard sit above the recommended lightness band and hard has no hue at all. So each
compound also has its own marker shape and the legend or axis names it: colour is never the only
cue. The gray of non-pace laps is kept far darker than the white of the hard tyre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from src.analytics.laps import (
    MIN_REFERENCE_LAPS, OUTLIER_THRESHOLD_SEC, PACE, REASON_OUTLIER, RacePace, compound_rank,
)
from src.analytics.neutralisations import RED_FLAG
from src.gui.analysis_theme import COMPOUND_COLOURS, DIM, NO_COMPOUND_COLOUR, PANEL, TEXT

# Chart chrome on the window's dark panel
SURFACE = PANEL
INK = TEXT               # titles, values
INK_2 = "#B4B4B4"        # labels, legend
MUTED = DIM              # axes, notes
GRID = "#2C2C2A"         # hairline grid
AXIS = "#3A3A38"
CONTEXT = "#5C5C5A"      # laps that are not pace laps: context, de-emphasised
WASH = "#FFFFFF"         # neutralised periods, at low opacity
TIP_FACE, TIP_EDGE = "#262626", "#3A3A3A"

COMPOUND_MARKERS = {"SOFT": "o", "MEDIUM": "s", "HARD": "D", "INTERMEDIATE": "^", "WET": "v"}
NO_COMPOUND_MARKER = "P"

MIN_DRIVER_LAPS = 3      # driver pace chart: a median of fewer pace laps than this is not shown
HOVER_RADIUS_PX = 12     # a 24 px hit target around every point


@dataclass
class HoverPoints:
    """Points one axes drew, in data coordinates, with the tooltip text of each."""
    ax: Any
    x: np.ndarray
    y: np.ndarray
    text: List[str]


# --- small helpers ------------------------------------------------------------------------

def _missing(compound) -> bool:
    return compound is None or (not isinstance(compound, str) and pd.isna(compound))


def compound_colour(compound) -> str:
    return NO_COMPOUND_COLOUR if _missing(compound) else COMPOUND_COLOURS.get(str(compound), NO_COMPOUND_COLOUR)


def compound_marker(compound) -> str:
    return NO_COMPOUND_MARKER if _missing(compound) else COMPOUND_MARKERS.get(str(compound), NO_COMPOUND_MARKER)


def compound_name(compound) -> str:
    return "No compound" if _missing(compound) else str(compound).capitalize()


def lap_clock(seconds: float) -> str:
    """84.468 -> '1:24.468', the way timing screens print a lap (rounded to the millisecond first: 59.9999 -> 1:00.000)."""
    minutes, secs = divmod(round(float(seconds), 3), 60.0)
    return f"{int(minutes)}:{secs:06.3f}"


def nearest(points_px: np.ndarray, x: float, y: float, radius: float) -> Optional[int]:
    """Index of the point closest to (x, y) in pixels, if it is within `radius`; else None."""
    if len(points_px) == 0:
        return None
    d2 = ((np.asarray(points_px, dtype=float) - np.array([x, y], dtype=float)) ** 2).sum(axis=1)
    i = int(np.argmin(d2))
    return i if d2[i] <= radius ** 2 else None


def _start(fig: Figure):
    fig.clear()
    fig.set_layout_engine("constrained")
    fig.set_facecolor(SURFACE)


def _style(ax):
    """Recessive chrome: hairline solid grid, two muted spines, small muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig: Figure, text: str):
    fig.suptitle(text, x=0.008, ha="left", color=INK, fontsize=11, fontweight="bold")


def _empty(fig: Figure, text: str):
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=10, transform=ax.transAxes)


# --- one driver's laps --------------------------------------------------------------------

def lap_tooltip(row: Dict[str, Any], threshold_sec: float = OUTLIER_THRESHOLD_SEC) -> str:
    """What one lap was and why it is, or is not, a pace lap."""
    lines = [f"Lap {int(row['lap_number'])} · {row['driver_code']}"]
    t = row["lap_time_sec"]
    lines.append("no lap time" if pd.isna(t) else f"{t:.3f} s  ({lap_clock(t)})")
    tyre = compound_name(row.get("compound"))
    if not pd.isna(row.get("tyre_life")):
        tyre += f" · tyre age {int(row['tyre_life'])}"
    lines.append(tyre)
    status = row["pace_status"]
    if status == PACE and row["outlier_checked"]:
        lines.append(f"pace lap · {row['over_reference_sec']:+.2f} s vs reference {row['reference_sec']:.2f} s")
    elif status == PACE:
        lines.append(f"pace lap · not checked: fewer than {MIN_REFERENCE_LAPS} clean laps on this set of tyres")
    elif status == REASON_OUTLIER:
        lines.append(f"outlier · {row['over_reference_sec']:+.2f} s vs reference {row['reference_sec']:.2f} s "
                     f"(limit +{threshold_sec:g} s)")
    else:
        lines.append(f"not a pace lap: {status}")
    if row.get("is_deleted"):
        lines.append("lap time deleted (track limits): kept, it keeps its flag")
    return "\n".join(lines)


def _neutralised_bands(ax, pace: RacePace) -> int:
    """Shade every neutralised period; its name above the plot. Returns how many label rows were used."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    level, levels, prev_end = 0, 0, None
    for p in pace.periods.itertuples(index=False):
        ax.axvspan(p.first_lap - 0.5, p.last_lap + 0.5, color=WASH, alpha=0.10 if p.kind == RED_FLAG else 0.055,
                   linewidth=0, zorder=0)
        touching = prev_end is not None and p.first_lap <= prev_end + 1
        level = (level + 1) % 2 if touching else 0
        levels = max(levels, level + 1)
        ax.annotate(p.kind, xy=((p.first_lap + p.last_lap) / 2, 1.0), xycoords=trans, xytext=(0, 3 + 10 * level),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=MUTED,
                    annotation_clip=False)
        prev_end = p.last_lap
    return levels


def draw_lap_times(fig: Figure, pace: RacePace, driver: str,
                   threshold_sec: float = OUTLIER_THRESHOLD_SEC) -> List[HoverPoints]:
    """One driver's race, lap by lap (see the module note). Laps slower than the scale sit on its top edge (▲)."""
    _start(fig)
    laps = pace.laps[pace.laps["driver_code"] == driver].sort_values("lap_number")
    timed = laps[laps["lap_time_sec"].notna()].reset_index(drop=True)
    if timed.empty:
        _empty(fig, f"No timed laps for {driver}")
        return []
    ax = fig.add_subplot()
    _style(ax)

    is_pace = timed["is_pace_lap"].to_numpy(dtype=bool)
    is_out = timed["is_outlier"].to_numpy(dtype=bool)
    other = ~is_pace & ~is_out
    x = timed["lap_number"].to_numpy(dtype=float)
    t = timed["lap_time_sec"].to_numpy(dtype=float)
    core = t[is_pace] if is_pace.any() else t
    lo, hi = float(core.min()) - 1.0, float(core.max()) + threshold_sec + 1.5
    off = t > hi
    y = np.where(off, hi - 0.025 * (hi - lo), t)

    levels = _neutralised_bands(ax, pace)

    # the reference each checked lap was compared with, and the limit above it; the lines break where
    # laps in between were not checked (a safety car, a pit stop), rather than bridge the gap
    checked = timed[timed["outlier_checked"]]
    run_key = [checked["tyre_run"], checked["compound"].astype("string").fillna("-")]
    for _, run in checked.groupby(run_key, sort=False):
        ref = run.set_index("lap_number")["reference_sec"]
        ref = ref.reindex(range(int(ref.index.min()), int(ref.index.max()) + 1))
        ax.plot(ref.index, ref.to_numpy(), color=INK_2, linewidth=1.0, alpha=0.55, zorder=2)
        ax.plot(ref.index, ref.to_numpy() + threshold_sec, color=MUTED, linewidth=1.0, alpha=0.6,
                linestyle=(0, (3, 3)), zorder=2)

    # laps that are not pace laps: one gray, the reason is in the tooltip
    ax.scatter(x[other & ~off], y[other & ~off], s=14, color=CONTEXT, linewidths=0, zorder=1.5)
    ax.scatter(x[other & off], y[other & off], s=22, marker="^", color=CONTEXT, linewidths=0, zorder=1.5)

    compounds = sorted(timed.loc[is_pace | is_out, "compound"].drop_duplicates().tolist(), key=compound_rank)
    for compound in compounds:
        same = (timed["compound"].isna() if _missing(compound) else timed["compound"].eq(compound)).fillna(False)
        same = same.to_numpy(dtype=bool)
        colour, marker = compound_colour(compound), compound_marker(compound)
        m = same & is_pace
        ax.scatter(x[m], y[m], s=40, marker=marker, color=colour, edgecolors=SURFACE, linewidths=1.0, zorder=3)
        m = same & is_out
        ax.scatter(x[m & ~off], y[m & ~off], s=54, marker=marker, facecolors=SURFACE, edgecolors=colour,
                   linewidths=1.5, zorder=4)
        ax.scatter(x[m & off], y[m & off], s=54, marker="^", facecolors=SURFACE, edgecolors=colour,
                   linewidths=1.5, zorder=4)
    for i in np.flatnonzero(is_out):           # direct labels only on the outliers
        ax.annotate(f"+{timed.at[i, 'over_reference_sec']:.1f} s", (x[i], y[i]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2, zorder=5)

    race_laps = int(pace.laps["lap_number"].max())
    ax.set_xlim(0.5, race_laps + 0.5)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("lap", color=MUTED, fontsize=8.5)
    ax.set_ylabel("lap time (s)", color=MUTED, fontsize=8.5)

    n_pace, n_out, n_other = int(is_pace.sum()), int(is_out.sum()), int(other.sum())
    no_time = int(laps["lap_time_sec"].isna().sum())
    parts = [f"{n_pace} pace laps", f"{n_out} outlier" + ("" if n_out == 1 else "s"), f"{n_other} not pace laps"]
    if no_time:
        parts.append(f"{no_time} without a time (not drawn)")
    if off.any():
        parts.append(f"{int(off.sum())} off the scale (▲ on the top edge)")
    _title(fig, f"Lap times · {driver}")
    ax.set_title(" · ".join(parts), loc="left", color=MUTED, fontsize=8.5, pad=8 + 10 * levels)

    handles = [Line2D([], [], linestyle="none", marker=compound_marker(c), markersize=6.5,
                      markerfacecolor=compound_colour(c), markeredgecolor=SURFACE, label=compound_name(c))
               for c in compounds]
    if n_out:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                              markeredgecolor=INK_2, markeredgewidth=1.5, label=f"outlier (> {threshold_sec:g} s over)"))
    handles += [
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=CONTEXT, markeredgecolor="none",
               label="not a pace lap"),
        Line2D([], [], color=INK_2, linewidth=1.0, alpha=0.55, label="reference (median of 5 clean laps)"),
        Line2D([], [], color=MUTED, linewidth=1.0, alpha=0.6, linestyle=(0, (3, 3)), label=f"reference + {threshold_sec:g} s"),
    ]
    if len(pace.periods):
        handles.append(Patch(facecolor=to_rgba(WASH, 0.10), edgecolor="none", label="neutralised"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False, fontsize=8,
               labelcolor=INK_2, handletextpad=0.5, columnspacing=1.4)

    texts = [lap_tooltip(r, threshold_sec) for r in timed.to_dict("records")]
    return [HoverPoints(ax, x, y, texts)]


# --- raw pace by compound -----------------------------------------------------------------

def draw_compound_pace(fig: Figure, pace: RacePace) -> List[HoverPoints]:
    """Box per compound over every driver's pace laps. Raw: not fuel-corrected (see laps.py)."""
    _start(fig)
    table = pace.by_compound
    if table.empty:
        _empty(fig, "No pace laps")
        return []
    ax = fig.add_subplot()
    _style(ax)
    records = table.to_dict("records")
    stats = [{"med": r["median_sec"], "q1": r["q25_sec"], "q3": r["q75_sec"], "whislo": r["p05_sec"],
              "whishi": r["p95_sec"], "fliers": [], "label": compound_name(r["compound"])} for r in records]
    pos = np.arange(len(stats), dtype=float)
    parts = ax.bxp(stats, positions=pos, widths=0.42, patch_artist=True, showfliers=False, showcaps=False,
                   medianprops={"color": INK, "linewidth": 2.0, "solid_capstyle": "butt"},
                   whiskerprops={"color": MUTED, "linewidth": 1.0}, boxprops={"linewidth": 1.4})
    for box, r in zip(parts["boxes"], records):
        colour = compound_colour(r["compound"])
        box.set_facecolor(to_rgba(colour, 0.30))
        box.set_edgecolor(colour)
    for p, r in zip(pos, records):            # the median, written beside each box
        ax.annotate(f"{r['median_sec']:.1f}", (p + 0.21, r["median_sec"]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    labels = [f"{compound_name(r['compound'])}\n{r['laps']} laps · {r['drivers']} drivers\naround lap {r['median_lap']:.0f}"
              for r in records]
    ax.set_xticks(pos, labels)
    ax.tick_params(axis="x", length=0, labelcolor=INK_2)
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylabel("lap time (s)", color=MUTED, fontsize=8.5)
    _title(fig, "Raw pace by compound")
    ax.set_title("Every driver's pace laps. Raw, not fuel-corrected: a compound run late\n"
                 "in the race (lighter car, faster track) looks faster than it is.\n"
                 "Box = middle half of laps, line = median, whiskers = 5th to 95th percentile.",
                 loc="left", color=MUTED, fontsize=8, pad=8)

    texts = [f"{compound_name(r['compound'])}: median {r['median_sec']:.3f} s ({lap_clock(r['median_sec'])})\n"
             f"{r['laps']} pace laps by {r['drivers']} drivers, run around lap {r['median_lap']:.0f}\n"
             f"middle half {r['q25_sec']:.2f} to {r['q75_sec']:.2f} s · fastest {r['fastest_sec']:.3f} s\n"
             "raw: not fuel-corrected" for r in records]
    return [HoverPoints(ax, pos, table["median_sec"].to_numpy(dtype=float), texts)]


# --- each driver on each compound -----------------------------------------------------------

def driver_chart_rows(pace: RacePace, min_laps: int = MIN_DRIVER_LAPS) -> int:
    """How many driver rows the tallest compound panel has (to size the canvas)."""
    shown = pace.by_driver[pace.by_driver["laps"] >= min_laps]
    if shown.empty:
        return 0
    return int(shown.groupby(shown["compound"].astype("string").fillna("-")).size().max())


def draw_driver_pace(fig: Figure, pace: RacePace, highlight: Optional[str] = None,
                     min_laps: int = MIN_DRIVER_LAPS) -> List[HoverPoints]:
    """One panel per compound: each driver's median pace lap and the middle half of their laps, fastest on top."""
    _start(fig)
    table = pace.by_driver[pace.by_driver["laps"] >= min_laps]
    if table.empty:
        _empty(fig, f"No driver has {min_laps} or more pace laps on one compound")
        return []
    compounds = sorted(table["compound"].drop_duplicates().tolist(), key=compound_rank)
    axes = fig.subplots(1, len(compounds), squeeze=False)[0]
    tallest = driver_chart_rows(pace, min_laps)
    points = []
    for ax, compound in zip(axes, compounds):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        same = table["compound"].isna() if _missing(compound) else table["compound"].eq(compound).fillna(False)
        rows = table[same.to_numpy(dtype=bool)].sort_values("median_sec").reset_index(drop=True)
        n = len(rows)
        y = (tallest - 1 - np.arange(n)).astype(float)    # fastest at the top, same row height in every panel
        colour, marker = compound_colour(compound), compound_marker(compound)
        codes = rows["driver_code"].tolist()
        if highlight in codes:
            row_y = y[codes.index(highlight)]
            ax.axhspan(row_y - 0.5, row_y + 0.5, color=WASH, alpha=0.08, linewidth=0, zorder=0)
        ax.hlines(y, rows["q25_sec"], rows["q75_sec"], color=colour, linewidth=2.2, alpha=0.85, zorder=2)
        ax.scatter(rows["median_sec"], y, s=34, marker=marker, color=colour, edgecolors=SURFACE, linewidths=1.0,
                   zorder=3)
        ax.set_yticks(y, codes)
        ax.tick_params(axis="y", length=0, labelcolor=INK_2)
        for label, code in zip(ax.get_yticklabels(), codes):
            if code == highlight:
                label.set_color(INK)
                label.set_fontweight("bold")
        ax.set_ylim(-0.7, tallest - 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{compound_name(compound)} · {n} driver" + ("" if n == 1 else "s"), loc="left", color=INK,
                     fontsize=9)
        texts = [f"{r['driver_code']} on {compound_name(compound).lower()}: median {r['median_sec']:.3f} s "
                 f"({lap_clock(r['median_sec'])})\n{r['laps']} pace laps, run around lap {r['median_lap']:.0f}\n"
                 f"middle half {r['q25_sec']:.2f} to {r['q75_sec']:.2f} s · fastest {r['fastest_sec']:.3f} s"
                 for r in rows.to_dict("records")]
        points.append(HoverPoints(ax, rows["median_sec"].to_numpy(dtype=float), y, texts))
    _title(fig, "Driver pace by compound · fastest first")
    fig.supxlabel(f"median pace lap (s) · dot = median, bar = middle half of that driver's pace laps · raw, not "
                  f"fuel-corrected · drivers with {min_laps} or more pace laps on the compound",
                  color=MUTED, fontsize=8)
    return points
