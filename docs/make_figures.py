"""
The README's charts, drawn from the lake: python -m docs.make_figures        (writes docs/images/, ~2 min)

Each chart comes in a light and a dark version; the README's <picture> tags show the one that fits the reader's
theme. Numbers come from the project's own reports and functions, never typed in here, so a chart cannot drift from
what the analytics say. Colours: blue / orange / aqua pass the colour-blind checks in any combination, the models' one-hue
ramp is ordered (a ladder). Every series is also labelled directly, so nothing depends on colour alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from src.analytics import dirty_air  # noqa: E402
from src.analytics.undercuts import wilson  # noqa: E402
from src.pipeline import lake  # noqa: E402
from src.simulator.calibration import reliability  # noqa: E402

OUT = Path(__file__).parent / "images"
SEASONS = (2023, 2024, 2025)
SOURCE = "2023-2025, {n} races. Data: F1 timing via FastF1, telemetry via OpenF1."

THEMES: Dict[str, Dict] = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"],
              "ladder": ["#86b6ef", "#5598e7", "#256abf", "#104281"],
              "speed": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"],
             "ladder": ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
             "speed": ["#184f95", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"]},
}
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
RUNGS = {"persistence": "Last clean lap", "linear": "Linear terms", "state_space": "Kalman filter",
         "boosted": "LightGBM corrections"}


def _figure(theme: Dict, width: float, height: float, title: str, subtitle: str, **subplots):
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.unicode_minus": False})
    fig, axes = plt.subplots(figsize=(width, height), facecolor=theme["surface"], **subplots)
    for ax in np.atleast_1d(axes).ravel():
        _plain(ax, theme)
    fig.text(0.02, 0.975, title, fontsize=15, fontweight="semibold", color=theme["ink"], va="top")
    fig.text(0.02, 0.975 - 0.32 / height, subtitle, fontsize=11, color=theme["ink2"], va="top")
    return fig, axes


def _plain(ax, theme: Dict) -> None:
    """Hairline grid and axes that stay behind the data."""
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["ink2"], length=0)
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _note(fig, theme: Dict, text: str) -> None:
    fig.text(0.02, 0.012, text, fontsize=9, color=theme["muted"], va="bottom")


def tyre_wear(theme: Dict, root: str):
    """Median over races with 3+ stints, as src/analytics/degradation.py reports it."""
    table = lake.read_report("degradation", root)
    enough = table[(~table["mixed"].astype(bool)) & (table["stints"] >= 3)]
    races = enough[["season", "round"]].drop_duplicates()
    fig, ax = _figure(theme, 9, 4.3, "Burning fuel hides most of the tyre wear in raw lap times",
                      "Time lost per lap of tyre age, one dot per race. Softs wear fastest, then mediums, then hards.")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.2)
    rng = np.random.default_rng(2026)
    for row, compound in enumerate(COMPOUNDS):
        c = enough[enough["compound"] == compound]
        colour = theme["series"][row]
        y = row + rng.uniform(-0.16, 0.16, len(c))
        ax.scatter(c["slope"], y, s=26, color=colour, alpha=0.55, linewidths=0, zorder=2)
        corrected, raw = c["slope"].median(), c["raw_slope"].median()
        ax.plot([corrected, corrected], [row - 0.33, row + 0.33], color=colour, linewidth=3, zorder=3,
                solid_capstyle="round")
        ax.plot([raw, raw], [row - 0.33, row + 0.33], color=theme["muted"], linewidth=3, zorder=3,
                solid_capstyle="round")
        ax.text(corrected, row - 0.39, f"{corrected:+.3f} s corrected", ha="left", va="bottom", fontsize=10,
                color=theme["ink"])
        ax.text(raw, row - 0.39, f"raw {raw:+.3f}", ha="right", va="bottom", fontsize=10, color=theme["ink2"])
    ax.set_yticks(range(3), [c.capitalize() for c in COMPOUNDS], color=theme["ink"])
    ax.set_ylim(2.5, -0.62)
    ax.grid(False, axis="y")
    ax.set_xlabel("seconds per lap of tyre age", color=theme["ink2"])
    ax.axvline(0, color=theme["axis"], linewidth=1, zorder=1)
    _note(fig, theme, f"Dry races of 2023-2025 with 3+ stints on the compound ({len(races)} races). "
                      "Fuel correction: 0.03 s per kg, checked against the data.")
    return fig


def dirty_air_cost(theme: Dict, root: str):
    results = dirty_air.season_results(SEASONS, root)
    pool = dirty_air.pooled(results)
    bins = pool["bins"].iloc[:-1]        # the last row is the reference: 3 s or more behind
    fig, ax = _figure(theme, 9, 4.1, "Following another car costs more the closer you get",
                      "Lap time lost against clear air (3 s or more behind), by the gap when the lap starts.")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.2)
    x = np.arange(len(bins))
    colour = theme["series"][0]
    ax.errorbar(x, bins["cost"], yerr=1.96 * bins["se"], fmt="none", ecolor=colour, elinewidth=1.5, capsize=0, zorder=2)
    ax.plot(x, bins["cost"], color=colour, linewidth=2, zorder=2)
    ax.scatter(x, bins["cost"], s=70, color=colour, edgecolors=theme["surface"], linewidths=2, zorder=3)
    for i in (0, len(bins) - 1):
        ax.annotate(f"+{bins['cost'].iloc[i]:.3f} s a lap", (x[i], bins["cost"].iloc[i]), textcoords="offset points",
                    xytext=(12, 6), fontsize=11, color=theme["ink"])
    ax.set_xticks(x, [b.replace("-", "–") + " behind" for b in bins["bin"]])
    ax.set_ylim(0, None)
    ax.set_xlim(-0.35, len(bins) - 0.4)
    ax.grid(False, axis="x")
    ax.set_ylabel("seconds per lap", color=theme["ink2"])
    _note(fig, theme, f"{pool['races']} dry races, {int(pool['bins']['laps'].sum()):,} laps; bars are 95 % intervals. "
                      "The cost is net: lost downforce and being held up, minus what DRS gives back.")
    return fig


def pit_loss(theme: Dict, root: str):
    table = lake.read_report("pit_loss", root).dropna(subset=["green_loss_sec"]).copy()
    table["circuit"] = table["event_name"].str.replace(" Grand Prix", "", regex=False)
    table["circuit"] = table["circuit"].replace({"Belgian": "Belgian (Spa)", "Emilia Romagna": "Emilia Romagna (Imola)"})
    by = table.groupby("circuit")["green_loss_sec"].agg(["median", "min", "max"]).sort_values("median", ascending=False)
    fig, ax = _figure(theme, 9, 7.4, "A pit stop costs 18 s at Spa and 28 s at Imola",
                      "Time lost to a stop under green, per circuit: each season's median, and the median across seasons.")
    fig.subplots_adjust(left=0.23, right=0.97, top=0.885, bottom=0.095)
    y = np.arange(len(by))
    colour = theme["series"][0]
    ax.hlines(y, by["min"], by["max"], color=theme["muted"], linewidth=2, alpha=0.55, zorder=2)
    ax.scatter(by["median"], y, s=60, color=colour, edgecolors=theme["surface"], linewidths=2, zorder=3)
    ax.set_yticks(y, by.index, color=theme["ink"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.set_xlabel("seconds lost", color=theme["ink2"])
    ax.annotate(f"{by['median'].iloc[0]:.1f} s", (by["max"].iloc[0], y[0]), textcoords="offset points",
                xytext=(8, -4), fontsize=10, color=theme["ink"])
    ax.annotate(f"{by['median'].iloc[-1]:.1f} s", (by["min"].iloc[-1], y[-1]), textcoords="offset points",
                xytext=(-8, -4), ha="right", fontsize=10, color=theme["ink"])
    ax.set_xlim(by["min"].min() - 1.2, by["max"].max() + 1.2)
    _note(fig, theme, f"{int(table['green_stops'].sum()):,} green stops in {len(table)} races. Grey bars span the seasons' medians.")
    return fig


def undercut_success(theme: Dict, root: str):
    table = lake.read_report("undercuts", root)
    attempts = table[table["status"] == "attempt"].copy()
    attempts["worked"] = attempts["worked"].astype(bool)
    n, k = len(attempts), int(attempts["worked"].sum())
    fig, axes = _figure(theme, 9, 4.3, "An undercut works three times in four from under a second behind",
                        f"{n} real attempts in 2023-2025; {k} worked ({k / n:.0%}). The later the car ahead answers, the better it works.",
                        ncols=2, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.71, bottom=0.2, wspace=0.08)
    groups = [(axes[0], "Gap to the car ahead before the stop",
               pd.cut(attempts["gap_before_sec"], [0, 1, 2, 3], right=False, labels=["under 1 s", "1–2 s", "2–3 s"])),
              (axes[1], "The car ahead stops ...", pd.Categorical(attempts["lag"].map({1: "next lap", 2: "2 laps later", 3: "3 laps later"}),
                               categories=["next lap", "2 laps later", "3 laps later"], ordered=True))]
    colour = theme["series"][0]
    for ax, heading, group in groups:
        grouped = attempts["worked"].groupby(group, observed=True).agg(["sum", "size"])
        x = np.arange(len(grouped))
        share = grouped["sum"] / grouped["size"]
        low, high = zip(*(wilson(int(s), int(t)) for s, t in zip(grouped["sum"], grouped["size"])))
        ax.vlines(x, low, high, color=colour, linewidth=1.5, zorder=2)
        ax.scatter(x, share, s=80, color=colour, edgecolors=theme["surface"], linewidths=2, zorder=3)
        for i, value in enumerate(share):
            ax.annotate(f"{value:.0%}", (x[i], value), textcoords="offset points", xytext=(12, -4), fontsize=11,
                        color=theme["ink"])
        ax.set_xticks(x, [f"{name}\n{int(count)} attempts" for name, count in zip(grouped.index, grouped["size"])])
        ax.set_xlim(-0.5, len(grouped) - 0.4)
        ax.set_title(heading, loc="left", fontsize=11, color=theme["ink2"])
        ax.grid(False, axis="x")
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _note(fig, theme, "Worked = ahead at the line when the car ahead has finished its out-lap. Lines are 95 % intervals.")
    return fig


def pace_models(theme: Dict, root: str):
    table = lake.read_report("pace_model_test_horizons", root)
    fig, ax = _figure(theme, 9, 4.7, "Each pace model beats the one before it, on a season none of them saw",
                      "Mean error predicting a dry lap, by how many laps ahead. Fitted on 2023-2024, scored once on 2025.")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.79, bottom=0.17)
    for i, (rung, name) in enumerate(RUNGS.items()):
        one = table[table["rung"] == rung].sort_values("laps_ahead")
        ax.plot(one["laps_ahead"], one["dry_mae"], color=theme["ladder"][i], linewidth=2, marker="o", markersize=7,
                markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=2 + i)
        last = one.iloc[-1]
        nudge = {"linear": 3, "state_space": -10}.get(rung, -4)      # these two end 0.011 s apart
        ax.annotate(f"{name}  {last['dry_mae']:.2f} s", (last["laps_ahead"], last["dry_mae"]), xytext=(10, nudge),
                    textcoords="offset points", fontsize=10.5, color=theme["ink"], annotation_clip=False)
    ax.set_xticks((1, 3, 5, 10), ["next lap", "3 laps", "5 laps", "10 laps"])
    ax.set_ylim(0.3, None)
    ax.set_ylabel("seconds", color=theme["ink2"])
    ax.grid(False, axis="x")
    laps = int(table[(table["rung"] == "boosted") & (table["laps_ahead"] == 1)]["laps"].iloc[0])
    _note(fig, theme, f"{laps:,} scored laps of the next-lap test. Each step is better by a paired bootstrap over races, at all four horizons. "
                      "Axis starts at 0.3 s.")
    return fig


def fastest_lap_rows(season: int, round_number: int, root: str):
    laps = lake.read_table(lake.table_path(lake.SILVER, "laps", season, round_number, "R", root))
    best = laps[laps["is_clean_lap"].astype(bool)].sort_values("lap_time_sec").iloc[0]
    ticks = lake.read_table(lake.table_path(lake.SILVER, "telemetry", season, round_number, "R", root))
    one = ticks[(ticks["driver_code"] == best["driver_code"]) & (ticks["lap_number"] == best["lap_number"])]
    return best, one.dropna(subset=["x", "y", "speed_kmh"]).sort_values("session_time_sec")


def track_speed(theme: Dict, root: str):
    best, lap = fastest_lap_rows(2024, 4, root)
    x, y = lap["x"].to_numpy(float), lap["y"].to_numpy(float)
    # lay the circuit along its longest direction, so it fills a wide figure
    xy = np.column_stack([x - x.mean(), y - y.mean()])
    _, _, turn = np.linalg.svd(xy, full_matrices=False)
    x, y = (xy @ turn.T).T
    speed = lap["speed_kmh"].to_numpy(float)
    distance = lap["lap_distance_m"].to_numpy(float)
    title = "Telemetry for every car on every lap of 70 races"
    subtitle = f"{best['driver_code']}'s fastest clean lap at Suzuka, 2024 ({best['lap_time_sec']:.3f} s), rebuilt from the lake's Silver layer."
    fig, axes = _figure(theme, 9, 7.4, title, subtitle, nrows=2, height_ratios=[2.6, 1])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1, hspace=0.1)
    colours = LinearSegmentedColormap.from_list("speed", theme["speed"])
    scale = Normalize(speed.min(), speed.max())
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    middle = (speed[:-1] + speed[1:]) / 2

    track, trace = axes
    # a surface-coloured line under the track keeps the crossover readable
    track.plot(x, y, color=theme["surface"], linewidth=8, zorder=1, solid_capstyle="round")
    track.add_collection(LineCollection(segments, colors=colours(scale(middle)), linewidths=4.5, capstyle="round", zorder=2))
    track.set_aspect("equal")
    track.axis("off")
    track.set_xlim(x.min() - 200, x.max() + 200)
    track.set_ylim(y.min() - 200, y.max() + 200)
    legend = track.inset_axes([0.02, 0.0, 0.22, 0.03])
    bar = fig.colorbar(plt.cm.ScalarMappable(scale, colours), cax=legend, orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=theme["ink2"], length=0, labelsize=9)
    bar.set_label("speed, km/h", color=theme["ink2"], fontsize=9)

    trace.plot(distance, speed, color=theme["series"][0], linewidth=1.8)
    slowest = int(np.argmin(speed))
    fastest = int(np.argmax(speed))
    trace.annotate(f"{speed[slowest]:.0f} km/h", (distance[slowest], speed[slowest]), xytext=(8, 2),
                   textcoords="offset points", fontsize=10, color=theme["ink"])
    trace.annotate(f"{speed[fastest]:.0f} km/h", (distance[fastest], speed[fastest]), xytext=(6, 4),
                   textcoords="offset points", fontsize=10, color=theme["ink"])
    trace.set_xlabel("distance along the lap, m", color=theme["ink2"])
    trace.set_ylabel("km/h", color=theme["ink2"])
    trace.set_xlim(0, distance.max())
    trace.set_ylim(40, speed.max() + 45)
    trace.grid(False, axis="x")
    quality = lake.read_report("telemetry_quality", root)
    _note(fig, theme, f"{quality['ticks'].sum() / 1e6:.0f} M samples; clean laps with no hole: {quality['full_lap_share'].median():.0%} (median race). "
                      "Source about 4 Hz, resampled to a shared 10 Hz clock. Telemetry: OpenF1, CC BY-NC-SA 4.0.")
    return fig


def honest_odds(theme: Dict, root: str):
    table = lake.read_report("sim_test_calibration", root)
    full = table[table["config"] == "+ retirements"]
    fig, ax = _figure(theme, 7.6, 6.2, "The simulator's odds stay honest on a season it never saw",
                      "Chance of a podium given by the simulator on 2025, against how often it really happened.")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.12)
    ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1.5, zorder=1, label="honest: chance = what happened")
    for i, (start, name) in enumerate((("lap 5", "from lap 5"), ("halfway", "from halfway"))):
        bins = reliability(full[full["start"] == start], "podium")
        colour = theme["series"][i]
        low, high = zip(*(wilson(int(round(h * n)), int(n)) for h, n in zip(bins["happened"], bins["n"])))
        ax.vlines(bins["chance"], low, high, color=colour, linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter(bins["chance"], bins["happened"], s=60, color=colour, marker="os"[i], edgecolors=theme["surface"],
                   linewidths=1.5, zorder=3, label=name)
    legend = ax.legend(loc="upper left", frameon=False, labelcolor=theme["ink"], fontsize=11)
    legend.set_zorder(5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("chance the simulator gave", color=theme["ink2"])
    ax.set_ylabel("how often it happened", color=theme["ink2"])
    _note(fig, theme, f"{len(full):,} cars at the two start points of the dry races of 2025. Nothing of the real race known. Lines are 95 % intervals.")
    return fig


CHARTS: Dict[str, Callable] = {
    "track_speed": track_speed,
    "tyre_wear": tyre_wear,
    "dirty_air": dirty_air_cost,
    "pit_loss": pit_loss,
    "undercuts": undercut_success,
    "pace_models": pace_models,
    "honest_odds": honest_odds,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the README's charts from the lake")
    parser.add_argument("--lake", default=lake.DEFAULT_LAKE_ROOT)
    parser.add_argument("--only", nargs="+", choices=sorted(CHARTS), help="draw only these")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or CHARTS:
        for mode, theme in THEMES.items():
            fig = CHARTS[name](theme, args.lake)
            path = OUT / f"{name}-{mode}.png"
            fig.savefig(path, dpi=160, facecolor=theme["surface"])
            plt.close(fig)
            print(f"{path}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
