"""
Colours of the Analyze window (src/gui/analysis_theme.py)

One definition shared by the window (Qt) and its charts (matplotlib, src/gui/pace_charts.py),
so a compound or a panel can never be two slightly different colours.
"""

# Window chrome (dark)
BG, PANEL, BORDER = "#121212", "#1A1A1A", "#2A2A2A"
TEXT, DIM, ACCENT, GOOD, BAD = "#E0E0E0", "#8A8A8A", "#00E5FF", "#00E676", "#FF5252"

# Pirelli's sidewall colours: what every F1 viewer reads as soft, medium, hard, inter, wet.
COMPOUND_COLOURS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#E8E8E8",
    "INTERMEDIATE": "#39B54A", "WET": "#0067FF",
}
# Text drawn ON a compound colour (stint bars): white or black by the fill's lightness.
COMPOUND_TEXT = {"SOFT": "#FFFFFF", "MEDIUM": "#000000", "HARD": "#000000",
                 "INTERMEDIATE": "#FFFFFF", "WET": "#FFFFFF"}
NO_COMPOUND_COLOUR = "#9E9E9E"   # the feed gave no compound: a neutral, never a made-up tyre colour
