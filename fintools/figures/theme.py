"""Shared figure themes for papers, Word documents, and slides."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

FINS_COLORS = {
    "navy": "#1F3A5F",
    "crimson": "#B23A48",
    "forest": "#2E7D32",
    "gold": "#C99700",
    "teal": "#007C89",
    "violet": "#6B5B95",
    "steel": "#4A5568",
    "gray": "#8A8F98",
}

FT_BACKGROUND = "#FDF1E6"

FT_COLORS = {
    "maroon": "#990F3D",
    "pink": "#E95D8E",
    "light_blue": "#A7D4E8",
    "teal": "#0F766E",
    "blue": "#0F5499",
    "gold": "#F2B701",
    "green": "#0D7680",
    "purple": "#6F4E7C",
    "orange": "#D56F3E",
    "slate": "#4C78A8",
    "brown": "#8C6D31",
    "charcoal": "#262A33",
    "muted": "#6B625C",
    "axis": "#B8AEA7",
    "grid": "#E2D8CF",
    "recession": "#DAD6D0",
}

COLOR_CYCLE = [
    FINS_COLORS["navy"],
    FINS_COLORS["crimson"],
    FINS_COLORS["forest"],
    FINS_COLORS["gold"],
    FINS_COLORS["teal"],
    FINS_COLORS["violet"],
]

FT_COLOR_CYCLE = [
    FT_COLORS["maroon"],
    FT_COLORS["blue"],
    FT_COLORS["teal"],
    FT_COLORS["pink"],
    FT_COLORS["gold"],
    FT_COLORS["purple"],
    FT_COLORS["orange"],
    FT_COLORS["slate"],
    FT_COLORS["brown"],
    FT_COLORS["green"],
]


def theme_rc(
    profile: str = "paper",
    *,
    style: str = "fins",
    ft_background: bool = False,
) -> dict[str, object]:
    """Return Matplotlib rcParams for a named figure profile."""

    if style not in {"fins", "ft"}:
        raise ValueError("style must be one of: fins, ft")

    base: dict[str, object] = {
        "axes.edgecolor": "#2F3337",
        "axes.grid": False,
        "axes.labelcolor": "#1F2933",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=COLOR_CYCLE),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlecolor": "#111827",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "font.family": "DejaVu Sans",
        "grid.alpha": 0.55,
        "grid.color": "#D8DDE6",
        "grid.linewidth": 0.6,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.04,
        "xtick.color": "#4B5563",
        "xtick.labelsize": 9,
        "ytick.color": "#4B5563",
        "ytick.labelsize": 9,
    }

    if profile == "word_a4":
        base.update(
            {
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "figure.dpi": 150,
                "lines.linewidth": 2.0,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )
    elif profile == "slides":
        base.update(
            {
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "figure.dpi": 150,
                "grid.linewidth": 0.8,
                "legend.fontsize": 12,
                "lines.linewidth": 2.8,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )
    elif profile != "paper":
        raise ValueError("profile must be one of: paper, word_a4, slides")

    if style == "ft":
        background = FT_BACKGROUND if ft_background else "white"
        base.update(
            {
                "axes.edgecolor": FT_COLORS["axis"],
                "axes.facecolor": background,
                "axes.labelcolor": FT_COLORS["charcoal"],
                "axes.prop_cycle": cycler(color=FT_COLOR_CYCLE),
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.titlecolor": FT_COLORS["charcoal"],
                "figure.facecolor": background,
                "font.family": "DejaVu Sans",
                "grid.alpha": 0.9,
                "grid.color": FT_COLORS["grid"],
                "grid.linewidth": 0.7,
                "legend.frameon": False,
                "savefig.facecolor": background,
                "xtick.color": FT_COLORS["muted"],
                "ytick.color": FT_COLORS["muted"],
            }
        )

    return base


def apply_theme(
    profile: str = "paper",
    *,
    style: str = "fins",
    ft_background: bool = False,
) -> None:
    """Apply a figure profile globally for the current Python session."""

    mpl.rcParams.update(theme_rc(profile, style=style, ft_background=ft_background))


@contextmanager
def figure_style(
    profile: str = "paper",
    *,
    style: str = "fins",
    ft_background: bool = False,
) -> Iterator[None]:
    """Temporarily apply a figure profile within a plotting block."""

    with plt.rc_context(theme_rc(profile, style=style, ft_background=ft_background)):
        yield
