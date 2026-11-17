"""Shared chart style and colors so every figure in the project looks the same."""

import matplotlib.pyplot as plt

from hillstrom.paths import FIGURES_DIR

# Fixed color per arm, used in every chart. Control is a neutral reference, not a series.
ARM_COLORS = {"mens": "#2a78d6", "womens": "#eb6834", "control": "#8a8985"}
ARM_LABELS = {"mens": "Mens E-Mail", "womens": "Womens E-Mail", "control": "No E-Mail"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Recessive grid and axes, readable font sizes."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as reports/figures/<name>.png for the README."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
