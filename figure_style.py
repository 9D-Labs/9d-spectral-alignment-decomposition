"""Shared matplotlib style for spectral_alignment paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Tol bright qualitative (colorblind-safe)
ARCH_COLORS = {
    "resnet18": "#0077BB",
    "vgg11": "#33BBEE",
    "gpt2": "#EE7733",
    "tiny_imagenet": "#CC3311",
    "imagenet1k": "#AA4499",
}

ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "vgg11": "VGG-11",
    "gpt2": "GPT-2",
    "tiny_imagenet": "Tiny-ImageNet",
    "imagenet1k": "ImageNet-1K",
}

ARCH_MARKERS = {
    "resnet18": "o",
    "vgg11": "s",
    "gpt2": "^",
    "imagenet1k": "D",
    "tiny_imagenet": "v",
}

MODEL_COLORS = {
    "power_law": "#009988",
    "exponential": "#CC6677",
    "lognormal": "#4477AA",
    "log-normal": "#4477AA",
}

DATA_COLOR = "#222222"
FIT_COLOR = "#BB5566"
GRID_KW = {"alpha": 0.35, "ls": "-", "lw": 0.4, "color": "#CCCCCC"}
MINOR_GRID_KW = {"alpha": 0.2, "ls": "-", "lw": 0.25, "color": "#DDDDDD"}


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "medium",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": False,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def style_axes(ax, *, minor_grid: bool = True):
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="major", **GRID_KW)
    if minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", **MINOR_GRID_KW)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
        color="#333333",
    )


def save_figure(fig, path_stem, *, pad: float = 0.08):
    for ext in ("pdf", "png"):
        fig.savefig(
            path_stem.with_suffix(f".{ext}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad,
            facecolor="white",
        )
    plt.close(fig)


def jitter(x: np.ndarray, amount: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.uniform(-amount, amount, size=len(x))


def clip_curve_to_ylim(k, y, ylo, yhi):
    """Return k, y masked to visible y range for semilogy plots."""
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (y >= ylo) & (y <= yhi) & np.isfinite(y)
    return k[mask], y[mask]
