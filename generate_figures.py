#!/usr/bin/env python3
"""Generate publication figures for spectral_alignment.tex from bundled JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    ARCH_COLORS,
    ARCH_LABELS,
    ARCH_MARKERS,
    DATA_COLOR,
    FIT_COLOR,
    MODEL_COLORS,
    apply_style,
    clip_curve_to_ylim,
    jitter,
    panel_label,
    save_figure,
    style_axes,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def model_curves(k: np.ndarray, pl: dict, exp: dict, ln: dict):
    y_pl = np.exp(pl["log_c"]) * k ** (-pl["gamma"])
    y_exp = np.exp(exp["log_c"]) * np.exp(-exp["beta"] * k)
    logk = np.log(k)
    y_ln = np.exp(ln["log_c"] + ln["a"] * logk + ln["b"] * logk**2)
    return y_pl, y_exp, y_ln


def fig1_power_law():
    data = load("results_alpha_triangle.json")
    picks = [
        ("resnet18_triangle", "layer2.0.conv1", "ResNet conv"),
        ("resnet18_triangle", "fc", "ResNet FC"),
        ("gpt2_triangle", "block0.mlp.0", "GPT-2 MLP"),
        ("vgg11_triangle", "features.0", "VGG conv"),
    ]
    by_exp = {b["experiment"]: b["results"] for b in data}

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.6), constrained_layout=True)
    for ax, (letter, (exp, layer, title)) in zip(
        axes.flat, zip("abcd", picks)
    ):
        row = next(r for r in by_exp[exp] if r["name"] == layer)
        sigma = np.array(row["sigma"])
        h = np.array(row["h_exact"])
        mask = (sigma > 0) & (h > 0)
        sx, hx = sigma[mask], h[mask]
        alpha, log_c = np.polyfit(np.log(sx), np.log(hx), 1)
        s_fit = np.logspace(np.log10(sx.min()), np.log10(sx.max()), 80)
        h_fit = np.exp(log_c) * s_fit**alpha

        ax.loglog(sx, hx, "o", ms=5, color=DATA_COLOR, alpha=0.85, zorder=3)
        ax.loglog(s_fit, h_fit, "-", color=FIT_COLOR, lw=2.0, zorder=2)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$\sigma_k$")
        ax.set_ylabel(r"$h_k$")
        style_axes(ax, minor_grid=False)
        panel_label(ax, letter)
        ax.text(
            0.04,
            0.96,
            rf"$\alpha = {alpha:.2f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.95,
            ),
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DATA_COLOR, ms=6, label="HVP $h_k$"),
        plt.Line2D([0], [0], color=FIT_COLOR, lw=2, label=r"fit $h \propto \sigma^\alpha$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
    save_figure(fig, FIGURES / "fig1_power_law", pad=0.1)


def collect_triangle_points():
    pts = []
    for block in load("results_alpha_triangle.json"):
        arch = block["experiment"].replace("_triangle", "")
        for r in block["results"]:
            if "s_hessian" not in r and "predicted_s" not in r:
                continue
            s = r.get("s_hessian", r.get("s_measured"))
            pred = r.get("predicted_s", r["alpha"] * r["gamma"])
            pts.append((s, pred, arch, r["name"]))

    tin = load("results_tiny_imagenet_alpha_map.json")
    for r in tin["results"]:
        pts.append((r["s_hessian"], r["predicted_s"], "tiny_imagenet", r["name"]))

    in1k = load("results_imagenet1k_pretrained.json")
    for r in in1k["results"]:
        pred = r.get("predicted_s", r["alpha"] * r["gamma"])
        pts.append((r["s_hessian"], pred, "imagenet1k", r["name"]))
    return pts


def fig2_alpha_triangle():
    pts = collect_triangle_points()
    s = np.array([p[0] for p in pts])
    pred = np.array([p[1] for p in pts])
    archs = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(4.35, 4.1), constrained_layout=True)
    for arch in sorted(set(archs)):
        mask = np.array([a == arch for a in archs])
        ax.scatter(
            pred[mask],
            s[mask],
            s=36,
            alpha=0.82,
            c=ARCH_COLORS.get(arch, "#666666"),
            marker=ARCH_MARKERS.get(arch, "o"),
            label=ARCH_LABELS.get(arch, arch),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    lo = min(s.min(), pred.min()) * 0.85
    hi = max(s.max(), pred.max()) * 1.08
    xx = np.linspace(lo, hi, 100)
    ax.fill_between(xx, xx * 0.9, xx * 1.1, color="#E8EEF4", zorder=0, label=None)
    ax.plot(xx, xx, color="#333333", ls="--", lw=1.2, zorder=1, label=r"$s = \alpha\gamma$")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Predicted $s = \alpha\gamma$")
    ax.set_ylabel(r"Measured $s$ (Hessian decay)")
    style_axes(ax)

    rel = np.abs(s - pred) / np.maximum(s, 1e-6)
    ax.text(
        0.97,
        0.05,
        rf"median rel.\ error $= {np.median(rel)*100:.1f}\%$" + "\n" + rf"$n = {len(s)}$ layers",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
    )
    ax.legend(loc="upper left", frameon=True, borderaxespad=0.6)
    save_figure(fig, FIGURES / "fig2_alpha_triangle")


def fig3_rank_profiles():
    mfc = load("results_model_fit_comparison.json")
    tri = {
        b["experiment"]: {r["name"]: r for r in b["results"]}
        for b in load("results_alpha_triangle.json")
    }
    layers = [
        ("resnet18_triangle", "conv1", "conv1"),
        ("resnet18_triangle", "layer1.0.conv1", "layer1 conv"),
        ("resnet18_triangle", "fc", "FC head"),
    ]
    bic_labels = {"lognormal": "log-normal", "power_law": "power-law", "exponential": "exponential"}

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.75), constrained_layout=True)
    for ax, (letter, (exp, name, short)) in zip(axes, zip("abc", layers)):
        layer_mfc = next(l for l in mfc["layers"] if l["experiment"] == exp and l["name"] == name)
        winner = bic_labels.get(layer_mfc["best_model_bic"], layer_mfc["best_model_bic"])
        sigma = np.array(tri[exp][name]["sigma"])
        k = np.arange(1, len(sigma) + 1, dtype=float)
        fits = layer_mfc["fits"]
        y_pl, y_exp, y_ln = model_curves(k, fits["power_law"], fits["exponential"], fits["lognormal"])

        sigma_ref = sigma[sigma > 0]
        if name == "fc" and len(sigma_ref) > 3:
            sigma_ref = sigma_ref[:-1]  # last rank is a spectral outlier
        ylo = max(sigma_ref.min() * 0.3, 1e-10)
        yhi = sigma_ref.max() * 4
        ax.semilogy(k, sigma, "o", ms=4.5, color=DATA_COLOR, zorder=4, label="measured")

        for y, key, lbl in [
            (y_pl, "power_law", "power-law"),
            (y_exp, "exponential", "exponential"),
            (y_ln, "lognormal", "log-normal"),
        ]:
            kc, yc = clip_curve_to_ylim(k, y, ylo, yhi)
            is_winner = lbl == winner
            lw = 2.2 if is_winner else 1.2
            alpha_line = 1.0 if is_winner else 0.5
            ax.semilogy(
                kc,
                yc,
                "-",
                color=MODEL_COLORS[key],
                lw=lw,
                alpha=alpha_line,
                zorder=2 if lw > 2 else 1,
                label=lbl,
            )

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0.8, len(k) + 0.4)
        ax.set_title(f"{short}  ·  BIC: {winner}", fontsize=8.5, pad=5)
        ax.set_xlabel("rank $k$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\sigma_k$")
        style_axes(ax)
        panel_label(ax, letter)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.06))
    save_figure(fig, FIGURES / "fig3_rank_profiles", pad=0.12)


def fig4_phase_transition():
    traj = load("results_vgg16_trajectory.json")["trajectory"]
    epochs = np.array([t["epoch"] for t in traj])
    models = ["lognormal", "power_law", "exponential"]
    labels = {"lognormal": "log-normal", "power_law": "power-law", "exponential": "exponential"}
    counts = {m: [] for m in models}
    for snap in traj:
        w = snap["wins_by_bic"]
        n = snap["n_layers"]
        for m in models:
            counts[m].append(100 * w.get(m, 0) / n)

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    y0 = np.zeros(len(epochs))
    for m in models:
        y = np.array(counts[m])
        ax.fill_between(
            epochs,
            y0,
            y0 + y,
            step="mid",
            alpha=0.85,
            color=MODEL_COLORS.get(m, "#888888"),
            label=labels[m],
            linewidth=0,
        )
        y0 = y0 + y

    ax.set_xlim(epochs.min() - 1, epochs.max() + 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("layers (%)")
    style_axes(ax, minor_grid=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7)
    save_figure(fig, FIGURES / "fig4_phase_transition")


def layer_depth_index(name: str) -> float:
    if name == "conv1" or name.startswith("features.0"):
        return 0
    m = re.search(r"layer(\d+)", name)
    if m:
        sub = re.search(r"conv(\d+)", name)
        return int(m.group(1)) + (int(sub.group(1)) * 0.1 if sub else 0)
    m = re.search(r"block(\d+)", name)
    if m:
        return 10 + int(m.group(1))
    if "fc" in name or "classifier" in name or "output" in name:
        return 20
    m = re.search(r"features\.(\d+)", name)
    if m:
        return int(m.group(1)) / 3
    return 5


def fig5_alpha_depth():
    fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
    specs = [
        ("results_alpha_triangle.json", "resnet18_triangle", "ResNet-18", "resnet18"),
        ("results_alpha_triangle.json", "vgg11_triangle", "VGG-11", "vgg11"),
        ("results_alpha_triangle.json", "gpt2_triangle", "GPT-2", "gpt2"),
    ]
    for seed, (fname, exp, label, key) in enumerate(specs):
        block = next(b for b in load(fname) if b["experiment"] == exp)
        xs, ys = [], []
        for r in block["results"]:
            xs.append(layer_depth_index(r["name"]))
            ys.append(r["alpha"])
        xs = jitter(np.array(xs), amount=0.22, seed=seed + 7)
        ax.scatter(
            xs,
            ys,
            s=42,
            alpha=0.78,
            c=ARCH_COLORS[key],
            marker=ARCH_MARKERS[key],
            label=label,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

    ax.axhspan(1.85, 2.15, color="#4477AA", alpha=0.12, zorder=0)
    ax.axhline(2.0, color="#555555", ls="--", lw=1.0, zorder=1, label=r"$\alpha = 2$ (conv)")
    ax.set_xlabel("depth index (approx.)")
    ax.set_ylabel(r"curvature exponent $\alpha$")
    ax.set_xlim(-0.8, 21.5)
    ax.set_ylim(0.65, 3.35)
    style_axes(ax)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7, columnspacing=0.8)
    save_figure(fig, FIGURES / "fig5_alpha_depth")


def main():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating figures ->", FIGURES)
    fig1_power_law()
    fig2_alpha_triangle()
    fig3_rank_profiles()
    fig4_phase_transition()
    fig5_alpha_depth()
    print("Done.")


if __name__ == "__main__":
    main()
