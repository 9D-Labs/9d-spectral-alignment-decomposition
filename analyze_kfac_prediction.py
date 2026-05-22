#!/usr/bin/env python3
"""K-FAC / Kronecker quality vs curvature exponent (conv gap v2 + alpha triangle)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import FIT_COLOR, apply_style, panel_label, save_figure, style_axes

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Tol bright
C_SLACK = "#0077BB"
C_R2 = "#CC3317"


def main():
    apply_style()
    with open(RESULTS / "results_conv_gap_v2.json") as f:
        gap = json.load(f)

    rows = []
    for layer in gap["conv_results"]:
        conc = float(np.median(layer["concentration_ratio"]))
        rows.append(
            {
                "name": layer["name"],
                "alpha": layer["alpha_exact"],
                "alpha_gap": abs(layer["alpha_exact"] - 2.0),
                "conc_ratio": conc,
                "kronecker_slack": abs(1.0 - conc),
                "r2_exact": layer["r2_exact"],
            }
        )
    fc = gap.get("fc_control")
    if fc:
        rows.append(
            {
                "name": "fc",
                "alpha": fc["alpha_exact"],
                "alpha_gap": abs(fc["alpha_exact"] - 2.0),
                "conc_ratio": float(np.median(fc["concentration_ratio"])),
                "kronecker_slack": abs(1.0 - float(np.median(fc["concentration_ratio"]))),
                "r2_exact": fc["r2_exact"],
            }
        )

    alpha_gap = np.array([r["alpha_gap"] for r in rows])
    slack = np.array([r["kronecker_slack"] for r in rows])
    r2 = np.array([r["r2_exact"] for r in rows])
    corr_slack = float(np.corrcoef(alpha_gap, slack)[0, 1]) if len(rows) > 2 else np.nan
    corr_r2 = float(np.corrcoef(alpha_gap, r2)[0, 1]) if len(rows) > 2 else np.nan

    out = {
        "n_layers": len(rows),
        "corr_alpha_gap_vs_kronecker_slack": corr_slack,
        "corr_alpha_gap_vs_r2_h_sigma": corr_r2,
        "layers": rows,
    }
    with open(RESULTS / "results_kfac_prediction.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.05), constrained_layout=True)

    ax = axes[0]
    sizes = 50 + 120 * (1 - slack / max(slack.max(), 1e-6))
    ax.scatter(alpha_gap, slack, s=sizes, c=C_SLACK, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
    if len(rows) > 2:
        z = np.polyfit(alpha_gap, slack, 1)
        xs = np.linspace(alpha_gap.min(), alpha_gap.max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), color=C_SLACK, ls="--", lw=1.0, alpha=0.6, zorder=1)
    ax.set_xlabel(r"$|\alpha - 2|$")
    ax.set_ylabel(r"$|1 - \mathrm{conc.\ ratio}|$")
    ax.set_title(f"Kronecker slack  ($r = {corr_slack:.2f}$)", fontsize=8.5, pad=6)
    style_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(alpha_gap, r2, s=sizes, c=C_R2, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(0.97, color="#888888", ls=":", lw=1.0, zorder=1)
    ax.set_xlabel(r"$|\alpha - 2|$")
    ax.set_ylabel(r"$R^2(h_k, \sigma_k^\alpha)$")
    ax.set_ylim(0.93, 1.005)
    ax.set_title(f"Power-law fit  ($r = {corr_r2:.2f}$)", fontsize=8.5, pad=6)
    style_axes(ax)
    panel_label(ax, "b")

    FIGURES.mkdir(exist_ok=True)
    save_figure(fig, FIGURES / "fig6_kfac_prediction", pad=0.1)
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
