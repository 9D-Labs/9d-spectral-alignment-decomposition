# Spectral Alignment Decomposition

**Spectral Asymptotics of Neural Network Loss Landscapes: Decomposing the Curvature Exponent**

Anherutowa Calvo — [9D Labs](https://9dlabs.xyz)

---

## Overview

This repository provides the complete reproducibility package for the paper. It contains:

- **Paper source** (`paper/`) — LaTeX source, style file, and compiled PDF
- **Frozen results** (`results/`) — 15 JSON files covering all tables and figures
- **Automated claim verification** (`verify_claims.py`) — checks 27 quantitative claims against JSON data
- **Figure generation** (`generate_figures.py`, `analyze_kfac_prediction.py`) — reproduces all 6 main-text figures from frozen data
- **Table verification** (`regenerate_tables.py`) — prints summary statistics for manual comparison with paper tables

## Quick start (no GPU required)

```bash
pip install -r requirements.txt
make all     # verify claims → regenerate figures → compile PDF
```

Or step by step:

```bash
python verify_claims.py -v            # 27 quantitative claims vs. JSON
python regenerate_tables.py --verbose # summary statistics for tables
python generate_figures.py            # figures 1–5
python analyze_kfac_prediction.py     # figure 6 (K-FAC prediction quality)
cd paper && pdflatex spectral_alignment.tex && pdflatex spectral_alignment.tex
```

## What each script does

| Script | Purpose |
|--------|---------|
| `verify_claims.py` | Checks every number in the paper (α values, triangle errors, BIC counts, PR bounds, SN/Adam accuracies) against bundled JSON. Exits 1 on any mismatch. |
| `regenerate_tables.py` | Prints summary statistics for manual comparison with paper tables. |
| `generate_figures.py` | Produces Figs 1–5 (power law, alpha triangle, rank profiles, phase transition, α vs depth). |
| `analyze_kfac_prediction.py` | Produces Fig 6 (Kronecker factorization quality vs. curvature exponent). |
| `figure_style.py` | Shared Tol-bright palette, grid, and typography for all figures. |

## Paper claims → data sources

| Claim | Section | JSON file | Key field |
|-------|---------|-----------|-----------|
| α ≈ 2 for conv, ≈ 1 for transformers | §3 | `results_alpha_triangle.json` | `alpha` per layer |
| R² = 0.98 (median, 21 ResNet-18 layers) | §3, Abstract | `results_alpha_triangle.json` | `r2_h_exact` |
| Tiny-ImageNet: 49 conv, median α = 1.93 | §3 | `results_tiny_imagenet_alpha_map.json` | `summary.by_layer_type.conv` |
| ImageNet-1K: 14 layers, R² ≥ 0.97 | §3 | `results_imagenet1k_pretrained.json` | `r2_exact` |
| s = αγ, median error 1.9% (CIFAR) | §5 | `results_alpha_triangle.json` | `triangle_relative_error` |
| s = αγ, median error 1.0% (Tiny-IN) | §5 | `results_tiny_imagenet_alpha_map.json` | `triangle_relative_error` |
| s = αγ, median error 1.6% (IN-1K) | §5 | `results_imagenet1k_pretrained.json` | `triangle_relative_error` |
| BIC: 21/24 log-normal (CIFAR, k=20) | §6, App | `results_model_fit_comparison.json` | `summary.wins_by_bic` |
| VGG-16 spectral phase transition | §6 | `results_vgg16_trajectory.json` | `trajectory[].wins_by_bic` |
| PR width scaling: 2.5 → 3.2 (64→512) | App | `results_intrinsic_dim.json` | `eff_rank_curvature_pr` |
| Kronecker slack ↔ \|α−2\| | §7, Fig 6 | `results_kfac_prediction.json` | `corr_alpha_gap_vs_*` |
| Scalar preconditioner ≈ Adam (±0.2%) | App | `results_preconditioner_summary.json` | `summary.*` |

## Directory layout

```
spectral_alignment_decomposition/
├── README.md
├── LICENSE
├── Makefile
├── requirements.txt
├── verify_claims.py
├── regenerate_tables.py
├── generate_figures.py
├── analyze_kfac_prediction.py
├── figure_style.py
├── EXPERIMENTS.md              # measurement protocol for full reproduction
├── paper/
│   ├── spectral_alignment.tex
│   ├── spectral_alignment.pdf
│   └── neurips_2025.sty
├── figures/
│   ├── fig1_power_law.{pdf,png}
│   ├── fig2_alpha_triangle.{pdf,png}
│   ├── fig3_rank_profiles.{pdf,png}
│   ├── fig4_phase_transition.{pdf,png}
│   ├── fig5_alpha_depth.{pdf,png}
│   └── fig6_kfac_prediction.{pdf,png}
└── results/
    └── (15 JSON files)
```

## Measurement protocol

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed instructions on reproducing the experimental measurements from scratch (requires GPU).

## Hardware

All GPU experiments were run on **NVIDIA A10G** via [Modal](https://modal.com). CPU-only verification and figure generation runs on any machine with Python 3.9+ and the dependencies in `requirements.txt`.

## Citation

```bibtex
@article{calvo2026spectral,
  title={Spectral Asymptotics of Neural Network Loss Landscapes:
         Decomposing the Curvature Exponent},
  author={Calvo, Anherutowa},
  year={2026}
}
```

## License

Code: MIT. Paper: CC-BY 4.0.
