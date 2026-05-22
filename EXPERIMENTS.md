# Experimental Protocol

This document describes the measurement protocol for fully reproducing all experimental results from scratch. The frozen JSON files in `results/` already contain all outputs; this guide is for researchers who want to verify the measurements independently.

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA
- GPU with ≥ 24 GB VRAM (A10G or better)
- Dependencies: `pip install -r requirements.txt`

## Overview of experiments

All measurements follow the same core protocol described in Section 2 of the paper:

1. Train model (or load pretrained weights) to convergence
2. Fix a batch of B = 2048–4096 samples
3. Compute top-k gradient singular directions via SVD
4. Measure exact curvature h_k = v_k^T H v_k via double backpropagation (exact HVP)
5. Fit α by log-log regression on (σ_k, h_k)

Finite-difference Hessians are excluded (R² ≈ 0.17 with 50% spurious negatives).

---

## Experiment 1: Curvature exponent α (architecture sweep)

**Models:** ResNet-18, VGG-11, Pure MLP, GPT-2 (6-layer)
**Dataset:** CIFAR-10
**Output:** `results/results_alpha_triangle.json`

For each model:
- Train to convergence with standard hyperparameters
- At convergence, compute per-layer gradient SVD (top-20 directions)
- Measure exact HVP curvature along each singular direction
- Fit α, γ, and compute spectral transfer identity s = αγ

## Experiment 2: Conv gap and ablations

**Models:** ResNet-18, VGG-11, ViT-Tiny, GPT-2
**Dataset:** CIFAR-10, CIFAR-100
**Output:** `results/results_conv_gap.json`, `results/results_conv_gap_v2.json`, `results/results_hypothesis.json`, `results/results_predictions.json`

Key measurements:
- Concentration ratio (sum-of-products vs. product-of-sums Hessian)
- cos²θ_k (GN-to-exact gap)
- Controlled ablations: LayerNorm removal, output dimension variation, task structure (LM vs. classification)

## Experiment 3: Alpha triangle and intrinsic dimension

**Models:** ResNet-18, VGG-11, Pure MLP, GPT-2
**Dataset:** CIFAR-10
**Output:** `results/results_alpha_triangle.json`, `results/results_intrinsic_dim.json`

- Compute (α, γ, s) triad per layer
- Validate s = αγ identity
- Measure participation ratio PR_h and PR_grad
- Width scaling test: MLP with hidden dim 64, 128, 256, 512

## Experiment 4: Scale validation (Tiny-ImageNet)

**Model:** ResNet-50
**Dataset:** Tiny-ImageNet-200 (64×64, 200 classes)
**Output:** `results/results_tiny_imagenet_alpha_map.json`

- Train ResNet-50 for 25 epochs
- Measure α across all 49 conv layers + FC head
- Validate spectral transfer identity at this scale

## Experiment 5: Scale validation (ImageNet-1K)

**Model:** ResNet-50 (pretrained IMAGENET1K_V1)
**Dataset:** ImageNet-1K
**Output:** `results/results_imagenet1k_pretrained.json`

- Load torchvision pretrained weights
- Measure α on 14 representative conv layers + FC head
- No training — purely measurement on pretrained checkpoint

## Experiment 6: Gradient rank profiles and BIC model comparison

**Models:** ResNet-18 (CIFAR-10), ResNet-50 (Tiny-ImageNet)
**Output:** `results/results_model_fit_comparison.json`, `results/results_rmt_gamma_spectrum.json`

- Fit power-law, exponential, and log-normal models to rank-ordered σ_k
- Select best model via Bayesian Information Criterion (BIC)
- Compare init vs. trained distributions

## Experiment 7: Spectral phase transition (VGG-16 trajectory)

**Model:** VGG-16
**Dataset:** Tiny-ImageNet-200
**Output:** `results/results_vgg16_trajectory.json`

- Save checkpoints every 5 epochs during training
- At each checkpoint, compute per-layer BIC model selection
- Track the fraction of layers best fit by each model over training

## Experiment 8: Depth scaling

**Models:** ResNet-18, VGG-11, MLP
**Output:** `results/results_depth_scaling.json`

- Measure α variation across depth within each architecture

## Experiment 9: Kronecker factor spectra

**Model:** ResNet-50
**Dataset:** Tiny-ImageNet-200
**Output:** `results/results_kronecker_spectral.json`

- Extract Kronecker factors C_δ and C_A via backward hooks (im2col for conv)
- Compare factor eigenspectra to gradient SVD
- Measure factor-to-gradient BIC agreement and effective rank

## Experiment 10: Scalar preconditioner (negative control)

**Model:** ResNet-18
**Dataset:** CIFAR-10
**Output:** `results/results_preconditioner_summary.json`

- Adam baseline vs. Adam with per-layer learning rate η_ℓ ∝ (s_ℓ / s̄)^β
- β ∈ {0.5, 1.0, 1.5}, plus controls (inverse scaling, random noise)
- 3 seeds each, 100 epochs
- Confirms scalar per-layer scaling does NOT reproduce directional curvature correction
