#!/usr/bin/env python3
"""
Verify paper table statistics against bundled result JSON files.

Usage (from repo root or this directory):
    python papers/spectral_alignment/regenerate_tables.py
    python papers/spectral_alignment/regenerate_tables.py --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def check_alpha_triangle(verbose: bool) -> None:
    data = load("results_alpha_triangle.json")
    rows = []
    for block in data:
        for r in block["results"]:
            if "triangle_relative_error" not in r:
                continue
            rows.append(r)

    import numpy as np

    errs_all = [r["triangle_relative_error"] for r in rows]
    good = [r for r in rows if r["triangle_relative_error"] <= 0.10]
    errs_good = [r["triangle_relative_error"] for r in good]

    print("=== Alpha Triangle (CIFAR / multi-arch) ===")
    print(f"  All layers: {len(rows)}, median err = {np.median(errs_all)*100:.2f}%")
    print(f"  Well-conditioned (rel err <= 10%): {len(good)}, median err = {np.median(errs_good)*100:.2f}%  [paper: 1.9%]")
    print(f"  Max error (excluded layers): {max(errs_all)*100:.2f}%")

    if verbose and good:
        for r in sorted(good, key=lambda x: x["triangle_relative_error"])[:5]:
            print(f"    {r['name']}: rel_err={r['triangle_relative_error']*100:.2f}%")


def check_tiny_imagenet_triangle(verbose: bool) -> None:
    path = RESULTS / "results_tiny_imagenet_alpha_map.json"
    if not path.exists():
        print("\n=== Tiny-ImageNet triangle: skipped (no JSON) ===")
        return
    import numpy as np

    data = load("results_tiny_imagenet_alpha_map.json")
    rows = data.get("results", [])
    errs = [r["triangle_relative_error"] for r in rows if r.get("triangle_relative_error") is not None]
    good = [e for e in errs if e <= 0.10]
    print("\n=== Tiny-ImageNet ResNet-50 ===")
    print(f"  Layers: {len(errs)}, median triangle err = {np.median(errs)*100:.2f}%  [paper: 1.0%]")
    print(f"  Conv median alpha = {data['summary']['by_layer_type']['conv']['median_alpha']:.2f}  [paper: 1.93]")
    if verbose:
        fc = data["summary"]["by_layer_type"].get("fc", {})
        if fc:
            print(f"  FC alpha = {fc.get('median_alpha', fc.get('mean_alpha')):.2f}  [paper: 0.90]")


def check_gamma_bic(verbose: bool) -> None:
    p1 = RESULTS / "results_model_fit_comparison.json"
    rmt = RESULTS / "results_rmt_gamma_spectrum.json"
    if not p1.exists() and not rmt.exists():
        print("\n=== Gamma BIC: skipped ===")
        return
    print("\n=== Gamma rank-profile (BIC) ===")
    if p1.exists():
        s = load("results_model_fit_comparison.json")["summary"]
        print(f"  CIFAR top-20 (24 layers): {s['wins_by_bic']}  [paper: LN=21, PL=1, EXP=2]")
    if rmt.exists():
        d = load("results_rmt_gamma_spectrum.json")
        print(f"  T-IN init (k=100): {d['phase_2a_init']['summary']['wins_by_bic']}")
        print(f"  T-IN trained (k=100): {d['phase_1b_extended_trained']['summary']['wins_by_bic']}")
        if verbose:
            for snap in d.get("phase_2b_trajectory", []):
                print(f"    epoch {snap['epoch']}: {snap['wins_by_bic']}")


def check_intrinsic_dim(verbose: bool) -> None:
    data = load("results_intrinsic_dim.json")
    print("\n=== Intrinsic dimension (curvature PR) ===")
    for block in data:
        exp = block.get("experiment", "?")
        if exp not in ("resnet18_intrinsic", "gpt2_intrinsic", "scaling_test"):
            continue
        print(f"  [{exp}]")
        for r in block.get("results", []):
            name = r.get("name", r.get("width", "?"))
            pr = r.get("eff_rank_curvature_pr", r.get("curvature_pr"))
            if pr is not None:
                print(f"    {name}: PR_h={pr:.1f}")


def check_conv_gap(verbose: bool) -> None:
    data = load("results_conv_gap.json")
    print("\n=== Conv gap ===")
    items = data.get("conv_results", []) if isinstance(data, dict) else []
    for r in items:
        if not isinstance(r, dict):
            continue
        name = r.get("name", "?")
        alpha = r.get("alpha")
        c2_list = r.get("cos2_naive") or r.get("cos2_kfc")
        c2 = float(c2_list[0]) if c2_list else None
        if alpha is not None:
            extra = f", cos2[0]={c2:.2e}" if c2 else ""
            print(f"    {name}: alpha={float(alpha):.2f}{extra}")


def check_imagenet1k(verbose: bool) -> None:
    path = RESULTS / "results_imagenet1k_pretrained.json"
    if not path.exists():
        print("\n=== ImageNet-1K pretrained: skipped (no JSON) ===")
        return
    import numpy as np

    data = load("results_imagenet1k_pretrained.json")
    rows = data.get("results", [])
    errs = [r["triangle_relative_error"] for r in rows if r.get("triangle_relative_error") is not None]
    conv_alphas = [r["alpha"] for r in rows if "conv" in r["name"]]
    fc = [r for r in rows if r["name"] == "fc.weight"]
    print("\n=== ImageNet-1K Pretrained ResNet-50 ===")
    print(f"  Layers: {len(rows)}, median triangle err = {np.median(errs)*100:.2f}%  [paper: 1.6%]")
    print(f"  Conv mean alpha = {np.mean(conv_alphas):.2f}, median = {np.median(conv_alphas):.2f}  [paper: 2.10 / 2.15]")
    if fc:
        print(f"  FC alpha = {fc[0]['alpha']:.2f}  [paper: 2.83]")
    print(f"  BIC wins: {data['summary']['bic_wins']}")


def check_ablations(verbose: bool) -> None:
    hyp = load("results_hypothesis.json")
    print("\n=== Ablation: output dim / task structure ===")
    if isinstance(hyp, dict):
        for key in hyp:
            print(f"  [{key}]: {list(hyp[key].keys()) if isinstance(hyp[key], dict) else '...'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Results directory: {RESULTS}")
    assert RESULTS.exists(), f"Missing {RESULTS}; bundle JSON from experiments/."

    check_alpha_triangle(args.verbose)
    check_tiny_imagenet_triangle(args.verbose)
    check_imagenet1k(args.verbose)
    check_gamma_bic(args.verbose)
    check_intrinsic_dim(args.verbose)
    try:
        check_conv_gap(args.verbose)
    except (KeyError, TypeError) as e:
        print(f"\n=== Conv gap: skipped ({e}) ===")
    check_ablations(args.verbose)
    print("\nDone. Compare printed values to papers/spectral_alignment.tex.")


if __name__ == "__main__":
    main()
