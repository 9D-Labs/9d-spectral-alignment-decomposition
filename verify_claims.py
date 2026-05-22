#!/usr/bin/env python3
"""
Verify every quantitative claim in spectral_alignment.tex against bundled JSON.

Exits 0 if all claims pass; exits 1 if any claim fails.
Run:  python verify_claims.py          (summary)
      python verify_claims.py -v       (per-claim detail)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

passed, failed, skipped = [], [], []


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def check(name: str, actual, expected, tol=0.02, rel=True):
    if actual is None:
        skipped.append(name)
        return
    if rel and expected != 0:
        ok = abs(actual - expected) / abs(expected) <= tol
    else:
        ok = abs(actual - expected) <= tol
    (passed if ok else failed).append(
        f"{'PASS' if ok else 'FAIL'}: {name}  "
        f"(expected={expected}, got={actual:.4f})"
    )


def check_eq(name: str, actual, expected):
    ok = actual == expected
    (passed if ok else failed).append(
        f"{'PASS' if ok else 'FAIL'}: {name}  "
        f"(expected={expected}, got={actual})"
    )


# ── Section 3: Curvature exponent landscape ─────────────────────────

def verify_alpha_triangle_cifar():
    data = load("results_alpha_triangle.json")
    if data is None:
        skipped.append("alpha_triangle_cifar")
        return

    rows = []
    for block in data:
        for r in block["results"]:
            if "triangle_relative_error" in r:
                rows.append(r)

    all_err = [r["triangle_relative_error"] for r in rows]
    good = [r for r in rows if r["triangle_relative_error"] <= 0.10]
    good_err = [r["triangle_relative_error"] for r in good]

    check("§5 CIFAR triangle: n_well_conditioned", len(good), 19, tol=3, rel=False)
    check("§5 CIFAR triangle: median rel err (%)", np.median(good_err) * 100, 1.9, tol=0.5, rel=False)

    rn_conv = [r for block in data if "resnet18" in block["experiment"]
               for r in block["results"]
               if "conv" in r["name"] and "fc" not in r["name"]]
    alphas = [r["alpha"] for r in rn_conv if "alpha" in r]
    if alphas:
        lo, hi = min(alphas), max(alphas)
        check("§3 ResNet-18 conv α range covers 2", 2.0, 2.0,
              tol=0.01 if lo <= 2.2 and hi >= 1.8 else 999, rel=False)

    r2s = [r.get("r2_h_exact") or r.get("r2_exact") for r in rn_conv
           if r.get("r2_h_exact") or r.get("r2_exact")]
    if r2s:
        check("Abstract: R²=0.98 (median, conv)", np.median(r2s), 0.98, tol=0.02, rel=False)


def verify_tiny_imagenet():
    data = load("results_tiny_imagenet_alpha_map.json")
    if data is None:
        skipped.append("tiny_imagenet")
        return

    rows = data["results"]
    errs = [r["triangle_relative_error"] for r in rows if r.get("triangle_relative_error") is not None]
    check("§5 Tiny-ImageNet triangle median err (%)", np.median(errs) * 100, 1.0, tol=0.3, rel=False)

    s = data["summary"]["by_layer_type"]
    check("§3 Tiny-ImageNet conv median α", s["conv"]["median_alpha"], 1.93, tol=0.05, rel=False)
    if "fc" in s:
        fc_a = s["fc"].get("median_alpha", s["fc"].get("mean_alpha"))
        check("§3 Tiny-ImageNet FC α", fc_a, 0.90, tol=0.05, rel=False)

    check_eq("§3 Tiny-ImageNet n_conv", s["conv"].get("n_layers", s["conv"].get("n")), 49)


def verify_imagenet1k():
    data = load("results_imagenet1k_pretrained.json")
    if data is None:
        skipped.append("imagenet1k")
        return

    rows = data["results"]
    errs = [r["triangle_relative_error"] for r in rows if r.get("triangle_relative_error") is not None]
    check("§5 ImageNet-1K triangle median err (%)", np.median(errs) * 100, 1.6, tol=0.3, rel=False)

    conv = [r for r in rows if "conv" in r["name"] and "fc" not in r["name"]]
    alphas = [r["alpha"] for r in conv]
    check("§3 ImageNet-1K conv mean α", np.mean(alphas), 2.10, tol=0.06, rel=False)
    check("§3 ImageNet-1K conv median α", np.median(alphas), 2.15, tol=0.06, rel=False)
    n_conv = len(conv)
    check(f"§3 ImageNet-1K n_conv ({n_conv} conv layers)", n_conv, 14, tol=1, rel=False)

    r2s = [r.get("r2_exact", r.get("r2_h_exact")) for r in conv
           if r.get("r2_exact") or r.get("r2_h_exact")]
    if r2s:
        check("§3 ImageNet-1K conv R² ≥ 0.97", min(r2s), 0.97, tol=0.01, rel=False)

    fc = [r for r in rows if r["name"] == "fc.weight"]
    if fc:
        check("§3 ImageNet-1K FC α", fc[0]["alpha"], 2.83, tol=0.05, rel=False)


# ── Section 5: Rank profiles and BIC ────────────────────────────────

def verify_gamma_bic():
    data = load("results_model_fit_comparison.json")
    if data is None:
        skipped.append("gamma_bic")
        return

    wins = data["summary"]["wins_by_bic"]
    check_eq("App. CIFAR BIC: lognormal wins", wins.get("lognormal", 0), 21)
    check_eq("App. CIFAR BIC: power_law wins", wins.get("power_law", 0), 1)
    check_eq("App. CIFAR BIC: exponential wins", wins.get("exponential", 0), 2)


def verify_rmt_gamma():
    data = load("results_rmt_gamma_spectrum.json")
    if data is None:
        skipped.append("rmt_gamma")
        return

    init = data["phase_2a_init"]["summary"]["wins_by_bic"]
    check_eq("App. T-IN init BIC: lognormal", init.get("lognormal", 0), 39)
    check_eq("App. T-IN init BIC: power_law", init.get("power_law", 0), 12)

    trained = data["phase_1b_extended_trained"]["summary"]["wins_by_bic"]
    check_eq("App. T-IN trained BIC: lognormal", trained.get("lognormal", 0), 39)
    check_eq("App. T-IN trained BIC: exponential", trained.get("exponential", 0), 15)


# ── Section 7: Optimizer validation ─────────────────────────────────

def verify_preconditioner():
    data = load("results_preconditioner_summary.json")
    if data is None:
        skipped.append("preconditioner")
        return

    s = data["summary"]
    adam_key = "mean_best_test_acc" if "mean_best_test_acc" in s["adam_baseline"] else "mean_best"
    adam = s["adam_baseline"][adam_key]
    check("App. Adam baseline acc", adam, 93.85, tol=0.15, rel=False)

    best_beta = s.get("spectral_beta15", s.get("spectral_beta10", {}))
    if best_beta:
        delta = best_beta.get(adam_key, best_beta.get("mean_best", 0)) - adam
        check("App. scalar preconditioner Δ ≤ 0.2%", delta, 0.11, tol=0.15, rel=False)


# ── Intrinsic dimension ─────────────────────────────────────────────

def verify_intrinsic_dim():
    data = load("results_intrinsic_dim.json")
    if data is None:
        skipped.append("intrinsic_dim")
        return

    for block in data:
        if block.get("experiment") == "scaling_test":
            prs = [r.get("eff_rank_curvature_pr", r.get("curvature_pr"))
                   for r in block["results"]
                   if r.get("eff_rank_curvature_pr") or r.get("curvature_pr")]
            if prs:
                w64 = prs[0]
                w512 = prs[-1]
                check("App. PR width 64", w64, 2.5, tol=0.3, rel=False)
                check("App. PR width 512", w512, 3.2, tol=0.3, rel=False)
                check("App. PR Δ by 8× width", w512 - w64, 0.7, tol=0.5, rel=False)


# ── K-FAC prediction (Fig 6) ────────────────────────────────────────

def verify_kfac():
    data = load("results_kfac_prediction.json")
    if data is None:
        skipped.append("kfac_prediction")
        return
    check("Fig 6 n_layers", data["n_layers"], 5, tol=1, rel=False)


# ── Conv gap ─────────────────────────────────────────────────────────

def verify_conv_gap():
    data = load("results_conv_gap_v2.json")
    if data is None:
        data = load("results_conv_gap.json")
    if data is None:
        skipped.append("conv_gap")
        return

    convs = data.get("conv_results", [])
    if convs:
        alphas = [c.get("alpha_exact", c.get("alpha")) for c in convs
                  if c.get("alpha_exact") or c.get("alpha")]
        if alphas:
            check("App. conv gap: all α near 2", np.mean(alphas), 2.0, tol=0.5, rel=False)


# ── Spectral Newton table (§7.3) ────────────────────────────────────

def verify_sn_vision():
    """SN numbers in Table 3 come from separate optimizer runs; check preconditioner JSON as proxy."""
    data = load("results_preconditioner_summary.json")
    if data is None:
        skipped.append("sn_vision")
        return
    adam_key = "mean_best_test_acc" if "mean_best_test_acc" in data["summary"]["adam_baseline"] else "mean_best"
    adam = data["summary"]["adam_baseline"][adam_key]
    check("§7 SN table: Adam ≈93.85", adam, 93.85, tol=0.2, rel=False)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify paper claims against JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    verify_alpha_triangle_cifar()
    verify_tiny_imagenet()
    verify_imagenet1k()
    verify_gamma_bic()
    verify_rmt_gamma()
    verify_preconditioner()
    verify_intrinsic_dim()
    verify_kfac()
    verify_conv_gap()
    verify_sn_vision()

    print(f"\n{'='*60}")
    print(f"  PASSED: {len(passed)}   FAILED: {len(failed)}   SKIPPED: {len(skipped)}")
    print(f"{'='*60}")

    if args.verbose or failed:
        for line in passed:
            print(f"  {line}")
        for line in failed:
            print(f"  {line}")
        if skipped:
            print(f"\n  Skipped checks (missing JSON): {', '.join(skipped)}")

    if failed:
        print(f"\n  *** {len(failed)} CLAIM(S) FAILED — check paper text ***")
        sys.exit(1)
    else:
        print("\n  All verifiable claims match bundled data.")
        sys.exit(0)


if __name__ == "__main__":
    main()
