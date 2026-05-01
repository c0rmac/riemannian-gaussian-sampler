#!/usr/bin/env python3
"""Visualise anisotropic SO(d) orientation frames from Phase II sampling.

Usage:
    python3 visualise_so_samples.py <samples.h5>

Produces a PNG alongside the input file showing:
  - Per-coordinate histograms of the first and second column of Q.
  - A 2-D scatter of first-column components in the low-precision vs
    high-precision subspaces, making the anisotropy visually clear.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <samples.h5>", file=sys.stderr)
        sys.exit(1)

    hdf5_path = Path(sys.argv[1])
    out_path  = hdf5_path.with_suffix(".png")

    with h5py.File(hdf5_path, "r") as f:
        d         = int(f.attrs["d"])
        N         = int(f.attrs["N"])
        gamma_min = float(f.attrs["gamma_min"])
        gamma_max = float(f.attrs["gamma_max"])
        gamma     = f["gamma"][:]    # (d, d)
        frames    = f["frames"][:]   # (N, d, d) row-major
        angles    = f["angles"][:]   # (N, m)

    gamma_diag = np.diag(gamma)
    half       = d // 2
    low_idx    = list(range(half))        # indices with gamma_low
    high_idx   = list(range(half, d))     # indices with gamma_high

    # frames[n, row, col] — first column is frames[:, :, 0]
    q1 = frames[:, :, 0]   # (N, d) — first column of each frame

    # -------------------------------------------------------------------------
    # Figure layout: top row = per-coordinate histograms of q1
    #                bottom-left = scatter low vs high projection
    #                bottom-right = angle distribution histogram
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(4 * d, 8), constrained_layout=True)
    fig.suptitle(
        f"SO({d}) anisotropic orientation frames  (N={N})\n"
        f"Γ diagonal ≈ [{gamma_diag[0]:.2f}×{half}, "
        f"{gamma_diag[-1]:.2f}×{d-half}]   "
        f"γ_min={gamma_min:.3f}  γ_max={gamma_max:.3f}",
        fontsize=12,
    )

    gs = fig.add_gridspec(2, max(d, 2))

    # Row 0: per-coordinate histograms of q1
    for i in range(d):
        ax = fig.add_subplot(gs[0, i])
        color = "#4C72B0" if i < half else "#C44E52"
        label = f"γ={gamma_diag[i]:.2f}\n({'low' if i < half else 'high'})"
        ax.hist(q1[:, i], bins=40, color=color, alpha=0.85, density=True)
        ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel(f"$q_{{1,{i+1}}}$", fontsize=11)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-1.05, 1.05)

    # Row 1, col 0..half-1: 2-D scatter — low-precision projection vs
    #                                      high-precision projection
    ax_scatter = fig.add_subplot(gs[1, : max(d // 2, 1)])
    low_proj  = np.sqrt(np.sum(q1[:, low_idx]  ** 2, axis=1))   # ‖q1_low‖
    high_proj = np.sqrt(np.sum(q1[:, high_idx] ** 2, axis=1))   # ‖q1_high‖

    ax_scatter.hexbin(low_proj, high_proj, gridsize=30,
                      cmap="Blues", mincnt=1)
    ax_scatter.set_xlabel(f"‖$q_1$ projected onto low-Γ subspace‖", fontsize=11)
    ax_scatter.set_ylabel(f"‖$q_1$ projected onto high-Γ subspace‖", fontsize=11)
    ax_scatter.set_title("Projection norms (q₁)", fontsize=11)
    # Unit circle constraint: low² + high² = 1
    t = np.linspace(0, 1, 200)
    ax_scatter.plot(t, np.sqrt(np.clip(1 - t**2, 0, 1)),
                    "k--", lw=1, alpha=0.6, label="‖q₁‖=1")
    ax_scatter.legend(fontsize=9)

    # Row 1, col half..end: angle distribution
    m = angles.shape[1]
    ax_ang = fig.add_subplot(gs[1, max(d // 2, 1) :])
    for j in range(m):
        ax_ang.hist(angles[:, j], bins=50, alpha=0.6, density=True,
                    label=f"θ_{j+1}")
    ax_ang.set_xlabel("angle θ (radians)", fontsize=11)
    ax_ang.set_title("Phase I angle distribution", fontsize=11)
    ax_ang.legend(fontsize=9)

    plt.savefig(str(out_path), dpi=150)
    print(f"Saved visualisation to {out_path}")


if __name__ == "__main__":
    main()
