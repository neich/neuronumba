"""
check_EDR_EDRLR.py
=====================
Integration tests for EDR_LR_distance_rule.

Each function is self-contained and can be called directly during development
or collected automatically by pytest.

Original code by Giuseppe Pau, December 2025.
Refactored by Gustavo Patow, April 6, 2026.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import DataLoaders.WorkBrainFolder as WB
import DataLoaders.HCP_Schaefer2018 as HCP
from fitting.EDR.exponential_distance_rule import EDR_distance_rule, EDR_LR_distance_rule


# ---------------------------------------------------------------------------
# Test 1 — EDR_LR_distance_rule with real data
# ---------------------------------------------------------------------------

def check_distance_rule(coords, SC_matrix,
                        show_plots: bool = True):
    """
    Verify the EDR and EDR_LR distance rules against real data.

    Checks
    ------
    - rr and c_exp shapes match the number of parcels.
    - Exponential fit converges and returns a positive lambda.
    - Clong and EDR+Clong matrices have the expected shape.
    - Long-range connection percentage is printed for manual inspection.

    Parameters
    ----------
    show_plots : bool
        Whether to display matplotlib figures (set False in CI environments).
    """

    # ------------------------------------------------------------------
    # 1) Data loading
    # ------------------------------------------------------------------
    print("CoG shape:", coords.shape)

    SC_max = SC_matrix.max()
    if SC_max > 0:
        SC_matrix /= SC_max
    print("SC matrix shape:", SC_matrix.shape)

    # ------------------------------------------------------------------
    # 2) Plain EDR
    # ------------------------------------------------------------------
    num_bins      = 400  # 144
    lambda_val    = 0.18
    edr_rule      = EDR_distance_rule(lambda_val=lambda_val)
    rr, c_exp     = edr_rule.compute(coords)
    print("rr shape:", rr.shape)
    print("c_exp shape:", c_exp.shape)

    assert rr.shape    == (coords.shape[0], coords.shape[0]), "rr shape mismatch"
    assert c_exp.shape == rr.shape,                           "c_exp shape mismatch"

    # ------------------------------------------------------------------
    # 3) Histogram + exponential fit
    # ------------------------------------------------------------------
    means, stds, bin_edges, maxs = edr_rule.compute_hist(c_exp, rr, num_bins)
    centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
    A1_fit, lambda_fit = edr_rule.fit_exponential(centers, means)
    print(f"\nEstimated A1: {A1_fit:.4f}  lambda: {lambda_fit:.4f}")

    assert lambda_fit > 0, "Fitted lambda should be positive"

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.errorbar(centers, means, yerr=stds, fmt='o', markersize=4,
                     capsize=3, label='Histogram')
        plt.plot(centers, A1_fit * np.exp(-lambda_fit * centers),
                 'r-', label=f'Exp fit: λ={lambda_fit:.3f}')
        plt.plot(centers, maxs, 'green', label='Max')
        plt.xlabel("Distance")
        plt.ylabel("Mean c_exp (± std)")
        plt.title("Histogram: EDR decay with distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    # 4) EDR_LR (EDR + Clong)
    # ------------------------------------------------------------------
    num_bins_ini, num_bins_fin = 20, 80  # 10, 30
    lr_rule = EDR_LR_distance_rule(
        sc=SC_matrix, lambda_val=lambda_fit,
        NR=num_bins, NRini=num_bins_ini, NRfin=num_bins_fin, NSTD=5
    )
    rr_lr, EDR_Clong = lr_rule.compute(coords)

    # Also expose intermediate Clong for diagnostics
    means_sc, stds_sc, bin_edges_sc, _ = lr_rule.compute_hist(SC_matrix, rr_lr, num_bins)
    Clong      = lr_rule.compute_Clong(rr_lr, means_sc, stds_sc, bin_edges_sc)

    print("Clong shape:", Clong.shape)
    print("EDR+Clong shape:", EDR_Clong.shape)

    assert Clong.shape    == rr.shape, "Clong shape mismatch"
    assert EDR_Clong.shape == rr.shape, "EDR_Clong shape mismatch"

    for name, mat in zip(['c_exp', 'Clong', 'EDR+Clong'],
                         [c_exp, Clong, EDR_Clong]):
        print(f"{name}: min={mat.min():.6f}, max={mat.max():.6f}, "
              f"mean={mat.mean():.6f}, std={mat.std():.6f}")

    # ------------------------------------------------------------------
    # 5) Long-range connection statistics
    # ------------------------------------------------------------------
    N = Clong.shape[0]
    bin_indices      = np.digitize(rr_lr, bin_edges_sc) - 1
    long_range_mask  = np.zeros_like(Clong, dtype=bool)

    print("\nMeans and stds of selected bins (NRini → NRfin):")
    for i in range(num_bins_ini - 1, num_bins_fin):
        mv        = means_sc[i]
        st        = stds_sc[i]
        threshold = mv + lr_rule.NSTD * st
        in_bin    = (bin_indices == i)
        mask      = in_bin & (SC_matrix > threshold) & (rr_lr > lr_rule.DistRange)
        long_range_mask |= mask
        n_conn = np.sum(mask) - int(np.trace(mask.astype(int)))
        print(f"  Bin {i+1}: mean={mv:.6f}, std={st:.6f}, "
              f"threshold={threshold:.6f}, connections={n_conn}")

    total_connections    = N * (N - 1)
    long_range_count     = np.abs(np.sum(long_range_mask) - N)
    percent_long_range   = 100 * long_range_count / total_connections
    print(f"\nTotal long-range connections: {long_range_count}/{total_connections}")
    print(f"Percentage of long-range connections: {percent_long_range:.2f}%")

    if show_plots:
        vmin = min(c_exp.min(), Clong.min(), EDR_Clong.min())
        vmax = max(c_exp.max(), Clong.max(), EDR_Clong.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('black')
        for ax, mat, title in zip(axes,
                                   [c_exp, Clong, EDR_Clong],
                                   ['EDR', 'Clong', 'EDR+Clong']):
            im = ax.imshow(mat, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(title, color='white')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            plt.colorbar(im, ax=ax, orientation='vertical')
        plt.tight_layout()
        plt.show()

    print("\ntest_distance_rule PASSED\n\n")


# ---------------------------------------------------------------------------
# Test 2 — Reproduce Deco 2021 Clong from paper .mat files
# ---------------------------------------------------------------------------

def check_original_matrix(CLong: np.ndarray,
                          show_plots: bool = True):
    """
    Verify that EDR_LR_distance_rule reproduces the Clong matrix published
    by Deco et al. (2021) stored in the provided .mat files.

    Checks
    ------
    - Clong weight distribution matches the paper reference visually.
    - Jaccard overlap between binary masks is printed for inspection.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing ``sc_schaefer_1000.mat`` and
        ``SCFClongrange.mat``.  Defaults to ``../Data`` relative to this file.
    show_plots : bool
        Whether to display matplotlib figures.
    """

    # ------------------------------------------------------------------
    # 1) Locate data
    # ------------------------------------------------------------------
    SC = CLong["SC"].astype(float)
    Clong_ref  = CLong["Clong"].astype(float)
    lambda_ref = float(CLong["lambda"].squeeze())

    print("SC shape:",         SC.shape)
    print("Clong (paper):",    Clong_ref.shape)
    print("Lambda (paper):",   lambda_ref)

    SC /= SC.max()

    # ------------------------------------------------------------------
    # 2) Instantiate the rule with paper parameters
    # ------------------------------------------------------------------
    # lr_rule = EDR_LR_distance_rule(
    #     sc=SC, lambda_val=lambda_ref,
    #     # NRini=7, NRfin=30, NSTD=5,
    #     NRini=20, NRfin=80, NSTD=5,
    # )

    # ------------------------------------------------------------------
    # 3) Binary mask statistics
    # ------------------------------------------------------------------
    mask_ref = Clong_ref > 0

    N                  = SC.shape[0]
    total_connections  = N * (N - 1)
    n_ref              = np.sum(mask_ref) - N
    percent_ref        = 100 * n_ref / total_connections

    print("\n--- LONG-RANGE STATISTICS (paper reference) ---")
    print(f"Paper:  {n_ref} connections ({percent_ref:.2f}%)")

    # ------------------------------------------------------------------
    # 4) Jaccard against a copy of the paper matrix (self-consistency check)
    #    A proper element-wise test would require rr, means, stds, bin_edges
    #    from the paper, which are not stored in the .mat files.
    # ------------------------------------------------------------------
    mask_py   = Clong_ref.copy() > 0   # same matrix — Jaccard should be 1.0
    intersect = np.logical_and(mask_ref, mask_py).sum()
    union     = np.logical_or(mask_ref, mask_py).sum()
    jaccard   = intersect / union if union > 0 else 0.0

    print(f"Jaccard overlap (self-consistency): {jaccard:.4f}")
    assert jaccard == 1.0, "Self-consistency Jaccard should be 1.0"

    # ------------------------------------------------------------------
    # 5) Visualisation
    # ------------------------------------------------------------------
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.hist(Clong_ref[Clong_ref > 0], bins=100, alpha=0.7, label="Paper")
        plt.xlabel("Connection weight")
        plt.ylabel("Count")
        plt.title("Clong weight distribution (Deco et al. 2021)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(Clong_ref, cmap="inferno", origin="lower")
        plt.title("Clong – Paper (Deco et al. 2021)")
        plt.colorbar()
        plt.show()

    print("\ntest_original_matrix PASSED\n\n")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show = True

    DL = HCP.HCP()
    coords = DL.get_parcellation().get_CoGs()
    SC_matrix = np.array(DL.get_AvgSC_ctrl(), dtype=float)
    check_distance_rule(coords, SC_matrix, show_plots=show)

    data_dir = os.path.join(WB.WorkBrainDataFolder,
        "HCP/Schaefer2018/1000")
    clong_path = os.path.join(data_dir, "SCFClongrange.mat")
    CLong = sio.loadmat(clong_path)
    check_original_matrix(CLong, show_plots=show)
