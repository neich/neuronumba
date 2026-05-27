"""Run paired (hand-written, DSL) simulations and show that they agree.

For each pair we:
  1. Run the hand-written model and the DSL spec on the same connectivity,
     params, and deterministic integrator.
  2. Plot the two trajectories overlaid (visually identical).
  3. Plot the difference, which should sit at machine-precision noise.

Run:  PYTHONPATH=src python examples/dsl_models/compare_simulations.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.simulator.integrators.euler import EulerDeterministic
from neuronumba.simulator.models import Hopf, Deco2014

# Make the dsl_models package importable when run as a script. The script
# lives inside the package, so we add its parent directory to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsl_models import HopfDSL, Deco2014DSL  # noqa: E402


def make_W(n: int = 8, seed: int = 0) -> np.ndarray:
    """Symmetric, zero-diagonal connectivity normalized to max 1.0."""
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    W /= W.max()
    return W


def run(model_cls, weights, obs_var: str, *, t_max: float = 1000.0,
        dt: float = 0.1, sampling_period: float = 1.0, **model_kwargs):
    """Run a deterministic simulation and return the observed signal."""
    np.random.seed(0)  # simulate_nodelay draws random tract lengths internally
    model = model_cls(**model_kwargs)
    integrator = EulerDeterministic(dt=dt)
    return simulate_nodelay(
        model, integrator, weights, obs_var,
        sampling_period=sampling_period,
        t_max_neuronal=t_max,
        t_warmup=0.0,
    )


def plot_pair(ax_overlay, ax_diff, ref, dsl, *, title: str, n_show: int = 4):
    """Overlay (left axis) + difference (right axis) for one model pair."""
    t = np.arange(ref.shape[0])
    for i in range(min(n_show, ref.shape[1])):
        ax_overlay.plot(t, ref[:, i], color="black", lw=1.0, alpha=0.6,
                        label="hand-written" if i == 0 else None)
        ax_overlay.plot(t, dsl[:, i], color="tab:red", lw=0.8, ls="--",
                        label="DSL" if i == 0 else None)
    ax_overlay.set_title(f"{title} — trajectories (first {n_show} ROIs)")
    ax_overlay.set_xlabel("sample index")
    ax_overlay.legend(loc="upper right", fontsize=9)

    diff = ref - dsl
    max_abs = np.abs(diff).max()
    ax_diff.plot(t, diff, lw=0.6)
    ax_diff.set_title(f"{title} — difference (max |Δ| = {max_abs:.2e})")
    ax_diff.set_xlabel("sample index")
    ax_diff.axhline(0.0, color="black", lw=0.5, alpha=0.3)


def main():
    n_rois = 8
    W = make_W(n=n_rois)

    # Hopf: diffusive coupling, observe state variable x.
    hopf_ref = run(Hopf, W, obs_var="x", g=0.5)
    hopf_dsl = run(HopfDSL, W, obs_var="x", g=0.5)

    # Deco2014: linear coupling, observe state variable S_e.
    deco_ref = run(Deco2014, W, obs_var="S_e", g=0.5, auto_fic=False)
    deco_dsl = run(Deco2014DSL, W, obs_var="S_e", g=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plot_pair(axes[0, 0], axes[0, 1], hopf_ref, hopf_dsl, title="Hopf")
    plot_pair(axes[1, 0], axes[1, 1], deco_ref, deco_dsl, title="Deco2014")
    fig.suptitle("DSL vs hand-written: deterministic Euler, n=8 ROIs", y=1.0)
    fig.tight_layout()

    print(f"Hopf      max |Δ|: {np.abs(hopf_ref - hopf_dsl).max():.3e}")
    print(f"Deco2014  max |Δ|: {np.abs(deco_ref - deco_dsl).max():.3e}")
    plt.show()


if __name__ == "__main__":
    main()
