"""
neuronumba/observables/edge_centric_metastability.py
------------------------------
Edge-Centric Metastability (ECM) observable for NeuroNumba.

ECM measures the temporal variability of instantaneous functional
connectivity patterns — a sensitive index of brain dynamics that goes
beyond static FC.

Reference:
    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

    Deco et al. (2017). The dynamics of resting fluctuations in the brain:
    metastability and its dynamical cortical core. Scientific Reports.

Usage
-----
Two calling styles, matching the NeuroNumba ObservableFMRI convention:

    # Style 1 — attribute-based (generic Observable pattern):
    obs = ECM()
    obs.bold_signal = bold   # shape (T, N), already z-scored
    result = obs.compute()   # {'ECM': float}

    # Style 2 — backwards-compatible entry point:
    obs = ECM()
    result = obs.from_fmri(bold)   # {'ECM': float}

Input convention
----------------
bold_signal : np.ndarray, shape (T, N)
    Rows = timepoints, columns = ROIs — matching the NeuroNumba convention
    (note: Neuroreduce uses the transposed convention N × T internally).
    The signal should already be z-scored across time before calling compute().
    If it is not z-scored, ECM values will not be comparable across subjects.
"""

import numpy as np
from numpy import linalg as LA

from neuronumba.observables.base_observable import ObservableFMRI


class ECM(ObservableFMRI):
    """
    Edge-Centric Metastability (ECM) observable.

    ECM = differential entropy of the FCD matrix variance, approximated
    assuming a Gaussian distribution over the lower-triangle values:

        H = 0.5 * log(2π * Var(FCD_lower_tri)) + 0.5

    The FCD (functional connectivity dynamics) matrix is the (T × T)
    matrix of cosine similarities between instantaneous FC patterns:

        FCD[t1, t2] = cosine_similarity(edge_t1, edge_t2)

    where  edge_t = lower_triangle(outer(bold[t], bold[t]))  is the
    instantaneous co-activation pattern at timepoint t.

    Higher ECM → more dynamic, less stationary brain activity.

    Parameters
    ----------
    bold_signal : np.ndarray, shape (T, N)
        BOLD signal, rows = timepoints, columns = ROIs.
        Should be z-scored across time (axis=0) before calling compute().

    Returns
    -------
    dict with key 'ECM' → float
        ECM value H. Typically negative (log of a small variance).
    """

    def _compute_from_fmri(self, bold_signal: np.ndarray) -> dict:
        """
        Compute ECM from a validated (T, N) BOLD signal.

        Parameters
        ----------
        bold_signal : np.ndarray, shape (T, N)
            Already validated by ObservableFMRI._compute().
            Should be z-scored across time before calling this.

        Returns
        -------
        dict
            {'ECM': float}
        """
        T, N = bold_signal.shape

        # ── Build edge matrix: (n_edges × T) ─────────────────────────────────
        # Each column t is the instantaneous co-activation pattern:
        #   edge_t = lower_triangle( outer(bold[t], bold[t]) )  shape: (N*(N-1)/2,)
        # Stacking across time gives EdgesL with shape (n_edges, T).
        #
        # Assumption: the outer product outer(bold[t], bold[t]) captures
        # the instantaneous pairwise co-activation between all ROI pairs.
        # The lower triangle (excluding diagonal) gives n_edges = N*(N-1)/2
        # directed edge weights at each timepoint.
        i_n, j_n = np.tril_indices(N, k=-1)   # lower-triangle indices of (N, N)
        n_edges   = len(i_n)                    # = N*(N-1)//2

        EdgesL = np.zeros((n_edges, T))
        for t in range(T):
            outer_t      = np.outer(bold_signal[t], bold_signal[t])   # (N, N)
            EdgesL[:, t] = outer_t[i_n, j_n]                          # (n_edges,)

        # ── FCD matrix: (T × T) cosine similarity between edge patterns ───────
        # FCDQ[t1, t2] = (EdgesL[:,t1] · EdgesL[:,t2]) /
        #                (||EdgesL[:,t1]|| * ||EdgesL[:,t2]||)
        #
        # Assumption: cosine similarity normalises for differences in overall
        # activation magnitude across timepoints, matching the original Deco
        # lab implementation.
        norms = LA.norm(EdgesL, axis=0, keepdims=True)   # (1, T)
        norms = np.where(norms == 0, 1.0, norms)         # avoid division by zero
        FCDQ  = (EdgesL.T @ EdgesL) / (norms.T @ norms)  # (T, T)

        # ── ECM = differential entropy of lower-triangle FCD variance ─────────
        # Lower triangle (off-diagonal) only — the diagonal is trivially 1.
        i_lt, j_lt = np.tril_indices(T, k=-1)
        var_fcd     = np.var(FCDQ[i_lt, j_lt])

        # Differential entropy of Gaussian with variance σ²:
        #   H = 0.5 * log(2πe * σ²) = 0.5 * log(2π * σ²) + 0.5
        # Assumption: Gaussian approximation, matching the original code.
        ecm = float(0.5 * np.log(2 * np.pi * var_fcd) + 0.5)

        return {'ECM': ecm}
