from abc import abstractmethod
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from neuronumba.basic.attr import HasAttr, Attr


class DistanceRule(HasAttr):
    """
    Base class for distance-based connectivity rules.

    Provides common attributes (lambda_val) and utility methods for
    histogram computation and exponential fitting.  Subclasses must implement
    ``compute_Dist_Rule``.

    Code by Giuseppe Pau, December 27, 2025.
    Refactored by Gustavo Patow, January 13, 2025.
    """

    # Exponential decay parameter lambda (fitted or externally provided)
    lambda_val = Attr(required=False, default=None)

    def compute(self, cog_dist: np.ndarray, lambda_val: float = None):
        """
        Resolve lambda_val and delegate to ``compute_Dist_Rule``.
        """
        if lambda_val is None:
            if self.lambda_val is None:
                raise ValueError(
                    "lambda_val is not set. "
                    "Run fit_exponential or provide it explicitly."
                )
            lambda_val = self.lambda_val
        return self.compute_Dist_Rule(cog_dist, lambda_val)

    @abstractmethod
    def compute_Dist_Rule(self, cog_dist: np.ndarray, lambda_val: float):
        """
        Compute a distance-based connectivity rule.

        Parameters
        ----------
        cog_dist : np.ndarray
            Coordinates of centres of gravity (N x 3).
        lambda_val : float
            Exponential decay parameter.

        Returns
        -------
        rr : np.ndarray
            Pairwise distance matrix (N x N).
        c_exp : np.ndarray
            Resulting connectivity matrix (N x N).
        """
        pass

    def compute_hist(self, C: np.ndarray, rr: np.ndarray, NR: int):
        """
        Compute statistics of a connectivity matrix binned by distance.

        Parameters
        ----------
        C : np.ndarray
            Connectivity matrix (e.g. SC or EDR).
        rr : np.ndarray
            Pairwise distance matrix.
        NR : int
            Number of distance bins.

        Returns
        -------
        means : np.ndarray
            Mean connectivity per distance bin.
        stds : np.ndarray
            Standard deviation per distance bin.
        bin_edges : np.ndarray
            Distance bin edges.
        maxs : np.ndarray
            Maximum connectivity value per bin.
        """
        rr_flat = rr.flatten()
        C_flat = C.flatten()
        min_dist = rr_flat.min()
        max_dist = rr_flat.max()

        means, bin_edges, _ = binned_statistic(
            rr_flat, C_flat, statistic='mean',
            bins=NR, range=(min_dist, max_dist)
        )
        stds, _, _ = binned_statistic(
            rr_flat, C_flat, statistic='std',
            bins=NR, range=(min_dist, max_dist)
        )
        maxs, _, _ = binned_statistic(
            rr_flat, C_flat, statistic='max',
            bins=NR, range=(min_dist, max_dist)
        )
        return means, stds, bin_edges, maxs

    def fit_exponential(self, centers: np.ndarray, means: np.ndarray,
                        start_index: int = 24):
        """
        Fit an exponential decay to the binned connectivity profile.

        Model:  C(d) = A1 * exp(-lambda * d)

        Parameters
        ----------
        centers : np.ndarray
            Bin centre distances.
        means : np.ndarray
            Mean connectivity values per bin.
        start_index : int
            Index from which the fit begins (ignores very short distances).

        Returns
        -------
        A1_fit : float
            Fitted amplitude constant.
        lambda_fit : float
            Fitted exponential decay parameter.
        """
        def expfunc(x, A1, A2):
            return A1 * np.exp(-A2 * x)

        xdata = centers[start_index:]
        ydata = means[start_index:]

        popt, _ = curve_fit(expfunc, xdata, ydata,
                            p0=[0.15, 0.18],
                            bounds=([-100, -100], [100, 100]))
        return popt[0], popt[1]   # A1_fit, lambda_fit


# ---------------------------------------------------------------------------
# EDR — plain Exponential Distance Rule
# ---------------------------------------------------------------------------

class EDR_distance_rule(DistanceRule):
    """
    Exponential Distance Rule (EDR).

    Computes pairwise Euclidean distances and the exponential-decay matrix:

        C_ij = exp(-lambda * ||r_i - r_j||)

    with the diagonal set to 1.
    """

    def __init__(self, lambda_val: float = None):
        super().__init__(lambda_val=lambda_val)

    def compute_Dist_Rule(self, cog_dist: np.ndarray, lambda_val: float):
        """
        Parameters
        ----------
        cog_dist : np.ndarray
            Coordinates of centres of gravity, shape (N, 3).
        lambda_val : float
            Exponential decay parameter.

        Returns
        -------
        rr : np.ndarray
            Pairwise distance matrix, shape (N, N).
        c_exp : np.ndarray
            EDR connectivity matrix, shape (N, N).
        """
        N = cog_dist.shape[0]
        rr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(cog_dist[i] - cog_dist[j])

        c_exp = np.exp(-lambda_val * rr)
        np.fill_diagonal(c_exp, 1.0)
        return rr, c_exp


# ---------------------------------------------------------------------------
# EDR_LR — EDR + long-range connections (Deco et al., Curr. Biol. 2021)
# ---------------------------------------------------------------------------

class EDR_LR_distance_rule(DistanceRule):
    """
    Exponential Distance Rule extended with rare long-range connections.

    Reference
    ---------
    Deco G. et al., "Rare long-range cortical connections enhance human
    information processing", Current Biology 31(20), 4436-4448.e5 (2021).
    https://doi.org/10.1016/j.cub.2021.07.064

    This rule augments the plain EDR matrix with a long-range connectivity
    matrix (Clong).  A connection (i,j) enters Clong when all of the
    following hold:

    - its distance bin falls within [NRini, NRfin]  (1-based, inclusive)
    - rr[i,j] > DistRange
    - sc[i,j] > mean(bin) + NSTD * std(bin)

    The combined matrix is:

        EDR_Clong = clip(exp(-lambda * rr) + Clong, 0, 1)

    with the diagonal set to 1.

    Intended usage
    --------------
    >>> rule = EDR_LR_distance_rule(sc=sc_norm, NRini=7, NRfin=30, NSTD=5)
    >>> rule.compute(cog_dist=COGs, lambda_val=0.18)


    Parameters
    ----------
    sc : np.ndarray, shape (N, N)
        Normalised structural connectivity matrix (values in [0, 1]).
    NR : int
        Number of distance bins for the SC histogram (default 144).
    NRini : int
        First 1-based bin index for long-range candidates (default 7).
    NRfin : int
        Last 1-based bin index for long-range candidates (default 30).
    DistRange : float
        Minimum distance for a connection to be considered long-range
        (default 0).
    NSTD : float
        Standard-deviation threshold (default 3, as in Deco 2021).

    Original MATLAB code by Gustavo Deco, 2021.
    Integrated by Giuseppe Pau, December 2025.
    Refactored by Gustavo Patow, 2025.
    """

    sc        = Attr(required=True)
    NR        = Attr(required=False, default=400)  # 144
    NRini     = Attr(required=False, default=20)  # 7
    NRfin     = Attr(required=False, default=80)  # 30
    DistRange = Attr(required=False, default=0)
    # Number of standard deviations used as threshold for long-range connections
    NSTD      = Attr(required=False, default=3)

    def __init__(self, sc: np.ndarray = None, lambda_val: float = None,
                 # NR: int = 144, NRini: int = 7, NRfin: int = 30,
                 NR: int = 400, NRini: int = 20, NRfin: int = 80,
                 DistRange: float = 0, NSTD: float = 3):
        super().__init__(lambda_val=lambda_val, NSTD=NSTD)
        if sc is not None:
            self.sc = sc / np.max(sc)
        self.NR        = NR
        self.NRini     = NRini
        self.NRfin     = NRfin
        self.DistRange = DistRange

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rr(self, cog_dist: np.ndarray) -> np.ndarray:
        """Pairwise Euclidean distance matrix from CoG coordinates."""
        N = cog_dist.shape[0]
        rr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(cog_dist[i] - cog_dist[j])
        return rr

    def compute_Clong(self, rr: np.ndarray,
                      means: np.ndarray, stds: np.ndarray,
                      bin_edges: np.ndarray,
                      ) -> np.ndarray:
        """
        Build the long-range connectivity matrix (Clong).

        Parameters
        ----------
        rr : np.ndarray
            Pairwise distance matrix, shape (N, N).
        means, stds : np.ndarray
            Per-bin mean and std from ``compute_hist``.
        bin_edges : np.ndarray
            Bin edges from ``compute_hist``.

        Returns
        -------
        Clong : np.ndarray, shape (N, N)
        """
        N = rr.shape[0]
        Clong = np.zeros((N, N))

        # Convert 1-based MATLAB indices → 0-based Python
        NRini_py = self.NRini - 1
        NRfin_py = self.NRfin - 1

        bin_indices = np.digitize(rr, bin_edges) - 1  # 0-based bin index

        for i in range(N):
            for j in range(N):
                bin_id = bin_indices[i, j]
                if bin_id < NRini_py or bin_id > NRfin_py:
                    continue
                if rr[i, j] <= self.DistRange:
                    continue
                if self.sc[i, j] > means[bin_id] + self.NSTD * stds[bin_id]:
                    Clong[i, j] = self.sc[i, j]

        return Clong

    # ------------------------------------------------------------------
    # DistanceRule interface
    # ------------------------------------------------------------------

    def compute_Dist_Rule(self, cog_dist: np.ndarray, lambda_val: float):
        """
        Full EDR + long-range pipeline.

        Steps
        -----
        1. Compute pairwise distances rr from cog_dist.
        2. Build the plain EDR matrix  C = exp(-lambda * rr).
        3. Histogram sc vs distance (NR bins) and fit exponential → A1.
        4. Identify long-range connections → Clong.
        5. Return rr and clip(C + Clong, 0, 1) with diagonal = 1.

        Parameters
        ----------
        cog_dist : np.ndarray
            Coordinates of centres of gravity, shape (N, 3).
        lambda_val : float
            Exponential decay parameter.

        Returns
        -------
        rr : np.ndarray
            Pairwise distance matrix, shape (N, N).
        EDR_Clong : np.ndarray
            Combined connectivity matrix, shape (N, N).
        """
        if self.sc is None:
            raise ValueError(
                "sc (structural connectivity matrix) must be set "
                "before calling compute()."
            )

        rr = self._compute_rr(cog_dist)
        C  = np.exp(-lambda_val * rr)

        means, stds, bin_edges, _ = self.compute_hist(self.sc, rr, self.NR)
        # centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # A1, _   = self.fit_exponential(centers, means)

        Clong     = self.compute_Clong(rr, means, stds, bin_edges)
        EDR_Clong = np.clip(C + Clong, 0.0, 1.0)
        np.fill_diagonal(EDR_Clong, 1.0)
        return rr, EDR_Clong
