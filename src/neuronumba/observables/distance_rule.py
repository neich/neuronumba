from abc import abstractmethod
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from neuronumba.basic.attr import HasAttr, Attr


class DistanceRule(HasAttr):
    """
    Base class for distance-based connectivity rules.

    This class provides:
    - common attributes (e.g. NSTD, lambda_val)
    - utility methods for histogram computation and exponential fitting

    Subclasses must implement the `compute_Dist_Rule` method.

    Code by Giuseppe Pau, December 27, 2025
    Refactored by Gustavo Patow, January 13, 2025
    """

    # Number of standard deviations used as threshold for long-range connections
    NSTD = Attr(required=False, default=1)
    # Exponential decay parameter lambda (can be fitted or externally provided)
    lambda_val = Attr(required=False, default=None)

    def compute(self,
                cog_dist: np.ndarray,
                lambda_val: float = None):
        if lambda_val is None:
            if self.lambda_val is None:
                raise ValueError(
                    "lambda_val is not set. Run fit_exponential or provide it explicitly."
                )
            lambda_val = self.lambda_val
        return self.compute_Dist_Rule(cog_dist, lambda_val)

    @abstractmethod
    def compute_Dist_Rule(self, cog_dist: np.ndarray, lambda_val: float):
        """
        Abstract method to compute a distance-based connectivity rule.
        """
        pass

    def compute_hist(self, C, rr, NR):
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
            rr_flat, C_flat,
            statistic='mean',
            bins=NR,
            range=(min_dist, max_dist)
        )

        stds, _, _ = binned_statistic(
            rr_flat, C_flat,
            statistic='std',
            bins=NR,
            range=(min_dist, max_dist)
        )

        maxs, _, _ = binned_statistic(
            rr_flat, C_flat,
            statistic='max',
            bins=NR,
            range=(min_dist, max_dist)
        )

        return means, stds, bin_edges, maxs

    def fit_exponential(self, centers, means, start_index=24):
        """
        Fit an exponential decay to the binned connectivity profile.

        The model is:
            C(d) = A1 * exp(-lambda * d)

        Parameters
        ----------
        centers : np.ndarray
            Bin center distances.
        means : np.ndarray
            Mean connectivity values per bin.
        start_index : int
            Index from which the fit is performed (to ignore short distances).

        Returns
        -------
        lambda_fit : float
            Estimated exponential decay parameter lambda.
        """

        def expfunc(x, A1, A2):
            return A1 * np.exp(-A2 * x)

        xdata = centers[start_index:]
        ydata = means[start_index:]

        # Initial guess and bounds for nonlinear fitting
        A0 = [0.15, 0.18]
        bounds = ([-100, -100], [100, 100])

        popt, _ = curve_fit(expfunc, xdata, ydata, p0=A0, bounds=bounds)

        # The second fitted parameter corresponds to lambda
        lambda_fit = popt[1]
        return lambda_fit


class EDR_distance_rule(DistanceRule):
    """
    Exponential Distance Rule (EDR).

    Computes:
    - pairwise Euclidean distances
    - exponential decay connectivity matrix
    """
    def __init__(self, lambda_val: float):
        super().__init__(lambda_val=lambda_val)

    def compute_Dist_Rule(self,
                          cog_dist: np.ndarray,
                          lambda_val: float):
        """
        Compute distance matrix and EDR connectivity.

        Parameters
        ----------
        cog_dist : np.ndarray
            Coordinates of centers of gravity (N x 3).
        lambda_val : float, optional
            Exponential decay parameter. If None, self.lambda_val is used.

        Returns
        -------
        rr : np.ndarray
            Pairwise distance matrix.
        c_exp : np.ndarray
            Exponential distance rule connectivity matrix.
        """
        N = cog_dist.shape[0]
        rr = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(cog_dist[i, :] - cog_dist[j, :])

        c_exp = np.exp(-lambda_val * rr)
        np.fill_diagonal(c_exp, 1.0)

        return rr, c_exp


class EDR_LR_distance_rule(DistanceRule):
    """
    Extension of the EDR model including long-range connections (Clong). From
    Deco, Gustavo et al., Rare long-range cortical connections enhance human
    information processing, Current Biology, Volume 31, Issue 20, 4436 - 4448.e5
    DOI: 10.1016/j.cub.2021.07.064

    original code by Gustavo Deco, 2021
    """

    c_exp = Attr(dependant=True)
    rr = Attr(dependant=True)

    def __init__(self, sc=None):
        super().__init__()
        self.sc = sc

    def _init_dependant(self):
        super()._init_dependant()
        # First, we need the rr and Clong matrices

    def compute_Dist_Rule(self,
                          cog_dist: np.ndarray,
                          lambda_val: float):
        """
        Compute EDR + long-range connectivity matrix.

        Parameters
        ----------
        rr : np.ndarray
            Distance matrix.
        Clong : np.ndarray
            Long-range connectivity matrix.
        lambda_val : float, optional
            Exponential decay parameter.

        Returns
        -------
        EDR_Clong : np.ndarray
            Combined EDR + Clong connectivity matrix.
        """
        C = np.exp(-lambda_val * rr)
        EDR_Clong = C + Clong
        EDR_Clong = np.clip(EDR_Clong, 0, 1)
        np.fill_diagonal(EDR_Clong, 1.0)
        return rr, EDR_Clong

    def compute_Clong(self, rr, means, stds, bin_edges,
                      NRini, NRfin, DistRange=0, A1=None):
        """
        Compute the long-range connectivity matrix (Clong).

        A connection is considered long-range if:
        - its distance bin is between NRini and NRfin
        - its distance is greater than DistRange
        - its strength exceeds mean + NSTD * std within the bin

        Parameters
        ----------
        rr : np.ndarray
            Distance matrix.
        sc_matrix : np.ndarray
            Structural connectivity matrix.
        means, stds : np.ndarray
            Mean and standard deviation per distance bin.
        bin_edges : np.ndarray
            Distance bin edges.
        NRini, NRfin : int
            First and last bin indices (MATLAB-style, 1-based).
        DistRange : float
            Minimum distance threshold.
        A1 : float
            Normalization constant (from exponential fit).

        Returns
        -------
        Clong : np.ndarray
            Long-range connectivity matrix.
        """
        if A1 is None:
            raise ValueError("A1 (from exponential fit) must be provided.")

        NSTD = self.NSTD
        N = rr.shape[0]
        Clong = np.zeros((N, N))

        # Convert MATLAB-style indices (1-based) to Python (0-based)
        NRini_py = NRini - 1
        NRfin_py = NRfin - 1

        bin_indices = np.digitize(rr, bin_edges) - 1

        for i in range(N):
            for j in range(N):
                bin_id = bin_indices[i, j]

                if bin_id < NRini_py or bin_id > NRfin_py:
                    continue
                if rr[i, j] <= DistRange:
                    continue

                mv = means[bin_id]
                st = stds[bin_id]

                if self.sc[i, j] > mv + NSTD * st:
                    Clong[i, j] = self.sc[i, j]

        Clong /= A1
        return Clong
