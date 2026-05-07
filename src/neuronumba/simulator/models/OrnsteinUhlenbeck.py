# ==========================================================================
# ==========================================================================
# Multivariate Ornstein–Uhlenbeck (OU) Process
#
# The multivariate OU process is a linear stochastic model defined by:
#
#   dx = -A · x dt + B dW
#
# where:
#   x   : state vector (n_rois,)
#   A   : drift/mean-reversion matrix — must have all eigenvalues with
#         strictly positive real parts so that -A is Hurwitz stable.
#         In the whole-brain context A is typically built as:
#             A = (1/tau) * I  -  g * W_SC
#         but the user may supply A directly (default) or let the model
#         build it from W_SC, g, and tau.
#   B   : diffusion coefficient (scalar * I, i.e. isotropic noise)
#   dW  : vector Wiener increment
#
# Stability condition (for dX = -A·X):
#   The system is stable iff all eigenvalues of A have POSITIVE real parts,
#   i.e. Re(λᵢ(A)) > 0 for all i  (equivalently, -A is Hurwitz).
#
# Stabilization strategies (StabilizationMethod enum):
#   NONE                - no correction; raises ValueError if A is not stable
#   SPECTRAL_RADIUS     - normalize off-diagonal part so spectral radius = 1;
#                         stable iff g < 1/tau  [only valid for W_SC path]
#   SPECTRAL_PROJECTION - push any Re(λ) ≤ 0 eigenvalue to +epsilon
#   DIAGONAL_DOMINANCE  - set A_ii = row_sum + epsilon  [DEFAULT]
#                         preserves off-diagonal topology exactly
#   SYMMETRIZED_SHIFT   - symmetrize A then shift spectrum to +epsilon;
#                         loses directionality
#
# Input modes (set at construction via use_SC_direct, default True):
#   Direct  (default) : user calls configure(A=...)      — A is used as-is
#                       after optional stabilization
#   SC path           : user calls configure(W_SC=...)   — model builds
#                       A = diag(1/tau) - g * W_SC first
#
# References:
#   [Uhlenbeck_Ornstein_1930] G.E. Uhlenbeck, L.S. Ornstein
#       "On the Theory of the Brownian Motion"
#       Phys. Rev. 36, 823 (1930)
#
#   [Risken_1996] H. Risken
#       "The Fokker-Planck Equation: Methods of Solution and Applications"
#       Springer, 2nd ed., 1996
#
# ==========================================================================
# ==========================================================================

from enum import Enum, auto
from typing import Dict, List, Optional

import numba as nb
import numpy as np
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL

# ==========================================================================
# Stabilization method options
# ==========================================================================
class StabilizationMethod(Enum):
    """
    Strategy used to enforce stability of A in  dX = -A·X dt + B dW.

    Stability requires Re(λᵢ(A)) > 0 for all i  (i.e. -A is Hurwitz).

    NONE                - trust the user; raise ValueError if A is unstable
    SPECTRAL_RADIUS     - normalize W_SC by its spectral radius so that
                          stability holds iff g < 1/tau  (SC path only)
    SPECTRAL_PROJECTION - push any non-positive-real eigenvalue of A to
                          +epsilon
    DIAGONAL_DOMINANCE  - set A_ii = (off-diagonal row sum) + epsilon;
                          preserves off-diagonal topology [DEFAULT]
    SYMMETRIZED_SHIFT   - symmetrize A then shift spectrum above +epsilon;
                          loses directionality
    """
    NONE                = auto()
    SPECTRAL_RADIUS     = auto()
    SPECTRAL_PROJECTION = auto()
    DIAGONAL_DOMINANCE  = auto()
    SYMMETRIZED_SHIFT   = auto()


# ==========================================================================
# Matrix Stabilizer
# ==========================================================================
class Matrix_Stabilizer():
    """
    Stabilization strategies:
    We need A to be Hurwitz, i.e. Re(λᵢ(A)) > 0 for all i (stability of dX = -A·X).
    There are several ways to enforce this, each with different trade-offs.

    Comparison at a Glance (asked Claude IA)
    ----------------------------------------------------------------------------------------------------
    Method                 Hurwitz        Preserves     Preserves        Best used when...
                           on W?          asymmetry?    off-diagonal
                                                        structure?
    ----------------------------------------------------------------------------------------------------
    Spectral radius norm   ❌            ✅             ✅               We tune g and τ explicitly
                           (depends on g, τ)
    Spectral projection    ✅            ✅             ⚠️ partial       We need exact spectral control
    Diagonal dominance     ✅            ✅             ✅ off-diagonal  We want to preserve topology
    Symmetrization + shift ✅            ❌             ⚠️ averaged      The model assumes undirected SC
    ----------------------------------------------------------------------------------------------------
    """

    def __init__(self,
                 stabilization: StabilizationMethod = StabilizationMethod.DIAGONAL_DOMINANCE,
                 epsilon: float = 0.01,
                 ):
        self.stabilization = stabilization
        self.epsilon = epsilon

    def _assert_stable(self, A: np.ndarray) -> np.ndarray:
        """NONE: pass through; raise if any eigenvalue has Re(λ) ≤ 0."""
        if not self._is_stable(A):
            raise ValueError(
                "StabilizationMethod.NONE selected but A has eigenvalues "
                "with Re(λ) ≤ 0.  Pre-stabilize your matrix or choose a "
                "stabilization strategy."
            )
        return A

    def _stabilize_spectral_radius(self, A: np.ndarray) -> np.ndarray:
        """
        SPECTRAL_RADIUS (SC path only):
        Re-normalize the off-diagonal (coupling) part of A so that its
        spectral radius equals 1, then rebuild A with the original diagonal.
        Stability is guaranteed iff g < 1/tau.
        """
        # Decompose: A = diag(1/tau)  +  A_offdiag  where A_offdiag = -g*W
        diag_A     = np.diag(A)
        A_offdiag  = A - np.diag(diag_A)          # = -g * W  (off-diagonal only)

        sr = np.max(np.real(np.linalg.eigvals(A_offdiag)))
        # sr is negative (A_offdiag = -g*W with g,W ≥ 0); use its magnitude
        sr_abs = abs(sr)
        if sr_abs < 1e-12:
            return A                               # no coupling, already stable

        A_offdiag_norm = A_offdiag / sr_abs        # spectral radius of coupling = 1
        return np.diag(diag_A) + A_offdiag_norm

    def _stabilize_spectral_projection(self, A: np.ndarray) -> np.ndarray:
        """
        SPECTRAL_PROJECTION:
        Decompose A, push any Re(λ) ≤ 0 eigenvalue to +epsilon, reconstruct.
        Works on both input modes.
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stabilized = np.where(np.real(eigenvalues) <= 0,
                              self.epsilon + 0j, eigenvalues)
        A_stable = eigenvectors @ np.diag(stabilized) @ np.linalg.inv(eigenvectors)
        return np.real(A_stable)

    def _stabilize_diagonal_dominance(self, A: np.ndarray) -> np.ndarray:
        """
        DIAGONAL_DOMINANCE  [DEFAULT]:
        Set A_ii = (sum of |off-diagonal row entries|) + epsilon.
        By the Gershgorin circle theorem all eigenvalues then lie in discs
        centred at the (positive) diagonal, guaranteeing Re(λ) > 0.
        Preserves the off-diagonal topology of A exactly.
        """
        A_new = A.copy()
        np.fill_diagonal(A_new, 0.0)
        row_sums = np.sum(np.abs(A_new), axis=1)
        np.fill_diagonal(A_new, row_sums + self.epsilon)
        return A_new

    def _stabilize_symmetrized_shift(self, A: np.ndarray) -> np.ndarray:
        """
        SYMMETRIZED_SHIFT:
        Symmetrize A as (A + Aᵀ)/2, then shift the spectrum so that
        the minimum eigenvalue equals +epsilon.
        Loses directionality; appropriate only for undirected SC.
        """
        A_sym    = (A + A.T) / 2.0
        lam_min  = np.min(np.linalg.eigvalsh(A_sym))
        shift    = self.epsilon - lam_min          # shift so min eigenvalue = +epsilon
        return A_sym + shift * np.eye(A.shape[0])

    def stabilize(self, A_raw: np.ndarray) -> np.ndarray:
        print(f"\n{'='*58}")
        print(f"StabilizationModel — stabilize(A_raw)")
        print(f"  Stabilization : {self.stabilization.name}")
        print(f"  epsilon       : {self.epsilon}")
        print(f"  N (ROIs)      : {A_raw.shape[0]}")
        self._print_stability("A_raw  (before correction)", A_raw)

        if self.stabilization == StabilizationMethod.NONE:
            self.A_stable = self._assert_stable(A_raw)
        elif self.stabilization == StabilizationMethod.SPECTRAL_RADIUS:
            if self.use_SC_direct:
                raise ValueError(
                    "StabilizationMethod.SPECTRAL_RADIUS is only meaningful "
                    "on the W_SC path (use_SC_direct=False), because it "
                    "operates on the off-diagonal coupling part separately."
                )
            self.A_stable = self._stabilize_spectral_radius(A_raw)
        elif self.stabilization == StabilizationMethod.SPECTRAL_PROJECTION:
            self.A_stable = self._stabilize_spectral_projection(A_raw)
        elif self.stabilization == StabilizationMethod.DIAGONAL_DOMINANCE:
            self.A_stable = self._stabilize_diagonal_dominance(A_raw)
        elif self.stabilization == StabilizationMethod.SYMMETRIZED_SHIFT:
            self.A_stable = self._stabilize_symmetrized_shift(A_raw)

        self._print_stability("A_stable (after correction)", self.A_stable)
        print(f"{'='*58}\n")

        return self.A_stable

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stable(A: np.ndarray, tol: float = 1e-9) -> bool:
        """Return True iff all eigenvalues of A have Re(λ) > 0."""
        return bool(np.all(np.real(np.linalg.eigvals(A)) > tol))

    @staticmethod
    def _print_stability(label: str, A: np.ndarray):
        eigs   = np.real(np.linalg.eigvals(A))
        stable = np.all(eigs > 0)
        print(f"  {label}:")
        print(f"    min Re(λ) = {np.min(eigs):.6f}  "
              f"max Re(λ) = {np.max(eigs):.6f}  "
              f"| stable: {stable}")

# ==========================================================================
# Ornstein-Uhlenbeck model
# ==========================================================================
class OrnsteinUhlenbeck(LinearCouplingModel):
    """
    Multivariate Ornstein-Uhlenbeck (OU) whole-brain model.

    SDE convention:
        dX = -A_stable · X dt + B dW

    A_stable is the single source of truth built by configure() and used
    by get_numba_dfun() and get_jacobian().  It is always exposed as
    self.A_stable for external inspection.

    Input modes
    -----------
    use_SC_direct=True  (default)
        Call  configure(A=my_matrix).
        The provided matrix is treated as A and stabilized if needed.

    use_SC_direct=False
        Call  configure(W_SC=my_sc).
        The model builds  A = diag(1/tau) - g * W_SC  internally,
        then stabilizes the result.

    Parameters
    ----------
    stabilization : StabilizationMethod
        Strategy to enforce Re(λ(A)) > 0.  Default: DIAGONAL_DOMINANCE.
    epsilon : float
        Positive margin from zero used by eigenvalue-based methods.
        Default: 0.01.
    use_SC_direct : bool
        If True (default), configure() expects a pre-built A matrix.
        If False, configure() expects a raw W_SC matrix.
    zero_diagonal : bool
        When use_SC_direct=False, zero the diagonal of W_SC before
        building A.  Default: True.

    State variables      : x
    Observable variables : x
    Coupling variables   : x  (linear, via structural connectivity)
    """

    # ------------------------------------------------------------------
    # Variable bookkeeping
    # ------------------------------------------------------------------
    _state_var_names      = ['x']
    _coupling_var_names   = ['x']
    _observable_var_names = ['x']

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    # tau = Attr(
    #     default=20.0,
    #     attributes=Model.Tag.REGIONAL,
    #     doc="Mean-reversion time constant (ms). Only used when "
    #         "use_SC_direct=False to build A from W_SC.",
    # )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    # def __init__(self,
    #              use_SC_direct: bool = True,
    #              zero_diagonal: bool = True,
    #              **kwargs):
    #     super().__init__(**kwargs)
    #     # self.use_SC_direct  = use_SC_direct
    #     # self.zero_diagonal = zero_diagonal
    #     # self.A_stable: Optional[np.ndarray] = None   # set by configure()

    # ------------------------------------------------------------------
    # Configuration  — must be called once before simulation
    # ------------------------------------------------------------------
    # def configure(self,
    #               weights: Optional[np.ndarray] = None,
    #               **kwargs) -> np.ndarray:
    #     """
    #     Build and stabilize the drift matrix A for  dX = -A·X dt + B dW.
    #
    #     Call with the argument that matches use_SC_direct:
    #         configure(weights=my_matrix)
    #
    #     Parameters
    #     ----------
    #     weights : (n, n) pre-built drift matrix  [used when use_SC_direct=True] or
    #               raw structural connectivity  [used when use_SC_direct=False]
    #
    #     Returns
    #     -------
    #     A_stable : (n, n) stabilized drift matrix, also stored in self.A_stable
    #     """
    #     if weights is None:  # if it was already provided at the constructor
    #         weights = self.weights
    #     if self.use_SC_direct:
    #         A_raw = weights.copy().astype(float)
    #     else:
    #         A_raw = self._build_A_from_SC(weights)
    #
    #     super().configure(weights=self.A_stable, **kwargs)

    # ------------------------------------------------------------------
    # Internal: build A from W_SC  (SC path only)
    # ------------------------------------------------------------------
    # def _build_A_from_SC(self, W_SC: np.ndarray) -> np.ndarray:
    #     """
    #     Build  A = diag(1/tau) - g * W_SC.
    #
    #     For dX = -A·X this gives the standard OU mean-reversion:
    #         dX_i/dt = -(1/tau)*x_i + g * (W_SC @ x)_i
    #     """
    #     W = W_SC.copy().astype(float)
    #     if self.zero_diagonal:
    #         np.fill_diagonal(W, 0)
    #
    #     N = W.shape[0]
    #     tau = np.asarray(self.tau)
    #     inv_tau = np.ones(N) / float(tau) if tau.ndim == 0 else 1.0 / tau
    #
    #     return np.diag(inv_tau) - self.g * W

    # ------------------------------------------------------------------
    # Initialisation helpers  (framework bookkeeping)
    # ------------------------------------------------------------------

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        self._init_dependant_automatic()

    @property
    def get_state_vars(self) -> Dict[str, int]:
        return OrnsteinUhlenbeck.state_vars

    @property
    def get_observablevars(self) -> Dict[str, int]:
        return OrnsteinUhlenbeck.observable_vars

    @property
    def get_c_vars(self) -> List[int]:
        return OrnsteinUhlenbeck.c_vars

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Initial state array of shape (n_state_vars, n_rois).
        Small Gaussian noise around zero so ROIs start from distinct positions.
        """
        state = np.ones((OrnsteinUhlenbeck.n_state_vars, n_rois))
        state[0] = 0.01 * np.random.randn(n_rois)
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        """Initial observable array of shape (n_observable_vars, n_rois)."""
        observed = np.empty((OrnsteinUhlenbeck.n_observable_vars, n_rois))
        observed[0] = 0.0
        return observed

    # ------------------------------------------------------------------
    # Numba differential function
    # ------------------------------------------------------------------

    def get_numba_dfun(self):
        """
        Return the Numba-compiled differential function for the OU model.

        Implements:
            dX/dt = -A_stable · X

        A_stable is captured in the closure; the coupling argument from the
        base class is intentionally ignored — A_stable already encodes the
        full connectivity and mean-reversion.

        Signature:
            dfun(state, coupling) -> (derivatives, observables)
            shapes: (n_state_vars, n_rois), (n_state_vars, n_rois)
        """

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def OU_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Parameters
            ----------
            state    : (1, n_rois) — current activity x
            coupling : (1, n_rois) — unused; A encodes all coupling

            Returns
            -------
            derivatives : (1, n_rois) — dX/dt = -A @ x
            observables : (1, n_rois) — x  (activity is its own observable)
            """
            x  = state[0, :]
            Ax = coupling[0, :]
            dx = -Ax             # dx/dt = -A · x = -Ax

            derivatives = np.empty((1, x.shape[0]))
            observables = np.empty((1, x.shape[0]))
            derivatives[0, :] = dx
            observables[0, :] = x
            return derivatives, observables

        return OU_dfun

    # ------------------------------------------------------------------
    # noise template
    # ------------------------------------------------------------------

    def get_noise_template(self):
        #           ['x']
        return np.r_[ 1.]

    # ------------------------------------------------------------------
    # Analytical Jacobian
    # ------------------------------------------------------------------

    @overrides
    def get_jacobian(self, sc: np.ndarray) -> np.ndarray:
        """
        Return the analytical Jacobian of the OU model.

        For  dx = -A · x  the Jacobian is simply  J = -A
        (the dynamics are linear so J is constant everywhere).

        The sc argument is accepted for API compatibility with the base
        class but is ignored — A_stable already encodes the SC.

        Returns
        -------
        jacobian : (n_rois, n_rois)
        """
        A = -self.g * self.weights
        return A