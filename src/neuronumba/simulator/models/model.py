from abc import ABC, abstractmethod
from enum import IntEnum
from typing import NamedTuple

import numba as nb
import numpy as np
from overrides import overrides

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class VarInfo(NamedTuple):
    """Information about a model variable (state or observable)."""
    is_state: bool        # True if state variable, False if observable
    buffer_index: int     # Index into the state/observable output buffer
    original_index: int   # Index into the state_vars or observable_vars dict


class Model(HasAttr, ABC):

    class Tag(HasAttr.Tag):
        """Attribute tags for Model parameters.

        REGIONAL: per-ROI scalar params, packed into the parameter matrix (self.m).
        GLOBAL: non-ROI array params, packed into the auxiliary dict (self.m_aux).
        """
        REGIONAL = 'regional'
        GLOBAL = 'global'

    weights = Attr(required=True)
    n_rois = Attr(dependant=True)
    m = Attr(dependant=True)
    m_aux = Attr(dependant=True)

    # ---------------------------------------------------------------
    # Model interface: subclasses declare these lists of variable names
    # ---------------------------------------------------------------

    _state_var_names: list[str] = []
    """Names of state variables, in row order.
    Example: ['S_e', 'S_i'] means state[0,:] is S_e, state[1,:] is S_i.
    """

    _coupling_var_names: list[str] = []
    """Names of state variables that participate in inter-region coupling.
    Must be a subset of _state_var_names.
    Example: ['S_e'] means only S_e is coupled across regions.
    """

    _observable_var_names: list[str] = []
    """Names of observable variables computed inside dfun but not integrated.
    Example: ['Ie', 're'] for excitatory current and firing rate.
    """

    _state_var_bounds: dict = {}
    """Optional bounds per state variable name: {name: (lo, hi)}.
    Variables not listed are unbounded. Use math.inf / -math.inf for one-sided bounds."""

    # ---------------------------------------------------------------
    # Derived class attributes (auto-computed by __init_subclass__)
    # ---------------------------------------------------------------

    state_vars: dict = {}
    """Dict mapping state variable name -> row index. Auto-computed from _state_var_names."""

    n_state_vars: int = 0
    """Number of state variables. Auto-computed from _state_var_names."""

    c_vars: list = []
    """List of integer indices into the state array for coupled variables.
    Auto-computed from _coupling_var_names and _state_var_names.
    """

    observable_vars: dict = {}
    """Dict mapping observable variable name -> row index. Auto-computed from _observable_var_names."""

    n_observable_vars: int = 0
    """Number of observable variables. Auto-computed from _observable_var_names."""

    _state_var_lo: np.ndarray = np.empty(0)
    _state_var_hi: np.ndarray = np.empty(0)
    _has_bounds: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only process subclasses that declare their own _state_var_names
        if '_state_var_names' in cls.__dict__:
            cls.state_vars = {name: i for i, name in enumerate(cls._state_var_names)}
            cls.n_state_vars = len(cls._state_var_names)
        if '_observable_var_names' in cls.__dict__:
            cls.observable_vars = {name: i for i, name in enumerate(cls._observable_var_names)}
            cls.n_observable_vars = len(cls._observable_var_names)
        if '_coupling_var_names' in cls.__dict__:
            # Resolve names to integer indices using state_vars
            cls.c_vars = [cls.state_vars[name] for name in cls._coupling_var_names]
        if '_state_var_names' in cls.__dict__ or '_state_var_bounds' in cls.__dict__:
            n = cls.n_state_vars
            lo = np.full(n, -np.inf, dtype=np.float64)
            hi = np.full(n, np.inf, dtype=np.float64)
            bounds = cls._state_var_bounds or {}
            for name, (b_lo, b_hi) in bounds.items():
                if name not in cls.state_vars:
                    raise ValueError(f"Bound declared for unknown state var <{name}> in {cls.__name__}")
                idx = cls.state_vars[name]
                lo[idx] = b_lo
                hi[idx] = b_hi
            cls._state_var_lo = lo
            cls._state_var_hi = hi
            cls._has_bounds = bool(bounds)

    def configure(self, **kwargs):
        self.P, self.P_aux = type(self)._build_parameter_enum()
        super().configure(**kwargs)
        return self

    def _init_dependant(self):
        super()._init_dependant()
        self.n_rois = self.weights.shape[0]

    @classmethod
    def _build_parameter_enum(cls):
        attrs = [name for name, value in cls._get_attributes().items() if Model.Tag.REGIONAL in value.attributes]
        attrs_aux = [name for name, value in cls._get_attributes().items() if Model.Tag.GLOBAL in value.attributes]
        p = IntEnum('P', {k: i for i, k in enumerate(attrs)})
        p_aux = IntEnum('P_aux', {k: i for i, k in enumerate(attrs_aux)})
        return p, p_aux

    def get_var_info(self, v_list: list[str] = None):
        v_list = v_list or []
        result = {}
        i_state = 0
        i_obs = 0
        for v in v_list:
            if v in self.state_vars:
                result[v] = VarInfo(is_state=True, buffer_index=i_state, original_index=self.state_vars[v])
                i_state += 1
            elif v in self.observable_vars:
                result[v] = VarInfo(is_state=False, buffer_index=i_obs, original_index=self.observable_vars[v])
                i_obs += 1
            else:
                raise AttributeError(f"Variable <{v}> is not in the state or observable variables list!")
        return result

    @abstractmethod
    def get_numba_dfun(self):
        """Return a @nb.njit compiled function that computes state derivatives and observables.

        Signature: UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:])
            (state, coupling) -> (d_state, observed)

        - state:    shape (n_state_vars, n_rois)
        - coupling: shape (len(c_vars), n_rois)
        - d_state:  shape (n_state_vars, n_rois) — time derivatives
        - observed: shape (n_observable_vars, n_rois) or (1,1) if no observables
        """

    @abstractmethod
    def initial_state(self, n_rois):
        """Return the initial state array of shape (n_state_vars, n_rois)."""

    @abstractmethod
    def get_numba_coupling(self):
        """Return a @nb.njit compiled coupling function.

        Signature: f8[:,:](f8[:,:])
            (state_coupled) -> coupling

        - state_coupled: shape (len(c_vars), n_rois)
        - coupling:      shape (len(c_vars), n_rois)
        """

    def get_numba_validate(self):
        """Return a @nb.njit function that clips state to declared bounds in place.

        Signature: f8[:,:](f8[:,:])  (state) -> state
        If the model declares no bounds, returns an identity closure.
        """
        if not self._has_bounds:
            @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE)
            def validate(state):
                return state
            return validate

        lo = self._state_var_lo
        hi = self._state_var_hi
        bounded_idx = np.array(
            [i for i in range(lo.shape[0])
             if np.isfinite(lo[i]) or np.isfinite(hi[i])],
            dtype=np.int64,
        )

        @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
        def validate(state):
            n = bounded_idx.shape[0]
            n_rois = state.shape[1]
            for k in range(n):
                i = bounded_idx[k]
                lo_i = lo[i]
                hi_i = hi[i]
                for j in range(n_rois):
                    state[i, j] = min(hi_i, max(lo_i, state[i, j]))
            return state

        return validate

    def get_jacobian(self, sc):
        """
        :return: the jacobian matrix of the model.
        """
        raise NotImplementedError

    def get_noise_matrix(self, sigma, N):
        """Compute the covariance noise matrix of the model.

        Args:
            sigma: Noise amplitude (scalar).
            N: Number of brain regions.

        Returns:
            Covariance noise matrix Qn of shape (n_state_vars * N, n_state_vars * N).
        """
        Qn = (sigma ** 2) * np.eye(self.n_state_vars * N)
        return Qn

    def as_array(self, param):
        return np.atleast_1d(param)

    def _init_dependant_automatic(self):
        self.m = np.empty((len(self.P), self.n_rois))
        self.m_aux = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64[:])
        for p in list(self.P):
            name = p.name
            index = p.value
            self.m[index] = self.as_array(getattr(self, name))
        for p in list(self.P_aux):
            name = p.name
            index = p.value
            self.m_aux[index] = getattr(self, name)



class LinearCouplingModel(Model):
    g = Attr(default=1.0, attributes=Model.Tag.REGIONAL)

    weights_t = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        # Make sure we store a copy and not a view
        self.weights_t = self.weights.T.copy()

    @overrides
    def get_numba_coupling(self):
        """
        This is the default coupling for most models, linear coupling using the weights matrix

        :param g: global linear coupling
        :return:
        """

        wtg = self.g * self.weights_t.copy()

        # TODO: why adding the signature raises a numba warning about state_coupled being a non contiguous array?
        @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE)
        def linear_coupling(state):
            """

            :param state: (n_cvars, n_rois) this is a subset of the full state that
                                    contains only the variables to couple
            :return:
            """
            r = state @ wtg
            return r

        return linear_coupling
