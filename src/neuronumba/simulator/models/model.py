import os
from enum import IntEnum

import numba as nb
import numpy as np
from overrides import overrides

from neuronumba.basic.attr import HasAttr, Attr, AttrEnum
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class Model(HasAttr):
    Type = AttrEnum(['Model', 'ModelAux'])

    weights = Attr(required=True)
    n_rois = Attr(dependant=True)
    m = Attr(dependant=True)

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cls = type(self)
        p, p_aux = cls._build_parameter_enum()
        setattr(cls, 'P', p)
        setattr(cls, 'P_aux', p_aux)

    def configure(self, **kwargs):
        # Kind of a hack to avoid dynamic class variables not being copied when using multiprocessing on Windows
        if os.name == "nt":
            cls = type(self)
            p, p_aux = cls._build_parameter_enum()
            setattr(cls, 'P', p)
            setattr(cls, 'P_aux', p_aux)
        super().configure(**kwargs)
        return self

    def _init_dependant(self):
        super()._init_dependant()
        self.n_rois = self.weights.shape[0]

    @classmethod
    def _build_parameter_enum(cls):
        attrs = [name for name, value in cls._get_attributes().items() if Model.Type.Model in value.attributes]
        attrs_aux = [name for name, value in cls._get_attributes().items() if Model.Type.ModelAux in value.attributes]
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
                result[v] = (True, i_state, self.state_vars[v])
                i_state += 1
            elif v in self.observable_vars:
                result[v] = (False, i_obs, self.observable_vars[v])
                i_obs += 1
            else:
                raise AttributeError(f"Variable <{v}> is not in the state or observable variables list!")
        return result

    def get_numba_dfun(self):
        """Return a @nb.njit compiled function that computes state derivatives and observables.

        Signature: UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:])
            (state, coupling) -> (d_state, observed)

        - state:    shape (n_state_vars, n_rois)
        - coupling: shape (len(c_vars), n_rois)
        - d_state:  shape (n_state_vars, n_rois) — time derivatives
        - observed: shape (n_observable_vars, n_rois) or (1,1) if no observables
        """
        raise NotImplementedError

    def initial_state(self, n_rois):
        """Return the initial state array of shape (n_state_vars, n_rois)."""
        raise NotImplementedError

    def get_numba_coupling(self):
        """Return a @nb.njit compiled coupling function.

        Signature: f8[:,:](f8[:,:])
            (state_coupled) -> coupling

        - state_coupled: shape (len(c_vars), n_rois)
        - coupling:      shape (len(c_vars), n_rois)
        """
        raise NotImplementedError

    def get_jacobian(self, sc):
        """
        :return: the jacobian matrix of the model.
        """
        raise NotImplementedError

    def get_noise_matrix(self, sigma, N):
        """
        computes the covariance noise matrix of the model

        :param sigma: the noise amplitude, format one value, type float

        :return: the covariance noise matrix Qn, format (n_eqs n_roi, n_eqs n_roi)
        """
        # =============== Build Qn
        cls = type(self)
        n_eqs = cls.n_state_vars
        Qn = (sigma ** 2) * np.eye(n_eqs * N)  # covariance matrix of the noise
        return Qn

    def as_array(self, param):
        if isinstance(param, np.ndarray):
            return param
        else:
            return np.r_[param]

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
    g = Attr(default=1.0, attributes=Model.Type.Model)

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
