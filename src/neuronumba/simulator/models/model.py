import os
from enum import IntEnum

import numba as nb
import numpy as np
from overrides import overrides

from neuronumba.basic.attr import HasAttr, Attr, AttrEnum


class ParameterEnum(object):
    def __init__(self):
        self._index = 0

    def add_param(self, name):
        setattr(self, name, self._index)
        self._index += 1


class Model(HasAttr):
    Type = AttrEnum(['Model'])

    weights = Attr(required=True)
    n_rois = Attr(dependant=True)
    m = Attr(dependant=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cls = type(self)
        setattr(cls, 'P', cls._build_parameter_enum())

    def configure(self, **kwargs):
        # Kind of a hack to avoid dynamic class variables not being copied when using multiprocessing on Windows
        if os.name == "nt":
            cls = type(self)
            setattr(cls, 'P', cls._build_parameter_enum())
        super().configure(**kwargs)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_rois = self.weights.shape[0]

    @classmethod
    def _build_var_dict(cls, var_list: list[str]):
        return {v_name: index for index, v_name in enumerate(var_list)}

    @classmethod
    def _build_parameter_enum(cls):
        attrs = [name for name, value in cls._get_attributes().items() if Model.Type.Model in value.attributes]
        p = IntEnum('P', {k: i for i, k in enumerate(attrs)})
        return p

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

    # def get_state_sub(self, v_list: list[str] = None):
    #     v_list = v_list or []
    #     for v in v_list:
    #         if v not in self.state_vars:
    #             raise AttributeError(f"Variable <{v}> is not in the state variables list!")
    #     return {v: (self.state_vars[v], i) for i, v in enumerate(v_list)}
    #
    # def get_observed_sub(self, v_list: list[str] = None):
    #     v_list = v_list or []
    #     for v in v_list:
    #         if v not in self.observable_vars:
    #             raise AttributeError(f"Variable <{v}> is not in the observed list!")
    #     return {v: (self.observable_vars[v], i) for i, v in enumerate(v_list)}

    def get_numba_coupling(self):
        """
        :return: numba function with signature nb.f8[:, :](nb.f8[:, :]) (state) -> coupling
        """
        raise NotImplementedError

    def compute_linear_matrix(self, sc, sigma):
        """
        :return:
        """
        raise NotImplementedError

    def as_array(self, param):
        if isinstance(param, np.ndarray):
            return param
        else:
            return np.r_[param]

    def _init_dependant_automatic(self):
        self.m = np.empty((len(self.P), self.n_rois))
        for p in list(self.P):
            name = p.name
            index = p.value
            self.m[index] = self.as_array(getattr(self, name))


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
        @nb.njit #(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :]))
        def linear_coupling(state):
            """

            :param state: (n_cvars, n_rois) this is a subset of the full state that
                                    contains only the variables to couple
            :return:
            """
            r = state @ wtg
            return r

        return linear_coupling
