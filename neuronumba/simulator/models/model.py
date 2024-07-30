from enum import Enum, IntEnum

import numba as nb
import numpy as np

from neuronumba.basic.attr import HasAttr, Attr, AttrType

class ParameterEnum(object):
    def __init__(self):
        self._index = 0

    def add_param(self, name):
        setattr(self, name, self._index)
        self._index += 1


class Model(HasAttr):

    n_rois = Attr(required=True)
    m = Attr(dependant=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cls = type(self)
        setattr(cls, 'P', cls._build_parameter_enum())

    @classmethod
    def _build_var_dict(cls, var_list: list[str]):
        return {v_name: index for index, v_name in enumerate(var_list)}

    @classmethod
    def _build_parameter_enum(cls):
        attrs = [name for name, value in cls._get_attributes().items() if value.attr_type == AttrType.Model]
        p = IntEnum('P', {k: i for i, k in enumerate(attrs)})
        return p


    def get_state_sub(self, v_list: list[str] = None):
        v_list = v_list or []
        for v in v_list:
            if v not in self.state_vars:
                raise AttributeError(f"Variable <{v}> is not in the state variables list!")
        return {v: (self.state_vars[v], i) for i, v in enumerate(v_list)}

    def get_observed_sub(self, v_list: list[str] = None):
        v_list = v_list or []
        for v in v_list:
            if v not in self.observable_vars:
                raise AttributeError(f"Variable <{v}> is not in the observed list!")
        return {v: (self.observable_vars[v], i) for i, v in enumerate(v_list)}

    def get_numba_coupling(self, g: nb.f8):
        """
        This is the default coupling for most models, linear coupling using the weights matrix

        :param g: global linear coupling
        :return:
        """
        @nb.njit #(nb.f8[:](nb.f8[:, :], nb.f8[:]))
        def linear_coupling(weights, state):
            r = weights @ state
            return r * g

        return linear_coupling

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
    g = Attr(default=1.0, attr_type=AttrType.Model)

    def get_numba_coupling(self):
        """
        This is the default coupling for most models, linear coupling using the weights matrix

        :param g: global linear coupling
        :return:
        """

        g = self.g

        @nb.njit #(nb.f8[:](nb.f8[:, :], nb.f8[:]))
        def linear_coupling(weights, state):
            r = weights @ state
            return r * g

        return linear_coupling
