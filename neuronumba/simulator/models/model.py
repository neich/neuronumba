import numba as nb
import numpy as np

from neuronumba.basic.attr import HasAttr, Attr


class Model(HasAttr):

    n_rois = Attr(required=True)

    @classmethod
    def _build_var_dict(cls, var_list: list[str]):
        return {v_name: index for index, v_name in enumerate(var_list)}

    def get_state_sub(self, v_list: list[str] = None):
        v_list = v_list or []
        return {v: (self.state_vars[v], i) for i, v in enumerate(v_list)}

    def get_observed_sub(self, v_list: list[str] = None):
        v_list = v_list or []
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


