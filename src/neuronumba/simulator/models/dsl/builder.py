"""
Source generators and `build_model` — the entry point that turns a `ModelSpec`
into a numba-friendly `Model` subclass.
"""
from __future__ import annotations

import ast
import textwrap
from typing import Any, Dict, List, Type

import numba as nb
import numpy as np

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL
from neuronumba.simulator.models.model import Model, LinearCouplingModel

from .dependents import _topo_sort_dependents
from .materialize import _materialize_module
from .rewriter import _DfunRewriter
from .spec import ModelSpec


def _build_dfun_source(spec: ModelSpec) -> str:
    """Return executable Python source for a function `dfun(state, coupling)`
    that uses captured `m` and `P` variables (added by exec env)."""

    state_index = {sv.name: i for i, sv in enumerate(spec.state_vars)}
    coupling_index = {cv.name: i for i, cv in enumerate(spec.coupling_vars)}
    param_names = {p.name for p in spec.parameters}
    obs_names = set(spec.observables)

    src = textwrap.dedent(spec.equations).strip()
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        raise ValueError(
            f"Could not parse equations for model '{spec.name}': {e}"
        ) from e

    rewriter = _DfunRewriter(
        state_index, coupling_index, param_names, obs_names
    )
    new_body = [rewriter.visit(stmt) for stmt in tree.body]

    if rewriter.errors:
        raise ValueError(
            "Equation errors in model '{}':\n  - {}".format(
                spec.name, "\n  - ".join(rewriter.errors)
            )
        )

    # Verify that every state var has a derivative `d_<name>` introduced.
    expected_d = {f"d_{sv.name}" for sv in spec.state_vars}
    user_derivatives = {
        node.targets[0].id
        for node in new_body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in expected_d
    }
    missing = expected_d - user_derivatives
    if missing:
        raise ValueError(
            f"Model '{spec.name}': missing derivative(s) {missing}. "
            f"You must assign d_<state_var> for every state variable."
        )

    # Verify each declared observable has been assigned.
    user_assigns = {
        node.targets[0].id
        for node in new_body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    missing_obs = obs_names - user_assigns
    if missing_obs:
        raise ValueError(
            f"Model '{spec.name}': declared observable(s) {missing_obs} are "
            f"never assigned in equations."
        )

    # Build the return tuple:
    #   (np.stack((d_<sv0>, d_<sv1>, ...)), np.stack((<obs0>, <obs1>, ...)))
    d_stack_args = ", ".join(f"d_{sv.name}" for sv in spec.state_vars)
    if spec.observables:
        obs_stack_args = ", ".join(spec.observables)
        ret_obs = f"np.stack(({obs_stack_args},))"
    else:
        ret_obs = "np.empty((1, 1))"

    body_src = "\n".join(ast.unparse(stmt) for stmt in new_body)

    # Header bindings: state vars from `state` array, params from `m`.
    state_binds = "\n".join(
        f"        {sv.name} = state[{i}, :]"
        for i, sv in enumerate(spec.state_vars)
        if sv.name in rewriter.used_state
    )
    param_binds = "\n".join(
        f"        {p} = m[np.intp(P.{p})]"
        for p in sorted(rewriter.used_params)
    )

    indented_body = textwrap.indent(body_src, "        ")

    # Factory function: `m` and `P` are passed in and become real closure
    # variables for the inner dfun — exactly the pattern the hand-written
    # neuronumba models use.
    parts = [f"def make_{spec.name}_dfun(m, P):"]
    parts.append(f"    def {spec.name}_dfun(state, coupling):")
    if state_binds:
        parts.append(state_binds)
    if param_binds:
        parts.append(param_binds)
    parts.append(indented_body)
    parts.append(f"        return np.stack(({d_stack_args},)), {ret_obs}")
    parts.append(f"    return {spec.name}_dfun")
    fn_src = "\n".join(parts) + "\n"
    return fn_src


def _build_coupling_kernel(spec: ModelSpec):
    """Return a factory function that, given a configured model instance,
    builds and returns a numba-jitted coupling closure."""

    kinds = {cv.kind for cv in spec.coupling_vars}

    if kinds == {"linear"}:
        def make_coupling(self):
            wtg = self.g * self.weights_t.copy()

            @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE)
            def _linear_coupling(state):
                return state @ wtg
            return _linear_coupling
        return make_coupling

    if kinds == {"diffusive"}:
        def make_coupling(self):
            wt = self.weights_t.copy()
            ink = wt.sum(axis=1)
            g = self.g

            @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE)
            def _diffusive_coupling(state):
                r = state @ wt
                return g * (r - ink * state)
            return _diffusive_coupling
        return make_coupling

    raise NotImplementedError(
        f"Unsupported coupling kinds {sorted(kinds)}. The DSL ships 'linear' "
        f"and 'diffusive'; subclass `Model` directly and override "
        f"`get_numba_coupling` for anything else."
    )


def build_model(spec: ModelSpec) -> Type[Model]:
    """Compile a ModelSpec into a `Model` subclass equivalent to a hand-written
    neuronumba model. The returned class has the same external contract."""

    # Validate that coupling vars are state vars.
    state_names = [sv.name for sv in spec.state_vars]
    for cv in spec.coupling_vars:
        if cv.name not in state_names:
            raise ValueError(
                f"Coupling var '{cv.name}' is not declared as a state var."
            )

    # Build the dfun source. We embed it into a complete module that imports
    # numpy (so numba can later compile from a real file path).
    dfun_body_src = _build_dfun_source(spec)
    full_src = (
        "# Auto-generated by nndsl. Do not edit.\n"
        "import numpy as np\n"
        "\n"
        f"{dfun_body_src}\n"
    )
    gen_module = _materialize_module(spec.name, full_src)
    factory = getattr(gen_module, f"make_{spec.name}_dfun")

    # Build the class body programmatically.
    cls_dict: Dict[str, Any] = {}

    # Class-level model-info attributes.
    cls_dict["_state_var_names"] = [sv.name for sv in spec.state_vars]
    cls_dict["_coupling_var_names"] = [cv.name for cv in spec.coupling_vars]
    cls_dict["_observable_var_names"] = list(spec.observables)
    bounds = {sv.name: sv.bounds for sv in spec.state_vars if sv.bounds is not None}
    if bounds:
        cls_dict["_state_var_bounds"] = bounds

    # Parameter Attrs. Independent params get `default`; dependents get
    # `dependant=True` so neuronumba's existing machinery treats them like
    # regional params that the model itself fills in during `_init_dependant`.
    independent_params = [p for p in spec.parameters if not p.is_dependent]
    dependent_params = [p for p in spec.parameters if p.is_dependent]
    for p in independent_params:
        attr_kwargs: Dict[str, Any] = {
            "default": p.default,
            "doc": p.doc,
        }
        if p.required:
            attr_kwargs["required"] = True
        attr_kwargs["attributes"] = (
            Model.Tag.REGIONAL if p.regional else Model.Tag.GLOBAL
        )
        cls_dict[p.name] = Attr(**attr_kwargs)
    for p in dependent_params:
        # `dependant=True` tells HasAttr/Model to skip default-setting and
        # required-checking; we'll fill these in during _init_dependant.
        cls_dict[p.name] = Attr(
            dependant=True,
            doc=p.doc,
            attributes=(Model.Tag.REGIONAL if p.regional else Model.Tag.GLOBAL),
        )

    # Topo-sort dependents once at build time and compile each formula.
    sorted_deps = _topo_sort_dependents(spec)
    compiled_formulas = [
        (p.name, compile(p.formula, f"<formula:{spec.name}.{p.name}>", "eval"))
        for p in sorted_deps
    ]

    # _init_dependant: evaluate formulas in order, on the instance. Runs at
    # configure() time, i.e. on construction and any subsequent configure() call,
    # which is the existing protocol for "parameter was modified" in neuronumba.
    #
    # Mutable container that will hold the generated class once `type()` returns
    # it below. Captured by the closure so `super()` always resolves against the
    # DSL-generated class, not against `type(self)`. Without this indirection,
    # subclassing the generated class would re-enter this method via super()
    # forever (because `type(self)` would be the user's subclass).
    _class_cell: List[Type[Model]] = [None]

    def _init_dependant(self):
        # Run the base-class hook first (sets n_rois, weights_t for
        # LinearCouplingModel, etc.). Without this, `self.weights` exists but
        # things like `self.weights_t` don't yet.
        super(_class_cell[0], self)._init_dependant()
        # Evaluation env: numpy + every parameter currently on self (independent
        # ones from the user, plus dependents already computed earlier in the
        # topo order).
        env: Dict[str, Any] = {"np": np}
        # Snapshot all params that exist so far.
        for p in spec.parameters:
            val = getattr(self, p.name, None)
            if val is not None:
                env[p.name] = val
        for name, code_obj in compiled_formulas:
            try:
                value = eval(code_obj, env)
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating dependent parameter '{name}' in model "
                    f"'{spec.name}': {e}"
                ) from e
            setattr(self, name, value)
            env[name] = value
    cls_dict["_init_dependant"] = _init_dependant

    # `g` is pulled in by LinearCouplingModel; we don't redeclare it.

    initial_overrides = dict(spec.initial_state_overrides)
    state_initials = [sv.initial for sv in spec.state_vars]

    def initial_state(self, n_rois: int) -> np.ndarray:
        out = np.empty((len(state_initials), n_rois))
        for i, init_val in enumerate(state_initials):
            out[i] = init_val
        for name, val in initial_overrides.items():
            if name in cls_dict["_state_var_names"]:
                out[cls_dict["_state_var_names"].index(name)] = val
        return out
    cls_dict["initial_state"] = initial_state

    # Capture spec metadata in the closure for error reporting.
    _spec_name = spec.name
    _spec_equations = spec.equations
    _source_file = gen_module.__file__

    def get_numba_dfun(self):
        m_local = self.m.copy()
        P_local = self.P
        py_dfun = factory(m_local, P_local)
        try:
            jitted = nb.njit(
                nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL,
            )(py_dfun)
        except nb.errors.NumbaError as e:
            raise RuntimeError(
                f"Could not compile dfun for DSL model '{_spec_name}'.\n"
                f"Generated source: {_source_file}\n"
                f"Spec equations:\n{_spec_equations}\n"
                f"Original numba error: {e}"
            ) from e
        return jitted
    cls_dict["get_numba_dfun"] = get_numba_dfun

    coupling_factory = _build_coupling_kernel(spec)
    cls_dict["get_numba_coupling"] = coupling_factory

    # Choose base class. LinearCouplingModel only adds `g` + `weights_t`. We
    # always inherit from it: even diffusive Hopf needs `g` and `weights_t`,
    # and for 'linear' it gives us the default coupling (which we override
    # only when we want — but we override always for clarity here).
    base = LinearCouplingModel

    cls_dict["__doc__"] = (
        f"DSL-generated whole-brain model: {spec.name}\n\n"
        f"State: {[sv.name for sv in spec.state_vars]}\n"
        f"Coupling: {[(cv.name, cv.kind) for cv in spec.coupling_vars]}\n"
        f"Observables: {spec.observables}\n"
        f"Parameters: {[p.name for p in spec.parameters]}\n"
    )
    cls_dict["__module__"] = "neuronumba.simulator.models.dsl.generated"

    cls = type(spec.name, (base,), cls_dict)
    cls.__nndsl_source_file__ = gen_module.__file__
    _class_cell[0] = cls
    return cls


def dump_generated(spec: ModelSpec) -> str:
    """Return the generated dfun source for inspection."""
    return _build_dfun_source(spec)


def get_source_file(cls: Type[Model]) -> str:
    """Return the on-disk path to the generated dfun source for a built model.

    Useful when numba reports a compile error pointing at the synthesized
    file: open the path, read the offending line, fix the spec.
    """
    try:
        return cls.__nndsl_source_file__
    except AttributeError:
        raise TypeError(
            f"{cls!r} does not look like a DSL-built model "
            f"(no `__nndsl_source_file__` attribute)."
        ) from None
