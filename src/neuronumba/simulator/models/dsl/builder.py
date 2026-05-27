"""
Source generators and `build_model` — the entry point that turns a `ModelSpec`
into a numba-friendly `Model` subclass.

Architecture: every name referenced in the dfun (parameters, helpers, etc.)
resolves through ``model.<name>`` at the factory level. The factory takes a
single argument — the model instance — and the inner dfun closes over the
captured locals. There is no `m`/`P`/`m_aux` packing for DSL-built classes;
parameters live as regular Python attributes on the instance.
"""
from __future__ import annotations

import ast
import textwrap
from typing import Any, Dict, List, Tuple, Type

import numba as nb
import numpy as np

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL
from neuronumba.simulator.models.model import Model, LinearCouplingModel

from .dependents import _topo_sort_dependents
from .materialize import _materialize_module
from .rewriter import ALLOWED_NP_FUNCS, _DfunRewriter
from .spec import ModelSpec


def _build_dfun_source(spec: ModelSpec) -> Tuple[str, List[str]]:
    """Return ``(source, used_names)`` for the dfun factory of `spec`.

    The source defines a ``make_<spec.name>_dfun(model)`` factory whose inner
    function is the dfun. ``used_names`` is the sorted list of param + helper
    names referenced by the equations; each gets a ``name = model.name``
    binding at the factory level so the inner dfun captures it via closure.
    """

    state_index = {sv.name: i for i, sv in enumerate(spec.state_vars)}
    coupling_index = {cv.name: i for i, cv in enumerate(spec.coupling_vars)}
    param_names = {p.name for p in spec.parameters}
    obs_names = set(spec.observables)
    helper_names = {h.__name__ for h in spec.helpers}

    src = textwrap.dedent(spec.equations).strip()
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        raise ValueError(
            f"Could not parse equations for model '{spec.name}': {e}"
        ) from e

    rewriter = _DfunRewriter(
        state_index, coupling_index, param_names, obs_names,
        helper_names=helper_names,
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

    # State-var bindings live inside the inner dfun (read from `state`).
    state_binds = "\n".join(
        f"        {sv.name} = state[{i}, :]"
        for i, sv in enumerate(spec.state_vars)
        if sv.name in rewriter.used_state
    )

    # All parameters and helpers bind once at the factory level via
    # `model.<name>`. Numba captures them as compile-time constants when the
    # inner dfun is JIT'd. Helper functions (also model attributes since
    # build_model stashes them on the class) come through the same path.
    used_names = sorted(rewriter.used_params | rewriter.used_helpers)
    factory_binds = "\n".join(
        f"    {name} = model.{name}" for name in used_names
    )

    indented_body = textwrap.indent(body_src, "        ")

    parts = [f"def make_{spec.name}_dfun(model):"]
    if factory_binds:
        parts.append(factory_binds)
    parts.append(f"    def {spec.name}_dfun(state, coupling):")
    if state_binds:
        parts.append(state_binds)
    parts.append(indented_body)
    parts.append(f"        return np.stack(({d_stack_args},)), {ret_obs}")
    parts.append(f"    return {spec.name}_dfun")
    fn_src = "\n".join(parts) + "\n"
    return fn_src, used_names


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

    if kinds == {"delayed"}:
        # `HistoryDelays.get_numba_sample` already computed
        # g * sum_j W[i,j] * state[k, j, t-delay[i,j]] and returned shape
        # (n_cvars, n_rois). The coupling step is therefore identity.
        def make_coupling(self):
            @nb.njit(nb.f8[:, :](nb.f8[:, :]), cache=NUMBA_CACHE)
            def _delayed_coupling(state):
                return state
            return _delayed_coupling
        return make_coupling

    if "delayed" in kinds:
        raise NotImplementedError(
            f"Mixing 'delayed' coupling with other kinds ({sorted(kinds)}) is "
            f"not supported in this version. Either make all coupling vars "
            f"'delayed' or none."
        )

    raise NotImplementedError(
        f"Unsupported coupling kinds {sorted(kinds)}. The DSL ships 'linear', "
        f"'diffusive', and 'delayed'; subclass `Model` directly and override "
        f"`get_numba_coupling` for anything else."
    )


# Names that the generated dfun source uses directly (function args, np
# alias) and that helpers must therefore not shadow.
_RESERVED_HELPER_NAMES = {"state", "coupling", "model", "np"}


def _validate_helpers(spec: ModelSpec) -> None:
    """Reject helper lists that would clash with other names in the dfun.

    A helper conflicts with the dfun namespace if its `__name__` matches a
    state var, coupling var, parameter, observable, whitelisted numpy
    function, a reserved token, or any inherited `LinearCouplingModel`
    method/attribute name (helpers are stashed as class attributes; a
    collision would shadow the inherited member). Two helpers with the same
    `__name__` are also a conflict.
    """
    if not spec.helpers:
        return

    seen: Dict[str, int] = {}
    for i, h in enumerate(spec.helpers):
        name = getattr(h, "__name__", None)
        if not name:
            raise ValueError(
                f"helpers[{i}] has no __name__; pass a regular function, not "
                f"a lambda or callable object."
            )
        if name in seen:
            raise ValueError(
                f"helpers[{i}] and helpers[{seen[name]}] both have "
                f"__name__='{name}'; helper names must be unique."
            )
        seen[name] = i

    # Names already on the base class (Model methods, HasAttr machinery).
    # Stashing a helper with the same __name__ would shadow them and break
    # the model contract.
    base_attrs = {n for n in dir(LinearCouplingModel) if not n.startswith("__")}

    forbidden = {
        sv.name for sv in spec.state_vars
    } | {
        cv.name for cv in spec.coupling_vars
    } | {
        p.name for p in spec.parameters
    } | set(spec.observables) | _RESERVED_HELPER_NAMES | ALLOWED_NP_FUNCS | base_attrs

    clashes = set(seen) & forbidden
    if clashes:
        raise ValueError(
            f"Helper name(s) {sorted(clashes)} clash with state vars, "
            f"coupling vars, parameters, observables, reserved tokens, "
            f"whitelisted numpy functions, or base-class methods. "
            f"Rename the helpers."
        )


def _coupling_kernel_jacobian(g: float, W: np.ndarray, kind: str) -> np.ndarray:
    """Closed-form Jacobian of one coupling kernel.

    Returns ``C`` of shape ``(N, N)`` where
    ``C[i, j] = d coupling[i] / d state_at_coupling_var[j]``.
    """
    if kind == "linear":
        # coupling[i] = sum_j state[j] * g * W[i, j]
        return g * W
    if kind == "diffusive":
        # coupling[i] = g * (sum_j state[j] * W[i, j] - ink[i] * state[i])
        # ink[i] = sum_l W[l, i]  (col-sum of W; row-sum if W is symmetric)
        ink = W.T.sum(axis=1)
        return g * (W - np.diag(ink))
    if kind == "delayed":
        # The Jacobian of a delay-differential system isn't a single matrix
        # — it's a delay-coupled spectrum. Linear stability for delayed
        # systems needs different tooling (DDE-Biftool, Lambert-W methods).
        raise NotImplementedError(
            "get_jacobian is not supported for 'delayed' coupling kernels. "
            "The Jacobian of a delay-differential system is not a single "
            "matrix. Use a no-delay variant (replace 'delayed' with 'linear' "
            "or 'diffusive') for linear stability analysis, or analyse the "
            "system with DDE-aware tooling."
        )
    raise NotImplementedError(
        f"Jacobian not implemented for coupling kind '{kind}'. "
        f"Subclass and override `get_jacobian` for custom kernels."
    )


def _compute_jacobian(model, state, eps: float = 1e-6) -> np.ndarray:
    """Two-piece numerical Jacobian: local FD + closed-form coupling assembly.

    Used as the body of `get_jacobian` on every DSL-built class. See the
    docstring on the class method for shape and indexing conventions.
    """
    state = np.asarray(state, dtype=np.float64)
    nsv = model.n_state_vars
    N = model.n_rois
    if state.shape != (nsv, N):
        raise ValueError(
            f"state must have shape ({nsv}, {N}); got {state.shape}"
        )

    c_vars = list(model.c_vars)
    ncv = len(c_vars)

    # Get the current dfun + coupling closures (capture current params).
    dfun = model.get_numba_dfun()
    coupling_factory = model.get_numba_coupling()
    coupling = coupling_factory(state[c_vars, :].copy())  # shape (ncv, N)

    # 1) Local partials: dD_u/d(state_v) at each region.
    #    The dfun is per-region pointwise (no cross-region dependence),
    #    so perturbing the entire row of state_v gives the per-region
    #    derivative with no cross-talk.
    local = np.empty((nsv, nsv, N))
    for v in range(nsv):
        s_plus = state.copy(); s_plus[v] += eps
        s_minus = state.copy(); s_minus[v] -= eps
        out_plus, _ = dfun(s_plus, coupling.copy())
        out_minus, _ = dfun(s_minus, coupling.copy())
        local[:, v, :] = (out_plus - out_minus) / (2.0 * eps)

    # 2) Coupling partials: dD_u/d(coupling_k) at each region.
    coup = np.empty((nsv, ncv, N))
    for k in range(ncv):
        c_plus = coupling.copy(); c_plus[k] += eps
        c_minus = coupling.copy(); c_minus[k] -= eps
        out_plus, _ = dfun(state.copy(), c_plus)
        out_minus, _ = dfun(state.copy(), c_minus)
        coup[:, k, :] = (out_plus - out_minus) / (2.0 * eps)

    # 3) Closed-form coupling-kernel Jacobians, one (N, N) matrix per cv.
    coupling_jacobians = [
        _coupling_kernel_jacobian(model.g, model.weights, kind)
        for kind in model._coupling_var_kinds
    ]

    # 4) Assemble the network Jacobian:
    #    J[u*N + i, v*N + j] = local[u, v, i] * delta_ij
    #                        + sum_k (v == c_vars[k]) * coup[u, k, i] * C_k[i, j]
    J = np.zeros((nsv * N, nsv * N))
    for u in range(nsv):
        for v in range(nsv):
            J[u * N:(u + 1) * N, v * N:(v + 1) * N] = np.diag(local[u, v, :])
    for k in range(ncv):
        v = c_vars[k]
        C_k = coupling_jacobians[k]
        for u in range(nsv):
            J[u * N:(u + 1) * N, v * N:(v + 1) * N] += coup[u, k, :, None] * C_k

    return J


def build_model(spec: ModelSpec) -> Type[Model]:
    """Compile a ModelSpec into a `Model` subclass with the same external
    contract as a hand-written neuronumba model."""

    # Validate that coupling vars are state vars.
    state_names = [sv.name for sv in spec.state_vars]
    for cv in spec.coupling_vars:
        if cv.name not in state_names:
            raise ValueError(
                f"Coupling var '{cv.name}' is not declared as a state var."
            )

    # Validate helpers up-front.
    _validate_helpers(spec)

    # Build the dfun source. Embed in a module that imports numpy so numba
    # can compile from a real file path.
    dfun_body_src, _used_names = _build_dfun_source(spec)
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
    cls_dict["_coupling_var_kinds"] = [cv.kind for cv in spec.coupling_vars]
    cls_dict["_observable_var_names"] = list(spec.observables)
    bounds = {sv.name: sv.bounds for sv in spec.state_vars if sv.bounds is not None}
    if bounds:
        cls_dict["_state_var_bounds"] = bounds

    # Parameter Attrs. No `attributes=Tag.REGIONAL` — DSL params do not flow
    # through the inherited `_init_dependant_automatic` packing path
    # (see the no-op override below). The Attr is purely for default-setting
    # and `__init__` kwarg validation.
    independent_params = [p for p in spec.parameters if not p.is_dependent]
    dependent_params = [p for p in spec.parameters if p.is_dependent]
    for p in independent_params:
        attr_kwargs: Dict[str, Any] = {"default": p.default, "doc": p.doc}
        if p.required:
            attr_kwargs["required"] = True
        cls_dict[p.name] = Attr(**attr_kwargs)
    for p in dependent_params:
        # `dependant=True` skips default-setting and tells HasAttr to refuse
        # the param as an __init__ kwarg. We populate it in _init_dependant.
        cls_dict[p.name] = Attr(dependant=True, doc=p.doc)

    # Stash helpers as class attributes so the factory can resolve them
    # via `model.<name>`. They're shared across all instances.
    for helper in spec.helpers:
        cls_dict[helper.__name__] = helper

    # Override the inherited m/m_aux packing to a no-op. DSL parameters live
    # as plain instance attributes; the dfun captures them via closure.
    def _init_dependant_automatic(self):
        pass
    cls_dict["_init_dependant_automatic"] = _init_dependant_automatic

    # Topo-sort dependents once at build time and pre-compile each formula.
    sorted_deps = _topo_sort_dependents(spec)
    compiled_formulas = [
        (p.name, compile(p.formula, f"<formula:{spec.name}.{p.name}>", "eval"))
        for p in sorted_deps
    ]

    on_configure_cb = spec.on_configure

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

    # Capture spec metadata for error reporting.
    _spec_name = spec.name
    _spec_equations = spec.equations
    _source_file = gen_module.__file__

    def get_numba_dfun(self):
        py_dfun = factory(self)
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

    def get_jacobian(self, state, eps: float = 1e-6) -> np.ndarray:
        """Network Jacobian d(d_state)/d(state) at the operating point ``state``.

        Computed via centered finite differences on the dfun, combined with
        the closed-form linearization of the coupling kernel (which the DSL
        already knows from each `CouplingVar.kind`).

        Args:
            state: shape ``(n_state_vars, n_rois)`` operating point.
            eps: finite-difference step. Default ``1e-6`` gives roughly
                ``1e-10`` accuracy with centered differences.

        Returns:
            Jacobian of shape ``(n_state_vars * n_rois, n_state_vars * n_rois)``.
            Indexing follows the convention ``J[u*N + i, v*N + j] = d(dfun_u
            at region i)/d(state_v at region j)``.
        """
        return _compute_jacobian(self, state, eps)
    cls_dict["get_jacobian"] = get_jacobian

    # Inherit from LinearCouplingModel: we get `g`, `weights_t`, the base
    # `_init_dependant` (sets weights_t/n_rois), and `get_numba_validate`.
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

    # Define `_init_dependant` AFTER class creation so the closure captures
    # the real `cls` directly (no `_class_cell[0]` indirection). Defines
    # super() resolution to walk past `cls` regardless of the runtime type
    # of `self` — crucial when users subclass the DSL-generated class.
    def _init_dependant(self):
        super(cls, self)._init_dependant()
        # Evaluation env: numpy + model-context attrs + every parameter
        # currently on self (independents from the user, dependents already
        # computed earlier in the topo order).
        env: Dict[str, Any] = {"np": np}
        for ctx_name in ("weights", "weights_t", "g", "n_rois"):
            val = getattr(self, ctx_name, None)
            if val is not None:
                env[ctx_name] = val
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
            # Contiguify any 2D-or-higher array result so the dfun's
            # closure capture sees C-contiguous memory (numba prefers it).
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                value = np.ascontiguousarray(value)
            setattr(self, name, value)
            env[name] = value
        # User imperative hook runs last — sees fully-evaluated dependents,
        # can freely mutate self. Mutations are visible to the dfun the next
        # time `get_numba_dfun()` is called (factory reads `model.<name>`
        # fresh at that point).
        if on_configure_cb is not None:
            on_configure_cb(self)
    cls._init_dependant = _init_dependant

    return cls


def dump_generated(spec: ModelSpec) -> str:
    """Return the generated dfun source for inspection."""
    src, _ = _build_dfun_source(spec)
    return src


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
