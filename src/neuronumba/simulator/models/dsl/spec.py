"""
Spec dataclasses for the neuronumba DSL.

These are the user-facing declaration objects: a `ModelSpec` describes a whole
model in terms of its `StateVar`s, `CouplingVar`s, `Parameter`s, and an
equations string. They are pure data — no compilation logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class StateVar:
    name: str
    initial: float = 0.0
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class CouplingVar:
    """A state variable that participates in inter-region coupling.

    kind:
      'linear'    — coupling = g * (W^T @ S)                    (standard SC coupling)
      'diffusive' — coupling = g * (W^T @ S - sum(W^T,1) * S)   (Hopf-style)
      'delayed'   — coupling = g * sum_j W[i,j] * state[j, t-delay[i,j]]

    For coupling shapes outside this set, subclass `Model` directly and override
    `get_numba_coupling`.
    """
    name: str
    kind: str = "linear"


@dataclass
class Parameter:
    """A model parameter.

    A parameter is *independent* if it carries a `default` (or is `required`) and
    no `formula`. Independent params are user-settable.

    A parameter is *dependent* if it has a `formula`: a Python expression that
    may reference other parameters (independent or dependent, as long as no
    cycles), the model-context names ``weights``, ``weights_t``, ``g``,
    ``n_rois``, and ``np``. Dependents are NOT user-settable; they're computed
    at ``configure()`` time and re-computed every time the user calls
    ``configure()`` (i.e. every time params are modified).

    Whatever Python value the formula returns is what the parameter is — a
    scalar, a per-region 1D array, or a 2D matrix all work. The dfun captures
    the value by closure and numba's type inference handles it. There is no
    explicit shape declaration; if the formula returns ``np.eye(n_rois)``,
    the parameter is a matrix.

    Examples:
        Parameter("J_ee", default=10.0)
        Parameter("g_ee", default=2.5)
        Parameter("a_e",  default=0.25)
        Parameter("J_N_ee", formula="J_ee + g_ee * np.log(a_e)")  # scalar dependent
        Parameter("A", formula="(-g * weights + np.diag(weights.sum(axis=1))) / tau")  # matrix dependent
    """
    name: str
    default: Optional[float] = None
    required: bool = False
    doc: str = ""
    formula: Optional[str] = None

    def __post_init__(self):
        if self.formula is not None:
            if self.default is not None:
                raise ValueError(
                    f"Parameter '{self.name}' has both `default` and `formula`; "
                    f"a dependent parameter cannot have a default value."
                )
            if self.required:
                raise ValueError(
                    f"Parameter '{self.name}' has `formula` and is `required`; "
                    f"dependent parameters are computed, not user-supplied."
                )
        else:
            if self.default is None and not self.required:
                raise ValueError(
                    f"Parameter '{self.name}' needs either `default`, "
                    f"`required=True`, or a `formula`."
                )

    @property
    def is_dependent(self) -> bool:
        return self.formula is not None


@dataclass
class ModelSpec:
    name: str
    state_vars: List[StateVar]
    coupling_vars: List[CouplingVar]
    observables: List[str]
    parameters: List[Parameter]
    equations: str  # multiline; assignments of intermediates and `d_<state>` rates
    initial_state_overrides: Dict[str, float] = field(default_factory=dict)

    # Imperative escape hatch. Called as `on_configure(model)` once per
    # `configure()`, AFTER dependent-parameter evaluation. Use it for setup
    # that doesn't fit the declarative formula form: fsolve-based FIC,
    # iterative steady-state computation, anything that needs Python control
    # flow. Mutations to the model instance (e.g. `model.J = computed`) are
    # visible to the dfun the next time `get_numba_dfun()` is called.
    #
    # For parameters set by this callback, declare them with a placeholder
    # `default=` value; the callback overwrites it. The user can still
    # supply an override at construction time, in which case the callback
    # would also overwrite that — design accordingly.
    on_configure: Optional[Callable[[Any], None]] = None

    # User-supplied @nb.njit functions usable from `equations`. Each must
    # have a unique `__name__` that doesn't collide with state/coupling
    # variables, parameter names, declared observables, whitelisted numpy
    # functions, the reserved tokens (`state`, `coupling`, `model`, `np`),
    # or any inherited `Model`/`LinearCouplingModel` method name. Helpers
    # are stashed as class attributes on the generated class and referenced
    # via `model.<name>` in the dfun factory; numba sees the closure-captured
    # callable and dispatches to it during compilation.
    #
    # Use case: models with custom subroutines (e.g. Zerlaut's
    # `erfc_approx`, `threshold_func`, `get_fluct_regime_vars`).
    helpers: List[Callable] = field(default_factory=list)
