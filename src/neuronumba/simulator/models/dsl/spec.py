"""
Spec dataclasses for the neuronumba DSL.

These are the user-facing declaration objects: a `ModelSpec` describes a whole
model in terms of its `StateVar`s, `CouplingVar`s, `Parameter`s, and an
equations string. They are pure data — no compilation logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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

    For coupling shapes outside this set, subclass `Model` directly and override
    `get_numba_coupling`.
    """
    name: str
    kind: str = "linear"


@dataclass
class Parameter:
    """A model parameter.

    A parameter is *independent* if it carries a `default` (or is `required`) and
    no `formula`. Independent params are user-settable and packed into `self.m`.

    A parameter is *dependent* if it has a `formula`: a Python expression that
    may reference other parameters (independent or dependent, as long as no
    cycles). Dependents are NOT user-settable; they're computed at
    `configure()` time and re-computed every time the user calls `configure()`
    (i.e. every time params are modified — which is already the protocol the
    existing neuronumba models follow).

    Examples:
        Parameter("J_ee", default=10.0)
        Parameter("g_ee", default=2.5)
        Parameter("a_e",  default=0.25)
        Parameter("J_N_ee", formula="J_ee + g_ee * np.log(a_e)")  # Montbrio-style

    Formulas may use any of: numpy (as `np`), the standard math operators, and
    references to other parameter names. They evaluate in NumPy land (not numba)
    so anything numpy supports works, including matrix ops.
    """
    name: str
    default: Optional[float] = None
    regional: bool = True
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
