"""
Incremental builder for DSL ModelSpec.

`ModelBuilder` lets you assemble a model piece by piece — adding state vars,
coupling vars, parameters, observables, helpers, and equations one call at a
time — instead of constructing a full `ModelSpec` literal up front. Every
mutating method returns `self`, so calls can be chained or run imperatively.

Two terminal methods:
  - `.spec()`  → returns the assembled `ModelSpec` (no compilation).
  - `.build()` → compiles via `build_model(spec)` and returns the model class.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .builder import build_model
from .spec import CouplingVar, ModelSpec, Parameter, StateVar


class ModelBuilder:
    """Fluent, incremental builder for `ModelSpec`.

    Example:
        Cls = (ModelBuilder("Hopf")
            .add_state("x", initial=0.0)
            .add_state("y", initial=0.0)
            .add_coupling("x", kind="diffusive")
            .add_param("a", default=-0.5)
            .add_param("omega", default=0.3)
            .add_equation("d_x = (a - x*x - y*y)*x - omega*y + coupling.x")
            .add_equation("d_y = (a - x*x - y*y)*y + omega*x")
            .build())
    """

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("ModelBuilder requires a non-empty `name` string")
        self._name: str = name
        self._states: List[StateVar] = []
        self._couplings: List[CouplingVar] = []
        self._params: List[Parameter] = []
        self._observables: List[str] = []
        self._helpers: List[Callable] = []
        self._equations: List[str] = []
        self._initial_overrides: Dict[str, float] = {}
        self._on_configure: Optional[Callable] = None

    def add_state(
        self,
        name: str,
        initial: float = 0.0,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> "ModelBuilder":
        if any(s.name == name for s in self._states):
            raise ValueError(f"Duplicate state variable: {name!r}")
        self._states.append(StateVar(name=name, initial=initial, bounds=bounds))
        return self

    def add_coupling(self, name: str, kind: str = "linear") -> "ModelBuilder":
        if any(c.name == name for c in self._couplings):
            raise ValueError(f"Duplicate coupling variable: {name!r}")
        self._couplings.append(CouplingVar(name=name, kind=kind))
        return self

    def add_param(
        self,
        name: str,
        default: Optional[float] = None,
        formula: Optional[str] = None,
        required: bool = False,
        doc: str = "",
    ) -> "ModelBuilder":
        if any(p.name == name for p in self._params):
            raise ValueError(f"Duplicate parameter: {name!r}")
        self._params.append(Parameter(
            name=name,
            default=default,
            formula=formula,
            required=required,
            doc=doc,
        ))
        return self

    def add_observable(self, name: str) -> "ModelBuilder":
        if name in self._observables:
            raise ValueError(f"Duplicate observable: {name!r}")
        self._observables.append(name)
        return self

    def add_helper(self, fn: Callable) -> "ModelBuilder":
        if not callable(fn):
            raise TypeError("add_helper expects a callable")
        if any(getattr(h, "__name__", None) == getattr(fn, "__name__", None)
               for h in self._helpers):
            raise ValueError(f"Helper {getattr(fn, '__name__', fn)!r} already added")
        self._helpers.append(fn)
        return self

    def add_equation(self, line: str) -> "ModelBuilder":
        """Append one line to the equations block."""
        if not isinstance(line, str) or not line.strip():
            raise ValueError("add_equation expects a non-empty string")
        self._equations.append(line.rstrip())
        return self

    def set_equations(self, source: str) -> "ModelBuilder":
        """Replace the equations block wholesale (for multi-line literals)."""
        if not isinstance(source, str):
            raise TypeError("set_equations expects a string")
        self._equations = [source]
        return self

    def override_initial(self, name: str, value: float) -> "ModelBuilder":
        self._initial_overrides[name] = float(value)
        return self

    def on_configure(self, fn: Callable) -> "ModelBuilder":
        if not callable(fn):
            raise TypeError("on_configure expects a callable")
        self._on_configure = fn
        return self

    def spec(self) -> ModelSpec:
        if not self._states:
            raise ValueError(
                "Model has no state variables; call add_state() at least once"
            )
        if not self._equations:
            raise ValueError(
                "Model has no equations; call add_equation() or set_equations()"
            )
        return ModelSpec(
            name=self._name,
            state_vars=list(self._states),
            coupling_vars=list(self._couplings),
            observables=list(self._observables),
            parameters=list(self._params),
            equations="\n".join(self._equations),
            initial_state_overrides=dict(self._initial_overrides),
            on_configure=self._on_configure,
            helpers=list(self._helpers),
        )

    def build(self):
        return build_model(self.spec())
