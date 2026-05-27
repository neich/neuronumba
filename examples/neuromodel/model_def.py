"""Pure-data structures describing a neuronumba model in progress.

No Qt and no neuronumba imports here on purpose, so this module is cheap to
import and easy to test in isolation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


_INF_TOKENS = {"inf": math.inf, "+inf": math.inf, "-inf": -math.inf}


def _f_to_json(v: float):
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _json_to_f(v, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, str):
        return _INF_TOKENS.get(v.strip().lower(), default)
    return float(v)


@dataclass
class ParamDef:
    name: str
    default: float = 0.0
    tag: str = "regional"
    doc: str = ""


@dataclass
class StateVarDef:
    name: str
    initial: float = 0.001
    lo: float = -math.inf
    hi: float = math.inf
    sigma: float = 0.0
    is_coupling: bool = False

    @property
    def has_bounds(self) -> bool:
        return math.isfinite(self.lo) or math.isfinite(self.hi)


@dataclass
class ObservableVarDef:
    name: str
    doc: str = ""


@dataclass
class DependantDef:
    name: str
    formula: str = ""   # Python code; must assign to a variable named `name`
    doc: str = ""


@dataclass
class ModelDef:
    name: str = "MyModel"
    state_vars: list[StateVarDef] = field(default_factory=list)
    observable_vars: list[ObservableVarDef] = field(default_factory=list)
    params: list[ParamDef] = field(default_factory=list)
    dependants: list[DependantDef] = field(default_factory=list)
    equations: str = ""

    @property
    def coupling_var_names(self) -> list[str]:
        return [v.name for v in self.state_vars if v.is_coupling]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state_vars": [
                {
                    "name": v.name,
                    "initial": v.initial,
                    "lo": _f_to_json(v.lo),
                    "hi": _f_to_json(v.hi),
                    "sigma": v.sigma,
                    "is_coupling": v.is_coupling,
                }
                for v in self.state_vars
            ],
            "observable_vars": [{"name": o.name, "doc": o.doc} for o in self.observable_vars],
            "params": [
                {"name": p.name, "default": p.default, "tag": p.tag, "doc": p.doc}
                for p in self.params
            ],
            "dependants": [
                {"name": d.name, "formula": d.formula, "doc": d.doc}
                for d in self.dependants
            ],
            "equations": self.equations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelDef":
        return cls(
            name=d.get("name", "MyModel"),
            state_vars=[
                StateVarDef(
                    name=v["name"],
                    initial=float(v.get("initial", 0.0)),
                    lo=_json_to_f(v.get("lo"), -math.inf),
                    hi=_json_to_f(v.get("hi"), math.inf),
                    sigma=float(v.get("sigma", 0.0)),
                    is_coupling=bool(v.get("is_coupling", False)),
                )
                for v in d.get("state_vars", [])
            ],
            observable_vars=[ObservableVarDef(**o) for o in d.get("observable_vars", [])],
            params=[ParamDef(**p) for p in d.get("params", [])],
            dependants=[DependantDef(**dd) for dd in d.get("dependants", [])],
            equations=d.get("equations", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelDef":
        return cls.from_dict(json.loads(s))
