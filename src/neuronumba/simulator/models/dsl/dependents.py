"""
Dependent-parameter analysis.

Parses each formula's AST, builds a DAG of dependencies, topo-sorts, and
returns dependent parameters in evaluation order. Cycles and self-references
raise ValueError.
"""
from __future__ import annotations

import ast
from typing import Dict, List

from .rewriter import ALLOWED_NP_FUNCS
from .spec import ModelSpec, Parameter


# Model-context attributes available inside dependent-param formulas. These
# are real attributes on the configured model instance (set by
# `LinearCouplingModel._init_dependant` before our hook runs); we expose
# them by name so formulas can reference e.g. `g * weights` or
# `np.eye(n_rois)`. They're allowed in formulas but do NOT create
# parameter-graph dependencies (they aren't declared params).
MODEL_CONTEXT_NAMES = frozenset({"weights", "weights_t", "g", "n_rois"})


def _formula_deps(name: str, src: str, allowed: set) -> List[str]:
    """Return the parameter names referenced in `src`. Validate identifiers."""
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Parameter '{name}' formula is not valid Python: {e}")

    deps: List[str] = []
    seen = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node):
            n = node.id
            if n == "np":
                return
            if n in ALLOWED_NP_FUNCS:
                return
            if n in MODEL_CONTEXT_NAMES:
                # Allowed but doesn't create a param-graph dependency.
                return
            if n not in allowed:
                raise ValueError(
                    f"Parameter '{name}': formula references unknown identifier "
                    f"'{n}' (line {node.lineno}). Must be another declared "
                    f"parameter, numpy, or one of {sorted(MODEL_CONTEXT_NAMES)}."
                )
            if n not in seen:
                deps.append(n)
                seen.add(n)

        def visit_Attribute(self, node):
            # Allow `np.something`, recurse into base.
            self.generic_visit(node)

    V().visit(tree)
    if name in deps:
        raise ValueError(f"Parameter '{name}' formula references itself.")
    return deps


def _topo_sort_dependents(spec: ModelSpec) -> List[Parameter]:
    """Return dependent parameters in evaluation order. Raise on cycles."""
    by_name = {p.name: p for p in spec.parameters}
    all_names = set(by_name.keys())

    deps_of: Dict[str, List[str]] = {}
    for p in spec.parameters:
        if p.is_dependent:
            deps_of[p.name] = _formula_deps(p.name, p.formula, all_names)

    # Kahn's algorithm restricted to the dependent subgraph.
    dependent_names = set(deps_of.keys())
    indeg: Dict[str, int] = {n: 0 for n in dependent_names}
    rev: Dict[str, List[str]] = {n: [] for n in dependent_names}
    for n, ds in deps_of.items():
        for d in ds:
            if d in dependent_names:
                indeg[n] += 1
                rev[d].append(n)

    ready = [n for n, k in indeg.items() if k == 0]
    order: List[str] = []
    while ready:
        n = ready.pop()
        order.append(n)
        for m in rev[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)

    if len(order) != len(dependent_names):
        unresolved = dependent_names - set(order)
        raise ValueError(
            f"Cycle detected among dependent parameters: {sorted(unresolved)}"
        )

    return [by_name[n] for n in order]
