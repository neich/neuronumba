"""
AST rewriter for DSL equation bodies.

Transforms identifiers in user-supplied equations to the
`m[np.intp(P.x)]` / `state[i, :]` / `coupling[k, :]` form used by the
hand-written neuronumba dfuns, and validates that every identifier is
declared (state var, parameter, intermediate, or whitelisted numpy func).
"""
from __future__ import annotations

import ast
from typing import Dict, List


# Numpy functions allowed inside dfun bodies. Using a whitelist is the cheapest
# way to catch typos and to refuse names that numba won't compile anyway.
ALLOWED_NP_FUNCS = {
    "exp", "log", "log1p", "sqrt", "sin", "cos", "tan",
    "tanh", "sinh", "cosh", "abs", "where",
    "minimum", "maximum", "clip",
    "stack", "empty", "zeros", "ones",
    "pi",
}


class _DfunRewriter(ast.NodeTransformer):
    """Walks the user's equation AST and rewrites Names / Attributes."""

    def __init__(
        self,
        state_index: Dict[str, int],
        coupling_index: Dict[str, int],
        param_names: set,
        observable_names: set,
    ):
        self.state_index = state_index
        self.coupling_index = coupling_index
        self.param_names = param_names
        self.observable_names = observable_names
        self.intermediates: set = set()       # names introduced by user assignments
        self.used_state: set = set()
        self.used_params: set = set()
        self.errors: List[str] = []

    # --- assignment LHS handling ------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        # Targets: only simple names allowed (e.g. `Ie = ...` or `dSe = ...`).
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.errors.append(
                f"Line {node.lineno}: only single-name assignments are supported"
            )
            return node

        target = node.targets[0].id
        # If the target is `d_<state>`, mark it as a derivative (no rewriting needed —
        # we collect derivatives at emit time and assemble the return value ourselves).
        # Anything else is an intermediate (or an observable).
        if not (
            target.startswith("d_")
            and target[2:] in self.state_index
        ):
            self.intermediates.add(target)
        # Recurse into the RHS only.
        node.value = self.visit(node.value)
        return node

    # --- name resolution --------------------------------------------------
    def visit_Name(self, node: ast.Name):
        n = node.id

        if n in self.state_index:
            # Don't rewrite. State vars get bound as locals (`Se = state[0,:]`).
            self.used_state.add(n)
            return node

        if n in self.param_names:
            # Don't rewrite parameters. They get bound as locals at the top of
            # the generated function (e.g. `tau_e = m[np.intp(P.tau_e)]`).
            self.used_params.add(n)
            return node

        # Allowed: intermediates introduced earlier, observables, the special
        # tokens 'state', 'coupling', 'm', 'P', 'np', and known numpy functions.
        if n in self.intermediates:
            return node
        if n in {"state", "coupling", "m", "P", "np"}:
            return node
        if n in ALLOWED_NP_FUNCS:
            return node

        # Anything else is an undeclared identifier — error.
        self.errors.append(
            f"Line {node.lineno}: unknown identifier '{n}'. "
            f"Not a state var, parameter, intermediate, or allowed numpy func."
        )
        return node

    # --- coupling.<name> rewriting ----------------------------------------
    def visit_Attribute(self, node: ast.Attribute):
        # Match `coupling.S_e` etc. -> `coupling[k, :]`
        if isinstance(node.value, ast.Name) and node.value.id == "coupling":
            cname = node.attr
            if cname not in self.coupling_index:
                self.errors.append(
                    f"Line {node.lineno}: coupling.{cname} but '{cname}' is not "
                    f"declared as a coupling variable."
                )
                return node
            k = self.coupling_index[cname]
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id="coupling", ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[ast.Constant(value=k),
                              ast.Slice(lower=None, upper=None, step=None)],
                        ctx=ast.Load(),
                    ),
                    ctx=node.ctx,
                ),
                node,
            )
        # Otherwise allow (e.g. `np.exp`) — but recurse into the value side.
        node.value = self.visit(node.value)
        return node
