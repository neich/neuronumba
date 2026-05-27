"""
AST rewriter for DSL equation bodies.

Transforms identifiers in user-supplied equations and validates that every
identifier is declared (state var, parameter, intermediate, helper, or
whitelisted numpy func). State vars and coupling refs get rewritten to
indexed access; everything else is left as-is and bound at the factory
level via `model.<name>` closure capture.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional


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
        helper_names: Optional[set] = None,
    ):
        self.state_index = state_index
        self.coupling_index = coupling_index
        self.param_names = param_names
        self.observable_names = observable_names
        self.helper_names = helper_names or set()
        self.intermediates: set = set()       # names introduced by user assignments
        self.used_state: set = set()
        self.used_params: set = set()
        self.used_helpers: set = set()
        self.errors: List[str] = []

    # --- assignment LHS handling ------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        # Single-target assigns: either `name = ...` (intermediate / derivative)
        # or `a, b, c = helper(...)` (tuple-unpacking from a multi-return helper).
        # Chained assignment (`a = b = ...`) and starred unpacking are rejected.
        if len(node.targets) != 1:
            self.errors.append(
                f"Line {node.lineno}: only single-target assignments are supported"
            )
            return node

        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            target = tgt.id
            # If the target is `d_<state>`, mark it as a derivative (no rewriting
            # needed — we collect derivatives at emit time). Anything else is an
            # intermediate (or an observable).
            if not (
                target.startswith("d_")
                and target[2:] in self.state_index
            ):
                self.intermediates.add(target)
        elif isinstance(tgt, ast.Tuple):
            # `mu_V, sigma_V, T_V = get_fluct_regime_vars(...)` — every name in
            # the tuple becomes an intermediate. Nested tuples and starred
            # targets are rejected for simplicity.
            for elt in tgt.elts:
                if not isinstance(elt, ast.Name):
                    self.errors.append(
                        f"Line {node.lineno}: tuple-unpacking targets must be "
                        f"plain names; got {ast.dump(elt)}"
                    )
                    return node
                self.intermediates.add(elt.id)
        else:
            self.errors.append(
                f"Line {node.lineno}: assignment target must be a name or a "
                f"tuple of names; got {type(tgt).__name__}"
            )
            return node

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
            # Parameters get bound at factory level via `model.<name>` closure
            # capture; we don't rewrite the AST node.
            self.used_params.add(n)
            return node

        # Allowed: intermediates introduced earlier, observables, the special
        # tokens 'state', 'coupling', 'model', 'np', known numpy functions,
        # and user-supplied helper functions.
        if n in self.intermediates:
            return node
        if n in {"state", "coupling", "model", "np"}:
            return node
        if n in ALLOWED_NP_FUNCS:
            return node
        if n in self.helper_names:
            self.used_helpers.add(n)
            return node

        # Anything else is an undeclared identifier — error.
        self.errors.append(
            f"Line {node.lineno}: unknown identifier '{n}'. "
            f"Not a state var, parameter, intermediate, helper, or allowed "
            f"numpy func."
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
