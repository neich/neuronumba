# DSL roadmap — features deferred from v0.1

This document lists features that were deliberately left out of v0.1 in
order to ship the 80% case quickly. They're acknowledged here so users know
the gaps are real but tracked, not forgotten.

For each item we record: what it would look like as an API, why it was
deferred, and the open design questions that would need to be answered
before implementing it.

---

## 1. `on_configure` Python hook

**Status:** ✅ **DONE** in v0.1. See [README.md](README.md#the-on_configure-escape-hatch)
and `tests/test_dsl_on_configure.py`. Resolution of the open questions
below is recorded inline in the spec docstring on `ModelSpec.on_configure`.

**API sketch:**
```python
def auto_fic(self):
    self.J = FICHerzog2022().compute_J(self.weights, self.g)

deco_spec = ModelSpec(
    ...,
    on_configure=auto_fic,   # plain Python callable, runs after _init_dependant
)
```

**Why deferred:** the formula-based dependent-parameter mechanism already
covers most setup needs. `on_configure` is the imperative escape hatch for
fsolve-based FIC, steady-state computations, and similar — but we'd rather
ship after observing concrete user pain than guess the API up front.

**Today's workaround:** subclass the DSL-built class and override
`_init_dependant`. See [test_dsl_subclass.py](../../../../../tests/test_dsl_subclass.py)
for the pattern.

**Open questions:**
- Does the callback run before or after dependent-parameter evaluation?
  (Probably before, so the imperative setup can populate values that
  dependent formulas read.)
- Multiple hooks composable, or just one?
- Should the callback receive `self` or a reduced view of the model state?

---

## 2. `helpers=[fn1, fn2]` on `ModelSpec`

**Status:** ✅ **DONE** in v0.1. See [README.md](README.md#helpers-user-nbnjit-functions)
and `tests/test_dsl_helpers.py`. Helpers are bound as closure variables
inside the generated dfun factory; numba compiles closure-captured `@nb.njit`
callees natively. The validator catches name clashes (params, state vars,
np funcs, reserved tokens) at build time and rejects nameless callables
(lambdas) up front.

The Zerlaut port itself isn't done — that requires writing the spec.
Helpers were the missing infrastructure piece.

**API sketch:**
```python
@nb.njit(...)
def erfc_approx(x):
    ...

zerlaut_spec = ModelSpec(
    ...,
    helpers=[erfc_approx, threshold_func],
    equations="""
        s = erfc_approx(some_combination)
        ...
    """,
)
```

**Why deferred:** the AST rewriter currently rejects any name not in the
declared parameter/state/observable/whitelisted-numpy set. Helpers require
adding a per-spec namespace, plumbing the helper symbols through the
generated source, and ensuring numba can see them at compile time. Roughly
50-100 LOC of careful work, plus equivalence tests against a hand-written
Zerlaut.

**Today's workaround:** for models that need `@njit` helpers (Zerlaut),
subclass `Model` directly. See `src/neuronumba/simulator/models/zerlaut.py`.

**Open questions:**
- Validation: should helpers be type-checked at build time or trusted?
- Naming: `helpers=[fn]` (function objects, name from `__name__`) vs
  `helpers={'erfc': fn}` (explicit names)?

---

## 3. `kind="delayed"` coupling

**Status:** ✅ **DONE** in v0.1. See [README.md](README.md#coupling-kinds)
and `tests/test_dsl_delayed.py`.

**Notes from the implementation:**
- The pre-existing `HistoryDense` was broken (invalid numba types,
  inverted buffer indexing) and unused. We fixed and renamed it to
  `HistoryDelays`.
- `Simulator.run` now branches on `isinstance(history, HistoryDelays)`
  to pass `g`, `delays = lengths/speed`, and `dt` automatically.
- The DSL's `kind="delayed"` coupling kernel is identity — `HistoryDelays`
  does the W-weighted, g-scaled, per-pair-delayed sum and returns a
  `(n_cv, n_rois)` array directly.
- `get_jacobian()` raises `NotImplementedError` for delayed kernels.
- v0.2 limitation: all coupling vars must share the same kind. Mixing
  `delayed` with `linear`/`diffusive` raises at `build_model` time.

**API sketch:**
```python
coupling_vars=[CouplingVar("S_e", kind="delayed")]
```

**Why deferred:** the existing `linear` and `diffusive` kernels assume
`HistoryNoDelays`; delayed coupling needs a delayed coupled-state buffer
threaded through the simulator. The coupling-kernel factory in
[builder.py](builder.py) would need a new branch that reads from the delay
history rather than the current state.

**Today's workaround:** build a hand-written `Model` subclass with custom
`get_numba_coupling`. The DSL spec produces the dfun; you can mix.

**Open questions:**
- Does the spec carry the delay parameter, or does it come from the
  `Connectivity` (lengths/speed)?
- Mixed-kind coupling: can a single spec have one delayed and one linear
  coupling var?

---

## 4. Non-scalar dependent parameters

**Status:** ✅ **DONE** in v0.1. See [README.md](README.md#matrix-parameters-2d)
and `tests/test_dsl_matrix_params.py`.

**Notes from the implementation:**
- Added `shape="matrix"` to `Parameter`. Matrix params get NO Tag —
  they live as plain attributes on the model and are captured into the
  dfun's closure (same pattern as `helpers`), bypassing `self.m`
  (wrong shape) and `self.m_aux` (1D-only).
- Dependent-param formulas now have access to `weights`, `weights_t`,
  `g`, and `n_rois` (the `MODEL_CONTEXT_NAMES` set). Required for the
  OU-style use case where `A` is derived from connectivity.
- The factory signature gained a `_matrix_params` argument; the
  generated source emits `name = _matrix_params[i]` bindings at the
  factory level alongside helper bindings.
- Validated against a hand-derived OU operator
  `(-g * W + diag(rowsum)) / tau` and a generic `A @ x` rate.

**API sketch:**
```python
Parameter("A", formula="np.linalg.solve(...)", shape="matrix")
```

**Why deferred:** the current dependent-parameter machinery routes scalar
results through `self.m` (the regional parameter matrix). OU's `A` matrix
is shape `(n_rois, n_rois)` and belongs in `self.m_aux` instead. Routing is
a small change but needs care to keep the regional-vs-global distinction
clean.

**Today's workaround:** subclass `Model` and compute `A` in
`_init_dependant`.

**Open questions:**
- Inferred shape from the formula's output, or declared explicitly?
- How does the dfun reference an `m_aux` matrix parameter? Today's
  rewriter only emits `m[np.intp(P.x)]` bindings.

---

## 5. Auto-Jacobian via numpy finite differences

**Status:** ✅ **DONE** in v0.1. See [README.md](README.md#jacobians) and
`tests/test_dsl_jacobian.py`. The rest of this section is preserved as the
design rationale.

**API sketch:**
```python
m = HopfDSL(g=0.5).configure(weights=W)
J = m.get_jacobian(state)
# state shape: (n_state_vars, n_rois)
# J shape:     (n_state_vars * n_rois, n_state_vars * n_rois)
```

The DSL has everything it needs to build this without symbolic math: it
already knows the dfun, the coupling kind, the state-var layout, and the
coupling-var layout. The Jacobian splits cleanly into two pieces and we
can compute each numerically.

### Implementation sketch

```python
def get_jacobian(self, state):
    """Network Jacobian d(d_state)/d(state) at the operating point `state`."""
    N = self.n_rois
    nsv = self.n_state_vars
    ncv = len(self.c_vars)

    coupling_factory = self.get_numba_coupling()
    coupling = coupling_factory(state[self.c_vars, :])

    dfun = self.get_numba_dfun()
    base, _ = dfun(state.copy(), coupling.copy())

    # 1) Local partials  dD_u/dv  for each state var v.
    #    Shape: (nsv, nsv, N) — a per-region (nsv x nsv) Jacobian block.
    eps = 1e-6
    local = np.empty((nsv, nsv, N))
    for j in range(nsv):
        s_pert = state.copy()
        s_pert[j] += eps
        out, _ = dfun(s_pert, coupling.copy())
        local[:, j, :] = (out - base) / eps

    # 2) Coupling partials  dD_u/d(coupling.v)  for each coupling var v.
    #    Shape: (nsv, ncv, N).
    coup = np.empty((nsv, ncv, N))
    for k in range(ncv):
        c_pert = coupling.copy()
        c_pert[k] += eps
        out, _ = dfun(state.copy(), c_pert)
        coup[:, k, :] = (out - base) / eps

    # 3) Assemble network Jacobian.
    #    For state vars u, v at regions i, j:
    #        J[u_i, v_j] = local[u, v, i] * δ_ij
    #                    + sum over coupling vars k of coup[u, k, i] * C[k, i, j]
    #    where C[k, i, j] is the (i, j) entry of the linearization of the
    #    k-th coupling kernel — known analytically per kind:
    #        linear:    C[k, i, j] = g * W[j, i]
    #        diffusive: C[k, i, j] = g * (W[j, i] - δ_ij * sum_l W[l, i])
    # ... (loop and stack into a (nsv*N, nsv*N) matrix)
```

### Why not SymPy

SymPy's `lambdify` NumPy printer doesn't always emit numba-compatible
code. `Piecewise → np.select` (numba doesn't support `np.select`).
`Heaviside`, `KroneckerDelta`, complex broadcasts, and special functions
need custom printers. Every model that uses an unusual function becomes a
debugging trip into SymPy printer internals. Plus ~50 MB dependency for
something we don't otherwise need.

The numerical approach gets us:
- **No new dependency.** Pure numpy (already required).
- **~1e-7 accuracy.** Plenty for the downstream linear-noise covariance
  estimation that Hopf/Deco's analytic Jacobians feed.
- **Works for any model the DSL can build.** Including Zerlaut once
  `helpers=[]` lands, with no per-special-function derivative table.
- **Coupling structure is already known to the DSL.** No need to
  symbolically differentiate the coupling kernel — just write the closed
  form per kind.

### What we give up vs analytic

- ~6-7 digits of precision instead of full machine precision.
- Slightly slower (each Jacobian costs `(n_state_vars + n_coupling_vars)`
  dfun evaluations, where the analytic form is one numpy expression).
  Negligible at simulation-time cost; matters only if you call
  `get_jacobian` in a tight loop.

### Validation strategy

Equivalence test against the hand-written analytical Jacobians for Hopf
and Deco2014:

```python
def test_jacobian_finite_diff_matches_analytic(weights, hopf_dsl_cls):
    m_ref = Hopf(g=0.5).configure(weights=weights)
    m_dsl = hopf_dsl_cls(g=0.5).configure(weights=weights)
    state = np.zeros((2, weights.shape[0]))  # Hopf linearizes at origin
    J_ref = m_ref.get_jacobian(0.5 * weights)  # SC pre-multiplied
    J_dsl = m_dsl.get_jacobian(state)
    assert np.allclose(J_ref, J_dsl, atol=1e-5, rtol=1e-5)
```

The 1e-5 tolerance accommodates the finite-difference truncation error.

### Open questions

- **Step size.** Default to a fixed `eps=1e-6` or auto-scale per state-var
  magnitude (Numerical Recipes recommends `eps = sqrt(machine_eps) * |x|`
  with a floor)?
- **Centered vs forward differences.** Centered is twice the cost but
  ~10× more accurate. Probably worth it for a once-per-configure call.
- **Operating point inference.** Hopf linearizes at origin; Deco at a
  steady state. Should `get_jacobian()` (no args) attempt to infer a
  reasonable default, or always require an explicit `state`?

**Today's workaround:** hand-write `get_jacobian` on a subclass.

---

## 6. LaTeX or markdown-flavored equation syntax

**Status:** small. Pure UX. Lowest priority.

**API sketch:**
```python
ModelSpec(
    ...,
    equations_latex=r"\dot{x} = (a - x^2 - y^2) x - \omega y + ...",
)
```

**Why deferred:** zero expressive power gain over the current Python form,
and the Python form is already very readable. Worth doing only if users
specifically ask for it.

**Today's workaround:** the existing string form is already nearly
math-notation-equivalent if you mentally drop the `np.`s.

**Open questions:**
- LaTeX vs Markdown vs a third format?
- Bidirectional (round-trip back to LaTeX for documentation)?

---

## Suggested implementation order

Items 1 (`on_configure`), 2 (`helpers=[]`), 3 (`kind="delayed"`), 4
(non-scalar dependent params), and 5 (auto-Jacobian) shipped in v0.1.
Remaining queue:

1. **LaTeX syntax** — pure UX; lowest priority.

Only one v0.2 item left.
