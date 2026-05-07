# DSL: declarative whole-brain models for neuronumba

A `ModelSpec` describes a neuronumba model as data: state variables, coupling
variables, parameters, and an equations string. `build_model(spec)` compiles
that into a `Model` subclass that plugs into `Simulator` exactly like a
hand-written model — same numba compilation, same cache behavior, same
trajectories.

## Why a DSL

Every model in `neuronumba/simulator/models/` is a `Model` subclass that
hand-implements four things: state-variable names, coupling kernel, parameter
list, and a numba-compiled `dfun`. Most of that is boilerplate — parameter
unpacking, state-variable indexing, return-tuple assembly. A "small
variation" — swap a sigmoid for a softplus in Deco2014 — currently means a
new file.

The DSL pitch: write only the math, the parameter list, and the topology
declarations. The boilerplate is generated. The DSL-built class satisfies
the same `Model` interface as a hand-written one — proven on Hopf, Deco2014,
Naskar2021, and Montbrio with random-input dfun comparisons that agree to
machine precision (see `tests/test_dsl_equivalence.py`).

## Hello world

```python
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)

hopf_spec = ModelSpec(
    name="HopfDSL",
    state_vars=[
        StateVar("x", initial=0.1),
        StateVar("y", initial=0.1),
    ],
    coupling_vars=[
        CouplingVar("x", kind="diffusive"),
        CouplingVar("y", kind="diffusive"),
    ],
    observables=[],
    parameters=[
        Parameter("a",          default=-0.5),
        Parameter("omega",      default=0.3),
        Parameter("I_external", default=0.0),
    ],
    equations="""
        d_x = (a - x*x - y*y) * x - omega * y + coupling.x + I_external
        d_y = (a - x*x - y*y) * y + omega * x + coupling.y
    """,
)

HopfDSL = build_model(hopf_spec)

m = HopfDSL(g=0.5).configure(weights=W)
```

The hand-written `Hopf` is 152 LOC. The DSL spec is ~30 LOC of declaration.
Both produce identical trajectories.

## ModelSpec fields

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Class name of the generated model |
| `state_vars` | `list[StateVar]` | Row order matters: defines `state[0, :]`, `state[1, :]`, ... |
| `coupling_vars` | `list[CouplingVar]` | Subset of state vars that are coupled across regions |
| `observables` | `list[str]` | Names of intermediates the simulator should expose for monitoring |
| `parameters` | `list[Parameter]` | Independent + dependent (formula-driven) parameters |
| `equations` | `str` | Python-source body. See "The equation language" below |
| `initial_state_overrides` | `dict[str, float]` | Optional per-state-var initial-value overrides |

`StateVar(name, initial=0.0, bounds=None)`. `bounds` is an optional
`(lo, hi)` tuple — passed straight to neuronumba's bounds-clamping pipeline.

## Parameters

A parameter has one of two flavors. Both kinds become regular Python
attributes on the configured model instance; the dfun captures them by
closure.

### Independent

A parameter is independent if it carries `default=` (or `required=True`) and
no `formula`. Independent parameters are user-settable through the
constructor:

```python
Parameter("tau_e", default=100.0)        # default value
Parameter("g",     required=True)        # caller must supply
```

### Dependent

A parameter is dependent if it carries a `formula`. The formula is a Python
expression that may reference other parameters, `np`, and the model-context
names `weights`, `weights_t`, `g`, `n_rois`. Dependents are computed at
`configure()` time and recomputed every time the user calls `configure()`
again — that's the existing neuronumba protocol for "a parameter was
modified":

```python
Parameter("J_ee",   default=10.0),
Parameter("g_ee",   default=2.5),
Parameter("a_e",    default=0.25),
Parameter("J_N_ee", formula="J_ee + g_ee * np.log(a_e)"),

# Matrix-shaped dependents: just return a 2D ndarray from the formula.
Parameter("A", formula="(-g * weights + np.diag(weights.sum(axis=1))) / tau"),
```

Order of declaration doesn't matter — dependents are topologically sorted
at build time. Cycles and self-references raise `ValueError`. Multi-level
dependencies work: a dependent can reference another dependent.

### Shapes

There's no explicit shape declaration. Whatever the user supplies (or the
formula returns) is the parameter's value: scalar, per-region 1D array, or
2D matrix. The dfun captures it via closure and numba's type inference
handles it. Use `A @ x` for matrix-vector, `tau_e * S_e` for scalar or
per-region — the same expression works whether `tau_e` is a scalar or an
`(n_rois,)` array.

### Why no `default + formula`?

Setting both is a signal of confused intent: a dependent value cannot also
have a default. The validator rejects this combination at `Parameter`
construction time. Same for `required=True` + `formula`.

## The `on_configure` escape hatch

Some setup doesn't fit the declarative `formula=` form: FIC needs `fsolve`,
steady-state computations need iteration, parameters might need to be
derived from `self.weights` and `self.g` together. For those cases pass an
`on_configure` callback to `ModelSpec`:

```python
from neuronumba.fitting.fic.fic import FICHerzog2022

def auto_fic(self):
    self.J = FICHerzog2022().compute_J(self.weights, self.g)

deco_spec = ModelSpec(
    ...,
    parameters=[
        ...,
        Parameter("J", default=1.0),    # placeholder; auto_fic overwrites
    ],
    on_configure=auto_fic,
)
```

The callback fires once per `configure()`, *after* dependent-parameter
evaluation. Whatever the callback assigns to `self` (e.g. `self.J = ...`)
persists on the instance and is visible to the dfun the next time
`get_numba_dfun()` is called — the factory reads `model.<name>` fresh
at that point.

When to use:
- **Imperative computation** that needs Python control flow (loops,
  fsolve, iterative steady-state).
- **Cross-parameter setup** that depends on multiple inputs in non-formula
  ways (e.g. solving a linear system).

When **not** to use:
- A simple formula like `J_N_ee = J_ee + g_ee * np.log(a_e)` — use
  `Parameter(..., formula=...)` instead. The DSL handles ordering and
  recompute-on-modify automatically.

## Helpers: user `@nb.njit` functions

Some models need custom subroutines that aren't in the whitelisted numpy
function set — e.g. Zerlaut's `erfc_approx` or threshold-piecewise
functions. Pass them via `helpers=[...]`:

```python
import numba as nb

@nb.njit(nb.f8[:](nb.f8[:]), cache=False)
def erfc_approx(x):
    # ... implementation ...

@nb.njit(nb.f8[:](nb.f8[:], nb.f8[:]), cache=False)
def threshold_func(x, theta):
    # ... implementation ...

zerlaut_spec = ModelSpec(
    ...,
    helpers=[erfc_approx, threshold_func],
    equations="""
        s = erfc_approx(some_combination)
        rate = threshold_func(I, theta)
        d_S = ...
    """,
)
```

The DSL binds each helper as a closure variable inside the generated dfun,
so numba sees and compiles them as regular jitted callees.

### Constraints

- **Each helper must be `@nb.njit`-decorated.** Passing a plain Python
  function will compile-fail when numba reaches the call site.
- **Each helper needs a unique `__name__`.** Lambdas and other callables
  without a usable name are rejected up front.
- **Names must not clash** with state vars, coupling vars, parameter
  names, declared observables, whitelisted numpy functions, or the
  reserved tokens (`state`, `coupling`, `m`, `P`, `np`).
- **Unused helpers are silently dropped** — list one in `helpers=[...]`
  and never call it from the equations and the generated dfun simply
  doesn't bind it.

## Coupling kinds

The DSL ships two pre-built kernels:

### `linear`

```
coupling = g * (W^T @ S)
```

Standard structural-connectivity coupling. Most neuronumba models use this.

### `diffusive`

```
coupling = g * (W^T @ S - sum(W^T, axis=1) * S)
```

Hopf-style: subtracts the local self-loop contribution from each region.

### `delayed`

```
coupling[i] = g * sum_j W[i, j] * state[j, t - delay[i, j]]
```

Conduction-delay coupling: each region receives signals from others
shifted in time by `delay[i, j] = lengths[i, j] / speed`. Use this when
modelling realistic propagation along white-matter tracts where the
~5-30 ms delay matters for phase relations.

To run a delayed simulation, swap the simulator's history class:

```python
from neuronumba.simulator.history import HistoryDelays
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.simulator import Simulator

sim = Simulator(
    connectivity=Connectivity(weights=W, lengths=L, speed=10.0),
    model=DelayedDSL(g=0.5),
    history=HistoryDelays(),       # not HistoryNoDelays
    integrator=integrator,
    monitors=[monitor],
)
sim.run(0, t_max)
```

The `Simulator` automatically passes `lengths/speed` (and the model's `g`,
the integrator's `dt`) to `HistoryDelays.configure()`.

**Limitations in v0.2:**
- All coupling vars in a spec must use the same kind. Mixing `delayed`
  with `linear` or `diffusive` raises at `build_model` time.
- `get_jacobian()` raises `NotImplementedError` for delayed kernels —
  the Jacobian of a delay-differential system isn't a single matrix.

For anything else — custom expressions, mixed kernels — subclass `Model`
directly and override `get_numba_coupling`. The DSL deliberately keeps
the coupling surface narrow; complex coupling deserves the imperative
escape hatch.

## The equation language

The `equations` string is Python source with extra constraints. It's parsed
with `ast.parse`, validated, and rewritten — not interpreted as a new
language. So if you can write it in numba, you can write it here.

### What's allowed

- **State variables** declared in `state_vars`. Inside equations, `x` refers
  to `state[i, :]` where `i` is the row index. The DSL handles the unpacking.
- **Parameters** by name. The DSL emits `tau_e = model.tau_e` bindings at
  the top of the generated factory; the inner dfun captures them by closure.
- **Intermediates** introduced by assignment: `Ie = ...` is a fine local
  variable. It must be assigned before it's used.
- **`coupling.<name>`** for declared coupling vars. Rewritten to
  `coupling[k, :]` where `k` is the index of that coupling var.
- **`np.<func>`** for these whitelisted numpy functions:
  `exp, log, log1p, sqrt, sin, cos, tan, tanh, sinh, cosh, abs, where,
  minimum, maximum, clip, stack, empty, zeros, ones, pi`.
- Standard arithmetic operators.

### What's required

- Every state variable `x` must have a `d_x = ...` assignment.
- Every declared `observables` entry must be assigned.

### Names with special meaning

- `d_<name>` is the time derivative of state variable `<name>`.
- `coupling.<name>` is the per-region coupled state for that coupling var.
- `np` is `numpy`.

### Errors caught at build time

The AST rewriter validates names before numba ever runs. Typos surface with
line numbers in the spec, not as cryptic numba traceback frames:

```
ValueError: Equation errors in model 'HopfDSL':
  - Line 2: unknown identifier 'omeg'. Not a state var, parameter,
    intermediate, or allowed numpy func.
```

## Inspecting generated code

`dump_generated(spec)` returns the synthesized dfun source as a string.
Useful when numba complains about compilation, or just to verify the rewriter
did what you expected:

```python
from neuronumba.simulator.models.dsl import dump_generated
print(dump_generated(hopf_spec))
```

The materialized `.py` file lives under `$TMPDIR/nndsl_generated/` and is the
real file numba caches against — same spec hash → same file → same cached
compilation. Two helpers for navigating the cache:

```python
from neuronumba.simulator.models.dsl import get_source_file, get_cache_dir

print(get_cache_dir())                # current cache root
print(get_source_file(HopfDSL))       # path to the .py for a built model
```

### Configurable cache directory

By default the cache lives under `$TMPDIR/nndsl_generated/`. On shared
compute clusters where `$TMPDIR` is per-job and gets cleaned, this defeats
numba's compilation cache between runs. Override with the
`NEURONUMBA_DSL_CACHE_DIR` environment variable to point at persistent
storage:

```bash
export NEURONUMBA_DSL_CACHE_DIR=/scratch/$USER/nndsl_cache
```

The env var is read on every cache-dir access, so changing it mid-session
takes effect for subsequent `build_model` calls.

### Numba compile errors

If numba can't compile a generated dfun, the `RuntimeError` you get includes
the spec name, the path to the synthesized `.py` file, and the original
spec equations — open the file, find the line numba was unhappy with, fix
the spec.

### Cleaning up stale generated files

Every spec change writes a new `.py` file. `cleanup_cache(max_age_days=7)`
removes files older than the cutoff. Files currently held in the in-process
import cache are never removed — deleting them out from under Python could
break already-built model classes:

```python
from neuronumba.simulator.models.dsl import cleanup_cache
removed = cleanup_cache(max_age_days=1)  # nuke yesterday's iterations
```

## Jacobians

DSL-built models ship a numerical `get_jacobian(state)` that returns the
network Jacobian at a given operating point:

```python
m = HopfDSL(g=0.5, a=-0.5, omega=0.3).configure(weights=W)
J = m.get_jacobian(np.zeros((m.n_state_vars, m.n_rois)))
# shape: (n_state_vars * n_rois, n_state_vars * n_rois)
# index:  J[u*N + i, v*N + j] = d(d_state_u at region i)/d(state_v at region j)
```

Implementation: centered finite differences on the dfun for local partials
(d_state w.r.t. state, d_state w.r.t. coupling) plus the closed-form
linearization of the coupling kernel (which the DSL knows from each
`CouplingVar.kind`). No SymPy, no new dependencies, ~1e-7 accuracy with the
default step. Validated against a brute-force naive Jacobian for Hopf, Deco,
and Naskar in `tests/test_dsl_jacobian.py`.

Tune the FD step size with the optional `eps` keyword if your model has
parameters at unusual scales:

```python
J = m.get_jacobian(state, eps=1e-8)  # tighter; risk of roundoff
```

If you need an analytic Jacobian (machine-precision, faster) you can still
hand-write `get_jacobian` on a subclass — the DSL-built class is just a
regular `Model` subclass, so override and you're done.

## When to subclass `Model` directly

The DSL covers the 80% case. Reach for a hand-written `Model` (or
`LinearCouplingModel`) subclass when:

1. **You need `@njit` helper functions inside the dfun.** Zerlaut's
   `get_fluct_regime_vars` and `erfc_approx` are 30-line jitted helpers that
   the DSL doesn't yet support.
2. **You need imperative setup in `_init_dependant`.** Naskar/Deco2014
   compute FIC iteratively via `FICHerzog2022().compute_J(weights, g)` — that
   needs Python control flow, not a single expression. You can also
   *subclass a DSL-built class* and override `_init_dependant` to call
   `super()` and then do your imperative setup. See
   `tests/test_dsl_subclass.py`.
3. **You need delays beyond linear/diffusive/delayed coupling** (e.g.
   per-coupling-var custom delays, mixed-kind specs).

These are documented as deferred items in [ROADMAP.md](ROADMAP.md), which
covers what each missing feature would look like, why it was deferred, and
the design questions to settle before implementing.

## Reference specs

Four canonical neuronumba models live as DSL specs in
`examples/dsl_models/`:

- [hopf_dsl.py](../../../../../examples/dsl_models/hopf_dsl.py)
- [deco2014_dsl.py](../../../../../examples/dsl_models/deco2014_dsl.py)
- [naskar2021_dsl.py](../../../../../examples/dsl_models/naskar2021_dsl.py)
- [montbrio_dsl.py](../../../../../examples/dsl_models/montbrio_dsl.py)

These also serve as regression-test references — see
`tests/test_dsl_equivalence.py`. They're the best starting point for porting
a new model.

## Public API

```python
from neuronumba.simulator.models.dsl import (
    # Spec types
    ModelSpec, StateVar, CouplingVar, Parameter,
    # Compilation
    build_model, dump_generated,
    # Cache & introspection
    get_source_file, get_cache_dir, cleanup_cache,
)
```

Everything else in the subpackage is internal — names prefixed with `_` and
helpers under `rewriter.py`, `dependents.py`, `materialize.py`,
`builder.py`. Don't depend on them; they may move between releases.
