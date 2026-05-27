# Reference DSL specs

This directory contains the canonical neuronumba models expressed in the DSL.
Each spec produces a `Model` subclass that is **bit-equivalent** to its
hand-written counterpart in `src/neuronumba/simulator/models/`.

| Spec | Hand-written | LOC ratio |
|---|---|---|
| [hopf_dsl.py](hopf_dsl.py) | `hopf.py` | 38 vs 152 |
| [deco2014_dsl.py](deco2014_dsl.py) | `deco2014.py` | 67 vs 416 |
| [naskar2021_dsl.py](naskar2021_dsl.py) | `naskar2021.py` | 69 vs 104 |
| [montbrio_dsl.py](montbrio_dsl.py) | `montbrio.py` | 88 vs 160 |
| [zerlaut_dsl.py](zerlaut_dsl.py) (1st + 2nd order) | `zerlaut.py` | ~320 vs 800 |

There is also [compare_simulations.py](compare_simulations.py): a runnable
script that simulates pairs of DSL and hand-written models on the same inputs
and plots their trajectories side-by-side.

## Dual purpose

These specs are both **examples** and **test references**:

- **Examples** — read them top-to-bottom to learn the DSL. Copy any of them
  into your own code and modify; you'll have a working model in minutes.
- **Test references** — `tests/test_dsl_equivalence.py` imports each spec,
  builds the DSL model, and compares it against the hand-written model on
  random inputs. Disagreement fails the test.

## Drift policy

If you change a hand-written model (bug fix, numerical refinement), update the
DSL spec here in the same commit. The equivalence test will fail loudly if you
forget — read the failure message: it tells you both files involved.

If you change a DSL spec (e.g. for a new feature you want to add), the
equivalence test still has to pass against the hand-written reference. If they
genuinely diverge, that's a model change and belongs in a separate PR.

## What each spec illustrates

Pick the spec that's closest to what you want to build:

- **`hopf_dsl.py`** — minimal 2-variable oscillator. Shows the basic shape
  of `state_vars` / `coupling_vars` / `parameters` / `equations` and the
  `kind="diffusive"` coupling kernel.
- **`deco2014_dsl.py`** — single-coupling-var rate model with declared
  observables (`Ie`, `re`) and an EPS-fix using `np.where` in the equations.
- **`naskar2021_dsl.py`** — extends Deco-style dynamics with a third state
  variable that itself follows a slow plasticity rule.
- **`montbrio_dsl.py`** — 6-variable Montbrio model with cross-coupling and
  recurrent dependent parameters (`Parameter("J_N_ee", formula="...")`).
- **`zerlaut_dsl.py`** — uses user-supplied `@nb.njit` helpers
  (`get_fluct_regime_vars`, `threshold_func`, `TF`, `erfc_approx`) imported
  from `neuronumba.simulator.models.zerlaut`. Demonstrates tuple-unpacking
  from a multi-return helper, helper composition (`TF` calls the others
  internally), and 1D-array parameters as defaults (length-10 polynomial
  coefficient vectors).

For matrix-valued parameters and conduction-delay coupling, see
`tests/test_dsl_matrix_params.py` and `tests/test_dsl_delayed.py`.

## Usage

```python
from examples.dsl_models import HopfDSL  # if examples/ is on sys.path
# or
from neuronumba.simulator.models.dsl import build_model
from examples.dsl_models.hopf_dsl import hopf_spec
HopfDSL = build_model(hopf_spec)

m = HopfDSL(g=0.5).configure(weights=W)
```

For incremental construction (instead of writing a full `ModelSpec` literal),
use `ModelBuilder`:

```python
from neuronumba.simulator.models.dsl import ModelBuilder

HopfDSL = (ModelBuilder("Hopf")
    .add_state("x").add_state("y")
    .add_coupling("x", kind="diffusive")
    .add_param("a", default=-0.5)
    .add_param("omega", default=0.3)
    .add_equation("d_x = (a - x*x - y*y)*x - omega*y + coupling.x")
    .add_equation("d_y = (a - x*x - y*y)*y + omega*x")
    .build())
```
