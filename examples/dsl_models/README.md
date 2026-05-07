# Reference DSL specs

This directory contains the canonical neuronumba models expressed in the DSL.
Each spec produces a `Model` subclass that is **bit-equivalent** to its
hand-written counterpart in `src/neuronumba/simulator/models/`.

| Spec | Hand-written | LOC ratio |
|---|---|---|
| [hopf_dsl.py](hopf_dsl.py) | `hopf.py` | ~30 vs 152 |
| [deco2014_dsl.py](deco2014_dsl.py) | `deco2014.py` | ~50 vs 416 |
| [naskar2021_dsl.py](naskar2021_dsl.py) | `naskar2021.py` | ~60 vs 104 |
| [montbrio_dsl.py](montbrio_dsl.py) | `montbrio.py` | ~70 vs 160 |

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

## Models not included

- **Zerlaut** — needs `@njit` helper subroutines (`get_fluct_regime_vars`,
  `threshold_func`, `erfc_approx`). The DSL doesn't support that yet — see the
  Phase 6 deferred items in `IMPLEMENTATION_PLAN.md`.
- **OrnsteinUhlenbeck** — needs non-scalar matrix parameters (`A` derived from
  weights/g/tau). Deferred to v0.2.

For these, subclass `Model` directly.

## Usage

```python
from examples.dsl_models import HopfDSL  # if examples/ is on sys.path
# or
from neuronumba.simulator.models.dsl import build_model
from examples.dsl_models.hopf_dsl import hopf_spec
HopfDSL = build_model(hopf_spec)

m = HopfDSL(g=0.5).configure(weights=W)
```
