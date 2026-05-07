"""Reference DSL specs for neuronumba's canonical models.

These specs reproduce the existing hand-written models exactly. They serve two
purposes simultaneously:

1. **User-facing examples.** Read them to learn how to express a model in the
   DSL. Each file is self-contained and small (~30-300 LOC vs 100-800 LOC for
   the hand-written equivalents).

2. **Regression test references.** `tests/test_dsl_equivalence.py` imports
   these specs and compares the resulting DSL-built models against the
   hand-written ones bit-by-bit. If you change a hand-written model (e.g. a
   numerical bug fix), the corresponding spec here must follow, or the
   equivalence test will fail.
"""
from .deco2014_dsl import deco_spec, Deco2014DSL
from .hopf_dsl import hopf_spec, HopfDSL
from .montbrio_dsl import montbrio_spec, MontbrioDSL
from .naskar2021_dsl import naskar_spec, Naskar2021DSL
from .zerlaut_dsl import (
    zerlaut_first_order_spec,
    zerlaut_second_order_spec,
    ZerlautAdaptationFirstOrderDSL,
    ZerlautAdaptationSecondOrderDSL,
)

__all__ = [
    "Deco2014DSL", "deco_spec",
    "HopfDSL", "hopf_spec",
    "MontbrioDSL", "montbrio_spec",
    "Naskar2021DSL", "naskar_spec",
    "ZerlautAdaptationFirstOrderDSL", "zerlaut_first_order_spec",
    "ZerlautAdaptationSecondOrderDSL", "zerlaut_second_order_spec",
]
