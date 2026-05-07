"""Shared fixtures for the DSL test suite.

Spec definitions live in `examples/dsl_models/` (single source of truth for
both user-facing examples and equivalence-test references). This file just
re-exports them as fixtures.

Built classes are session-scoped so numba's compile cache amortizes across
test modules.
"""
import os
import sys

import numpy as np
import pytest

# Make the `examples/dsl_models` package importable. `examples/` itself is not
# a Python package (it's a flat collection of demo scripts); we add it to
# sys.path so `import dsl_models` resolves the sub-package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))

from dsl_models import (  # noqa: E402
    hopf_spec as _hopf_spec,
    deco_spec as _deco_spec,
    naskar_spec as _naskar_spec,
    montbrio_spec as _montbrio_spec,
    HopfDSL as _HopfDSL,
    Deco2014DSL as _Deco2014DSL,
    Naskar2021DSL as _Naskar2021DSL,
    MontbrioDSL as _MontbrioDSL,
)


def _make_W(n=8, seed=0):
    """Symmetric, zero-diagonal connectivity normalized to max 1.0."""
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    W /= W.max()
    return W


@pytest.fixture(scope="session")
def n_rois():
    return 8


@pytest.fixture(scope="session")
def weights(n_rois):
    return _make_W(n=n_rois)


# ----- canonical model specs (re-exported from examples/dsl_models) ----------

@pytest.fixture(scope="session")
def hopf_spec():
    return _hopf_spec


@pytest.fixture(scope="session")
def deco_spec():
    return _deco_spec


@pytest.fixture(scope="session")
def naskar_spec():
    return _naskar_spec


@pytest.fixture(scope="session")
def montbrio_spec():
    return _montbrio_spec


# ----- materialized model classes --------------------------------------------
# Re-export the built classes from `examples/dsl_models/` directly. Tests
# exercise the exact same class objects a user would import.

@pytest.fixture(scope="session")
def hopf_dsl_cls():
    return _HopfDSL


@pytest.fixture(scope="session")
def deco_dsl_cls():
    return _Deco2014DSL


@pytest.fixture(scope="session")
def naskar_dsl_cls():
    return _Naskar2021DSL


@pytest.fixture(scope="session")
def montbrio_dsl_cls():
    return _MontbrioDSL
