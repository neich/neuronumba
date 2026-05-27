"""Tests for matrix-shaped dependent parameters.

After the closure-capture refactor, there's no special "matrix" marker — a
formula that returns a 2D ndarray simply produces a matrix-valued parameter.
The DSL captures it via `model.<name>` like any other parameter; numba's
type inference handles `A @ x` style expressions in the dfun.

Use case: OrnsteinUhlenbeck-style models where the dynamics need a full
(n_rois, n_rois) operator derived from `weights`/`g`/`tau`.
"""
import numpy as np
import numba as nb

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


def test_matrix_param_computed_and_visible_in_dfun():
    """Build a model whose dfun reads a 2D matrix via closure capture."""
    n = 4
    spec = ModelSpec(
        name="MatrixModel",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("scale", default=2.0),
            # A is the (n_rois, n_rois) identity matrix scaled by `scale`.
            Parameter("A", formula="scale * np.eye(weights.shape[0])"),
        ],
        equations="d_x = -x + A @ x + coupling.x",
    )
    Cls = build_model(spec)
    W = np.zeros((n, n))
    m = Cls(g=0.0).configure(weights=W)

    # The matrix lives on `self`.
    assert m.A.shape == (n, n)
    np.testing.assert_allclose(m.A, 2.0 * np.eye(n))

    # The dfun should compile and use the matrix correctly.
    f = m.get_numba_dfun()
    state = np.array([[0.5, 1.0, -0.5, 2.0]])
    ds, _ = f(state, np.zeros((1, n)))
    # d_x = -x + 2.0 * I @ x = -x + 2*x = x (in this trivial case)
    np.testing.assert_allclose(ds[0], state[0], atol=1e-12)


def test_matrix_param_lives_on_instance():
    """A 2D dependent param is just an instance attribute — no special storage."""
    n = 3
    spec = ModelSpec(
        name="MatrixOnInstance",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("alpha", default=1.0),
            Parameter("M", formula="np.eye(weights.shape[0])"),
        ],
        equations="d_x = alpha * (M @ x) + coupling.x",
    )
    Cls = build_model(spec)
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))

    assert isinstance(m.M, np.ndarray)
    assert m.M.shape == (n, n)
    np.testing.assert_allclose(m.M, np.eye(n))


def test_matrix_param_recomputes_on_configure():
    """Like other dependents, matrix params re-evaluate when configure() runs."""
    n = 4
    spec = ModelSpec(
        name="MatrixModel_Recompute",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("k", default=1.0),
            Parameter("A", formula="k * np.eye(weights.shape[0])"),
        ],
        equations="d_x = -(A @ x) + coupling.x",
    )
    Cls = build_model(spec)
    W = np.zeros((n, n))
    m = Cls(g=0.0).configure(weights=W)
    np.testing.assert_allclose(m.A, np.eye(n))

    m.k = 5.0
    m.configure(weights=W)
    np.testing.assert_allclose(m.A, 5.0 * np.eye(n))


def test_matrix_param_OU_like():
    """Smoke test of the OU-style use case: A derived from weights/g/tau.

    OU dynamics: dx/dt = -A x  with  A = (-W * g + diag(rowsum)) / tau.
    """
    n = 5
    rng = np.random.default_rng(0)
    W = rng.random((n, n)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)

    spec = ModelSpec(
        name="OULike",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("tau", default=10.0),
            Parameter(
                "A",
                formula="(-g * weights + np.diag(weights.sum(axis=1))) / tau",
            ),
        ],
        # Ignore the linear-coupling output here; the matrix A captures the
        # full network operator.
        equations="d_x = -(A @ x)",
    )
    Cls = build_model(spec)
    m = Cls(g=0.5).configure(weights=W)

    expected_A = (-0.5 * W + np.diag(W.sum(axis=1))) / 10.0
    np.testing.assert_allclose(m.A, expected_A, atol=1e-12)

    f = m.get_numba_dfun()
    state = rng.standard_normal((1, n)) * 0.1
    ds, _ = f(state, np.zeros((1, n)))
    expected = -(expected_A @ state[0])
    np.testing.assert_allclose(ds[0], expected, atol=1e-10)


def test_matrix_param_can_combine_with_helpers():
    """Helpers and matrix params should compose without interference."""
    n = 4

    @nb.njit(nb.f8[:](nb.f8[:]), cache=False)
    def squash(x):
        return np.tanh(x)

    spec = ModelSpec(
        name="MatrixHelperCombo",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("k", default=2.0),
            Parameter("A", formula="k * np.eye(weights.shape[0])"),
        ],
        equations="d_x = squash(A @ x) + coupling.x",
        helpers=[squash],
    )
    Cls = build_model(spec)
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))
    f = m.get_numba_dfun()
    state = np.array([[0.1, 0.2, 0.3, 0.4]])
    ds, _ = f(state, np.zeros((1, n)))
    np.testing.assert_allclose(ds[0], np.tanh(2.0 * state[0]), atol=1e-12)
