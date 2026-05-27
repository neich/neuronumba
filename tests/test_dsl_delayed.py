"""Tests for `kind="delayed"` coupling.

Covers:
  - DSL spec validation (build_model accepts kind="delayed", rejects mixed kinds).
  - get_jacobian raises NotImplementedError for delayed kernels.
  - End-to-end equivalence: with effectively-zero delays (huge speed),
    HistoryDelays + DSL kind="delayed" produces the same trajectory as
    HistoryNoDelays + DSL kind="linear" on the same spec.
  - Sanity: with non-trivial delays, the trajectory differs.
"""
import numpy as np
import pytest

from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryDelays, HistoryNoDelays
from neuronumba.simulator.integrators.euler import EulerDeterministic
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


def _delayed_hopf_spec(name):
    return ModelSpec(
        name=name,
        state_vars=[StateVar("x", initial=0.1), StateVar("y", initial=0.1)],
        coupling_vars=[
            CouplingVar("x", kind="delayed"),
            CouplingVar("y", kind="delayed"),
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


def _linear_hopf_spec(name):
    """Same spec but with kind='linear' — for the zero-delay equivalence test."""
    return ModelSpec(
        name=name,
        state_vars=[StateVar("x", initial=0.1), StateVar("y", initial=0.1)],
        coupling_vars=[
            CouplingVar("x", kind="linear"),
            CouplingVar("y", kind="linear"),
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


def _run(model, weights, lengths, speed, history, t_max=50.0, dt=0.1):
    np.random.seed(0)
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)
    integrator = EulerDeterministic(dt=dt)
    monitor = RawSubSample(period=1.0, monitor_vars=model.get_var_info(["x"]))
    sim = Simulator(
        connectivity=con, model=model, history=history,
        integrator=integrator, monitors=[monitor],
    )
    sim.run(0, t_max)
    return monitor.data("x")


def test_kind_delayed_builds():
    Cls = build_model(_delayed_hopf_spec("DelayedDSL_Build"))
    assert "x" in Cls._coupling_var_names
    assert Cls._coupling_var_kinds == ["delayed", "delayed"]


def test_kind_delayed_mixed_with_linear_rejected():
    spec = ModelSpec(
        name="MixedKindsDSL",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[
            CouplingVar("x", kind="delayed"),
        ],
        observables=[],
        parameters=[Parameter("a", default=1.0)],
        equations="d_x = -x * a + coupling.x",
    )
    # Single delayed cv is fine.
    build_model(spec)

    # Now make it mixed: add a linear coupling on a second state var.
    mixed = ModelSpec(
        name="MixedKindsDSL2",
        state_vars=[StateVar("x", initial=0.0), StateVar("y", initial=0.0)],
        coupling_vars=[
            CouplingVar("x", kind="delayed"),
            CouplingVar("y", kind="linear"),
        ],
        observables=[],
        parameters=[Parameter("a", default=1.0)],
        equations="""
            d_x = -x * a + coupling.x
            d_y = -y + coupling.y
        """,
    )
    with pytest.raises(NotImplementedError, match="Mixing 'delayed'"):
        build_model(mixed)


def test_jacobian_raises_for_delayed_coupling(weights):
    Cls = build_model(_delayed_hopf_spec("DelayedDSL_Jac"))
    m = Cls(g=0.5).configure(weights=weights)
    state = np.zeros((2, weights.shape[0]))
    with pytest.raises(NotImplementedError, match="not supported for 'delayed'"):
        m.get_jacobian(state)


def test_zero_delay_equivalent_to_linear(weights):
    """`speed=1e6` makes all delays round to zero. HistoryDelays + delayed-DSL
    must then match HistoryNoDelays + linear-DSL, bit-for-bit."""
    n = weights.shape[0]
    rng = np.random.default_rng(0)
    lengths = rng.random((n, n)) * 10.0 + 1.0
    huge_speed = 1e6  # delays << dt → all i_delays round to 0

    DelCls = build_model(_delayed_hopf_spec("DelayedDSL_ZeroDelay"))
    LinCls = build_model(_linear_hopf_spec("LinearDSL_ZeroDelay"))

    delayed_traj = _run(
        DelCls(g=0.5), weights, lengths, huge_speed, HistoryDelays(),
    )
    linear_traj = _run(
        LinCls(g=0.5), weights, lengths, huge_speed, HistoryNoDelays(),
    )

    np.testing.assert_allclose(delayed_traj, linear_traj, atol=1e-12, rtol=1e-12)


def test_nontrivial_delay_changes_trajectory(weights):
    """With finite speed, the delayed trajectory must differ from no-delay."""
    n = weights.shape[0]
    rng = np.random.default_rng(0)
    lengths = rng.random((n, n)) * 10.0 + 1.0

    DelCls = build_model(_delayed_hopf_spec("DelayedDSL_Real"))

    # speed=1.0 → delays of order 10 ms with our random lengths; with dt=0.1 ms
    # that's ~100 steps of delay. Definitely not zero.
    fast = _run(
        DelCls(g=0.5), weights, lengths, speed=1e6, history=HistoryDelays(),
    )
    slow = _run(
        DelCls(g=0.5), weights, lengths, speed=1.0, history=HistoryDelays(),
    )

    # The two runs must be DIFFERENT — coupling timing changed.
    diff = np.abs(fast - slow).max()
    assert diff > 1e-6, f"expected delay to perturb trajectory, max|Δ|={diff}"
