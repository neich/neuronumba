"""Tests for the DSL's numerical Jacobian.

The DSL builds the network Jacobian in two pieces:
  1. Local partials of the dfun w.r.t. each state var and each coupling var,
     via centered finite differences.
  2. Closed-form linearization of the coupling kernel (linear / diffusive).

These tests verify both pieces by:
  - Comparing the structured DSL Jacobian against a "naive" full-network
    finite-difference reference that doesn't exploit locality (catches
    assembly bugs).
  - Comparing the DSL Jacobian for Hopf at the origin against the
    hand-written analytical Jacobian (catches both math and assembly bugs).
"""
import numpy as np
import pytest


def _naive_full_network_fd_jacobian(model, state, eps=1e-6):
    """Reference: brute-force finite difference treating dfun(state)+coupling
    as one big function of the flat state. Slow, but free of structural
    assumptions."""
    nsv, N = state.shape
    c_vars = list(model.c_vars)
    dfun = model.get_numba_dfun()
    coupling_factory = model.get_numba_coupling()

    def total(flat_state):
        s = flat_state.reshape(nsv, N)
        c = coupling_factory(s[c_vars, :].copy())
        ds, _ = dfun(s.copy(), c)
        return ds.ravel()

    flat = state.ravel().copy()
    base = total(flat)
    J = np.empty((flat.size, flat.size))
    for k in range(flat.size):
        x_plus = flat.copy();  x_plus[k]  += eps
        x_minus = flat.copy(); x_minus[k] -= eps
        J[:, k] = (total(x_plus) - total(x_minus)) / (2.0 * eps)
    return J


def test_jacobian_shape(weights, n_rois, hopf_dsl_cls):
    m = hopf_dsl_cls(g=0.5).configure(weights=weights)
    nsv = m.n_state_vars
    state = np.zeros((nsv, n_rois))
    J = m.get_jacobian(state)
    assert J.shape == (nsv * n_rois, nsv * n_rois)


def test_jacobian_rejects_wrong_shape(weights, hopf_dsl_cls):
    m = hopf_dsl_cls(g=0.5).configure(weights=weights)
    with pytest.raises(ValueError, match="state must have shape"):
        m.get_jacobian(np.zeros((3, 5)))  # wrong shape


def test_jacobian_hopf_at_origin_matches_analytic(weights, hopf_dsl_cls):
    """DSL Jacobian at the origin matches the closed-form derivation.

    Hopf dfun at the origin (x=y=0):
        d(d_x)/dx = a + d(coupling.x)/dx     (the cubic terms vanish)
        d(d_x)/dy = -omega
        d(d_y)/dx = +omega
        d(d_y)/dy = same as d(d_x)/dx
    Diffusive coupling linearization: d(coupling.x[i])/dx[j] = g*(W[i,j] - delta_ij*ink[i]).

    Note: we deliberately build the reference here rather than calling
    `Hopf.get_jacobian`. The hand-written method has an internal factor-of-2pi
    inconsistency between its dfun (uses `omega`) and its analytical Jacobian
    (uses `2*pi*omega`), so it can't serve as a faithful oracle for any
    Jacobian that's consistent with the dfun.
    """
    N = weights.shape[0]
    g = 0.5
    a = -0.5
    omega = 0.3

    # Diffusive-coupling linearization for Hopf's coupling vars.
    ink = weights.T.sum(axis=1)
    C = g * (weights - np.diag(ink))

    # Closed-form blocks.
    block_diag = a * np.eye(N) + C            # (x->x) and (y->y)
    block_off_xy = -omega * np.eye(N)         # (x<-y) cross
    block_off_yx = +omega * np.eye(N)         # (y<-x) cross
    J_expected = np.block([
        [block_diag,    block_off_xy],
        [block_off_yx,  block_diag],
    ])

    m_dsl = hopf_dsl_cls(g=g, a=a, omega=omega).configure(weights=weights)
    J_dsl = m_dsl.get_jacobian(np.zeros((2, N)))

    np.testing.assert_allclose(J_dsl, J_expected, atol=1e-7, rtol=1e-7)


def test_jacobian_structured_matches_naive_fd_hopf(weights, hopf_dsl_cls):
    """Structured DSL Jacobian == brute-force network-wide FD (Hopf, random state)."""
    m = hopf_dsl_cls(g=0.5).configure(weights=weights)
    rng = np.random.default_rng(123)
    state = rng.standard_normal((2, weights.shape[0])) * 0.05  # small to keep regime stable

    J_struct = m.get_jacobian(state)
    J_naive = _naive_full_network_fd_jacobian(m, state)
    np.testing.assert_allclose(J_struct, J_naive, atol=1e-7, rtol=1e-7)


def test_jacobian_structured_matches_naive_fd_deco(weights, deco_dsl_cls):
    """Same check on Deco2014 — exercises linear coupling and observables."""
    m = deco_dsl_cls(g=0.5).configure(weights=weights)
    rng = np.random.default_rng(7)
    # Stay inside the model's bounds (S_e, S_i in [0, 1]).
    state = rng.uniform(0.1, 0.5, size=(2, weights.shape[0]))

    J_struct = m.get_jacobian(state)
    J_naive = _naive_full_network_fd_jacobian(m, state)
    # Deco's sigmoid has steep regions; bump tolerance slightly.
    np.testing.assert_allclose(J_struct, J_naive, atol=1e-5, rtol=1e-5)


def test_jacobian_structured_matches_naive_fd_naskar(weights, naskar_dsl_cls):
    """Three-state-var sanity check: Naskar2021 (S_e, S_i, J)."""
    m = naskar_dsl_cls(g=0.5).configure(weights=weights)
    rng = np.random.default_rng(11)
    state = np.empty((3, weights.shape[0]))
    state[0] = rng.uniform(0.1, 0.5, size=weights.shape[0])  # S_e
    state[1] = rng.uniform(0.1, 0.5, size=weights.shape[0])  # S_i
    state[2] = rng.uniform(0.5, 1.5, size=weights.shape[0])  # J

    J_struct = m.get_jacobian(state)
    J_naive = _naive_full_network_fd_jacobian(m, state)
    np.testing.assert_allclose(J_struct, J_naive, atol=1e-5, rtol=1e-5)


def test_jacobian_recomputes_after_param_change(weights, hopf_dsl_cls):
    """Changing `a` and reconfiguring must change the Jacobian's diagonal."""
    m = hopf_dsl_cls(g=0.5).configure(weights=weights)
    state = np.zeros((2, weights.shape[0]))
    J1 = m.get_jacobian(state)

    m.a = 0.5
    m.configure(weights=weights)
    J2 = m.get_jacobian(state)

    # The diagonal of the (x, x) block contains a + (coupling diagonal).
    # Changing a from -0.5 to +0.5 shifts those entries by exactly +1.0.
    N = weights.shape[0]
    diag_diff = np.diag(J2[:N, :N]) - np.diag(J1[:N, :N])
    np.testing.assert_allclose(diag_diff, 1.0, atol=1e-7)
