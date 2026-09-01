"""
Synthetic connectivity architectures for Hopf-network experiments.

This module factors the synthetic structural-connectivity construction out of
the original MATLAB demo. The common configuration lives in the abstract
SyntheticArchitecture base class, while RingArchitecture and
SmallWorldArchitecture implement the concrete connectivity rules.

The defaults reproduce the parameter values used in the paper:
  Deco et al. (2021), Current Biology — "Revisiting the Global Workspace
  orchestrating the hierarchical organization of the human brain"
from the Matlab file run_demo_ring_hopf_G_commented.m:
    n_nodes = 1000
    tmax = 1000
    distance_scale = 10.0
    spatial_decay = 1.0
    shortcut_probability = 0.05
    shortcut_weight = 0.25

`tmax` is intentionally part of the shared experiment configuration because
the original code couples matrix construction and simulation setup in one
script, and because the requested OO interface includes it. It does not alter
the connectivity matrix itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from neuronumba.basic.attr import HasAttr, Attr



@dataclass
class SyntheticArchitecture(HasAttr):
    """
    Abstract base class for synthetic network architectures.

    Parameters
    ----------
    n_nodes : int
        Number of nodes / matrix dimension.
    tmax : int
        Number of recorded simulation time points.  Stored as part of the
        experiment configuration; it does not affect matrix construction.
    distance_scale : float
        Divisor used to convert ring distance in node steps into the distance
        units used by the original MATLAB script.
    spatial_decay : float
        Exponential decay constant lambda for the short-range ring coupling.
    seed : int or None
        Optional random seed used by stochastic subclasses.
    """

    n_nodes: Attr(required=False, default=1000)  # int = 1000
    tmax: Attr(required=False, default=1000)  # int = 1000
    distance_scale: Attr(required=False, default=10.0)  # float = 10.0
    spatial_decay: Attr(required=False, default=1.0)  # float = 1.0
    seed: Attr(required=False, default=None)  # Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be >= 2.")
        if self.tmax < 2:
            raise ValueError("tmax must be >= 2.")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be > 0.")
        if self.spatial_decay < 0:
            raise ValueError("spatial_decay must be >= 0.")

    @property
    def rng(self) -> np.random.Generator:
        """Return the random-number generator associated with this object."""
        # Cache the generator lazily so repeated calls continue the same stream.
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def ring_distance_matrix(self) -> np.ndarray:
        """
        Return the shortest-arc distance matrix for nodes arranged on a ring.

        This vectorises the MATLAB double loop:

            d1 = abs(j - i)
            d2 = N - d1
            rr(i,j) = min(d1,d2) / 10
        """
        idx = np.arange(self.n_nodes)
        d1 = np.abs(idx[:, None] - idx[None, :])
        d2 = self.n_nodes - d1
        return np.minimum(d1, d2) / self.distance_scale

    def exponential_ring_matrix(self, decay: Optional[float] = None) -> np.ndarray:
        """
        Build the exponentially decaying short-range ring connectivity matrix.

        The diagonal is set to zero because the dynamical coupling contains no
        self-connections.
        """
        lam = self.spatial_decay if decay is None else float(decay)
        if lam < 0:
            raise ValueError("decay must be >= 0.")

        rr = self.ring_distance_matrix()
        matrix = np.exp(-lam * rr)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate and return this architecture's connectivity matrix."""
        raise NotImplementedError


@dataclass
class RingArchitecture(SyntheticArchitecture):
    """
    Pure short-range ring architecture.

    Connectivity decays exponentially with the shortest ring distance:
        C_ij = exp(-lambda * r_ij),  i != j
        C_ii = 0
    """

    def generate(self) -> np.ndarray:
        return self.exponential_ring_matrix()


@dataclass
class SmallWorldArchitecture(RingArchitecture):
    """
    Ring architecture augmented with random long-range shortcuts.

    Parameters
    ----------
    shortcut_probability : float
        Probability that a candidate off-diagonal node pair receives a
        shortcut.
    shortcut_weight : float
        Weight assigned to every shortcut.
    preserve_stronger_local_edges : bool
        If False (default), a shortcut overwrites the existing local-ring
        weight, matching the MATLAB assignment ``C2(l,k) = 0.25``.
        If True, the resulting edge is max(local_weight, shortcut_weight).
    symmetric_shortcuts : bool
        If False (default), shortcuts are directed, matching the MATLAB code.
        If True, every sampled shortcut is mirrored.
    """

    shortcut_probability: Attr(required=False, default=0.05)  # float = 0.05
    shortcut_weight: Attr(required=False, default=0.25)  # float = 0.25
    preserve_stronger_local_edges: Attr(required=False, default=False)  # bool = False
    symmetric_shortcuts: Attr(required=False, default=False)  # bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.shortcut_probability <= 1.0:
            raise ValueError("shortcut_probability must lie in [0, 1].")
        if self.shortcut_weight < 0:
            raise ValueError("shortcut_weight must be >= 0.")

    def generate(self) -> np.ndarray:
        """
        Generate the small-world matrix.

        The original MATLAB implementation performs N^2 Bernoulli trials and,
        on every success, chooses an unrelated random ordered node pair (l,k)
        and sets C2(l,k)=0.25.  Since the trigger indices (i,j) are discarded,
        that construction is statistically equivalent in spirit to drawing
        random shortcut locations, but its parameter beta is not literally the
        per-edge inclusion probability.

        Here the OO interface gives `shortcut_probability` its natural and
        requested meaning: each off-diagonal pair is independently eligible
        for a shortcut with that probability.  With the original defaults the
        expected shortcut density remains ~5%.
        """
        matrix = super().generate()

        mask = self.rng.random((self.n_nodes, self.n_nodes)) < self.shortcut_probability
        np.fill_diagonal(mask, False)

        if self.symmetric_shortcuts:
            mask = np.logical_or(mask, mask.T)

        if self.preserve_stronger_local_edges:
            matrix[mask] = np.maximum(matrix[mask], self.shortcut_weight)
        else:
            # Match the MATLAB semantics of assigning, rather than adding,
            # the fixed shortcut weight.
            matrix[mask] = self.shortcut_weight

        np.fill_diagonal(matrix, 0.0)
        return matrix

@dataclass
class RingWithLongRangeConnectionsArchitecture(RingArchitecture):
    """
    Exponentially coupled ring augmented with randomly located long-range shortcuts.

    This class reproduces the shortcut-generation rule of the original
    MATLAB demo:

        C2 = C

        for every one of N^2 Bernoulli trials:
            with probability beta:
                choose a random ordered pair of distinct nodes (l, k)
                set C2[l, k] = shortcut_weight

    Important
    ---------
    The Bernoulli trial indices are NOT the nodes receiving the shortcut.
    Instead, every successful trial selects a new random ordered node pair.

    Consequently:

    - the number of shortcut *attempts* is Binomial(N^2, beta);
    - shortcut locations are sampled with replacement;
    - the same edge can therefore be selected more than once;
    - shortcuts are directed;
    - the selected weight replaces the original ring weight;
    - the selected pair is not explicitly required to be geometrically
      long-range, although uniformly random pairs are typically far apart
      on a large ring.

    Parameters
    ----------
    shortcut_probability : float
        Probability beta that each of the N^2 trials generates a shortcut.

    shortcut_weight : float
        Weight assigned to a selected shortcut edge.
    """

    shortcut_probability: Attr(required=False, default=0.05)  # float = 0.05
    shortcut_weight: Attr(required=False, default=0.25)  # float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()

        if not 0.0 <= self.shortcut_probability <= 1.0:
            raise ValueError(
                "shortcut_probability must lie in [0, 1]."
            )

        if self.shortcut_weight < 0:
            raise ValueError(
                "shortcut_weight must be >= 0."
            )

    def generate(self) -> np.ndarray:
        """
        Generate the ring-plus-long-range-random-shortcuts connectivity matrix.
        """

        # Start from the exponentially decaying ring matrix:
        #
        #     C2 = C;
        #
        matrix = super().generate()

        # MATLAB performs N^2 independent Bernoulli trials:
        #
        #     for i = 1:N
        #         for j = 1:N
        #             if rand < beta
        #                 ...
        #
        # Therefore, the number of successful shortcut attempts follows
        # Binomial(N^2, beta).
        n_trials = self.n_nodes * self.n_nodes

        n_shortcuts = self.rng.binomial(
            n=n_trials,
            p=self.shortcut_probability,
        )

        if n_shortcuts == 0:
            return matrix

        # For each successful trial, MATLAB performs:
        #
        #     out = randperm(N);
        #     k = out(1);
        #     l = out(end);
        #
        # Hence (l, k) is a uniformly sampled ordered pair of
        # DISTINCT nodes.
        #
        # We generate the same distribution more efficiently without
        # constructing a complete random permutation for every shortcut.

        source = self.rng.integers(
            0,
            self.n_nodes,
            size=n_shortcuts,
        )

        # Draw one of the remaining N-1 nodes.
        target = self.rng.integers(
            0,
            self.n_nodes - 1,
            size=n_shortcuts,
        )

        # Shift indices >= source so that target != source.
        target += target >= source

        # MATLAB:
        #
        #     C2(l,k) = 0.25;
        #
        # This is assignment, NOT addition. Repeated selections of the
        # same edge simply overwrite it with the same value.
        matrix[source, target] = self.shortcut_weight

        # Kept explicitly for correspondence with the MATLAB code,
        # although source != target already guarantees this.
        np.fill_diagonal(matrix, 0.0)

        return matrix