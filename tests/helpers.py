"""Test doubles and reference formulas shared by the test files."""

from math import pi

import numpy as np

from shor.classical import classical_order


def qpe_theory(phi, t):
    """Exact outcome probabilities of t-bit phase estimation on the phase phi.

    P(y) = sin^2(pi 2^t d) / (4^t sin^2(pi d)) with d = phi - y / 2^t.
    """
    d = phi - np.arange(2**t) / 2**t
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.sin(pi * 2**t * d) ** 2 / (4.0**t * np.sin(pi * d) ** 2)
    probs[np.isnan(probs)] = 1.0  # d is an integer: phi is exactly y / 2^t
    return probs


def order_finding_theory(r, t):
    """Outcome probabilities of order finding: the phases s / r, s = 0 .. r-1, equally likely."""
    return sum(qpe_theory(s / r, t) for s in range(r)) / r


class FixedBases:
    """Stands in for the rng so a test can choose the bases a that get tried."""

    def __init__(self, *bases):
        self._bases = iter(bases)

    def randrange(self, start, stop):
        a = next(self._bases)
        assert start <= a < stop
        return a


class CountingOrder:
    """Wraps an order finder and records the bases it was asked about."""

    def __init__(self, find_order=classical_order):
        self._find_order = find_order
        self.calls = []

    def __call__(self, a, N):
        self.calls.append(a)
        return self._find_order(a, N)
