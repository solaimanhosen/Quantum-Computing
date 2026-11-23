"""Test doubles shared by the test files."""

from shor.classical import classical_order


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
