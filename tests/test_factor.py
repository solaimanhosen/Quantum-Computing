"""M1 tests for shor/factor.py (the classical driver around order finding)."""

import random

import pytest

from shor.classical import classical_order
from shor.factor import shor_factor


class FixedBases:
    """Stands in for the rng so a test can choose the bases a that get tried."""

    def __init__(self, *bases):
        self._bases = iter(bases)

    def randrange(self, start, stop):
        a = next(self._bases)
        assert start <= a < stop
        return a


class CountingOrder:
    """classical_order that records the bases it was asked about."""

    def __init__(self):
        self.calls = []

    def __call__(self, a, N):
        self.calls.append(a)
        return classical_order(a, N)


def is_valid_input(N):
    """Odd, with at least two distinct prime factors (independent of shor/)."""
    if N % 2 == 0:
        return False
    primes, p, m = set(), 3, N
    while p * p <= m:
        while m % p == 0:
            primes.add(p)
            m //= p
        p += 2
    if m > 1:
        primes.add(m)
    return len(primes) >= 2


def assert_nontrivial_factors(N, factors):
    p, q = factors
    assert p * q == N
    assert 1 < p <= q < N


def test_factors_every_valid_N_up_to_1000():
    # Exit test 1.
    rng = random.Random(2026)
    valid = [N for N in range(3, 1001) if is_valid_input(N)]
    assert valid[:6] == [15, 21, 33, 35, 39, 45]
    for N in valid:
        assert_nontrivial_factors(N, shor_factor(N, classical_order, rng))


def test_order_path_gives_the_factors_of_15():
    # a = 7 has r = 4; 7^2 = 4 (mod 15); gcd(3, 15) = 3 and gcd(5, 15) = 5.
    find_order = CountingOrder()
    assert shor_factor(15, find_order, FixedBases(7)) == (3, 5)
    assert find_order.calls == [7]


def test_lucky_base_skips_order_finding():
    # gcd(6, 15) = 3 is already a factor.
    find_order = CountingOrder()
    assert shor_factor(15, find_order, FixedBases(6)) == (3, 5)
    assert find_order.calls == []


def test_retries_when_half_power_is_minus_one():
    # a = 5 has r = 6 (mod 21) and 5^3 = 125 = -1 (mod 21), which gives nothing.
    find_order = CountingOrder()
    assert shor_factor(21, find_order, FixedBases(5, 2)) == (3, 7)
    assert find_order.calls == [5, 2]


def test_retries_when_order_is_odd():
    # a = 4 has r = 3 (mod 21). a = 2 has r = 6 and 2^3 = 8: gcd(7, 21) = 7.
    find_order = CountingOrder()
    assert shor_factor(21, find_order, FixedBases(4, 2)) == (3, 7)
    assert find_order.calls == [4, 2]


def test_retries_when_order_finding_fails_or_is_wrong():
    # A quantum order finder can return None (no usable shots) or a wrong value.
    answers = iter([None, 5, 4])
    factors = shor_factor(15, lambda a, N: next(answers), FixedBases(7, 7, 7))
    assert factors == (3, 5)


def test_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError):
        shor_factor(15, lambda a, N: None, FixedBases(7, 7, 7, 7, 7), max_attempts=5)


@pytest.mark.parametrize("N", [0, 1, 2, 14, 100, 13, 27, 49, 961])
def test_rejects_invalid_N(N):
    # Even numbers, primes and prime powers are not inputs for Shor's algorithm.
    with pytest.raises(ValueError):
        shor_factor(N, classical_order, random.Random(0))
