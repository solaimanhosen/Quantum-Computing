"""M1 tests for shor/factor.py (the classical driver around order finding)."""

import random
from math import gcd

import numpy as np
import pytest

from helpers import CountingOrder, FixedBases
from shor.classical import classical_order
from shor.factor import shor_factor


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


def test_order_path_factors_every_valid_N_up_to_1000():
    # Exit test 1 again, with the gcd shortcut out of the way: only bases
    # coprime to N are offered, so every factorisation comes from an order.
    for N in range(3, 1001):
        if not is_valid_input(N):
            continue
        coprime_bases = [a for a in range(2, N - 1) if gcd(a, N) == 1]
        find_order = CountingOrder()
        factors = shor_factor(N, find_order, FixedBases(*coprime_bases))
        assert_nontrivial_factors(N, factors)
        assert find_order.calls, N


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


def test_retries_when_given_an_even_multiple_of_the_order():
    # a = 7 has r = 4 (mod 15). With 8 instead, a^(8/2) = 1 and both gcds are
    # trivial; the driver must notice and try again rather than return (1, 15).
    answers = iter([8, 4])
    calls = []

    def find_order(a, N):
        calls.append(a)
        return next(answers)

    assert shor_factor(15, find_order, FixedBases(7, 7)) == (3, 5)
    assert calls == [7, 7]


def test_odd_multiple_of_the_order_still_factors():
    # 12 = 3 * 4: a^(12/2) = (a^2)^3 = a^2 * a^4 = a^2, the same half power.
    assert shor_factor(15, lambda a, N: 12, FixedBases(7)) == (3, 5)


def test_accepts_numpy_integer_order():
    factors = shor_factor(15, lambda a, N: np.int64(4), FixedBases(7))
    assert factors == (3, 5)
    assert all(type(f) is int for f in factors)


@pytest.mark.parametrize("bad", [0, -4])
def test_retries_on_non_positive_order(bad):
    answers = iter([bad, 4])
    assert shor_factor(15, lambda a, N: next(answers), FixedBases(7, 7)) == (3, 5)


def test_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError):
        shor_factor(15, lambda a, N: None, FixedBases(7, 7, 7, 7, 7), max_attempts=5)


@pytest.mark.parametrize("N", [0, 1, 2, 14, 100, 13, 27, 49, 961])
def test_rejects_invalid_N(N):
    # Even numbers, primes and prime powers are not inputs for Shor's algorithm.
    with pytest.raises(ValueError):
        shor_factor(N, classical_order, random.Random(0))
