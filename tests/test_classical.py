"""M1 tests for shor/classical.py (number theory and continued fractions)."""

import random
from math import gcd

import pytest

from shor.classical import (
    classical_order,
    convergent_denominators,
    is_prime_power,
    order_from_samples,
)


def distinct_prime_factors(N):
    """Independent ground truth for the tests: trial division."""
    primes, p = set(), 2
    while p * p <= N:
        while N % p == 0:
            primes.add(p)
            N //= p
        p += 1
    if N > 1:
        primes.add(N)
    return primes


def ideal_sample(s, r, t):
    """y = round(s * 2^t / r), the peak of the QPE distribution for phase s/r."""
    return (2 * s * 2**t + r) // (2 * r)


# ---------------------------------------------------------------- classical_order


@pytest.mark.parametrize(
    "a, N, r",
    [(7, 15, 4), (2, 15, 4), (14, 15, 2), (4, 21, 3), (2, 21, 6), (2, 35, 12), (2, 989, 154)],
)
def test_classical_order_known_values(a, N, r):
    assert classical_order(a, N) == r


def test_classical_order_is_the_smallest_exponent():
    for N in range(3, 60):
        for a in range(1, N):
            if gcd(a, N) != 1:
                continue
            r = classical_order(a, N)
            assert pow(a, r, N) == 1
            assert all(pow(a, k, N) != 1 for k in range(1, r))


def test_classical_order_rejects_non_coprime_base():
    with pytest.raises(ValueError):
        classical_order(6, 15)


# ----------------------------------------------------------------- is_prime_power


def test_is_prime_power_matches_trial_division():
    for N in range(2, 2000):
        assert is_prime_power(N) == (len(distinct_prime_factors(N)) == 1), N


def test_is_prime_power_small_cases():
    assert not is_prime_power(0)
    assert not is_prime_power(1)
    assert is_prime_power(2)  # a prime is p^1
    assert is_prime_power(27)
    assert is_prime_power(961)  # 31^2
    assert not is_prime_power(15)


def test_valid_shor_inputs_up_to_50():
    # Plan section 1: odd composites that are not prime powers.
    valid = [N for N in range(3, 51, 2) if not is_prime_power(N)]
    assert valid == [15, 21, 33, 35, 39, 45]


# -------------------------------------------------------- convergent_denominators


def test_convergents_of_five_sixteenths():
    # 5/16 = [0; 3, 5], convergents 0/1, 1/3, 5/16.
    assert convergent_denominators(5, 4, 100) == [1, 3, 16]
    assert convergent_denominators(5, 4, 10) == [1, 3]  # 16 >= N is dropped


def test_convergents_of_three_quarters():
    # 192/256 = 3/4 = [0; 1, 3], convergents 0/1, 1/1, 3/4.
    assert convergent_denominators(192, 8, 15) == [1, 1, 4]


def test_convergents_of_zero():
    assert convergent_denominators(0, 8, 15) == [1]


@pytest.mark.parametrize("N", [15, 21, 35, 143, 989])
def test_ideal_sample_contains_reduced_denominator(N):
    # With t = 2n, s/r in lowest terms is always a convergent of y / 2^t.
    t = 2 * N.bit_length()
    for r in range(1, N):
        for s in range(r):
            y = ideal_sample(s, r, t)
            assert r // gcd(s, r) in convergent_denominators(y, t, N), (r, s)


# ------------------------------------------------------------- order_from_samples


@pytest.mark.parametrize(
    "a, N", [(7, 15), (2, 15), (14, 15), (2, 21), (4, 21), (2, 35), (3, 35), (5, 143), (2, 989)]
)
def test_order_from_all_ideal_samples(a, N):
    # Exit test 2.
    r = classical_order(a, N)
    t = 2 * N.bit_length()
    ys = [ideal_sample(s, r, t) for s in range(r)]
    assert order_from_samples(ys, t, a, N) == r


def test_order_from_single_sample_with_coprime_s():
    a, N = 2, 989
    r = classical_order(a, N)  # 154 = 2 * 7 * 11
    t = 2 * N.bit_length()
    for s in range(1, r):
        if gcd(s, r) == 1:
            assert order_from_samples([ideal_sample(s, r, t)], t, a, N) == r


def test_order_needs_lcm_across_shots():
    # r = 154. s = 14 gives denominator 11 and s = 11 gives denominator 14.
    # Neither is r, and r / 11 = 14 is not a small multiple, but lcm(11, 14) = 154.
    a, N, r = 2, 989, 154
    t = 2 * N.bit_length()
    y_11, y_14 = ideal_sample(14, r, t), ideal_sample(11, r, t)
    assert order_from_samples([y_11], t, a, N) is None
    assert order_from_samples([y_11, y_14], t, a, N) == r


def test_order_from_small_multiple():
    # r = 154, s = 2 gives denominator 77; r = 2 * 77.
    a, N, r = 2, 989, 154
    t = 2 * N.bit_length()
    assert order_from_samples([ideal_sample(2, r, t)], t, a, N) == r


def test_order_ignores_zero_and_returns_none_without_information():
    a, N = 2, 989
    t = 2 * N.bit_length()
    assert order_from_samples([], t, a, N) is None
    assert order_from_samples([0, 0, 0], t, a, N) is None


def test_order_is_robust_to_junk_samples():
    # Shots far from every peak add wrong candidates; the result must still be
    # the true order, never a proper multiple of it.
    rng = random.Random(7)
    for a, N in [(2, 35), (2, 143), (2, 989)]:
        r = classical_order(a, N)
        t = 2 * N.bit_length()
        ys = [ideal_sample(s, r, t) for s in range(r)]
        ys += [rng.randrange(2**t) for _ in range(30)]
        assert order_from_samples(ys, t, a, N) == r
