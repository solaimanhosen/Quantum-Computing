"""M1 tests for shor/classical.py (number theory and continued fractions)."""

import random
from fractions import Fraction
from math import gcd

import numpy as np
import pytest

from shor.classical import (
    MAX_MULTIPLE,
    classical_order,
    convergents,
    is_prime_power,
    order_candidates,
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


def qpe_distribution(s, r, t):
    """Exact outcome probabilities of t-bit phase estimation on the phase s/r.

    With d = s/r - y/2^t, the amplitude of outcome y is the geometric sum
    2^-t * sum_k exp(2 pi i k d), so P(y) = sin^2(pi 2^t d) / (4^t sin^2(pi d)).
    """
    d = s / r - np.arange(2**t) / 2**t
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.sin(np.pi * 2**t * d) ** 2 / (4.0**t * np.sin(np.pi * d) ** 2)
    probs[np.isnan(probs)] = 1.0  # d is an integer: y is the exact peak
    return probs / probs.sum()


def qpe_shots(r, t, shots, rng):
    """Sample what an ideal order-finding circuit would measure: random s, then y."""
    ss = rng.integers(r, size=shots)
    return [rng.choice(2**t, p=qpe_distribution(int(s), r, t)) for s in ss]


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


def test_classical_order_of_one_is_one():
    assert classical_order(1, 15) == 1
    assert classical_order(16, 15) == 1


def test_classical_order_rejects_non_coprime_base():
    with pytest.raises(ValueError):
        classical_order(6, 15)


@pytest.mark.parametrize("N", [1, 0, -5])
def test_classical_order_rejects_modulus_below_two(N):
    # N = 1 used to loop forever: a % 1 is 0 and never becomes 1.
    with pytest.raises(ValueError):
        classical_order(1, N)


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


def test_order_is_below_half_of_N_for_valid_inputs():
    # The fact order_candidates relies on to accept both neighbours of a peak.
    for N in range(3, 400, 2):
        if is_prime_power(N):
            continue
        for a in range(1, N):
            if gcd(a, N) == 1:
                assert 2 * classical_order(a, N) < N, (a, N)


# -------------------------------------------------------------------- convergents


def test_convergents_of_five_sixteenths():
    # 5/16 = [0; 3, 5], convergents 0/1, 1/3, 5/16.
    assert convergents(5, 4, 100) == [(0, 1), (1, 3), (5, 16)]
    assert convergents(5, 4, 10) == [(0, 1), (1, 3)]  # 16 >= N is dropped


def test_convergents_of_three_quarters():
    # 192/256 = 3/4 = [0; 1, 3], convergents 0/1, 1/1, 3/4.
    assert convergents(192, 8, 15) == [(0, 1), (1, 1), (3, 4)]


def test_convergents_of_zero():
    assert convergents(0, 8, 15) == [(0, 1)]


def test_convergents_have_the_defining_properties():
    rng = random.Random(3)
    t = 16
    for _ in range(300):
        y = rng.randrange(2**t)
        cs = convergents(y, t, 2**t + 1)  # bound above 2^t: nothing is dropped
        x = Fraction(y, 2**t)
        # The expansion of a rational number ends at the number itself.
        assert Fraction(*cs[-1]) == x
        for (p0, q0), (p1, q1) in zip(cs, cs[1:]):
            assert abs(p1 * q0 - p0 * q1) == 1  # neighbours; implies lowest terms
            assert q1 >= q0
            assert abs(x - Fraction(p1, q1)) <= abs(x - Fraction(p0, q0))


def test_convergents_return_python_integers():
    # numpy integers would overflow for large t and break pow() downstream.
    cs = convergents(np.int64(95325), 20, 989)
    assert cs == [(0, 1), (1, 11)]
    assert all(type(v) is int for pair in cs for v in pair)


@pytest.mark.parametrize("y", [-1, 256, 1000])
def test_convergents_reject_out_of_range_y(y):
    # Catches a wrong t or a bit-order mistake in the caller early.
    with pytest.raises(ValueError):
        convergents(y, 8, 15)


@pytest.mark.parametrize("N", [15, 21, 35, 143, 989])
def test_ideal_sample_has_s_over_r_as_a_convergent(N):
    # With t = 2n, s/r in lowest terms is a convergent of the nearest y / 2^t.
    t = 2 * N.bit_length()
    for r in range(1, N):
        for s in range(r):
            g = gcd(s, r)
            assert (s // g, r // g) in convergents(ideal_sample(s, r, t), t, N), (r, s)


# --------------------------------------------------------------- order_candidates


@pytest.mark.parametrize("N", [15, 21, 35, 143, 989])
def test_candidates_from_the_nearest_y(N):
    t = 2 * N.bit_length()
    for r in range(1, N):
        for s in range(r):
            assert r // gcd(s, r) in order_candidates(ideal_sample(s, r, t), t, N), (r, s)


@pytest.mark.parametrize("N", [15, 21, 35, 143, 989])
def test_candidates_from_both_neighbours_of_a_peak(N):
    # Real QPE returns floor or ceil of s * 2^t / r, not always the nearer one.
    # Both must work for every order a valid input can have (r < N / 2).
    t = 2 * N.bit_length()
    for r in range(2, (N + 1) // 2):
        for s in range(1, r):
            floor, ceil = s * 2**t // r, -(-s * 2**t // r)
            for y in (floor, ceil):
                assert r // gcd(s, r) in order_candidates(y, t, N), (r, s, y)


def test_candidates_of_zero_and_its_neighbours():
    assert order_candidates(0, 8, 15) == {1}
    # One step away from the s = 0 peak is not on the peak.
    assert order_candidates(1, 8, 15) == set()
    assert order_candidates(255, 8, 15) == set()


def test_no_candidates_far_from_every_peak():
    # 30 lies between the peaks 256/9 = 28.4 and 256/8 = 32, more than one
    # step from both, and from every other fraction with denominator below 15.
    t, N, y = 8, 15, 30
    assert all(abs(y * q - p * 2**t) >= q for q in range(1, N) for p in range(q + 1))
    assert order_candidates(y, t, N) == set()


def test_candidates_are_always_inside_the_window():
    # Brute force over all fractions p/q with q < N, for every y. Not every
    # fraction in the window is a convergent, so this is a subset.
    t, N = 8, 15
    for y in range(2**t):
        in_window = {
            q
            for q in range(1, N)
            for p in range(q + 1)
            if gcd(p, q) == 1 and abs(y * q - p * 2**t) < q
        }
        assert order_candidates(y, t, N) <= in_window, y


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
    assert order_from_samples([y_14], t, a, N) is None
    assert order_from_samples([y_11, y_14], t, a, N) == r


def test_order_from_small_multiples_up_to_the_limit():
    # a = 2 has r = 60 (mod 143). A shot with gcd(s, r) = k gives r / k, and
    # the multiple k * (r / k) recovers r for k <= MAX_MULTIPLE only.
    a, N, r = 2, 143, 60
    assert classical_order(a, N) == r
    t = 2 * N.bit_length()
    for k in range(2, MAX_MULTIPLE + 1):
        assert order_from_samples([ideal_sample(k, r, t)], t, a, N) == r, k
    k = MAX_MULTIPLE + 1  # s = 5 gives the denominator 12, and 5 * 12 is not tried
    assert order_from_samples([ideal_sample(k, r, t)], t, a, N) is None


def test_zero_samples_carry_no_information():
    t = 8
    # r = 2 for a = 14 (mod 15). y = 0 must not lead to it through 2 * 1.
    assert order_from_samples([0], t, 14, 15) is None
    assert order_from_samples([0, 0, 0], t, 7, 15) is None
    assert order_from_samples([], t, 7, 15) is None
    # ... and must not get in the way of a good shot (64/256 = 1/4).
    assert order_from_samples([0, 64], t, 7, 15) == 4


def test_order_one():
    # a = 1 has order 1: its only phase is 0, so every ideal sample is y = 0.
    assert order_from_samples([0], 8, 1, 15) == 1
    assert order_from_samples([0, 0], 20, 1, 989) == 1


def test_shot_off_every_peak_gives_nothing():
    # y = 1 is next to no peak. An earlier version returned 4 here by trying
    # the multiples 2, 3, 4 of the trivial denominator 1.
    assert order_from_samples([1], 8, 7, 15) is None


def test_accepts_numpy_integers():
    # Samplers return numpy integers, which pow() rejects.
    ys = np.array([95325, 74898])
    r = order_from_samples(ys, 20, np.int64(2), np.int64(989))
    assert r == 154 and type(r) is int


def test_out_of_range_sample_is_rejected():
    with pytest.raises(ValueError):
        order_from_samples([64, 256], 8, 7, 15)


def test_junk_cannot_displace_the_true_order():
    # Every accepted candidate is a multiple of r, and the smallest is
    # returned. So once a good shot has put r among the candidates, no amount
    # of junk changes the answer.
    rng = random.Random(7)
    for a, N in [(2, 35), (2, 143), (2, 989)]:
        r = classical_order(a, N)
        t = 2 * N.bit_length()
        ys = [ideal_sample(1, r, t)] + [rng.randrange(2**t) for _ in range(200)]
        assert order_from_samples(ys, t, a, N) == r


def test_random_samples_do_not_reveal_the_order():
    # The post-processing must not find r by classical trial and error.
    # 20 uniformly random shots: an earlier version, which took every
    # convergent of every shot as a candidate, returned r = 154 in 96% of
    # trials. The window in order_candidates brings that below 5%.
    a, N, r = 2, 989, 154
    t = 2 * N.bit_length()
    rng = random.Random(11)
    trials = 300
    hits = sum(
        order_from_samples([rng.randrange(2**t) for _ in range(20)], t, a, N) == r
        for _ in range(trials)
    )
    assert hits < 0.05 * trials


@pytest.mark.parametrize("a, N", [(2, 21), (2, 35), (2, 143)])
def test_order_from_samples_of_the_exact_qpe_distribution(a, N):
    # What the quantum circuit will actually deliver: s uniform, y drawn from
    # the QPE distribution around s * 2^t / r, tails included.
    r = classical_order(a, N)
    t = 2 * N.bit_length()
    rng = np.random.default_rng(5)
    trials = 100
    hits = sum(order_from_samples(qpe_shots(r, t, 5, rng), t, a, N) == r for _ in range(trials))
    assert hits >= 0.95 * trials
